"""Server-side load generator for the ARTF orchestrator.

Provides endpoints to start, poll, and cancel load tests that exercise
the full mutation pipeline at scale (1K, 100K, or 1M requests staggered
over a 2-second window).

The generator runs in-process using asyncio with a configurable concurrency
pool (asyncio.Semaphore). It calls the internal mutation pipeline directly
via the orchestrator's own container invocation logic — no network hop to
itself.

Results are stored in-memory keyed by test ID with a 5-minute auto-expiry.
Only one load test may run at a time (HTTP 409 if already running).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Literal

import httpx
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from orchestrator.deal_yield_feedback import emit_load_test_deal_yield_outcome
from orchestrator.etl_trigger import SWEEP_DELAY_SECONDS, trigger_etl_sweep
from orchestrator.loadtest_instrumentation import (
    aggregate_run_model_version,
    emit_load_test_outcome,
)
from orchestrator.loadtest_targeting import (
    CanaryNotStagedError,
    CanaryNotSupportedError,
    build_override_headers,
    canary_supported,
    is_canary_staged,
    validate_challenger_target,
)
from shared.artf_types import Metadata, RTBRequest, RTBResponse
from shared.load_test_context import IS_LOAD_TEST_HEADER_NAME


def _shaded_price_from_mutations(mutations) -> float | None:
    """The price a bid-shading container actually chose, or None if it set none.

    Reads the adjust_bid payload of the container's own response (see
    shared/artf_types.py's Mutation.adjust_bid: AdjustBidPayload.price). None is
    a real "this request produced no priced bid" — the caller must not
    substitute a stand-in, since the price is what the synthetic auction outcome
    is scored against.
    """
    for mutation in mutations or []:
        adjust_bid = getattr(mutation, "adjust_bid", None)
        price = getattr(adjust_bid, "price", None) if adjust_bid is not None else None
        if price is not None:
            return float(price)
    return None


def _get_app_deps():
    """Lazy import to avoid circular dependency with orchestrator.app."""
    try:
        from orchestrator.app import CONTAINERS, _call_container_timed, _filter_containers
    except ImportError:
        from container.app import CONTAINERS, _call_container_timed, _filter_containers
    return CONTAINERS, _call_container_timed, _filter_containers


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Model types eligible for target_model_type selection (Governance panel's
# model-type selector today covers these two Triton-backed types plus the
# two rule-based ones — per Q4=B, all 4 stay selectable so a future
# Triton/canary rollout for the rule-based ones needs no API change).
# The two yield types are listed so the Yield Optimizer's genesis (constant-
# output) models can accumulate real, disclosed load-test-origin outcome data --
# without them, ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN have no route to real
# training data at all (a genesis model never emits a mutation on its own, so
# live traffic alone can never produce a signal to learn from either -- see
# deal_yield_feedback.emit_load_test_deal_yield_outcome()).
#
# These are spelled deal_yield_manager_floor/_margin, matching the TRAINING
# model types exactly (see governance_api.py) so a load-test run's recorded
# target_model_type is directly usable as a training target. A single
# "deal_yield_manager" type used to be recorded here, which matched neither
# real training model type and required a separate bridging map to translate
# between them; splitting the containers removed the need for that map.
_TARGET_MODEL_TYPES = (
    "dlrm_bid_shader",
    "widedeep_segment_activator",
    "ncf_deal_manager",
    "metrics_enricher",
    "deal_yield_manager_floor",
    "deal_yield_manager_margin",
)

# Maps a target_model_type to the CONTAINERS registry entry name it
# corresponds to (see orchestrator/app.py's CONTAINERS list — display names
# use hyphens per the deployment-redesign rename, internal keys stay as the
# model-architecture name per that feature's Q B1 decision).
_MODEL_TYPE_TO_CONTAINER_NAME = {
    "dlrm_bid_shader": "dlrm-bid-shader",
    "widedeep_segment_activator": "widedeep-segment-activator",
    "ncf_deal_manager": "ncf-deal-manager",
    "metrics_enricher": "metrics-enricher",
    "deal_yield_manager_floor": "yield-optimizer-floor",
    "deal_yield_manager_margin": "yield-optimizer-margin",
}


class LoadTestRequest(BaseModel):
    preset: Literal["100", "1k", "10k", "100k"]
    seed: int = 42
    duration_s: int = 30  # max duration in seconds; test stops after this even if not all requests sent
    # target_model_type/target_variant: optional (default preserves the
    # pre-existing behavior of not capturing outcome/version data at all).
    # When target_model_type is set, that container's responses are used for
    # BidShadingOutcomeEvent emission (source="load_test") and model_version
    # aggregation; the other containers are still called for fan-out
    # realism but ignored for outcome/version purposes (Q3=A).
    target_model_type: Literal[
        "dlrm_bid_shader", "widedeep_segment_activator",
        "ncf_deal_manager", "metrics_enricher",
        "deal_yield_manager_floor", "deal_yield_manager_margin",
    ] | None = None
    target_variant: Literal["current", "challenger"] = "current"
    # Traffic scenario shaping the synthetic requests (see TrafficProfile).
    # Validated against the registry in start_loadtest (422 on unknown) rather
    # than a Literal here, so the registry stays the single source of truth.
    # Defaults to the broad baseline mix, reproducing the pre-scenario behavior.
    scenario: str = "baseline"


class LoadTestStatus(BaseModel):
    id: str
    state: Literal["running", "complete", "cancelled", "error"]
    preset: str
    total_requests: int
    completed: int
    errors: int
    elapsed_ms: float
    rps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_avg: float
    latency_max: float
    histogram: dict  # {"lt_10ms": int, "10_30ms": int, "30_50ms": int, "gt_50ms": int}
    per_container: list[dict]  # [{name, avg_latency_ms, total_mutations}]
    # Additional stats
    warmup_avg_ms: float = 0.0  # avg latency of first 10% of requests
    steady_state_avg_ms: float = 0.0  # avg latency of last 50% of requests
    scaled_replicas: int = 1  # how many replicas were used during the test
    total_mutations: int = 0  # total mutations produced across all containers
    # FR-2/FR-7 fields (Train-from-Load-Test feature). model_version is the
    # majority/only version observed across target_model_type's requests in
    # this run ("" if no target_model_type was set, or no requests reached
    # it — a real "unknown", never fabricated).
    model_version: str = ""
    target_model_type: str = ""
    target_variant: str = "current"
    # Traffic scenario this run used (see TrafficProfile). Recorded so history
    # and comparisons can show which demand shape a run exercised.
    scenario: str = "baseline"
    canary_supported: bool = False
    canary_staged: bool = False
    # outcome_sample_count: how many BidShadingOutcomeEvents were actually emitted
    # for target_model_type during this run. Used by
    # LoadTestRunEligibilityService (Unit 3) to filter eligible runs without
    # re-deriving eligibility from raw sample data.
    outcome_sample_count: int = 0
    # outcome_samples: the real per-request primary-metric values (revenue
    # per bid: price_paid if won, else 0.0) for target_model_type's emitted
    # outcomes — real per-sample data for ComparisonService's ABEvaluator
    # call (FR-8), not a summary statistic. Capped at _MAX_STORED_SAMPLES
    # per run to keep the DynamoDB item within its size limit; this is a
    # disclosed bound (outcome_sample_count may exceed len(outcome_samples)
    # for large presets), not a silent truncation.
    outcome_samples: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Preset configuration
# ---------------------------------------------------------------------------

PRESET_CONFIG = {
    "100": {"total": 100, "concurrency": 10},
    "1k": {"total": 1_000, "concurrency": 25},
    "10k": {"total": 10_000, "concurrency": 50},
    "100k": {"total": 100_000, "concurrency": 100},
}

# Intents that trigger a full fan-out across all containers. Includes
# ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN so both yield containers are actually
# included in _filter_containers()'s fan-out below -- without these, load test
# traffic never reached either of them at all, regardless of target_model_type
# targeting. Each intent selects exactly one yield container now, so both must
# be listed or one of the two models gets no traffic.
ALL_INTENTS = [
    "ACTIVATE_SEGMENTS",
    "ACTIVATE_DEALS",
    "BID_SHADE",
    "ADD_METRICS",
    "ADD_CIDS",
    "ADJUST_DEAL_FLOOR",
    "ADJUST_DEAL_MARGIN",
]

# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_active_tests: dict[str, LoadTestStatus] = {}
_active_task: asyncio.Task | None = None
_cancel_flags: dict[str, bool] = {}

# Auto-expiry tracking: test_id -> expiry timestamp (monotonic)
_expiry_times: dict[str, float] = {}
_EXPIRY_SECONDS = 300  # 5 minutes

# Shared progress data for SSE streaming.
# The load test runner appends latencies here so the SSE handler can compute
# real-time stats without blocking the runner.
_progress_latencies: dict[str, list[float]] = {}  # test_id -> list of latencies (ms)
_progress_errors: dict[str, int] = {}  # test_id -> error count
_progress_completed: dict[str, int] = {}  # test_id -> completed count
_progress_start_time: dict[str, float] = {}  # test_id -> monotonic start time
_progress_per_container_latencies: dict[str, dict[str, list[float]]] = {}  # test_id -> {name: [latencies]}
_progress_per_container_mutations: dict[str, dict[str, int]] = {}  # test_id -> {name: mutation_count}


def _cleanup_expired() -> None:
    """Remove expired test results from the store."""
    now = time.monotonic()
    expired = [tid for tid, exp in _expiry_times.items() if now > exp]
    for tid in expired:
        _active_tests.pop(tid, None)
        _expiry_times.pop(tid, None)
        _cancel_flags.pop(tid, None)
        _progress_latencies.pop(tid, None)
        _progress_errors.pop(tid, None)
        _progress_completed.pop(tid, None)
        _progress_start_time.pop(tid, None)
        _progress_per_container_latencies.pop(tid, None)
        _progress_per_container_mutations.pop(tid, None)


# ---------------------------------------------------------------------------
# Seeded PRNG payload generator — uses scenario templates with variations
# ---------------------------------------------------------------------------

# A load test's synthetic requests are shaped by a selectable *traffic
# scenario* (TrafficProfile). A scenario is NOT a replay of real observed
# traffic — it is a deterministic synthetic profile that biases which publisher
# domains, IAB content categories, devices, PMP deals and bid-floor ranges the
# generated OpenRTB requests draw from, so a run can exercise the pipeline under
# a recognisable demand shape (a retail surge, a sports primetime window, an
# off-peak lull) rather than one flat uniform mix. Same (seed, scenario) => the
# same request sequence, so runs stay reproducible and comparable.
#
# The containers really process whatever these profiles generate, so the
# resulting latency and mutation behaviour is real for that request mix. The
# downstream win/price/CTR *outcome* model used for training/comparison (see
# loadtest_instrumentation._LOAD_TEST_SCENARIO) is deliberately fixed and is NOT
# changed by the traffic scenario — a scenario shapes the request inputs, never
# a pre-decided result.

# Device user agents keyed by form factor so a profile can skew the device mix
# by repeating entries (weighted random.choice), not just list them.
_USER_AGENTS = {
    "desktop_win": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0",
    "desktop_mac": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) Safari/17.0",
    "mobile_ios": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Mobile/15E148",
    "mobile_android": "Mozilla/5.0 (Linux; Android 14; Pixel 8) Chrome/120.0 Mobile",
    "tablet_ipad": "Mozilla/5.0 (iPad; CPU OS 17_0) AppleWebKit/605.1",
}

_BANNER_SIZES = [
    (728, 90), (300, 250), (160, 600), (320, 50),
    (970, 250), (300, 600), (468, 60), (336, 280),
]


@dataclass(frozen=True)
class TrafficProfile:
    """A selectable synthetic traffic shape for a load-test run.

    Every field is a *pool the generator samples from* (or a numeric range), not
    a fixed value — the scenario biases the distribution of generated requests,
    it does not hand-pick individual ones. ``user_agents`` may repeat an entry
    to skew the device mix via weighted ``random.choice``.
    """

    key: str
    label: str
    description: str
    domains: tuple[str, ...]
    iab_cats: tuple[str, ...]
    user_agents: tuple[str, ...]
    deal_ids: tuple[str, ...]
    # (min, max) uniform ranges for the three price-like fields the templates set.
    bidfloor_range: tuple[float, float]
    deal_floor_range: tuple[float, float]
    bid_price_range: tuple[float, float]


# Real IAB Content Taxonomy tier-1 category IDs are used throughout so the
# category signal the containers see is well-formed:
#   IAB1 Arts&Entertainment, IAB2 Automotive, IAB3 Business, IAB7 Health&Fitness,
#   IAB8 Food&Drink, IAB9 Hobbies&Interests, IAB12 News, IAB13 Personal Finance,
#   IAB17 Sports, IAB19 Technology, IAB20 Travel, IAB22 Shopping.

_BASELINE_PROFILE = TrafficProfile(
    key="baseline",
    label="Typical mixed traffic",
    description=(
        "A broad, balanced mix of publishers, content categories and devices — "
        "the default when no particular event is being modelled."
    ),
    domains=(
        "espn.com", "cnn.com", "techcrunch.com", "nytimes.com", "weather.com",
        "yelp.com", "reddit.com", "amazon.com", "walmart.com", "target.com",
    ),
    iab_cats=("IAB1", "IAB2", "IAB3", "IAB7", "IAB9", "IAB12", "IAB17", "IAB19", "IAB20"),
    user_agents=tuple(_USER_AGENTS.values()),
    deal_ids=(
        "deal-premium-auto", "deal-standard-sports", "deal-luxury-travel",
        "deal-tech-enterprise", "deal-finance-wealth", "deal-health-wellness",
    ),
    bidfloor_range=(1.0, 8.0),
    deal_floor_range=(1.0, 10.0),
    bid_price_range=(3.0, 12.0),
)

_BLACK_FRIDAY_PROFILE = TrafficProfile(
    key="black_friday",
    label="Black Friday — retail surge",
    description=(
        "Weighted toward retail/commerce publishers, shopping and personal-finance "
        "content, mobile devices, and elevated deal floors — the shape of a Black "
        "Friday demand spike. Shapes request content only; not a replay of real "
        "Black Friday traffic."
    ),
    domains=(
        "amazon.com", "walmart.com", "target.com", "bestbuy.com",
        "ebay.com", "etsy.com", "reddit.com",
    ),
    iab_cats=("IAB22", "IAB13", "IAB1", "IAB19"),
    # Mobile-heavy: shoppers on phones. iOS/Android repeated to skew the mix.
    user_agents=(
        _USER_AGENTS["mobile_ios"], _USER_AGENTS["mobile_ios"],
        _USER_AGENTS["mobile_android"], _USER_AGENTS["mobile_android"],
        _USER_AGENTS["desktop_win"], _USER_AGENTS["tablet_ipad"],
    ),
    deal_ids=(
        "deal-retail-doorbuster", "deal-electronics-blowout",
        "deal-premium-auto", "deal-finance-wealth",
    ),
    bidfloor_range=(3.0, 14.0),
    deal_floor_range=(4.0, 16.0),
    bid_price_range=(5.0, 20.0),
)

_NFL_SUNDAY_PROFILE = TrafficProfile(
    key="nfl_sunday",
    label="NFL Sunday — primetime sports",
    description=(
        "Sports publishers and content, automotive and food-and-drink advertisers, "
        "and a second-screen mobile/desktop device mix with elevated floors during "
        "the game window."
    ),
    domains=(
        "espn.com", "cbssports.com", "nfl.com", "foxsports.com",
        "bleacherreport.com", "yahoo.com",
    ),
    iab_cats=("IAB17", "IAB2", "IAB8", "IAB1"),
    user_agents=(
        _USER_AGENTS["mobile_ios"], _USER_AGENTS["mobile_android"],
        _USER_AGENTS["desktop_win"], _USER_AGENTS["desktop_mac"],
        _USER_AGENTS["tablet_ipad"],
    ),
    deal_ids=(
        "deal-sports-primetime", "deal-standard-sports",
        "deal-premium-auto", "deal-food-beverage",
    ),
    bidfloor_range=(2.5, 12.0),
    deal_floor_range=(3.0, 14.0),
    bid_price_range=(4.0, 16.0),
)

_HOLIDAY_PROFILE = TrafficProfile(
    key="holiday_season",
    label="Holiday season — Christmas shopping",
    description=(
        "Broad holiday retail, travel and gifting across shopping, travel and "
        "food content, mobile-leaning, with sustained elevated demand."
    ),
    domains=(
        "amazon.com", "walmart.com", "target.com", "etsy.com",
        "expedia.com", "booking.com", "nytimes.com",
    ),
    iab_cats=("IAB22", "IAB20", "IAB8", "IAB1", "IAB7"),
    user_agents=(
        _USER_AGENTS["mobile_ios"], _USER_AGENTS["mobile_ios"],
        _USER_AGENTS["mobile_android"], _USER_AGENTS["desktop_win"],
        _USER_AGENTS["tablet_ipad"],
    ),
    deal_ids=(
        "deal-holiday-gifting", "deal-luxury-travel",
        "deal-retail-doorbuster", "deal-food-beverage",
    ),
    bidfloor_range=(2.0, 11.0),
    deal_floor_range=(3.0, 13.0),
    bid_price_range=(4.0, 15.0),
)

_LATE_NIGHT_PROFILE = TrafficProfile(
    key="late_night_longtail",
    label="Late-night long-tail",
    description=(
        "Off-peak news and entertainment browsing on desktop and mobile, with "
        "thinner demand and lower floors."
    ),
    domains=(
        "reddit.com", "cnn.com", "nytimes.com", "techcrunch.com",
        "theverge.com", "weather.com", "yelp.com",
    ),
    iab_cats=("IAB1", "IAB12", "IAB19", "IAB9"),
    user_agents=(
        _USER_AGENTS["desktop_win"], _USER_AGENTS["desktop_mac"],
        _USER_AGENTS["mobile_ios"], _USER_AGENTS["mobile_android"],
    ),
    deal_ids=(
        "deal-standard-sports", "deal-tech-enterprise", "deal-health-wellness",
    ),
    bidfloor_range=(0.5, 4.0),
    deal_floor_range=(1.0, 5.0),
    bid_price_range=(1.5, 8.0),
)

# Registry. TRAFFIC_SCENARIO_KEYS is the tuple the API/UI validate against;
# DEFAULT_SCENARIO reproduces the pre-scenario broad uniform mix.
_TRAFFIC_PROFILES: dict[str, TrafficProfile] = {
    p.key: p
    for p in (
        _BASELINE_PROFILE,
        _BLACK_FRIDAY_PROFILE,
        _NFL_SUNDAY_PROFILE,
        _HOLIDAY_PROFILE,
        _LATE_NIGHT_PROFILE,
    )
}
TRAFFIC_SCENARIO_KEYS: tuple[str, ...] = tuple(_TRAFFIC_PROFILES)
DEFAULT_SCENARIO = _BASELINE_PROFILE.key


def get_traffic_profile(scenario: str | None) -> TrafficProfile:
    """Resolve a scenario key to its TrafficProfile.

    Raises KeyError for an unknown key so the route can answer 422 rather than
    silently substituting a different traffic shape than the caller asked for.
    ``None``/empty resolves to the baseline profile (the pre-scenario default).
    """
    if not scenario:
        return _BASELINE_PROFILE
    try:
        return _TRAFFIC_PROFILES[scenario]
    except KeyError as exc:
        raise KeyError(
            f"Unknown load-test scenario '{scenario}'. "
            f"Valid scenarios: {sorted(_TRAFFIC_PROFILES)}."
        ) from exc


def _rand_ifa(rng: random.Random) -> str:
    """A UUID-shaped device advertising ID (four rng draws, matching the
    original inline form so per-template draw order is unchanged)."""
    return (
        f"{rng.randint(10000000, 99999999):08x}-{rng.randint(1000, 9999):04x}-"
        f"4{rng.randint(100, 999):03x}-{rng.randint(1000, 9999):04x}-"
        f"{rng.randint(100000000000, 999999999999):012x}"
    )


def _rand_ip(rng: random.Random) -> str:
    return f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


# Scenario templates that produce real mutations from the containers. Each takes
# the run's TrafficProfile and draws domains/categories/devices/deals/floors
# from it, so the SAME template yields a Black-Friday-shaped or NFL-shaped
# request depending on the scenario.
def _tmpl_full(rng: random.Random, idx: int, p: TrafficProfile) -> dict:
    """Full fan-out (all intents, rich data)."""
    banner = rng.choice(_BANNER_SIZES)
    return {
        "id": f"lt-full-{idx}",
        "tmax": 100,
        "applicable_intents": ALL_INTENTS,
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [
                {
                    "id": f"imp-{idx}-0",
                    "banner": {"w": banner[0], "h": banner[1]},
                    "pos": rng.randint(1, 3),
                    "bidfloor": round(rng.uniform(*p.bidfloor_range), 2),
                    "pmp": {
                        "deals": [
                            {"id": rng.choice(p.deal_ids), "bidfloor": round(rng.uniform(*p.deal_floor_range), 2), "at": 1},
                            {"id": rng.choice(p.deal_ids), "bidfloor": round(rng.uniform(*p.deal_floor_range), 2), "at": 2},
                        ]
                    },
                }
            ],
            "site": {
                "domain": rng.choice(p.domains),
                "cat": [rng.choice(p.iab_cats), rng.choice(p.iab_cats)],
            },
            "user": {
                "id": f"user-{rng.randint(100000, 999999)}",
                "yob": rng.randint(1960, 2002),
                "gender": rng.choice(["M", "F"]),
                "data": [{"segment": [{"id": f"seg-{rng.randint(100, 999)}"}]}],
            },
            "device": {
                "ifa": _rand_ifa(rng),
                "ip": _rand_ip(rng),
                "ua": rng.choice(p.user_agents),
                "geo": {
                    "lat": round(rng.uniform(25.0, 48.0), 4),
                    "lon": round(rng.uniform(-122.0, -73.0), 4),
                    "type": 1,
                    "country": "USA",
                    "region": rng.choice(["CA", "NY", "TX", "FL", "IL"]),
                },
            },
        },
    }


def _tmpl_shade(rng: random.Random, idx: int, p: TrafficProfile) -> dict:
    """Bid shading scenario (with bid_response)."""
    return {
        "id": f"lt-shade-{idx}",
        "tmax": 100,
        "applicable_intents": ["BID_SHADE", "ADD_METRICS"],
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [{"id": f"imp-{idx}-0", "banner": {"w": 728, "h": 90}, "pos": 1, "bidfloor": round(rng.uniform(*p.bidfloor_range), 2)}],
            "site": {"domain": rng.choice(p.domains), "cat": [rng.choice(p.iab_cats)]},
            "user": {"id": f"user-{rng.randint(100000, 999999)}", "yob": rng.randint(1970, 2000)},
            "device": {"ua": rng.choice(p.user_agents), "ip": _rand_ip(rng)},
        },
        "bid_response": {
            "seatbid": [{"bid": [{"id": f"bid-{idx}", "impid": f"imp-{idx}-0", "price": round(rng.uniform(*p.bid_price_range), 2)}]}]
        },
    }


def _tmpl_segment(rng: random.Random, idx: int, p: TrafficProfile) -> dict:
    """Segment activation (behavioral + location)."""
    return {
        "id": f"lt-seg-{idx}",
        "tmax": 100,
        "applicable_intents": ["ACTIVATE_SEGMENTS", "ADD_METRICS"],
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [{"id": f"imp-{idx}-0", "banner": {"w": 300, "h": 250}, "pos": rng.randint(1, 3), "bidfloor": round(rng.uniform(*p.bidfloor_range), 2)}],
            "site": {"domain": rng.choice(p.domains), "cat": [rng.choice(p.iab_cats), rng.choice(p.iab_cats)]},
            "user": {"id": f"user-{rng.randint(100000, 999999)}", "yob": rng.randint(1965, 2000), "gender": rng.choice(["M", "F"])},
            "device": {
                "ifa": _rand_ifa(rng),
                "ip": _rand_ip(rng),
                "ua": rng.choice(p.user_agents),
                "geo": {"lat": round(rng.uniform(25.0, 48.0), 4), "lon": round(rng.uniform(-122.0, -73.0), 4), "type": 1, "country": "USA"},
            },
        },
    }


def _tmpl_identity(rng: random.Random, idx: int, p: TrafficProfile) -> dict:
    """Identity resolution + deals."""
    return {
        "id": f"lt-id-{idx}",
        "tmax": 100,
        "applicable_intents": ["ADD_CIDS", "ACTIVATE_DEALS", "ADD_METRICS"],
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [{"id": f"imp-{idx}-0", "banner": {"w": 320, "h": 50}, "pos": 1, "bidfloor": round(rng.uniform(*p.bidfloor_range), 2),
                     "pmp": {"deals": [{"id": rng.choice(p.deal_ids), "bidfloor": round(rng.uniform(*p.deal_floor_range), 2)}]}}],
            "site": {"domain": rng.choice(p.domains), "cat": [rng.choice(p.iab_cats)]},
            "user": {"id": f"user-{rng.randint(100000, 999999)}", "yob": rng.randint(1970, 1995), "data": [{"segment": [{"id": f"seg-{rng.randint(100, 999)}"}]}]},
            "device": {
                "ifa": _rand_ifa(rng),
                "ip": _rand_ip(rng),
                "ua": rng.choice(p.user_agents),
                "geo": {"country": "USA", "region": rng.choice(["CA", "NY", "TX", "FL", "IL"])},
            },
        },
    }


_SCENARIO_TEMPLATES = (_tmpl_full, _tmpl_shade, _tmpl_segment, _tmpl_identity)


def generate_payload(rng: random.Random, index: int, profile: TrafficProfile | None = None) -> dict:
    """Generate a varied RTBRequest by picking a template and drawing its
    content from the run's traffic ``profile``.

    Uses known-good request structures that produce real mutations from the
    containers. Same (seed, profile) always produces the same sequence. A None
    profile falls back to the baseline (broad uniform) mix.
    """
    profile = profile or _BASELINE_PROFILE
    template = rng.choice(_SCENARIO_TEMPLATES)
    return template(rng, index, profile)


# ---------------------------------------------------------------------------
# Latency statistics computation
# ---------------------------------------------------------------------------

def compute_latency_stats(latencies: list[float]) -> dict:
    """Compute min, max, avg, p50, p95, p99 from a list of latency values.

    Returns a dict with keys: latency_min, latency_max, latency_avg,
    latency_p50, latency_p95, latency_p99.
    """
    if not latencies:
        return {
            "latency_min": 0.0,
            "latency_max": 0.0,
            "latency_avg": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
        }
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "latency_min": round(sorted_lat[0], 3),
        "latency_max": round(sorted_lat[-1], 3),
        "latency_avg": round(sum(sorted_lat) / n, 3),
        "latency_p50": round(sorted_lat[int(n * 0.50)], 3),
        "latency_p95": round(sorted_lat[min(int(n * 0.95), n - 1)], 3),
        "latency_p99": round(sorted_lat[min(int(n * 0.99), n - 1)], 3),
    }


def compute_histogram(latencies: list[float]) -> dict:
    """Compute histogram buckets from latency values (in milliseconds).

    Buckets: lt_10ms, 10_30ms, 30_50ms, gt_50ms.
    Each value is assigned to exactly one bucket.
    """
    lt_10ms = 0
    b_10_30ms = 0
    b_30_50ms = 0
    gt_50ms = 0
    for lat in latencies:
        if lat < 10.0:
            lt_10ms += 1
        elif lat < 30.0:
            b_10_30ms += 1
        elif lat <= 50.0:
            b_30_50ms += 1
        else:
            gt_50ms += 1
    return {
        "lt_10ms": lt_10ms,
        "10_30ms": b_10_30ms,
        "30_50ms": b_30_50ms,
        "gt_50ms": gt_50ms,
    }


# ---------------------------------------------------------------------------
# Async load test runner
# ---------------------------------------------------------------------------

async def _scale_containers(replicas: int) -> None:
    """Scale all ARTF container deployments via Kubernetes API.
    
    Uses boto3 to call EKS, but since we can't easily scale deployments
    from inside a pod without a service account with RBAC, we skip this
    and rely on HPAs for auto-scaling. This is a no-op placeholder.
    """
    # HPAs handle scaling automatically based on CPU pressure.
    # Manual pre-scaling removed — it was causing hangs in the container.
    pass


async def _scale_down_after_delay(delay_s: int = 120) -> None:
    """No-op — HPAs handle scale-down automatically."""
    pass


# ---------------------------------------------------------------------------

async def _run_load_test(
    test_id: str,
    preset: str,
    seed: int,
    duration_s: int,
    target_model_type: str | None = None,
    target_variant: str = "current",
    scenario: str = DEFAULT_SCENARIO,
) -> None:
    """Execute the load test asynchronously.

    Pre-scales containers to handle load, runs the test, then schedules
    a scale-down after 2 minutes.

    Progress data is written to shared dicts so the SSE handler can compute
    real-time stats without blocking the runner.

    When target_model_type is set (FR-1/FR-2/FR-3, Story 1-2), the matching
    container's per-request responses are additionally used to emit real,
    origin-labeled BidShadingOutcomeEvents and to aggregate the run's observed
    model_version — the other containers in the fan-out are unaffected
    (Q3=A). When target_variant="challenger", the target container's calls
    carry the out-of-band challenger-targeting header (Q2=B) — validated
    BEFORE this function is even scheduled, in start_loadtest().
    """
    global _active_task

    CONTAINERS, _call_container_timed, _filter_containers = _get_app_deps()

    # Pre-scale containers based on preset
    config = PRESET_CONFIG[preset]
    scale_replicas = min(5, max(2, config["concurrency"] // 10))
    await _scale_containers(scale_replicas)

    config = PRESET_CONFIG[preset]
    total = config["total"]
    concurrency = config["concurrency"]

    rng = random.Random(seed)

    # Resolve the traffic scenario shaping this run's requests. Unknown keys are
    # rejected before the run starts (start_loadtest); this defensive fallback
    # keeps a direct _run_load_test caller on the baseline mix rather than
    # raising mid-run.
    try:
        profile = get_traffic_profile(scenario)
    except KeyError:
        profile = get_traffic_profile(DEFAULT_SCENARIO)

    # Initialize shared progress data
    _progress_latencies[test_id] = []
    _progress_errors[test_id] = 0
    _progress_completed[test_id] = 0
    _progress_start_time[test_id] = time.monotonic()
    _progress_per_container_latencies[test_id] = {c["name"]: [] for c in CONTAINERS}
    _progress_per_container_mutations[test_id] = {c["name"]: 0 for c in CONTAINERS}

    start_time = _progress_start_time[test_id]
    deadline = start_time + duration_s  # absolute monotonic deadline

    # Pre-generate all payloads for reproducibility, shaped by the scenario.
    payloads = [generate_payload(rng, i, profile) for i in range(total)]

    # Get all containers (full fan-out since applicable_intents includes all)
    active_containers = _filter_containers(ALL_INTENTS)

    # Resolve which CONTAINERS entry (by internal registry name) corresponds
    # to target_model_type, and the load-test-only headers for its calls
    # (BR-4 — only this run's calls to this specific container ever carry
    # the override; other containers/requests never do).
    target_container_name = (
        _MODEL_TYPE_TO_CONTAINER_NAME.get(target_model_type) if target_model_type else None
    )
    target_headers = build_override_headers(target_variant) if target_model_type else {}
    # X-Load-Test is sent on EVERY container call this run makes (unlike
    # target_headers above, which is scoped to only the one targeted
    # container). This is the signal the yield containers' bounded
    # exploration gates on (see shared/load_test_context.py's
    # module docstring) -- exploration must fire for load-test traffic
    # regardless of which container this run happens to be targeting for
    # outcome capture, but must never fire for real bid-serving traffic.
    load_test_headers = {IS_LOAD_TEST_HEADER_NAME: "1"}
    per_request_versions: list[str] = []
    outcome_sample_count = 0
    # Bounded to keep the persisted LoadTestStatus item within DynamoDB's
    # per-item size limit — outcome_sample_count (unbounded) still reports
    # the true total; outcome_samples is disclosed as capped, not silently
    # truncated (see LoadTestStatus.outcome_samples docstring).
    _MAX_STORED_SAMPLES = 2000
    outcome_samples: list[float] = []

    async def _execute_single(request_index: int, payload: dict, client: httpx.AsyncClient) -> None:  # nosemgrep: useless-inner-function
        nonlocal outcome_sample_count
        if _cancel_flags.get(test_id, False):
            return
        # Check deadline within batch to avoid blocking past duration
        if time.monotonic() > deadline:
            return

        payload_bytes = json.dumps(payload).encode()
        timeout_s = 5.0

        req_start = time.monotonic()
        try:
            tasks = []
            for c in active_containers:
                # Every container call in this run carries X-Load-Test: 1
                # (load_test_headers). Only the target_model_type
                # container's calls ALSO carry the target-variant override
                # header (target_headers merged in) — every other
                # container's calls are otherwise called exactly as before
                # (BR-4/Q3=A), just now with the plain load-test signal too.
                headers = dict(load_test_headers)
                if c["name"] == target_container_name:
                    headers.update(target_headers)
                tasks.append(
                    _call_container_timed(client, c, payload, payload_bytes, timeout_s, headers=headers)
                )
            invocations = await asyncio.gather(*tasks)

            req_latency = (time.monotonic() - req_start) * 1000.0
            _progress_latencies[test_id].append(req_latency)

            for inv in invocations:
                _progress_per_container_latencies[test_id][inv.name].append(inv.latency_ms)
                _progress_per_container_mutations[test_id][inv.name] += len(inv.mutations)
                if inv.status == "failed" or inv.status == "timeout":
                    _progress_errors[test_id] += 1
                if (
                    target_container_name
                    and inv.name == target_container_name
                    and inv.status == "ok"
                ):
                    per_request_versions.append(inv.model_version)
                    if target_model_type in (
                        "deal_yield_manager_floor", "deal_yield_manager_margin",
                    ):
                        # A yield container emits one mutation per deal it
                        # actually adjusts, so a single request can still yield
                        # 0..N mutations (N = number of PMP deals) even though
                        # each container now serves only one intent --
                        # reconstruct the RTBRequest/RTBResponse this container
                        # actually saw and let
                        # emit_load_test_deal_yield_outcome() build one
                        # DealYieldOutcomeEvent per adjust_deal mutation.
                        req_model = RTBRequest(**payload)
                        resp_model = RTBResponse(
                            id=req_model.id,
                            mutations=inv.mutations,
                            metadata=Metadata(model_version=inv.model_version),
                        )
                        sample_values = emit_load_test_deal_yield_outcome(
                            req_model, resp_model, run_id=test_id, request_index=request_index,
                            seed=seed,
                        )
                        outcome_sample_count += len(sample_values)
                        for sample_value in sample_values:
                            if len(outcome_samples) < _MAX_STORED_SAMPLES:
                                outcome_samples.append(sample_value)
                    else:
                        # Pass the price this container actually chose, plus the
                        # run's seed, so the synthetic auction outcome responds to
                        # the model instead of being drawn independently of it
                        # (see loadtest_instrumentation.generate_outcome_sample).
                        sample_value = emit_load_test_outcome(
                            test_id, request_index, inv.model_version, target_model_type,
                            seed=seed,
                            shaded_price=_shaded_price_from_mutations(inv.mutations),
                        )
                        # None means the container returned no price for this
                        # request, so there is no model decision to score.
                        if sample_value is not None:
                            outcome_sample_count += 1
                            if len(outcome_samples) < _MAX_STORED_SAMPLES:
                                outcome_samples.append(sample_value)

            _progress_completed[test_id] += 1
        except Exception:
            req_latency = (time.monotonic() - req_start) * 1000.0
            _progress_latencies[test_id].append(req_latency)
            _progress_errors[test_id] += 1
            _progress_completed[test_id] += 1

    # Dispatch in batches with deadline enforcement
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        ) as client:
            batch_size = concurrency
            for i in range(0, total, batch_size):
                # Check deadline and cancel flag before each batch
                if time.monotonic() > deadline:
                    break
                if _cancel_flags.get(test_id, False):
                    break

                batch = payloads[i:i + batch_size]
                await asyncio.gather(*[
                    _execute_single(i + offset, p, client) for offset, p in enumerate(batch)
                ])
    except asyncio.CancelledError:
        pass
    except Exception:
        pass

    elapsed_ms = (time.monotonic() - start_time) * 1000.0

    # Compute final statistics from shared progress data
    latencies = _progress_latencies.get(test_id, [])
    errors = _progress_errors.get(test_id, 0)
    completed = _progress_completed.get(test_id, 0)

    stats = compute_latency_stats(latencies)
    histogram = compute_histogram(latencies)

    # Per-container breakdown
    per_container = []
    for c in CONTAINERS:
        name = c["name"]
        c_lats = _progress_per_container_latencies.get(test_id, {}).get(name, [])
        avg_lat = round(sum(c_lats) / len(c_lats), 3) if c_lats else 0.0
        per_container.append({
            "name": name,
            "avg_latency_ms": avg_lat,
            "total_mutations": _progress_per_container_mutations.get(test_id, {}).get(name, 0),
        })

    # Determine final state
    if _cancel_flags.get(test_id, False):
        state = "cancelled"
    else:
        state = "complete"

    rps = round((completed / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0, 2)

    # Compute warm-up vs steady-state latency
    warmup_count = max(1, len(latencies) // 10)  # first 10%
    steady_count = max(1, len(latencies) // 2)  # last 50%
    warmup_avg = round(sum(latencies[:warmup_count]) / warmup_count, 1) if latencies else 0.0
    steady_avg = round(sum(latencies[-steady_count:]) / steady_count, 1) if latencies else 0.0

    # Total mutations across all containers
    total_muts = sum(c.get("total_mutations", 0) for c in per_container)

    # Aggregate this run's observed model_version from the target
    # container's per-request resolutions (FR-2). "" if target_model_type
    # wasn't set, or no requests reached it — a real "unknown", never
    # fabricated (see aggregate_run_model_version's docstring).
    aggregated_model_version = aggregate_run_model_version(per_request_versions)

    # Update the stored status
    _active_tests[test_id] = LoadTestStatus(
        id=test_id,
        state=state,
        preset=preset,
        total_requests=total,
        completed=completed,
        errors=errors,
        elapsed_ms=round(elapsed_ms, 2),
        rps=rps,
        latency_p50=stats["latency_p50"],
        latency_p95=stats["latency_p95"],
        latency_p99=stats["latency_p99"],
        latency_min=stats["latency_min"],
        latency_avg=stats["latency_avg"],
        latency_max=stats["latency_max"],
        histogram=histogram,
        per_container=per_container,
        warmup_avg_ms=warmup_avg,
        steady_state_avg_ms=steady_avg,
        scaled_replicas=scale_replicas,
        total_mutations=total_muts,
        model_version=aggregated_model_version,
        target_model_type=target_model_type or "",
        target_variant=target_variant,
        scenario=profile.key,
        canary_supported=canary_supported(target_model_type) if target_model_type else False,
        canary_staged=(
            await is_canary_staged(target_model_type)
            if target_model_type and canary_supported(target_model_type)
            else False
        ),
        outcome_sample_count=outcome_sample_count,
        outcome_samples=outcome_samples,
    )

    # Set expiry time
    _expiry_times[test_id] = time.monotonic() + _EXPIRY_SECONDS
    _active_task = None

    # Persist to DynamoDB
    _save_to_dynamodb(test_id, _active_tests[test_id])

    # Schedule an on-demand Glue sweep so this run becomes trainable in
    # minutes instead of waiting up to 6 hours for the next scheduled sweep
    # (see etl_trigger's module docstring). Only worth doing when this run
    # actually captured outcomes for a specific model -- with no samples there
    # is nothing new for the sweep to label.
    if target_model_type and outcome_sample_count > 0:
        asyncio.create_task(
            _trigger_etl_sweep_after_delay(target_model_type, SWEEP_DELAY_SECONDS)
        )

    # Schedule scale-down after 2 minutes
    asyncio.create_task(_scale_down_after_delay(120))


async def _trigger_etl_sweep_after_delay(model_type: str, delay_s: int) -> None:
    """Wait for Firehose to flush this run's outcomes to S3, then start a Glue
    sweep so the run passes the trainability gate.

    The delay is required for correctness, not politeness: sweeping before the
    outcomes land would mark the run trainable without its data being present
    (see etl_trigger's module docstring). Never raises — a failed sweep must
    not surface as an unhandled task exception; the run simply stays
    untrainable until the next scheduled sweep.
    """
    try:
        await asyncio.sleep(delay_s)
        await asyncio.to_thread(trigger_etl_sweep, model_type)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — best-effort background task
        print(f"[loadtest] on-demand ETL sweep failed for {model_type}: {exc}")


# ---------------------------------------------------------------------------
# DynamoDB persistence
# ---------------------------------------------------------------------------

_LOADTEST_TABLE = os.environ.get("LOADTEST_TABLE", "")

# In-memory history fallback (persists across requests within the same pod lifecycle)
_history_cache: list[dict] = []
_MAX_HISTORY_CACHE = 20


def _decimal_default(obj):
    """JSON serializer for Decimal objects returned by DynamoDB."""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_to_dynamodb(test_id: str, status: LoadTestStatus) -> None:
    """Persist load test results to DynamoDB and in-memory cache. Best-effort — doesn't block."""
    from datetime import datetime, timezone
    from decimal import Decimal

    # Always save to in-memory cache (survives DynamoDB failures)
    item = status.model_dump()
    item["timestamp"] = datetime.now(timezone.utc).isoformat()
    _history_cache.insert(0, item)
    if len(_history_cache) > _MAX_HISTORY_CACHE:
        _history_cache.pop()

    if not _LOADTEST_TABLE:
        print(f"[loadtest] LOADTEST_TABLE not set — result saved to in-memory cache only (id={test_id})")
        return
    try:
        import boto3
        dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        table = dynamodb.Table(_LOADTEST_TABLE)
        # Convert floats to Decimal (DynamoDB requirement)
        dynamo_item = json.loads(json.dumps(item), parse_float=Decimal, parse_int=Decimal)
        table.put_item(Item=dynamo_item)
        print(f"[loadtest] Saved to DynamoDB: {test_id}")
    except Exception as e:
        print(f"[loadtest] DynamoDB save failed (table={_LOADTEST_TABLE}): {e}")


def _get_history_from_dynamodb(limit: int = 20) -> list[dict]:
    """Retrieve recent load test results from DynamoDB, falling back to in-memory cache."""
    if not _LOADTEST_TABLE:
        # Return in-memory cache when DynamoDB is not configured
        return _history_cache[:limit]
    try:
        import boto3
        from decimal import Decimal
        dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        table = dynamodb.Table(_LOADTEST_TABLE)
        resp = table.scan(Limit=limit)
        items = resp.get("Items", [])
        # Convert Decimal back to float/int for JSON serialization
        items = json.loads(json.dumps(items, default=_decimal_default))
        # Sort by timestamp descending
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items[:limit]
    except Exception as e:
        print(f"[loadtest] DynamoDB read failed (table={_LOADTEST_TABLE}): {e}")
        # Fall back to in-memory cache
        return _history_cache[:limit]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def start_loadtest(request: Request) -> JSONResponse:
    """POST /v1/loadtest — Start a load test.

    Accepts {preset: "1k"|"100k"|"1m", seed: int, target_model_type?: str,
    target_variant?: "current"|"challenger"}.
    Returns {id: string} with HTTP 202.
    Returns HTTP 409 if a test is already running.
    Returns HTTP 422 for a malformed request, OR for a challenger-targeted
    request when the target model type has no canary infrastructure at all
    ("no canary supported") or no canary currently staged ("no canary
    staged") — reported plainly BEFORE the run starts, never silently
    falling back to testing stable (BR-5/FR-3).
    """
    global _active_task

    _cleanup_expired()

    # Enforce single concurrent test
    if _active_task is not None and not _active_task.done():
        return JSONResponse(
            {"error": "A load test is already running. Cancel it first or wait for completion."},
            status_code=409,
        )

    body = await request.json()
    try:
        req = LoadTestRequest(**body)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    if req.target_model_type and req.target_variant == "challenger":
        try:
            await validate_challenger_target(req.target_model_type)
        except CanaryNotSupportedError as exc:
            return JSONResponse({"error": str(exc), "reason": "no_canary_supported"}, status_code=422)
        except CanaryNotStagedError as exc:
            return JSONResponse({"error": str(exc), "reason": "no_canary_staged"}, status_code=422)

    # Validate the traffic scenario up front so an unknown key is a plain 422,
    # never a silent fallback to a different traffic shape than was requested.
    try:
        get_traffic_profile(req.scenario)
    except KeyError as exc:
        return JSONResponse({"error": str(exc), "reason": "unknown_scenario"}, status_code=422)

    test_id = f"lt-{uuid.uuid4().hex[:12]}"
    config = PRESET_CONFIG[req.preset]

    # Initialize status as running
    _active_tests[test_id] = LoadTestStatus(
        id=test_id,
        state="running",
        preset=req.preset,
        total_requests=config["total"],
        completed=0,
        errors=0,
        elapsed_ms=0.0,
        rps=0.0,
        latency_p50=0.0,
        latency_p95=0.0,
        latency_p99=0.0,
        latency_min=0.0,
        latency_avg=0.0,
        latency_max=0.0,
        histogram={"lt_10ms": 0, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 0},
        per_container=[],
        target_model_type=req.target_model_type or "",
        target_variant=req.target_variant,
        scenario=req.scenario,
    )
    _cancel_flags[test_id] = False

    # Start the async load test
    _active_task = asyncio.create_task(
        _run_load_test(
            test_id, req.preset, req.seed, req.duration_s,
            target_model_type=req.target_model_type,
            target_variant=req.target_variant,
            scenario=req.scenario,
        )
    )

    return JSONResponse({"id": test_id}, status_code=202)


async def get_loadtest(request: Request) -> JSONResponse:
    """GET /v1/loadtest/{id} — Poll final results."""
    _cleanup_expired()

    test_id = request.path_params["id"]
    status = _active_tests.get(test_id)
    if status is None:
        return JSONResponse({"error": "Load test not found"}, status_code=404)

    return JSONResponse(status.model_dump())


async def get_loadtest_history(request: Request) -> JSONResponse:
    """GET /v1/loadtest/history — Retrieve historical load test results from DynamoDB."""
    limit = int(request.query_params.get("limit", "20"))
    history = _get_history_from_dynamodb(limit)
    return JSONResponse({"history": history})


async def cancel_loadtest(request: Request) -> JSONResponse:
    """DELETE /v1/loadtest/{id} — Cancel a running test. Kills the async task immediately."""
    global _active_task
    test_id = request.path_params["id"]
    status = _active_tests.get(test_id)
    if status is None:
        return JSONResponse({"error": "Load test not found"}, status_code=404)

    if status.state != "running":
        return JSONResponse({"error": f"Load test is not running (state: {status.state})"}, status_code=400)

    # Set the cancel flag (cooperative cancellation)
    _cancel_flags[test_id] = True

    # Also forcefully cancel the asyncio task (kills pending HTTP calls)
    if _active_task is not None and not _active_task.done():
        _active_task.cancel()
        _active_task = None

    # Mark as cancelled immediately
    _active_tests[test_id] = LoadTestStatus(
        id=test_id,
        state="cancelled",
        preset=status.preset,
        total_requests=status.total_requests,
        completed=_progress_completed.get(test_id, 0),
        errors=_progress_errors.get(test_id, 0),
        elapsed_ms=round((time.monotonic() - _progress_start_time.get(test_id, time.monotonic())) * 1000, 2),
        rps=0.0,
        latency_p50=0.0,
        latency_p95=0.0,
        latency_p99=0.0,
        latency_min=0.0,
        latency_avg=0.0,
        latency_max=0.0,
        histogram={"lt_10ms": 0, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 0},
        per_container=[],
    )
    _expiry_times[test_id] = time.monotonic() + _EXPIRY_SECONDS

    return JSONResponse({"ok": True, "message": "Load test cancelled and task killed"})


# ---------------------------------------------------------------------------
# SSE streaming handler
# ---------------------------------------------------------------------------

async def _sse_event_generator(test_id: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for a running load test.

    Emits progress events every 500ms with current stats computed from
    the shared progress data. Emits a final 'complete' event when the
    test finishes. The test continues to completion regardless of whether
    the SSE client disconnects.

    Sends a keepalive comment every 15s to prevent CloudFront/ALB from
    closing the connection due to idle timeout (OriginReadTimeout=30s).
    """
    status = _active_tests.get(test_id)
    if status is None:
        return

    total = status.total_requests
    last_event_time = time.monotonic()

    while True:
        # Check if the test has completed
        current_status = _active_tests.get(test_id)
        if current_status and current_status.state != "running":
            # Emit final complete event with full LoadTestStatus
            yield f"event: complete\ndata: {json.dumps(current_status.model_dump())}\n\n"
            return

        # Compute current progress stats from shared data
        latencies = _progress_latencies.get(test_id, [])
        completed = _progress_completed.get(test_id, 0)
        errors = _progress_errors.get(test_id, 0)
        start_time = _progress_start_time.get(test_id)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0 if start_time else 0.0
        rps = round((completed / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0, 2)

        # Compute latency percentiles from accumulated data
        stats = compute_latency_stats(latencies)
        histogram = compute_histogram(latencies)

        progress_data = {
            "completed": completed,
            "total": total,
            "rps": rps,
            "elapsed_ms": round(elapsed_ms, 2),
            "latency_p50": stats["latency_p50"],
            "latency_p95": stats["latency_p95"],
            "latency_p99": stats["latency_p99"],
            "errors": errors,
            "histogram": histogram,
        }

        yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"
        last_event_time = time.monotonic()

        # Wait 500ms before next update
        await asyncio.sleep(0.5)


async def stream_loadtest(request: Request) -> StreamingResponse | JSONResponse:
    """GET /v1/loadtest/{id}/stream — SSE stream of load test progress.

    Returns text/event-stream with:
    - 'progress' events every 500ms containing current stats
    - A final 'complete' event with the full LoadTestStatus summary

    The load test continues to completion even if the client disconnects.
    Returns 404 if the test ID is not found.
    """
    test_id = request.path_params["id"]
    status = _active_tests.get(test_id)
    if status is None:
        return JSONResponse({"error": "Load test not found"}, status_code=404)

    # If the test is already complete, emit the final event immediately
    if status.state != "running":
        async def _completed_generator() -> AsyncGenerator[str, None]:
            yield f"event: complete\ndata: {json.dumps(status.model_dump())}\n\n"

        return StreamingResponse(
            _completed_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return StreamingResponse(
        _sse_event_generator(test_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
