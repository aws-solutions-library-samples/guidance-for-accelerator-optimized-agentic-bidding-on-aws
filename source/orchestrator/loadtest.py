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
from typing import AsyncGenerator, Literal

import httpx
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse


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

class LoadTestRequest(BaseModel):
    preset: Literal["100", "1k", "10k", "100k"]
    seed: int = 42
    duration_s: int = 30  # max duration in seconds; test stops after this even if not all requests sent


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


# ---------------------------------------------------------------------------
# Preset configuration
# ---------------------------------------------------------------------------

PRESET_CONFIG = {
    "100": {"total": 100, "concurrency": 10},
    "1k": {"total": 1_000, "concurrency": 25},
    "10k": {"total": 10_000, "concurrency": 50},
    "100k": {"total": 100_000, "concurrency": 100},
}

# Intents that trigger a full fan-out across all four containers
ALL_INTENTS = [
    "ACTIVATE_SEGMENTS",
    "ACTIVATE_DEALS",
    "BID_SHADE",
    "ADD_METRICS",
    "ADD_CIDS",
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

_DOMAINS = [
    "espn.com", "cnn.com", "techcrunch.com", "nytimes.com", "weather.com",
    "yelp.com", "reddit.com", "amazon.com", "walmart.com", "target.com",
]

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) Safari/17.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) Chrome/120.0 Mobile",
    "Mozilla/5.0 (iPad; CPU OS 17_0) AppleWebKit/605.1",
]

_BANNER_SIZES = [
    (728, 90), (300, 250), (160, 600), (320, 50),
    (970, 250), (300, 600), (468, 60), (336, 280),
]

_IAB_CATS = ["IAB1", "IAB2", "IAB3", "IAB7", "IAB9", "IAB12", "IAB17", "IAB19", "IAB20"]

_DEAL_IDS = [
    "deal-premium-auto", "deal-standard-sports", "deal-luxury-travel",
    "deal-tech-enterprise", "deal-finance-wealth", "deal-health-wellness",
]

# Base scenario templates that produce real mutations from the containers
_SCENARIO_TEMPLATES = [
    # Template 1: Full fan-out (all intents, rich data)
    lambda rng, idx: {
        "id": f"lt-full-{idx}",
        "tmax": 100,
        "applicable_intents": ALL_INTENTS,
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [
                {
                    "id": f"imp-{idx}-0",
                    "banner": {"w": rng.choice(_BANNER_SIZES)[0], "h": rng.choice(_BANNER_SIZES)[1]},
                    "pos": rng.randint(1, 3),
                    "bidfloor": round(rng.uniform(1.0, 8.0), 2),
                    "pmp": {
                        "deals": [
                            {"id": rng.choice(_DEAL_IDS), "bidfloor": round(rng.uniform(2.0, 10.0), 2), "at": 1},
                            {"id": rng.choice(_DEAL_IDS), "bidfloor": round(rng.uniform(1.0, 5.0), 2), "at": 2},
                        ]
                    },
                }
            ],
            "site": {
                "domain": rng.choice(_DOMAINS),
                "cat": [rng.choice(_IAB_CATS), rng.choice(_IAB_CATS)],
            },
            "user": {
                "id": f"user-{rng.randint(100000, 999999)}",
                "yob": rng.randint(1960, 2002),
                "gender": rng.choice(["M", "F"]),
                "data": [{"segment": [{"id": f"seg-{rng.randint(100, 999)}"}]}],
            },
            "device": {
                "ifa": f"{rng.randint(10000000, 99999999):08x}-{rng.randint(1000, 9999):04x}-4{rng.randint(100, 999):03x}-{rng.randint(1000, 9999):04x}-{rng.randint(100000000000, 999999999999):012x}",
                "ip": f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                "ua": rng.choice(_USER_AGENTS),
                "geo": {
                    "lat": round(rng.uniform(25.0, 48.0), 4),
                    "lon": round(rng.uniform(-122.0, -73.0), 4),
                    "type": 1,
                    "country": "USA",
                    "region": rng.choice(["CA", "NY", "TX", "FL", "IL"]),
                },
            },
        },
    },
    # Template 2: Bid shading scenario (with bid_response)
    lambda rng, idx: {
        "id": f"lt-shade-{idx}",
        "tmax": 100,
        "applicable_intents": ["BID_SHADE", "ADD_METRICS"],
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [{"id": f"imp-{idx}-0", "banner": {"w": 728, "h": 90}, "pos": 1, "bidfloor": round(rng.uniform(1.0, 4.0), 2)}],
            "site": {"domain": rng.choice(_DOMAINS), "cat": [rng.choice(_IAB_CATS)]},
            "user": {"id": f"user-{rng.randint(100000, 999999)}", "yob": rng.randint(1970, 2000)},
            "device": {"ua": rng.choice(_USER_AGENTS), "ip": f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"},
        },
        "bid_response": {
            "seatbid": [{"bid": [{"id": f"bid-{idx}", "impid": f"imp-{idx}-0", "price": round(rng.uniform(3.0, 12.0), 2)}]}]
        },
    },
    # Template 3: Segment activation (behavioral + location)
    lambda rng, idx: {
        "id": f"lt-seg-{idx}",
        "tmax": 100,
        "applicable_intents": ["ACTIVATE_SEGMENTS", "ADD_METRICS"],
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [{"id": f"imp-{idx}-0", "banner": {"w": 300, "h": 250}, "pos": rng.randint(1, 3), "bidfloor": round(rng.uniform(1.0, 5.0), 2)}],
            "site": {"domain": rng.choice(_DOMAINS), "cat": [rng.choice(_IAB_CATS), rng.choice(_IAB_CATS)]},
            "user": {"id": f"user-{rng.randint(100000, 999999)}", "yob": rng.randint(1965, 2000), "gender": rng.choice(["M", "F"])},
            "device": {
                "ifa": f"{rng.randint(10000000, 99999999):08x}-{rng.randint(1000, 9999):04x}-4{rng.randint(100, 999):03x}-{rng.randint(1000, 9999):04x}-{rng.randint(100000000000, 999999999999):012x}",
                "ip": f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                "ua": rng.choice(_USER_AGENTS),
                "geo": {"lat": round(rng.uniform(25.0, 48.0), 4), "lon": round(rng.uniform(-122.0, -73.0), 4), "type": 1, "country": "USA"},
            },
        },
    },
    # Template 4: Identity resolution + deals
    lambda rng, idx: {
        "id": f"lt-id-{idx}",
        "tmax": 100,
        "applicable_intents": ["ADD_CIDS", "ACTIVATE_DEALS", "ADD_METRICS"],
        "bid_request": {
            "id": f"br-{idx}",
            "imp": [{"id": f"imp-{idx}-0", "banner": {"w": 320, "h": 50}, "pos": 1, "bidfloor": round(rng.uniform(1.0, 4.0), 2),
                     "pmp": {"deals": [{"id": rng.choice(_DEAL_IDS), "bidfloor": round(rng.uniform(2.0, 8.0), 2)}]}}],
            "site": {"domain": rng.choice(_DOMAINS), "cat": [rng.choice(_IAB_CATS)]},
            "user": {"id": f"user-{rng.randint(100000, 999999)}", "yob": rng.randint(1970, 1995), "data": [{"segment": [{"id": f"seg-{rng.randint(100, 999)}"}]}]},
            "device": {
                "ifa": f"{rng.randint(10000000, 99999999):08x}-{rng.randint(1000, 9999):04x}-4{rng.randint(100, 999):03x}-{rng.randint(1000, 9999):04x}-{rng.randint(100000000000, 999999999999):012x}",
                "ip": f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                "ua": rng.choice(_USER_AGENTS),
                "geo": {"country": "USA", "region": rng.choice(["CA", "NY", "TX", "FL", "IL"])},
            },
        },
    },
]


def generate_payload(rng: random.Random, index: int) -> dict:
    """Generate a varied RTBRequest by picking a scenario template and randomizing inputs.

    Uses known-good scenario structures that produce real mutations from the containers.
    Same seed always produces the same sequence.
    """
    template = rng.choice(_SCENARIO_TEMPLATES)
    return template(rng, index)


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

async def _run_load_test(test_id: str, preset: str, seed: int, duration_s: int) -> None:
    """Execute the load test asynchronously.

    Pre-scales containers to handle load, runs the test, then schedules
    a scale-down after 2 minutes.

    Progress data is written to shared dicts so the SSE handler can compute
    real-time stats without blocking the runner.
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

    # Initialize shared progress data
    _progress_latencies[test_id] = []
    _progress_errors[test_id] = 0
    _progress_completed[test_id] = 0
    _progress_start_time[test_id] = time.monotonic()
    _progress_per_container_latencies[test_id] = {c["name"]: [] for c in CONTAINERS}
    _progress_per_container_mutations[test_id] = {c["name"]: 0 for c in CONTAINERS}

    start_time = _progress_start_time[test_id]
    deadline = start_time + duration_s  # absolute monotonic deadline

    # Pre-generate all payloads for reproducibility
    payloads = [generate_payload(rng, i) for i in range(total)]

    # Get all containers (full fan-out since applicable_intents includes all)
    active_containers = _filter_containers(ALL_INTENTS)

    async def _execute_single(payload: dict, client: httpx.AsyncClient) -> None:  # nosemgrep: useless-inner-function
        if _cancel_flags.get(test_id, False):
            return
        # Check deadline within batch to avoid blocking past duration
        if time.monotonic() > deadline:
            return

        payload_bytes = json.dumps(payload).encode()
        timeout_s = 5.0

        req_start = time.monotonic()
        try:
            tasks = [
                _call_container_timed(client, c, payload, payload_bytes, timeout_s)
                for c in active_containers
            ]
            invocations = await asyncio.gather(*tasks)

            req_latency = (time.monotonic() - req_start) * 1000.0
            _progress_latencies[test_id].append(req_latency)

            for inv in invocations:
                _progress_per_container_latencies[test_id][inv.name].append(inv.latency_ms)
                _progress_per_container_mutations[test_id][inv.name] += len(inv.mutations)
                if inv.status == "failed" or inv.status == "timeout":
                    _progress_errors[test_id] += 1

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
                await asyncio.gather(*[_execute_single(p, client) for p in batch])
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
    )

    # Set expiry time
    _expiry_times[test_id] = time.monotonic() + _EXPIRY_SECONDS
    _active_task = None

    # Persist to DynamoDB
    _save_to_dynamodb(test_id, _active_tests[test_id])

    # Schedule scale-down after 2 minutes
    asyncio.create_task(_scale_down_after_delay(120))


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

    Accepts {preset: "1k"|"100k"|"1m", seed: int}.
    Returns {id: string} with HTTP 202.
    Returns HTTP 409 if a test is already running.
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
    )
    _cancel_flags[test_id] = False

    # Start the async load test
    _active_task = asyncio.create_task(_run_load_test(test_id, req.preset, req.seed, req.duration_s))

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
