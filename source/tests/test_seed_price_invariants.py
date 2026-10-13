"""Price invariants over the deals seed for the scenario target winners.

Each cohort in source/training/ncf_sample/cohorts.json names the campaign and deal
that should win its scenario once the Deal Scorer activates the deal. The Yield
Optimizer may then move that deal's floor (multiplier in [0.5, 1.5]) and margin
(value in [-0.5, 0.5]). The winner must still clear the moved floor and still beat
the amt simulator seat, or a floor adjustment would remove the bid the scenario is
built around.

Worst case per deal, with f the deal floor and `at` its auction type as the deal
book gives it (OpenRTB 2.6 deal.at, the same codes as Deals API terms.pricetype):
- `at` 1 (first price): CPM margin, applied after the floor mutation: 1.5f + 0.5
- any other `at`: percent margin on the floor-adjusted deal floor: 1.5f x 1.5 = 2.25f

Prices are read from deployment/seed/deals_and_campaigns.json; nothing is restated.
"""
import json
import os
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import containers.yield_optimizer_margin.app as margin_app
from demand.artfhouse.catalog import CampaignCatalog, campaigns_from_records, highest_declared_cpm
from demand.artfhouse.eligibility import evaluate
from shared import artf_applier
from shared.artf_types import (
    AdjustDealPayload,
    IDsPayload,
    Intent,
    Margin,
    Mutation,
    Operation,
)
from shared.deal_library import DealBook, deal_from_record, not_on_request
from shared.yield_exploration import (
    FLOOR_MULTIPLIER_MAX,
    FLOOR_MULTIPLIER_MIN,
    MARGIN_VALUE_MAX,
    MARGIN_VALUE_MIN,
)

_SOURCE = Path(__file__).resolve().parents[1]
SAMPLES = _SOURCE / "frontend-react" / "public" / "samples"
COHORTS_FILE = _SOURCE / "training" / "ncf_sample" / "cohorts.json"

# The amt simulator seat answers at fixed prices per media type. Its prices live in
# the simulator image, not in this repository.
AMT_PRICE = {"banner": 3.25, "video": 12.50}
# Prebid's default price granularity tops out at 20; above it hb_pb is clamped.
TOP_PRICE_BUCKET = 20.0


def _load_seed():
    with open(os.environ["DEALS_SEED_FILE"], encoding="utf-8") as handle:
        return json.load(handle)


SEED = _load_seed()
COHORTS = json.loads(COHORTS_FILE.read_text())
CATALOG = CampaignCatalog(campaigns_from_records(SEED["campaigns"]))
CAMPAIGNS = {c.campaign_id: c for c in CATALOG.all()}
LIBRARY = {r["id"]: deal_from_record(r) for r in SEED["deals"]}
BOOK = DealBook(SEED["deals"])


def _sample(name):
    return json.loads((SAMPLES / name).read_text())


def worst_case_floor(dealfloor: float, at: int) -> float:
    """The highest deal floor the Yield Optimizer can produce for this deal."""
    moved = dealfloor * FLOOR_MULTIPLIER_MAX
    if at == 1:
        return moved + MARGIN_VALUE_MAX
    return moved * (1 + MARGIN_VALUE_MAX)


def adjusted_floor(dealfloor: float, at: int, multiplier: float, margin: float) -> float:
    """The deal floor after one floor mutation and one margin mutation in the yield stage."""
    moved = dealfloor * multiplier
    if at == 1:
        return moved + margin
    return moved + moved * margin


def _winner_cohorts():
    return [c for c in COHORTS["cohorts"] if c["winner"] is not None]


def _imp_floor_for(cohort):
    """The impression floor on the imp the winning deal is activated onto."""
    request = _sample(cohort["sample"])["bid_request"]
    library = BOOK.deals_for(request)
    for imp in request["imp"]:
        if imp.get("pmp") is None:
            continue
        candidates = not_on_request(library, imp["pmp"].get("deals") or [])
        if any(d.id == cohort["winner"]["deal"] for d in candidates):
            return float(imp.get("bidfloor") or 0.0)
    raise AssertionError(f"{cohort['id']}: {cohort['winner']['deal']} is not a library candidate")


# ------------------------------------------------------------ the cohort spec


def test_the_cohort_spec_declares_its_format_and_generator_parameters():
    assert COHORTS["format"] == "ncf-sample-cohorts/1"
    assert COHORTS["generator_seed"] == 20261007
    assert COHORTS["members_per_cohort"] == 60
    assert COHORTS["noise_rate"] == 0.05


def test_the_cohort_spec_carries_the_nine_cohorts():
    assert [c["id"] for c in COHORTS["cohorts"]] == [
        "sports-espn", "finance-ledger", "home-hearth", "parenting-weekly", "auto-cnn",
        "video-streaming", "yield-sports", "ridesharer-airport", "ridesharer-midday",
    ]
    assert len(_winner_cohorts()) == 7


@pytest.mark.parametrize("cohort", COHORTS["cohorts"], ids=lambda c: c["id"])
def test_every_cohort_names_its_sample_user_and_seed_deals(cohort):
    sample = _sample(cohort["sample"])
    assert cohort["scenario_users"] == [sample["bid_request"]["user"]["id"]]
    named = cohort["activate"] + cohort["suppress"] + cohort["keep"] + cohort["not_activated"]
    assert set(named) <= set(LIBRARY), set(named) - set(LIBRARY)
    assert len(named) == len(set(named)), "a deal is listed under two roles"
    offered = {
        d["id"]
        for imp in sample["bid_request"]["imp"]
        for d in (imp.get("pmp") or {}).get("deals") or []
    }
    # Suppress and keep apply to offered deals; activate to library candidates only.
    assert set(cohort["suppress"]) | set(cohort["keep"]) == offered
    assert not set(cohort["activate"]) & offered


@pytest.mark.parametrize("cohort", _winner_cohorts(), ids=lambda c: c["id"])
def test_every_winner_holds_an_activated_deal(cohort):
    campaign = CAMPAIGNS[cohort["winner"]["campaign"]]
    assert cohort["winner"]["deal"] in campaign.deal_ids
    assert cohort["winner"]["deal"] in cohort["activate"]


# --------------------------------------------------------- the price rules


@pytest.mark.parametrize("cohort", _winner_cohorts(), ids=lambda c: c["id"])
def test_every_winner_prices_at_least_one_and_a_half_times_its_deal_floor(cohort):
    campaign = CAMPAIGNS[cohort["winner"]["campaign"]]
    deal = LIBRARY[cohort["winner"]["deal"]]
    assert campaign.declared_cpm >= 1.5 * deal.bidfloor - 1e-9, (
        f"{campaign.campaign_id} {campaign.declared_cpm} < 1.5 x {deal.bidfloor}"
    )


@pytest.mark.parametrize("cohort", _winner_cohorts(), ids=lambda c: c["id"])
def test_every_winner_clears_its_worst_case_yield_floor(cohort):
    campaign = CAMPAIGNS[cohort["winner"]["campaign"]]
    deal = LIBRARY[cohort["winner"]["deal"]]
    worst = worst_case_floor(deal.bidfloor, deal.at)
    assert campaign.declared_cpm >= worst - 1e-9, (
        f"{campaign.campaign_id} {campaign.declared_cpm} < worst case {worst:.4f} "
        f"(floor {deal.bidfloor}, at {deal.at})"
    )


def test_the_worst_case_matches_the_plan_formula_for_each_target_winner():
    # 1.5f + 0.5 where at == 1, 2.25f otherwise.
    got = {
        c["winner"]["deal"]: round(
            worst_case_floor(LIBRARY[c["winner"]["deal"]].bidfloor, LIBRARY[c["winner"]["deal"]].at), 4
        )
        for c in _winner_cohorts()
    }
    expected = {}
    for c in _winner_cohorts():
        deal = LIBRARY[c["winner"]["deal"]]
        f = deal.bidfloor
        expected[deal.id] = round(1.5 * f + 0.5 if deal.at == 1 else 2.25 * f, 4)
    assert got == expected
    assert {d: LIBRARY[d].at for d in got} == {
        d: next(r for r in SEED["deals"] if r["id"] == d)["terms"]["pricetype"] for d in got
    }


@pytest.mark.parametrize("cohort", _winner_cohorts(), ids=lambda c: c["id"])
def test_every_winner_beats_the_amt_seat_and_clears_the_impression_floor(cohort):
    campaign = CAMPAIGNS[cohort["winner"]["campaign"]]
    assert campaign.declared_cpm > AMT_PRICE[campaign.media_type]
    assert campaign.declared_cpm >= _imp_floor_for(cohort)


def test_the_highest_ceiling_stays_under_the_top_price_bucket():
    assert highest_declared_cpm(CATALOG) < TOP_PRICE_BUCKET


@pytest.mark.parametrize("cohort", _winner_cohorts(), ids=lambda c: c["id"])
@settings(max_examples=300, derandomize=True, print_blob=True)
@given(
    multiplier=st.floats(FLOOR_MULTIPLIER_MIN, FLOOR_MULTIPLIER_MAX, allow_nan=False),
    margin=st.floats(MARGIN_VALUE_MIN, MARGIN_VALUE_MAX, allow_nan=False),
)
def test_property_the_winner_survives_any_yield_adjustment(cohort, multiplier, margin):
    """PBT-03: for any floor multiplier and margin the optimizer can emit, the target
    winner still clears the binding floor and still beats the amt seat."""
    campaign = CAMPAIGNS[cohort["winner"]["campaign"]]
    deal = LIBRARY[cohort["winner"]["deal"]]
    binding = max(_imp_floor_for(cohort), adjusted_floor(deal.bidfloor, deal.at, multiplier, margin))
    assert campaign.declared_cpm >= binding - 1e-9
    assert campaign.declared_cpm > AMT_PRICE[campaign.media_type]


# ----------------------- the worst case through the applier and the margin container


def _book_lookup(deal_id):
    deal = BOOK.by_id(deal_id)
    return deal.at if deal is not None else None


def _activated_working_request(cohort):
    """The sample request after the Deal Scorer's activation pair, applied as the orchestrator does."""
    request = _sample(cohort["sample"])["bid_request"]
    deal = LIBRARY[cohort["winner"]["deal"]]
    library = BOOK.deals_for(request)
    imp_id = next(
        imp["id"] for imp in request["imp"]
        if imp.get("pmp") is not None
        and deal.id in {d.id for d in not_on_request(library, imp["pmp"].get("deals") or [])}
    )
    result = artf_applier.apply(request, [
        Mutation(intent=Intent.ACTIVATE_DEALS, op=Operation.ADD, path=f"/imp/{imp_id}",
                 ids=IDsPayload(id=[deal.id])),
        Mutation(intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
                 path=f"/imp/{imp_id}/deals/{deal.id}", adjust_deal=AdjustDealPayload(bidfloor=deal.bidfloor)),
    ], deal_lookup=_book_lookup)
    assert all(d.applied for d in result.dispositions)
    return result.bid_request, imp_id


@pytest.mark.parametrize("cohort", _winner_cohorts(), ids=lambda c: c["id"])
def test_an_activated_winner_deal_carries_its_seed_auction_type(cohort):
    working, imp_id = _activated_working_request(cohort)
    imp = next(i for i in working["imp"] if i["id"] == imp_id)
    deal = next(d for d in imp["pmp"]["deals"] if d["id"] == cohort["winner"]["deal"])
    pricetype = next(r for r in SEED["deals"] if r["id"] == deal["id"])["terms"]["pricetype"]
    assert deal["at"] == pricetype


@pytest.mark.parametrize("cohort", _winner_cohorts(), ids=lambda c: c["id"])
def test_the_yield_worst_case_on_the_activated_deal_matches_the_plan_formula(cohort):
    """The highest floor and margin the Yield Optimizer can emit, applied the way the
    yield stage applies them (floor, then margin on the moved floor), with the margin
    type the margin container picks from the deal's `at`."""
    working, imp_id = _activated_working_request(cohort)
    deal_id = cohort["winner"]["deal"]
    imp = next(i for i in working["imp"] if i["id"] == imp_id)
    deal = next(d for d in imp["pmp"]["deals"] if d["id"] == deal_id)
    f = LIBRARY[deal_id].bidfloor
    calc = margin_app._margin_calculation_type(deal.get("at"))
    result = artf_applier.apply(working, [
        Mutation(intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE, path=f"/imp/{imp_id}/deals/{deal_id}",
                 adjust_deal=AdjustDealPayload(bidfloor=round(f * FLOOR_MULTIPLIER_MAX, 4))),
        Mutation(intent=Intent.ADJUST_DEAL_MARGIN, op=Operation.REPLACE, path=f"/imp/{imp_id}/deals/{deal_id}",
                 adjust_deal=AdjustDealPayload(margin=Margin(value=MARGIN_VALUE_MAX, calculation_type=calc))),
    ])
    assert all(d.applied for d in result.dispositions)
    moved = next(
        d for i in result.bid_request["imp"] if i["id"] == imp_id
        for d in i["pmp"]["deals"] if d["id"] == deal_id
    )["bidfloor"]
    plan = 1.5 * f + 0.5 if LIBRARY[deal_id].at == 1 else 2.25 * f
    assert moved == pytest.approx(plan)
    assert CAMPAIGNS[cohort["winner"]["campaign"]].declared_cpm >= moved - 1e-9


# --------------------------------------------- Courtside's single scenario


def _request_with_library_deals(sample_name):
    """The sample's request with every library candidate placed on its pmp imps."""
    request = json.loads(json.dumps(_sample(sample_name)["bid_request"]))
    library = BOOK.deals_for(request)
    for imp in request.get("imp") or []:
        if imp.get("pmp") is None:
            continue
        offered = imp["pmp"].setdefault("deals", [])
        offered.extend(d.as_openrtb() for d in not_on_request(library, offered))
    return request


def _courtside_eligible(request):
    return any(
        c.campaign.campaign_id == "camp-courtside" and c.eligible
        for imp in request.get("imp") or []
        for c in evaluate(imp, CATALOG)
    )


def test_courtside_is_eligible_only_on_banner_basic():
    eligible = [
        path.name
        for path in sorted(SAMPLES.glob("*.json"))
        if _sample(path.name).get("bid_request", {}).get("imp")
        and _courtside_eligible(_request_with_library_deals(path.name))
    ]
    assert eligible == ["banner-basic.json"]
