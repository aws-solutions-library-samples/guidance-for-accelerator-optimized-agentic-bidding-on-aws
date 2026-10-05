"""shared/artf_applier.py and shared/artf_stages.py.

The example tests are the Java applier's test vectors (ArtfMutationApplierTest.java)
translated, so the orchestrator's between-stage applier and the Prebid hook's
applier are pinned to the same answers. The property tests cover the two rules
the Java tests state as invariants: every mutation gets exactly one disposition,
and a mutation either applies fully or leaves the request untouched.
"""
from __future__ import annotations

import copy
import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import artf_applier, artf_stages  # noqa: E402
from shared.artf_types import (  # noqa: E402
    AddMetricsPayload,
    AdjustBidPayload,
    AdjustDealPayload,
    IDsPayload,
    Intent,
    Margin,
    MarginCalculationType,
    Metric,
    Mutation,
    Operation,
)


def _request() -> dict:
    return {
        "id": "auction-1",
        "cur": ["USD"],
        "imp": [
            {
                "id": "imp-1",
                "banner": {"w": 300, "h": 250},
                "bidfloor": 2.5,
                "bidfloorcur": "USD",
                "pmp": {"deals": [{"id": "deal-remnant-open", "bidfloor": 1.0, "at": 3}]},
            }
        ],
        "user": {"id": "user-1", "data": [{"id": "dmp", "segment": [{"id": "seg-1"}]}]},
    }


def _activate(deals, imp="imp-1") -> Mutation:
    return Mutation(intent=Intent.ACTIVATE_DEALS, op=Operation.ADD, path=f"/imp/{imp}", ids=IDsPayload(id=deals))


def _suppress(deals, imp="imp-1") -> Mutation:
    return Mutation(intent=Intent.SUPPRESS_DEALS, op=Operation.REMOVE, path=f"/imp/{imp}", ids=IDsPayload(id=deals))


def _floor(deal, value, imp="imp-1") -> Mutation:
    return Mutation(
        intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
        path=f"/imp/{imp}/deals/{deal}", adjust_deal=AdjustDealPayload(bidfloor=value),
    )


def _segments(ids) -> Mutation:
    return Mutation(intent=Intent.ACTIVATE_SEGMENTS, op=Operation.ADD, path="/user/data/segment", ids=IDsPayload(id=ids))


def _metrics(metrics, imp="imp-1") -> Mutation:
    return Mutation(
        intent=Intent.ADD_METRICS, op=Operation.ADD, path=f"/imp/{imp}/metric",
        add_metrics=AddMetricsPayload(metric=[Metric(**m) for m in metrics]),
    )


# ---------------------------------------------------------------------------
# ACTIVATE_DEALS: the defect this Express started from
# ---------------------------------------------------------------------------

def test_activate_deals_adds_the_deal_to_imp_pmp_deals():
    res = artf_applier.apply(_request(), [_activate(["deal-home-premium"])])
    assert res.dispositions[0].applied
    deals = res.bid_request["imp"][0]["pmp"]["deals"]
    assert [d["id"] for d in deals] == ["deal-remnant-open", "deal-home-premium"]
    assert res.dispositions[0].written == ("/bid_request/imp/0/pmp/deals/1",)


def test_activate_deals_does_not_duplicate_a_deal_already_offered():
    res = artf_applier.apply(_request(), [_activate(["deal-remnant-open"])])
    assert res.dispositions[0].applied
    assert len(res.bid_request["imp"][0]["pmp"]["deals"]) == 1


def test_activate_deals_creates_pmp_on_an_impression_without_one():
    req = _request()
    del req["imp"][0]["pmp"]
    res = artf_applier.apply(req, [_activate(["deal-x"])])
    assert res.bid_request["imp"][0]["pmp"]["deals"] == [{"id": "deal-x"}]


def test_activate_then_floor_on_the_activated_deal_lands_on_that_deal():
    """The Deal Scorer's pair: ACTIVATE_DEALS followed by ADJUST_DEAL_FLOOR for the
    library deal. The floor resolves only because the activation preceded it."""
    res = artf_applier.apply(_request(), [_activate(["deal-home-premium"]), _floor("deal-home-premium", 3.0)])
    assert all(d.applied for d in res.dispositions)
    deal = res.bid_request["imp"][0]["pmp"]["deals"][1]
    assert deal == {"id": "deal-home-premium", "bidfloor": 3.0, "bidfloorcur": "USD"}
    assert res.bid_request["imp"][0]["bidfloor"] == 3.0


def test_floor_before_activation_is_rejected_with_a_reason():
    res = artf_applier.apply(_request(), [_floor("deal-home-premium", 3.0), _activate(["deal-home-premium"])])
    assert not res.dispositions[0].applied
    assert "has no deal 'deal-home-premium'" in res.dispositions[0].reason
    assert res.dispositions[1].applied


# ---------------------------------------------------------------------------
# Java vectors
# ---------------------------------------------------------------------------

def test_a_deal_floor_mutation_writes_both_the_floor_and_the_currency():
    req = _request()
    del req["imp"][0]["bidfloorcur"]
    del req["imp"][0]["pmp"]["deals"][0]["at"]
    res = artf_applier.apply(req, [_floor("deal-remnant-open", 4.0)])
    deal = res.bid_request["imp"][0]["pmp"]["deals"][0]
    assert deal["bidfloor"] == 4.0 and deal["bidfloorcur"] == "USD"
    assert res.bid_request["imp"][0]["bidfloorcur"] == "USD"


def test_an_impression_floor_is_never_lowered_by_a_mutation():
    res = artf_applier.apply(_request(), [_floor("deal-remnant-open", 1.5)])
    assert res.dispositions[0].applied
    assert res.bid_request["imp"][0]["pmp"]["deals"][0]["bidfloor"] == 1.5
    assert res.bid_request["imp"][0]["bidfloor"] == 2.5, "2.5 already set, not lowered"


def test_a_non_positive_floor_is_rejected_rather_than_written_and_ignored():
    res = artf_applier.apply(_request(), [_floor("deal-remnant-open", 0.0)])
    assert not res.dispositions[0].applied
    assert "not greater than zero" in res.dispositions[0].reason
    assert res.bid_request == _request()


def test_suppression_marks_the_deal_rather_than_removing_it():
    res = artf_applier.apply(_request(), [_suppress(["deal-remnant-open"])])
    deals = res.bid_request["imp"][0]["pmp"]["deals"]
    assert len(deals) == 1
    assert deals[0]["ext"]["artf"]["suppressed"] is True
    assert res.dispositions[0].written == ("/bid_request/imp/0/pmp/deals/0/ext",)


def test_a_metric_outside_the_openrtb_range_is_rejected_rather_than_clamped():
    res = artf_applier.apply(_request(), [_metrics([{"type": "viewability", "value": 1.5, "vendor": "v"}])])
    assert not res.dispositions[0].applied
    assert "outside the [0.0, 1.0] range" in res.dispositions[0].reason
    assert "metric" not in res.bid_request["imp"][0]


def test_a_metric_inside_the_range_is_applied():
    res = artf_applier.apply(_request(), [_metrics([{"type": "viewability", "value": 0.8, "vendor": "nvidia-artf"}])])
    assert res.bid_request["imp"][0]["metric"] == [{"type": "viewability", "value": 0.8, "vendor": "nvidia-artf"}]
    assert res.dispositions[0].written == ("/bid_request/imp/0/metric/0",)


def test_segment_activation_appends_to_user_data_without_discarding_existing_data():
    res = artf_applier.apply(_request(), [_segments(["461", "460"])])
    data = res.bid_request["user"]["data"]
    assert data[0]["id"] == "dmp"
    assert data[1] == {"name": "artf", "segment": [{"id": "461"}, {"id": "460"}]}
    assert res.dispositions[0].written == ("/bid_request/user/data/1",)


def test_a_mutation_naming_an_unknown_impression_is_rejected():
    res = artf_applier.apply(_request(), [_activate(["d"], imp="nope")])
    assert not res.dispositions[0].applied
    assert "no impression with id 'nope'" in res.dispositions[0].reason


def test_an_unknown_path_is_rejected_with_the_permitted_list():
    m = Mutation(intent=Intent.ADD_METRICS, op=Operation.ADD, path="/site/cat", add_metrics=AddMetricsPayload(metric=[]))
    res = artf_applier.apply(_request(), [m])
    assert not res.dispositions[0].applied
    assert "Permitted:" in res.dispositions[0].reason


def test_an_empty_mutation_list_leaves_the_request_untouched_and_produces_no_dispositions():
    res = artf_applier.apply(_request(), [])
    assert res.bid_request == _request()
    assert res.dispositions == []


def test_inputs_are_not_mutated():
    req = _request()
    snapshot = copy.deepcopy(req)
    artf_applier.apply(req, [_activate(["deal-x"]), _segments(["1"])])
    assert req == snapshot


# ---------------------------------------------------------------------------
# Margin units: a PERCENT value is a fraction, as the margin container emits it
# ---------------------------------------------------------------------------

def _margin(deal, value, calc, imp="imp-1") -> Mutation:
    return Mutation(
        intent=Intent.ADJUST_DEAL_MARGIN, op=Operation.REPLACE, path=f"/imp/{imp}/deals/{deal}",
        adjust_deal=AdjustDealPayload(margin=Margin(value=value, calculation_type=calc)),
    )


def test_a_percent_margin_scales_the_deal_floor_by_the_fraction():
    res = artf_applier.apply(_request(), [_margin("deal-remnant-open", 0.5, MarginCalculationType.PERCENT)])
    assert res.bid_request["imp"][0]["pmp"]["deals"][0]["bidfloor"] == pytest.approx(1.5)


def test_a_cpm_margin_adds_to_the_deal_floor():
    res = artf_applier.apply(_request(), [_margin("deal-remnant-open", 0.75, MarginCalculationType.CPM)])
    assert res.bid_request["imp"][0]["pmp"]["deals"][0]["bidfloor"] == pytest.approx(1.75)


# ---------------------------------------------------------------------------
# BID_SHADE on the response
# ---------------------------------------------------------------------------

def _response() -> dict:
    return {"seatbid": [{"seat": "artfhouse", "bid": [{"id": "bid-1", "impid": "imp-1", "price": 6.35}]}]}


def _shade(price, seat="artfhouse", bid="bid-1") -> Mutation:
    return Mutation(intent=Intent.BID_SHADE, op=Operation.REPLACE, path=f"/seatbid/{seat}/bid/{bid}", adjust_bid=AdjustBidPayload(price=price))


def test_bid_shade_replaces_the_price_on_the_named_bid():
    res = artf_applier.apply(_request(), [_shade(4.2)], _response())
    assert res.dispositions[0].applied
    assert res.bid_response["seatbid"][0]["bid"][0]["price"] == 4.2
    assert res.dispositions[0].written == ("/bid_response/seatbid/0/bid/0/price",)


def test_bid_shade_without_a_bid_response_is_rejected_with_the_reason_the_java_hook_gives():
    res = artf_applier.apply(_request(), [_shade(4.2)], None)
    assert not res.dispositions[0].applied
    assert "carries no bid_response" in res.dispositions[0].reason


def test_bid_shade_on_an_unknown_bid_is_rejected():
    res = artf_applier.apply(_request(), [_shade(4.2, bid="bid-9")], _response())
    assert not res.dispositions[0].applied
    assert res.bid_response == _response()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

_deal_ids = st.sampled_from(["deal-remnant-open", "deal-home-premium", "deal-retail-run", "deal-x"])
_imp_ids = st.sampled_from(["imp-1", "imp-9"])

_mutation = st.one_of(
    st.lists(_deal_ids, min_size=0, max_size=3).flatmap(lambda ds: _imp_ids.map(lambda i: _activate(ds, i))),
    st.lists(_deal_ids, min_size=0, max_size=3).flatmap(lambda ds: _imp_ids.map(lambda i: _suppress(ds, i))),
    st.tuples(_deal_ids, st.floats(min_value=-1, max_value=20, allow_nan=False), _imp_ids).map(lambda t: _floor(*t)),
    st.lists(st.text(min_size=0, max_size=4), max_size=3).map(_segments),
    st.lists(
        st.fixed_dictionaries({"type": st.sampled_from(["viewability", "", "brand_safety"]),
                               "value": st.floats(min_value=-0.5, max_value=1.5, allow_nan=False)}),
        max_size=2,
    ).flatmap(lambda ms: _imp_ids.map(lambda i: _metrics(ms, i))),
    st.floats(min_value=-1, max_value=10, allow_nan=False).map(_shade),
)


@settings(max_examples=150, deadline=None)
@given(st.lists(_mutation, max_size=8))
def test_every_mutation_receives_exactly_one_disposition(mutations):
    res = artf_applier.apply(_request(), mutations, _response())
    assert len(res.dispositions) == len(mutations)
    for m, d in zip(mutations, res.dispositions):
        assert d.path == m.path and d.intent == m.intent
        assert d.applied or (d.reason and d.reason.strip())


@settings(max_examples=150, deadline=None)
@given(_mutation)
def test_a_single_mutation_either_applies_fully_or_leaves_the_request_untouched(m):
    before_req, before_resp = _request(), _response()
    res = artf_applier.apply(before_req, [m], before_resp)
    d = res.dispositions[0]
    if not d.applied:
        assert res.bid_request == before_req and res.bid_response == before_resp
        assert d.written == ()
    else:
        # An applied mutation may write nothing (activating a deal already on
        # the request, suppressing one that is not); whatever it did write
        # must be addressable in the result.
        for ptr in d.written:
            node = {"bid_request": res.bid_request, "bid_response": res.bid_response}
            for part in ptr.split("/")[1:]:
                node = node[int(part)] if isinstance(node, list) else node[part]


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

def test_the_six_built_in_containers_land_in_the_four_stages_in_order():
    assert artf_stages.stage_of({"ACTIVATE_SEGMENTS"}) == 1
    assert artf_stages.stage_of({"ADD_METRICS"}) == 1
    assert artf_stages.stage_of({"ACTIVATE_DEALS", "SUPPRESS_DEALS"}) == 2
    assert artf_stages.stage_of({"ADJUST_DEAL_FLOOR"}) == 3
    assert artf_stages.stage_of({"ADJUST_DEAL_MARGIN"}) == 3
    assert artf_stages.stage_of({"BID_SHADE"}) == 4


def test_stage_of_accepts_names_numeric_strings_and_ints():
    assert artf_stages.stage_of(["activate_deals"]) == 2
    assert artf_stages.stage_of(["2"]) == 2
    assert artf_stages.stage_of([Intent.BID_SHADE]) == 4
    assert artf_stages.stage_of([6]) == 4


def test_a_container_spanning_stages_runs_in_the_earliest_and_unknown_intents_run_first():
    assert artf_stages.stage_of({"ACTIVATE_SEGMENTS", "BID_SHADE"}) == 1
    assert artf_stages.stage_of({"SOMETHING_NEW"}) == 1
    assert artf_stages.stage_of(set()) == 1


def test_group_by_stage_preserves_input_order_within_a_stage_and_omits_empty_stages():
    entries = [("shader", {"BID_SHADE"}), ("segments", {"ACTIVATE_SEGMENTS"}),
               ("deals", {"ACTIVATE_DEALS"}), ("metrics", {"ADD_METRICS"})]
    grouped = artf_stages.group_by_stage(entries, lambda e: e[1])
    assert [(s, [e[0] for e in es]) for s, es in grouped] == [
        (1, ["segments", "metrics"]), (2, ["deals"]), (4, ["shader"]),
    ]
