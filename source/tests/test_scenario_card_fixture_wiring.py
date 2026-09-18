"""Every scenario card and the payload it points at have to agree.

There was already a wiring test, but it covered ONE scenario
(``test_parenting_scenario_fixture.py::TestScenarioCardWiring``). Everything else
was unguarded, so a card could name a payload file that did not exist, tag an
intent the payload never declares, or lose a text field the renderer reads -- and
the only symptom would be a blank card or a silently inert container in the UI.

This asserts the whole set, in both directions: no card without a payload, and no
payload without a card. The second half matters because a deleted scenario leaves
its payload behind in ``public/``, where it still gets built and uploaded.

The card list is read as TEXT rather than imported. It is JSX, so there is no
Python-importable form of it; the same approach is used by
``test_exclusion_reason_parity.py``.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

from shared.artf_types import RTBRequest  # noqa: E402

_SAMPLES = os.path.join(_ROOT, "frontend-react", "public", "samples")
_CARD_SOURCE = os.path.join(
    _ROOT, "frontend-react", "src", "components", "ScenarioCard.jsx"
)

# The three fields the card renders for every scenario. A scenario is a publisher
# bid request, and these are the three things the sell side decides on.
_REQUIRED_TEXT_FIELDS = ("name", "page", "audience", "demand")

# `surface` groups a scenario under one of the two toggles in the picker. The
# payload's `lifecycle` is the ground truth for which surface it really mutates,
# so the two have to agree: a card filed under the wrong surface would appear
# under the wrong toggle with nothing else to catch it.
_LIFECYCLE_SURFACE = {
    "LIFECYCLE_PUBLISHER_BID_REQUEST": "request",
    "LIFECYCLE_DSP_BID_RESPONSE": "response",
}


def _card_source() -> str:
    with open(_CARD_SOURCE, encoding="utf-8") as handle:
        return handle.read()


def _scenario_blocks() -> dict[str, str]:
    """Scenario id to the source text of its entry, sliced at the id boundaries."""
    source = _card_source()
    starts = [(m.group(1), m.start()) for m in re.finditer(r'id: "([^"]+)"', source)]
    assert starts, "no scenarios found in ScenarioCard.jsx"
    blocks: dict[str, str] = {}
    for i, (scenario_id, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(source)
        blocks[scenario_id] = source[start:end]
    return blocks


def _field(block: str, key: str) -> str | None:
    match = re.search(rf'{key}: "((?:[^"\\]|\\.)*)"', block)
    return match.group(1) if match else None


def _tags(block: str) -> set[str]:
    return set(re.findall(r'label: "([A-Z_]+)"', block))


def _payload_file(block: str) -> str | None:
    return _field(block, "file")


BLOCKS = _scenario_blocks()
IDS = sorted(BLOCKS)


def _payload(scenario_id: str) -> dict:
    name = _payload_file(BLOCKS[scenario_id])
    with open(os.path.join(_SAMPLES, name), encoding="utf-8") as handle:
        return json.load(handle)


class TestEveryCardHasAPayload:
    @pytest.mark.parametrize("scenario_id", IDS)
    def test_the_card_names_a_payload_file(self, scenario_id):
        assert _payload_file(BLOCKS[scenario_id]), f"{scenario_id} declares no file"

    @pytest.mark.parametrize("scenario_id", IDS)
    def test_the_payload_file_exists(self, scenario_id):
        name = _payload_file(BLOCKS[scenario_id])
        assert os.path.exists(os.path.join(_SAMPLES, name)), f"{name} is missing"

    @pytest.mark.parametrize("scenario_id", IDS)
    def test_the_payload_is_a_valid_artf_request(self, scenario_id):
        # Constructing the real type is the check. A payload the containers cannot
        # parse would fail at run time in the browser, with no test naming the file.
        request = RTBRequest(**_payload(scenario_id))
        assert request.bid_request is not None
        assert request.applicable_intents


class TestNoOrphanedPayloads:
    def test_every_sample_file_is_referenced_by_a_card(self):
        referenced = {_payload_file(b) for b in BLOCKS.values()}
        on_disk = {f for f in os.listdir(_SAMPLES) if f.endswith(".json")}
        orphans = on_disk - referenced
        assert not orphans, (
            f"payloads no card points at: {sorted(orphans)}. A removed scenario "
            "leaves its payload in public/, where it is still built and uploaded."
        )


class TestTheCardDescribesWhatThePayloadDeclares:
    @pytest.mark.parametrize("scenario_id", IDS)
    def test_the_tags_are_exactly_the_declared_intents(self, scenario_id):
        payload = _payload(scenario_id)
        assert _tags(BLOCKS[scenario_id]) == set(payload["applicable_intents"])

    @pytest.mark.parametrize("scenario_id", IDS)
    @pytest.mark.parametrize("field", _REQUIRED_TEXT_FIELDS)
    def test_the_rendered_text_fields_are_present_and_not_empty(self, scenario_id, field):
        value = _field(BLOCKS[scenario_id], field)
        assert value, f"{scenario_id} has no {field}"
        assert len(value) > 20, f"{scenario_id}'s {field} is a stub"

    def test_no_scenario_still_carries_the_old_single_desc_field(self):
        # `desc` was replaced by page/audience/demand. It was left behind once
        # already, and the renderer went on reading it -- so every card showed
        # nothing where its description belonged.
        stale = [sid for sid, block in BLOCKS.items() if _field(block, "desc")]
        assert not stale, f"still carrying desc: {stale}"

    def test_nothing_renders_scenario_desc_any_more(self):
        source = _card_source()
        assert "scenario.desc" not in source


class TestTheBidFloorTunerDoesNotRewriteTheScenario:
    """The floor slider's starting value must be the payload's own floor.

    `applyTunerToPayload` WRITES the slider's value onto `imp[0].bidfloor` on every
    submit. So a default that does not match the payload does not merely mislabel
    the floor — it changes the request. With a shared 1.5 every scenario ran at 1.5:
    home-lifestyle's $2.50 (which is what turns Openfield away), video-deals' $8.00,
    the $0.10 that gives the shader headroom. The card described one auction and the
    orchestrator received another.
    """

    @pytest.mark.parametrize("scenario_id", IDS)
    def test_the_card_declares_a_bid_floor_default(self, scenario_id):
        block = BLOCKS[scenario_id]
        assert re.search(r"defaults: \{[^}]*bidFloor:", block), (
            f"{scenario_id} has no defaults.bidFloor, so it would submit at the "
            f"shared default instead of its own floor"
        )

    @pytest.mark.parametrize("scenario_id", IDS)
    def test_it_equals_the_payload_floor(self, scenario_id):
        block = BLOCKS[scenario_id]
        declared = float(re.search(r"defaults: \{[^}]*bidFloor: ([\d.]+)", block).group(1))
        actual = float(_payload(scenario_id)["bid_request"]["imp"][0]["bidfloor"])
        assert declared == actual, (
            f"{scenario_id}: card starts the floor slider at {declared} but the "
            f"payload declares {actual}"
        )

    @pytest.mark.parametrize("scenario_id", IDS)
    def test_the_slider_can_represent_that_floor(self, scenario_id):
        # A range input clamps out-of-range values and snaps off-grid ones, so a
        # floor outside [min, max] or off the step grid would be silently moved.
        source = _card_source()
        bounds = re.search(
            r'type="range" min="([\d.]+)" max="([\d.]+)" step="([\d.]+)" value={bidFloor}',
            source,
        )
        assert bounds, "could not find the bid-floor range input"
        low, high, step = (float(g) for g in bounds.groups())

        floor = float(_payload(scenario_id)["bid_request"]["imp"][0]["bidfloor"])
        assert low <= floor <= high, f"{floor} is outside the slider's [{low}, {high}]"
        steps = (floor - low) / step
        assert abs(steps - round(steps)) < 1e-6, (
            f"{floor} is not on the slider's {step} grid from {low}, so it would snap"
        )


class TestTheSurfaceMatchesTheLifecycle:
    """Which toggle a scenario appears under, against what it actually mutates."""

    @pytest.mark.parametrize("scenario_id", IDS)
    def test_every_card_declares_a_surface(self, scenario_id):
        block = BLOCKS[scenario_id]
        match = re.search(r"surface: SURFACE_(REQUEST|RESPONSE)", block)
        assert match, f"{scenario_id} declares no surface"

    @pytest.mark.parametrize("scenario_id", IDS)
    def test_the_surface_agrees_with_the_payload_lifecycle(self, scenario_id):
        block = BLOCKS[scenario_id]
        declared = re.search(r"surface: SURFACE_(REQUEST|RESPONSE)", block).group(1).lower()
        lifecycle = _payload(scenario_id).get("lifecycle")
        assert lifecycle in _LIFECYCLE_SURFACE, f"{scenario_id} has lifecycle {lifecycle!r}"
        assert declared == _LIFECYCLE_SURFACE[lifecycle], (
            f"{scenario_id} is filed as {declared} but its payload declares {lifecycle}"
        )

    def test_both_surfaces_have_scenarios(self):
        # An empty toggle would render a dropdown with no options and no card.
        declared = [
            re.search(r"surface: SURFACE_(REQUEST|RESPONSE)", b).group(1).lower()
            for b in BLOCKS.values()
        ]
        assert "request" in declared
        assert "response" in declared


class TestResponseScenariosCanActuallyBeShaded:
    """A response scenario without a bid response produces nothing at all.

    ``containers/dlrm_bid_shader/app.py`` returns an empty mutation list the moment
    ``req.bid_response`` is falsy, and walks ``seatbid[].bid[]`` for prices. A
    payload missing either would run, succeed, and silently do nothing — which is
    indistinguishable from the deliberate "model declines to act" scenario.
    """

    @staticmethod
    def _response_ids():
        return [
            sid for sid, block in BLOCKS.items()
            if "surface: SURFACE_RESPONSE" in block
        ]

    def test_there_are_response_scenarios_to_check(self):
        assert self._response_ids(), "no response-surface scenarios found"

    def test_each_carries_a_bid_response_with_priced_bids(self):
        for scenario_id in self._response_ids():
            payload = _payload(scenario_id)
            seatbids = (payload.get("bid_response") or {}).get("seatbid") or []
            assert seatbids, f"{scenario_id} has no bid_response.seatbid"
            bids = [b for s in seatbids for b in (s.get("bid") or [])]
            assert bids, f"{scenario_id} has no bids to price"
            for bid in bids:
                # The shader skips a bid priced at or below zero outright.
                assert bid.get("price", 0) > 0, f"{scenario_id}: {bid.get('id')} has no price"

    def test_every_bid_points_at_an_impression_that_exists(self):
        # The shader looks the floor up by matching bid.impid to imp.id. A bid
        # naming an impression that is not there gets floor 0.0 silently, so the
        # floor-clamped scenario would stop clamping with nothing to show why.
        for scenario_id in self._response_ids():
            payload = _payload(scenario_id)
            imp_ids = {i.get("id") for i in payload["bid_request"].get("imp") or []}
            for seat in payload["bid_response"]["seatbid"]:
                for bid in seat.get("bid") or []:
                    assert bid.get("impid") in imp_ids, (
                        f"{scenario_id}: bid {bid.get('id')} names impid "
                        f"{bid.get('impid')!r}, which is not in the request"
                    )

    def test_each_declares_bid_shade(self):
        for scenario_id in self._response_ids():
            assert "BID_SHADE" in _payload(scenario_id)["applicable_intents"]
