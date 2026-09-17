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
