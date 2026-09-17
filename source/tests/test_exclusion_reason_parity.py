"""The exclusion vocabulary is shared by two languages, so it needs a shared test.

`ExclusionReason` is emitted by the Python demand endpoint and rendered by name in
the React offers column, which keeps its own copy of the set and its own map of
human wording. Nothing tied the two together, and they drifted:
`MEDIA_TYPE_UNSUPPORTED` was added to the enum and never added to the frontend.

The failure was silent and looked like the opposite of itself. The frontend's
classifier only accepts a reason present in its `HUMAN_REASON` map; anything else
falls through to a catch-all that prints "No offer -- no reason reported". So 12 of
30 excluded campaigns on the isv-ecosystem scenario told the reader that no reason
had been given, for campaigns whose reason the endpoint had reported perfectly well.

This test reads the JavaScript as text. That is deliberate: the alternative is a
Node process from pytest, and the property worth pinning -- "every enum member
appears in the frontend's set and in its wording map" -- is a textual one that a
regex answers without a second runtime.
"""

from __future__ import annotations

import os
import re

import pytest

from demand.artfhouse.exclusion import ExclusionReason

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_CLASSIFIER = os.path.join(
    _ROOT, "frontend-react", "src", "utils", "outcomeClassifier.js"
)


@pytest.fixture(scope="module")
def classifier_source() -> str:
    with open(_CLASSIFIER, encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture(scope="module")
def frontend_exclusion_values(classifier_source: str) -> set[str]:
    """The string values inside the frontend's EXCLUSION object."""
    match = re.search(
        r"export const EXCLUSION = Object\.freeze\(\{(.*?)\}\);",
        classifier_source,
        re.DOTALL,
    )
    assert match, "EXCLUSION not found in outcomeClassifier.js -- has it been renamed?"
    return set(re.findall(r'"([a-z_]+)"', match.group(1)))


def test_every_reason_the_endpoint_emits_is_known_to_the_frontend(frontend_exclusion_values):
    emitted = {r.value for r in ExclusionReason}
    missing = emitted - frontend_exclusion_values
    assert not missing, (
        f"the endpoint emits {sorted(missing)} and the frontend's EXCLUSION does not "
        f"list them, so those campaigns will render 'No offer -- no reason reported' "
        f"despite having reported a reason. Add them to EXCLUSION and to HUMAN_REASON "
        f"in {os.path.relpath(_CLASSIFIER, _ROOT)}."
    )


def test_the_frontend_lists_no_reason_the_endpoint_cannot_emit(frontend_exclusion_values):
    # The other direction matters too: a reason the frontend knows and the endpoint
    # never sends is dead wording that reads as a supported case.
    emitted = {r.value for r in ExclusionReason}
    extra = frontend_exclusion_values - emitted
    assert not extra, (
        f"the frontend lists {sorted(extra)}, which the endpoint cannot emit. Either "
        f"the enum lost a member or the frontend has stale wording."
    )


def test_every_reason_has_human_wording_not_just_membership(classifier_source):
    # Membership in EXCLUSION is not enough. classify() gates on HUMAN_REASON.has(),
    # so a value listed in EXCLUSION but absent from the Map still falls to the
    # catch-all -- which is exactly the shape of the original defect.
    match = re.search(
        r"const HUMAN_REASON = new Map\(\[(.*?)\]\);", classifier_source, re.DOTALL
    )
    assert match, "HUMAN_REASON not found -- has it been renamed?"
    mapped = match.group(1)
    for reason in ExclusionReason:
        # Referenced via the EXCLUSION constant, e.g. [EXCLUSION.BELOW_FLOOR, "..."].
        key = f"EXCLUSION.{reason.name}"
        assert key in mapped, (
            f"{reason.value} has no entry in HUMAN_REASON, so it will render as "
            f"'no reason reported'"
        )


def test_the_enum_docstring_states_the_real_count():
    # The docstring claimed "exactly four members" while the enum had five. A stated
    # count that contradicts the code is worse than no count: it tells a reader the
    # set is closed at a size it is not.
    from demand.artfhouse import exclusion

    words = {4: "four", 5: "five", 6: "six", 7: "seven"}
    count = len(list(ExclusionReason))
    assert words[count] in exclusion.__doc__, (
        f"the module docstring does not say '{words[count]}' but the enum has "
        f"{count} members"
    )
