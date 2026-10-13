"""The verbose-trace walk in prebid/verify_auction.py.

The staged fan-out check reads the ARTF hook's analytics tags out of PBS-Java's
module trace. The walk has to match PBS-Java's serialised shape, which is nested
five levels deep with lower-cased property names, and it has to ignore other
modules' tags. These tests pin that against a fixture shaped like the enricher's
output rather than against a live Prebid.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PREBID = Path(__file__).resolve().parents[1] / "prebid"
if str(_PREBID) not in sys.path:
    sys.path.insert(0, str(_PREBID))

from verify_auction import ARTF_MODULE_CODE, artf_activities  # noqa: E402


def _invocation(module: str, activities: list[dict]) -> dict:
    return {
        "hookid": {"module-code": module, "hook-impl-code": f"{module}-processed-auction-request"},
        "executiontime": 12,
        "status": "success",
        "action": "update",
        "analyticstags": {"activities": activities},
    }


def _trace(*invocations: dict) -> dict:
    return {
        "id": "r",
        "seatbid": [],
        "ext": {
            "prebid": {
                "modules": {
                    "trace": {
                        "stages": [
                            {
                                "stage": "processed-auction-request",
                                "outcomes": [
                                    {
                                        "entity": "auction-request",
                                        "groups": [{"invocationresults": list(invocations)}],
                                    }
                                ],
                            }
                        ]
                    }
                }
            }
        },
    }


MUTATIONS = {
    "name": "artf-mutations",
    "status": "success",
    "results": [
        {
            "status": "success",
            "values": {
                "applied": 3,
                "rejected": 1,
                "rejections": [
                    {
                        "intent": "ADJUST_DEAL_FLOOR",
                        "path": "/imp/imp-1/deals/deal-x",
                        "reason": "impression 'imp-1' has no deal 'deal-x'",
                    }
                ],
            },
        }
    ],
}


def test_finds_the_hooks_activities_in_a_verbose_trace():
    response = _trace(_invocation(ARTF_MODULE_CODE, [{"name": "artf-extension-point"}, MUTATIONS]))
    names = [a["name"] for a in artf_activities(response)]
    assert names == ["artf-extension-point", "artf-mutations"]


def test_ignores_other_modules_tags():
    response = _trace(
        _invocation("some-other-module", [{"name": "artf-mutations", "results": []}]),
        _invocation(ARTF_MODULE_CODE, [MUTATIONS]),
    )
    found = artf_activities(response)
    assert len(found) == 1
    assert found[0]["results"][0]["values"]["rejections"][0]["intent"] == "ADJUST_DEAL_FLOOR"


def test_returns_nothing_without_a_trace():
    assert artf_activities({"id": "r", "seatbid": []}) == []
    assert artf_activities({"id": "r", "ext": {"prebid": {}}}) == []
    assert artf_activities("not a dict") == []
    assert artf_activities(None) == []


def test_accepts_camel_cased_keys_too():
    response = _trace(
        {
            "hookId": {"moduleCode": ARTF_MODULE_CODE},
            "analyticsTags": {"activities": [MUTATIONS]},
        }
    )
    # The group key is camel-cased as well in this variant.
    group = response["ext"]["prebid"]["modules"]["trace"]["stages"][0]["outcomes"][0]["groups"][0]
    group["invocationResults"] = group.pop("invocationresults")
    assert [a["name"] for a in artf_activities(response)] == ["artf-mutations"]
