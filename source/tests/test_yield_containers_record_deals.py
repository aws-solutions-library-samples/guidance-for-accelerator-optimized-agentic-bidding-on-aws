"""The floor and margin containers accept the OpenRTB deals built from deal records.

A deal record (IAB Tech Lab Deals API v1.0) becomes an OpenRTB 2.6 Deal object
with ``id``, ``bidfloor``, ``bidfloorcur``, ``at``, ``guar`` and, where the record
sets a minimum, ``ext.mindealfloor``. These tests send exactly those objects
through both yield containers, directly and through the gRPC servicer's JSON
path, and pin that neither container can reach DynamoDB: their source imports
no AWS SDK, and a request still succeeds with the SDK's entry points patched to
raise.
"""
import ast
import importlib.util
import json
import os
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

from shared.artf_types import Intent, RTBRequest  # noqa: E402
from shared.server import _RTBExtensionPointServicer  # noqa: E402
import containers.yield_optimizer_floor.app as floor_app  # noqa: E402
import containers.yield_optimizer_margin.app as margin_app  # noqa: E402

_LIBRARY = os.path.join(_ROOT, "shared", "deal_library.py")


@pytest.fixture(scope="module")
def library():
    spec = importlib.util.spec_from_file_location("deal_library_for_yield_records", _LIBRARY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load()
    return module


def _app_request():
    return {"app": {"bundle": "com.ridesharer.app", "domain": "ridesharer.example", "cat": ["653"], "cattax": 9}}


@pytest.fixture(scope="module")
def record_deals(library):
    """Every RideSharer deal as the OpenRTB object built from its record."""
    deals = [d.as_openrtb() for d in library.deals_for(_app_request())]
    assert len(deals) == 4
    return deals


@pytest.fixture(scope="module")
def midday_deals(record_deals):
    return [d for d in record_deals if "ext" in d]


def _bid_request(deals):
    return {
        "id": "auction-r",
        "app": _app_request()["app"],
        "imp": [{"id": "imp-1", "banner": {"w": 320, "h": 480}, "bidfloor": 5.0,
                 "pmp": {"private_auction": 0, "deals": deals}}],
    }


def _req(deals):
    return RTBRequest(id="r", bid_request=_bid_request(deals),
                      applicable_intents=["ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"])


@pytest.fixture
def fixed_models(monkeypatch):
    monkeypatch.setattr(floor_app, "_predict_floor", lambda fv, target_variant=None: (0.3, "", ""))
    monkeypatch.setattr(margin_app, "_predict_margin", lambda fv, target_variant=None: (0.2, "", ""))


class _Context:
    def __init__(self):
        self.code = None

    def invocation_metadata(self):
        return ()

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def test_the_record_built_deals_carry_every_field(record_deals, midday_deals):
    for deal in record_deals:
        assert set(deal) >= {"id", "bidfloor", "bidfloorcur", "at", "guar"}
    assert [d["ext"]["mindealfloor"] for d in midday_deals] == [5.0, 5.0]


def test_the_floor_container_reads_the_minimum_from_the_record_deal(record_deals, fixed_models):
    floors = {m.path.rsplit("/", 1)[1]: m.adjust_deal.bidfloor for m in floor_app.mutate(_req(record_deals)).mutations}
    # 12.00 x 0.3 is 3.60. The two deals with a 5.00 minimum stop there; the two
    # without one go to 3.60.
    assert floors == {
        "deal-ridesharer-airport-open-travel": 3.6,
        "deal-ridesharer-airport-open-luggage": 3.6,
        "deal-ridesharer-eats-open-kitchen": 5.0,
        "deal-ridesharer-eats-open-noodles": 5.0,
    }


def test_the_margin_container_ignores_the_extra_record_fields(midday_deals, fixed_models):
    full = margin_app.mutate(_req(midday_deals)).mutations
    stripped = [{k: d[k] for k in ("id", "bidfloor", "at")} for d in midday_deals]
    bare = margin_app.mutate(_req(stripped)).mutations
    assert full and [m.model_dump() for m in full] == [m.model_dump() for m in bare]
    assert all(m.intent == Intent.ADJUST_DEAL_MARGIN for m in full)


@pytest.mark.parametrize("app", [floor_app, margin_app], ids=["floor", "margin"])
def test_the_grpc_json_path_accepts_the_record_deals(app, record_deals, fixed_models):
    payload = _req(record_deals).model_dump(mode="json")
    context = _Context()
    raw = _RTBExtensionPointServicer(app.mutate).GetMutations(json.dumps(payload).encode(), context)
    assert context.code is None
    assert json.loads(raw)["mutations"]


def test_the_grpc_json_path_delivers_the_minimum_to_the_floor_container(midday_deals, fixed_models):
    payload = _req(midday_deals).model_dump(mode="json")
    raw = _RTBExtensionPointServicer(floor_app.mutate).GetMutations(json.dumps(payload).encode(), _Context())
    floors = [m["adjust_deal"]["bidfloor"] for m in json.loads(raw)["mutations"]]
    assert floors == [5.0, 5.0]


# --------------------------------------------------- no table access, by design


def _module_path(name):
    """Source path of a ``shared.*`` or ``container.*`` import, else None."""
    parts = name.split(".")
    if parts[0] == "shared":
        base = os.path.join(_ROOT, "shared", *parts[1:])
    else:
        return None
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


def _reachable_sources(container):
    folder = os.path.join(_ROOT, "containers", container)
    todo = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".py")]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports(path):
            target = _module_path(name)
            if target:
                todo.append(target)
    return seen


@pytest.mark.parametrize("container", ["yield_optimizer_floor", "yield_optimizer_margin"])
def test_the_container_and_what_it_imports_never_import_an_aws_sdk(container):
    offenders = sorted(
        f"{os.path.relpath(path, _ROOT)} imports {name}"
        for path in _reachable_sources(container)
        for name in _imports(path)
        if name.split(".")[0] in ("boto3", "botocore", "aioboto3")
    )
    assert offenders == []


@pytest.mark.parametrize("container", ["yield_optimizer_floor", "yield_optimizer_margin"])
def test_the_container_ships_no_requirements_that_could_add_an_aws_sdk(container):
    assert not os.path.exists(os.path.join(_ROOT, "containers", container, "requirements.txt"))


def test_both_containers_handle_scenario_two_with_the_aws_sdk_unreachable(midday_deals, fixed_models, monkeypatch):
    import boto3

    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("an AWS SDK client was created on the request path")

    monkeypatch.setattr(boto3, "client", refuse)
    monkeypatch.setattr(boto3, "resource", refuse)
    assert floor_app.mutate(_req(midday_deals)).mutations
    assert margin_app.mutate(_req(midday_deals)).mutations
    assert calls == []
