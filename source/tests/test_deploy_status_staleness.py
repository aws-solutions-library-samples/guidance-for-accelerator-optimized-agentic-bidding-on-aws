"""Phase 1 marker and Phase 3 Deal Scorer engine checks in deployment/scripts/deploy_status.py.

deploy.sh passes --expect-deals-seed-sha and --expect-ncf-fingerprint (what the local
checkout would deploy). Phase 1 compares them with deploy-markers/phase1.json in the
model bucket; Phase 3 compares the fingerprint with the source-fingerprint metadata
on triton-models/ncf_deal_manager_stable/1/model.plan. A difference is `stale`, which
is not ok, so the gate runs the phase. The S3 client is a stub that records every
call, so these tests also pin that the probe only reads.
"""
import io
import json
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

_SCRIPTS = Path(__file__).resolve().parents[2] / "deployment" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import deploy_status  # noqa: E402
from deploy_status import (  # noqa: E402
    FAILED,
    IN_PROGRESS,
    MISSING,
    NCF_ENGINE_KEY,
    OK,
    PHASE1_MARKER_KEY,
    STALE,
    UNKNOWN,
    _SEVERITY,
    Probe,
    phase_status,
    render_shell,
)

SEED = "a" * 64
FP = "b" * 64
OTHER = "c" * 64
BUCKET = "t-nvidia-artf-recommenders-triton-models-0123abcd"
READ_ONLY = {"head_bucket", "list_objects_v2", "get_object", "head_object"}


def _err(code, op):
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


class FakeS3:
    """Answers head_bucket/list_objects_v2/get_object/head_object from a dict."""

    def __init__(self, objects=None, errors=None, keycount=1):
        self.objects = objects or {}
        self.errors = errors or {}
        self.keycount = keycount
        self.calls = []

    def head_bucket(self, Bucket):
        self.calls.append(("head_bucket", Bucket))
        return {}

    def list_objects_v2(self, Bucket, MaxKeys):
        self.calls.append(("list_objects_v2", Bucket))
        return {"KeyCount": self.keycount}

    def get_object(self, Bucket, Key):
        self.calls.append(("get_object", Bucket, Key))
        if Key in self.errors:
            raise self.errors[Key]
        if Key not in self.objects:
            raise _err("NoSuchKey", "GetObject")
        return {"Body": io.BytesIO(self.objects[Key]["body"])}

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", Bucket, Key))
        if Key in self.errors:
            raise self.errors[Key]
        if Key not in self.objects:
            raise _err("404", "HeadObject")
        return {"Metadata": dict(self.objects[Key].get("metadata") or {})}


def _probe(s3, *, seed="", fp=""):
    p = Probe.__new__(Probe)  # no AWS clients, no kubeconfig
    p.stack = "t-nvidia-artf-recommenders"
    p.uid = "0123abcd"
    p.prefix = "t"
    p._client = lambda service: s3 if service == "s3" else None
    p.expect_deals_seed_sha = seed
    p.expect_ncf_fingerprint = fp
    return p


def _marker(seed=SEED, fp=FP):
    body = json.dumps({"deals_seed_sha256": seed, "ncf_fingerprint": fp,
                       "written_at": "2026-10-07T00:00:00Z"}).encode()
    return {PHASE1_MARKER_KEY: {"body": body}}


def _engine(metadata):
    return {NCF_ENGINE_KEY: {"metadata": metadata}}


def _check(checks, name):
    found = [c for c in checks if c["name"] == name]
    assert len(found) == 1, checks
    return found[0]


def _assert_read_only(s3):
    assert {c[0] for c in s3.calls} <= READ_ONLY


# --- Phase 1: the marker ------------------------------------------------------

def test_matching_marker_keeps_phase1_ok():
    s3 = FakeS3(_marker())
    checks = _probe(s3, seed=SEED, fp=FP).phase1()
    assert _check(checks, "Phase 1 marker")["status"] == OK
    assert phase_status(checks) == OK
    assert ("get_object", BUCKET, PHASE1_MARKER_KEY) in s3.calls
    _assert_read_only(s3)


def test_missing_marker_is_stale():
    s3 = FakeS3({})
    checks = _probe(s3, seed=SEED, fp=FP).phase1()
    marker = _check(checks, "Phase 1 marker")
    assert marker["status"] == STALE
    assert PHASE1_MARKER_KEY in marker["detail"]
    assert phase_status(checks) == STALE
    _assert_read_only(s3)


def test_changed_deals_seed_is_stale():
    checks = _probe(FakeS3(_marker(seed=OTHER)), seed=SEED, fp=FP).phase1()
    marker = _check(checks, "Phase 1 marker")
    assert marker["status"] == STALE
    assert "deals seed cccccccccccc -> aaaaaaaaaaaa" in marker["detail"]


def test_changed_ncf_fingerprint_is_stale():
    checks = _probe(FakeS3(_marker(fp=OTHER)), seed=SEED, fp=FP).phase1()
    marker = _check(checks, "Phase 1 marker")
    assert marker["status"] == STALE
    assert "Deal Scorer fingerprint cccccccccccc -> bbbbbbbbbbbb" in marker["detail"]


def test_both_changed_are_both_named():
    detail = _check(_probe(FakeS3(_marker(seed=OTHER, fp=OTHER)), seed=SEED, fp=FP).phase1(),
                    "Phase 1 marker")["detail"]
    assert "deals seed" in detail and "Deal Scorer fingerprint" in detail


def test_only_the_given_expectation_is_compared():
    # A marker with another fingerprint, but only the seed sha was given.
    checks = _probe(FakeS3(_marker(fp=OTHER)), seed=SEED).phase1()
    assert _check(checks, "Phase 1 marker")["status"] == OK


def test_no_expectations_skips_the_marker():
    s3 = FakeS3({})
    checks = _probe(s3).phase1()
    assert [c["name"] for c in checks] == ["ONNX models in S3"]
    assert phase_status(checks) == OK
    assert not any(c[0] == "get_object" for c in s3.calls)


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b""])
def test_unreadable_marker_is_stale(body):
    s3 = FakeS3({PHASE1_MARKER_KEY: {"body": body}})
    assert _check(_probe(s3, seed=SEED, fp=FP).phase1(), "Phase 1 marker")["status"] == STALE


def test_marker_access_denied_is_unknown():
    s3 = FakeS3(errors={PHASE1_MARKER_KEY: _err("AccessDenied", "GetObject")})
    marker = _check(_probe(s3, seed=SEED, fp=FP).phase1(), "Phase 1 marker")
    assert marker["status"] == UNKNOWN
    assert "AccessDenied" in marker["detail"]


def test_empty_bucket_is_still_missing_before_the_marker():
    s3 = FakeS3(_marker(), keycount=0)
    checks = _probe(s3, seed=SEED, fp=FP).phase1()
    assert phase_status(checks) == MISSING
    assert not any(c[0] == "get_object" for c in s3.calls)


# --- Phase 3: the Deal Scorer engine -----------------------------------------

def _engine_check(s3, fp=FP):
    return _probe(s3, fp=fp)._ncf_engine()


def test_engine_with_the_local_fingerprint_is_ok():
    s3 = FakeS3(_engine({"source-fingerprint": FP, "provisioned-by": "bootstrap"}))
    check = _engine_check(s3)
    assert check["status"] == OK
    assert ("head_object", BUCKET, NCF_ENGINE_KEY) in s3.calls
    _assert_read_only(s3)


def test_bootstrap_engine_with_another_fingerprint_is_stale():
    check = _engine_check(FakeS3(_engine({"source-fingerprint": OTHER, "provisioned-by": "bootstrap"})))
    assert check["status"] == STALE
    assert "cccccccccccc" in check["detail"] and "bbbbbbbbbbbb" in check["detail"]


def test_engine_without_metadata_is_stale():
    # Built before engines carried metadata: a bootstrap engine with no fingerprint.
    assert _engine_check(FakeS3(_engine({})))["status"] == STALE


def test_promoted_engine_is_never_stale():
    check = _engine_check(FakeS3(_engine({"source-fingerprint": OTHER, "provisioned-by": "promotion"})))
    assert check["status"] == OK
    assert "promotion" in check["detail"]


def test_metadata_keys_are_compared_case_insensitively():
    s3 = FakeS3(_engine({"Source-Fingerprint": FP, "Provisioned-By": "bootstrap"}))
    assert _engine_check(s3)["status"] == OK


def test_missing_engine_is_missing():
    check = _engine_check(FakeS3({}))
    assert check["status"] == MISSING
    assert NCF_ENGINE_KEY in check["detail"]


def test_engine_head_error_is_unknown():
    s3 = FakeS3(errors={NCF_ENGINE_KEY: _err("403", "HeadObject")})
    assert _engine_check(s3)["status"] == UNKNOWN


def test_phase3_adds_the_engine_check_only_with_a_fingerprint(monkeypatch):
    s3 = FakeS3(_engine({"source-fingerprint": OTHER}))
    p = _probe(s3, fp=FP)
    monkeypatch.setattr(p, "_workloads", lambda: [{"name": "deployment x", "status": OK, "detail": ""}], raising=False)
    checks = p.phase3()
    assert [c["name"] for c in checks] == ["deployment x", "Deal Scorer engine"]
    assert phase_status(checks) == STALE

    p = _probe(FakeS3({}), fp="")
    monkeypatch.setattr(p, "_workloads", lambda: [{"name": "deployment x", "status": OK, "detail": ""}], raising=False)
    assert [c["name"] for c in p.phase3()] == ["deployment x"]


# --- status vocabulary and the shell contract ----------------------------------

def test_stale_ranks_just_above_ok():
    assert _SEVERITY[OK] < _SEVERITY[STALE] < _SEVERITY[UNKNOWN] < _SEVERITY[MISSING] \
        < _SEVERITY[IN_PROGRESS] < _SEVERITY[FAILED]
    assert phase_status([{"status": STALE}, {"status": MISSING}]) == MISSING
    assert phase_status([{"status": STALE}, {"status": IN_PROGRESS}]) == IN_PROGRESS
    assert phase_status([{"status": OK}, {"status": STALE}]) == STALE


def test_shell_output_carries_stale_and_its_reason():
    checks = _probe(FakeS3({}), seed=SEED, fp=FP).phase1()
    result = {"phases": {1: {"label": "Preparing models", "status": phase_status(checks), "checks": checks}},
              "kubeContext": ""}
    shell = render_shell(result)
    assert "DEPLOY_PHASE_1=stale" in shell
    assert "DEPLOY_BLOCKER_1='Phase 1 marker" in shell
    assert "DEPLOY_OVERALL=stale" in shell


def test_stale_has_a_mark_of_the_same_width():
    widths = {len(m) for m in deploy_status.MARKS.values()}
    assert STALE in deploy_status.MARKS
    assert len(widths) == 1


def test_cli_passes_the_expectations_to_the_probe(monkeypatch):
    seen = {}

    class _Probe:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def run(self):
            return {"phases": {}, "kubeContext": "", "notes": []}

    monkeypatch.setattr(deploy_status, "Probe", _Probe)
    monkeypatch.setattr(deploy_status, "boto3", object())
    deploy_status.main(["--prefix", "t", "--shell",
                        "--expect-deals-seed-sha", SEED, "--expect-ncf-fingerprint", FP])
    assert seen["expect_deals_seed_sha"] == SEED
    assert seen["expect_ncf_fingerprint"] == FP


def test_constants_match_deploy_sh_and_the_bootstrap_spec():
    root = Path(__file__).resolve().parents[2] / "deployment"
    deploy = (root / "deploy.sh").read_text()
    assert f'PHASE1_MARKER_KEY="{PHASE1_MARKER_KEY}"' in deploy
    spec = json.loads((root / "optimizer-bootstrap.json").read_text().replace("__FEATURE_SPEC_VERSION__", "1"))
    ncf = [m for m in spec["models"] if m["model_name"] == "ncf_deal_manager"][0]
    assert ncf["output_uri"].endswith("/" + NCF_ENGINE_KEY)
