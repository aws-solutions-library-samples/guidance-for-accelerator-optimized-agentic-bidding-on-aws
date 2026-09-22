"""Tests for the not-ready classification in deployment/scripts/deploy_status.py.

This is the honesty-critical part of the probe. It shipped reporting every
`readyReplicas < replicas` as FAILED, so a Triton pod 80 seconds into loading
TensorRT engines -- an entirely normal rollout -- was presented as a failed
deployment. It was observed doing exactly that against a real cluster, and the pod
reached 1/1 a minute later untouched.

Over-reporting FAILED is the same defect as under-reporting it. Both leave a reader
unable to trust the line, which is the whole point of the probe. So the distinction
gets tests:

  * FAILED only on evidence -- Kubernetes' own ProgressDeadlineExceeded verdict, or
    a container waiting on a reason that does not self-resolve.
  * Everything else that is merely not ready yet is IN_PROGRESS, which also makes a
    re-run WAIT for the rollout instead of re-applying manifests underneath it.
"""
import sys
from pathlib import Path

import pytest

# deployment/scripts is not a package, so it goes on sys.path directly
# (same approach as test_deploy_state.py).
_SCRIPTS = Path(__file__).resolve().parents[2] / "deployment" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from deploy_status import (  # noqa: E402
    FAILED,
    IN_PROGRESS,
    MISSING,
    OK,
    UNKNOWN,
    _SEVERITY,
    Probe,
)


def _probe(pod_reason=""):
    """A Probe with _pod_reason stubbed, so no kubectl call is made."""
    p = Probe.__new__(Probe)          # bypass __init__ and its AWS clients
    p._pod_reason = lambda _dep: pod_reason
    return p


PROGRESS_DEADLINE = [{"type": "Progressing", "status": "False",
                      "reason": "ProgressDeadlineExceeded"}]
PROGRESSING = [{"type": "Progressing", "status": "True",
                "reason": "ReplicaSetUpdated"}]


# --- IN_PROGRESS: not ready is not the same as broken -----------------------

def test_starting_pod_is_in_progress_not_failed():
    """The live regression: Triton loading engines must not read as FAILED."""
    status, why = _probe("pod Running: containers with unready status: [triton]") \
        ._rollout_status("triton-inference-server", PROGRESSING)
    assert status == IN_PROGRESS
    assert "triton" in why


def test_no_conditions_and_no_reason_is_in_progress():
    status, why = _probe("")._rollout_status("some-deployment", [])
    assert status == IN_PROGRESS
    assert why == "starting"          # never blank -- a status with no reason is useless


@pytest.mark.parametrize("reason", [
    "pod Pending: ContainerCreating",
    "pod Pending: PodInitializing",
    "pod Running: containers with unready status: [app]",
])
def test_transient_pod_states_are_in_progress(reason):
    status, _ = _probe(reason)._rollout_status("app", PROGRESSING)
    assert status == IN_PROGRESS


# --- FAILED: only on real evidence -----------------------------------------

def test_progress_deadline_exceeded_is_failed():
    """Kubernetes' own verdict that the rollout is stuck. Time-based, so it cannot
    be fooled by something merely slow to start."""
    status, why = _probe("pod Pending: Unschedulable") \
        ._rollout_status("triton-inference-server", PROGRESS_DEADLINE)
    assert status == FAILED
    assert "stuck" in why


@pytest.mark.parametrize("reason", [
    "pod Running: CrashLoopBackOff back-off 5m0s restarting failed container",
    "pod Pending: ImagePullBackOff Back-off pulling image",
    "pod Pending: ErrImagePull manifest unknown",
    "pod Pending: InvalidImageName couldn't parse image",
    "pod Pending: CreateContainerConfigError secret not found",
    "pod Pending: CreateContainerError",
])
def test_non_self_resolving_container_errors_are_failed(reason):
    """Waiting out a 10-minute progress deadline to state the obvious would be its
    own kind of unhelpful."""
    status, why = _probe(reason)._rollout_status("app", PROGRESSING)
    assert status == FAILED
    assert why == reason


def test_progress_deadline_wins_over_a_transient_looking_reason():
    status, _ = _probe("pod Pending: ContainerCreating") \
        ._rollout_status("app", PROGRESS_DEADLINE)
    assert status == FAILED


# --- severity ordering the phase roll-up depends on ------------------------

def test_in_progress_outranks_missing_but_not_failed():
    """A phase takes the worst of its checks. IN_PROGRESS must beat MISSING (being
    built is not being absent) and FAILED must beat everything, or a broken
    deployment hides behind a rolling one."""
    assert _SEVERITY[OK] < _SEVERITY[UNKNOWN] < _SEVERITY[MISSING] \
        < _SEVERITY[IN_PROGRESS] < _SEVERITY[FAILED]


def test_every_status_has_a_severity():
    for status in (OK, UNKNOWN, MISSING, IN_PROGRESS, FAILED):
        assert status in _SEVERITY


# --- the expected set: what the deploy APPLIES, not what eks/ contains ------
#
# These guard a defect that made the probe permanently wrong. The expected set was
# a glob over eks/*.yaml minus anything whose FILENAME contained "prebid", which
# missed amt-simulator-deployment.yaml -- applied only by deploy_prebid.sh
# (--with-prebid). A normal deploy therefore reported amt-simulator as missing
# forever. Because MISSING tells the gate to RUN the phase, Phase 3 re-ran on every
# single deploy and could never converge, and --status could never exit 0.

_DEPLOYMENT_DIR = Path(__file__).resolve().parents[2] / "deployment"


def _real_probe(prebid: bool):
    p = Probe.__new__(Probe)
    p.manifest_dir = str(_DEPLOYMENT_DIR / "eks")
    p.prebid = prebid
    return p


def test_default_deploy_does_not_expect_prebid_owned_workloads():
    """The regression, stated directly."""
    names = _real_probe(prebid=False)._manifest_deployments()
    assert names is not None
    assert "amt-simulator" not in names
    assert "prebid-server" not in names


def test_default_deploy_expects_the_workloads_deploy_sh_applies():
    names = _real_probe(prebid=False)._manifest_deployments()
    for expected in ("triton-inference-server", "orchestrator", "bid-pricer",
                     "yield-optimizer-floor", "yield-optimizer-margin"):
        assert expected in names


def test_with_prebid_expects_both_prebid_owned_workloads():
    names = _real_probe(prebid=True)._manifest_deployments()
    assert "amt-simulator" in names      # deploy_prebid.sh, second auction seat
    assert "prebid-server" in names


def test_the_real_deploy_sh_array_is_parseable():
    """Guards the None fallback from becoming the normal path. If _APPLIED_MANIFESTS
    is renamed or restructured, every phase-3 check degrades to UNKNOWN -- correct,
    but useless. Fail here instead, where the cause is obvious."""
    files = _real_probe(prebid=False)._expected_manifest_files()
    assert files, "could not read _APPLIED_MANIFESTS from the real deploy.sh"
    assert all(f.endswith((".yaml", ".yml")) for f in files)


def test_unreadable_deploy_sh_yields_none_not_a_guess(tmp_path):
    """Undeterminable must stay undeterminable. Falling back to 'expect whatever is
    in eks/' is what produced the bug; falling back to 'expect nothing' would mark
    the phase OK while a deployment was genuinely absent."""
    (tmp_path / "eks").mkdir()
    p = Probe.__new__(Probe)
    p.manifest_dir = str(tmp_path / "eks")
    p.prebid = False
    assert p._expected_manifest_files() is None   # no deploy.sh at all
    assert p._manifest_deployments() is None


def test_renamed_array_yields_none(tmp_path):
    (tmp_path / "eks").mkdir()
    (tmp_path / "deploy.sh").write_text("_MANIFESTS_TO_APPLY=(triton-deployment.yaml)\n")
    p = Probe.__new__(Probe)
    p.manifest_dir = str(tmp_path / "eks")
    p.prebid = False
    assert p._expected_manifest_files() is None
