#!/usr/bin/env python3
"""What is actually deployed, according to AWS.

This asks the account. It does not read a local progress file, a pid, or a log,
and it does not care whether a deploy is running, finished, or was killed
half-way. Every slow thing in this deployment is an AWS-side asynchronous
operation with a queryable status, so the account is the only thing that can
answer "where are we?" correctly:

  - CodeBuild builds      -> IN_PROGRESS / SUCCEEDED / FAILED
  - CloudFormation stacks -> *_IN_PROGRESS / *_COMPLETE / *_FAILED
  - EKS cluster/nodegroups-> CREATING / ACTIVE / DEGRADED
  - AgentCore runtimes    -> CREATING / READY / FAILED
  - ECR, S3, CloudFront, Cognito -> present or not

Why this replaced a local state file: the file recorded what the *script* had
done, which is not the same thing as what *exists*. It reported "Phase 3
complete" while the Triton pod had been Pending for 20 hours because no GPU node
ever joined the cluster. A file cannot know that. It also cannot be read from
another machine, after a reboot, or by a colleague -- and it is wrong the moment
anything changes outside the script.

Three consequences worth stating, because they are the point:

1. **Two terminals can run this at the same time.** It only reads. There is no
   process to attach to, nothing to lock, and no way to deadlock.
2. **IN_PROGRESS is a first-class answer**, distinct from "missing". That is what
   lets a caller wait for something instead of racing it or re-creating it.
3. **Unknown is reported as unknown.** If kubectl is not configured, Phase 3 says
   so rather than claiming success. A status probe that guesses is worse than no
   probe.

Usage:
    deploy_status.py --prefix bt1 [--region us-east-1] [--json]
                     [--no-retraining] [--with-prebid] [--skip-agentcore]

Exit status: 0 when everything expected is present and healthy, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - deploy.sh installs boto3 before calling
    boto3 = None
    ClientError = BotoCoreError = Exception


# --- status vocabulary -------------------------------------------------------
# Deliberately small, and deliberately includes UNKNOWN. Collapsing UNKNOWN into
# either OK or MISSING is how a probe starts lying.
OK = "ok"
IN_PROGRESS = "in_progress"
MISSING = "missing"
FAILED = "failed"
UNKNOWN = "unknown"

# Ordered worst-last: a phase's status is the worst of its checks, except that
# IN_PROGRESS outranks MISSING (something being built is not something absent).
_SEVERITY = {OK: 0, UNKNOWN: 1, MISSING: 2, IN_PROGRESS: 3, FAILED: 4}

def _marks() -> Dict[str, str]:
    """Check/cross marks, with an ASCII fallback.

    Same glyphs as deployment/lib/deploy_progress.sh, so a phase looks identical
    whether you are watching a deploy or asking `--status` afterwards. Check and
    cross rather than emoji: one column wide in every terminal, they survive a
    paste into a ticket, and they need no colour-pictograph font.

    The word stays next to the cross because a cross alone cannot distinguish
    "never created" from "created and broken", and those need different actions.
    Every value is padded to the same display width so the column scans.
    """
    good, bad, wait, huh = "\u2713", "\u2717", "\u00b7", "?"
    try:
        # Does this output stream actually support the glyphs? A mark that renders
        # as a replacement box is worse than 'x'.
        enc = sys.stdout.encoding or "ascii"
        "".join((good, bad, wait)).encode(enc)
    except (UnicodeEncodeError, LookupError):
        good, bad, wait = "+", "x", "."
    return {
        OK: f"{good}        ",
        IN_PROGRESS: f"{wait} working",
        MISSING: f"{bad} missing",
        FAILED: f"{bad} FAILED ",
        UNKNOWN: f"{huh} unknown",
    }


MARKS = _marks()

PHASE_LABELS = {
    1: "Preparing models",
    2: "Building containers & provisioning infrastructure",
    3: "Deploying workloads",
    4: "Setting up access",
    5: "Registering agents",
}

# ECR repositories, by display name (see display_name() in deploy.sh). The keys
# differ from the repo names; the repos use the display names.
CORE_IMAGES = (
    "bid-pricer",
    "audience-activator",
    "deal-scorer",
    "signals-enricher",
    "yield-optimizer-floor",
    "yield-optimizer-margin",
    "artf-template",
    "orchestrator",
    "model-optimizer",
    "agentcore",
)
RETRAINING_IMAGES = ("adaptive-bidding-agent",)

# The UI API proxy Lambda stack, created by deploy.sh Step 8.4 (Phase 3) after the
# manifests. It is the browser's only route to the orchestrator, which is a
# ClusterIP Service with no public address, so a missing or failed stack means a
# UI that loads but cannot reach anything.
UI_API_PROXY_STACK = "ui-api-proxy"
# The Service whose internal NLB the UI API proxy forwards to
# (deployment/eks/orchestrator-internal-nlb.yaml). Not prefixed: Kubernetes names
# are scoped by cluster, and each prefix has its own cluster.
ORCHESTRATOR_INTERNAL_SVC = "orchestrator-internal"

# CloudFormation stacks created by deploy_closed_loop.sh, in the order it creates
# them, so a partial Phase 5 reads as a prefix of this list.
CLOSED_LOOP_STACKS = (
    "feedback-pipeline",
    "glue-etl",
    "closed-loop-core",
    "agentcore-security",
    "governance-eventbridge",
)


def phase_status(checks: List[Dict[str, Any]]) -> str:
    if not checks:
        return UNKNOWN
    return max((c["status"] for c in checks), key=lambda s: _SEVERITY.get(s, 1))


# ---------------------------------------------------------------------------
# naming — must match deploy.sh exactly
# ---------------------------------------------------------------------------


def stack_name(prefix: str, base: str = "nvidia-artf-recommenders") -> str:
    """deploy.sh: STACK_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}nvidia-artf-recommenders"."""
    return f"{prefix}-{base}" if prefix else base


def stack_uid(stack: str, account: str, region: str) -> str:
    """deploy.sh: sha256("STACK:ACCOUNT:REGION")[:8]."""
    import hashlib

    return hashlib.sha256(f"{stack}:{account}:{region}".encode()).hexdigest()[:8]


def runtime_prefix(prefix: str) -> str:
    """deploy_closed_loop.sh: RUNTIME_NAME_PREFIX="${STACK_PREFIX//-/_}_"."""
    return (prefix.replace("-", "_") + "_") if prefix else ""


def prefixed(prefix: str, name: str) -> str:
    return f"{prefix}-{name}" if prefix else name


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


class Probe:
    def __init__(self, prefix: str, region: str, *, retraining: bool,
                 prebid: bool, agentcore: bool, manifest_dir: str,
                 profile: Optional[str] = None):
        self.prefix = prefix
        self.region = region
        self.profile = profile or None
        # The kubectl subprocesses below authenticate through the kubeconfig's exec
        # plugin (`aws eks get-token`), which reads AWS_PROFILE from the environment,
        # not from this process's boto3 session. Export it so both paths agree.
        if self.profile:
            os.environ["AWS_PROFILE"] = self.profile
        self.retraining = retraining
        self.prebid = prebid
        self.agentcore = agentcore
        self.manifest_dir = manifest_dir
        self.stack = stack_name(prefix)
        self.notes: List[str] = []
        self._session = (boto3.session.Session(region_name=region, profile_name=self.profile)
                         if boto3 else None)
        self.account = self._account()
        self.uid = stack_uid(self.stack, self.account, region) if self.account else ""
        self.cluster = f"{self.stack}-triton"
        # Resolved once, and never assumed. See _resolve_kube_context().
        self.kube_context, self.kube_reason = self._resolve_kube_context()

    # -- kubernetes targeting ---------------------------------------------

    def _resolve_kube_context(self) -> Tuple[Optional[str], str]:
        """Find the kubeconfig context for THIS cluster. Never trust the current one.

        This exists because of a real, and badly misleading, failure: `kubectl` was
        pointed at a different cluster in the same account (`nv5-...`) while this probe
        was asked about `bt1-...`. It compared bt1's EC2 instances against nv5's
        registered nodes and reported, with confidence, that a GPU node had failed to
        join and that Triton had been Pending for 20 hours. Both statements were about
        the wrong cluster.

        `deploy.sh` itself documents the cause: every deploy rewrites the shared
        `~/.kube/config`, so whichever ran last owns the current context. A probe that
        reads the current context is therefore reading "whatever was deployed most
        recently", not "the thing I was asked about".

        So the context is selected by matching the cluster ARN/name, and if no such
        context exists the answer is UNKNOWN with the reason -- never a guess.
        """
        try:
            proc = subprocess.run(["kubectl", "config", "view", "-o", "json"],
                                  capture_output=True, text=True, timeout=20)
        except FileNotFoundError:
            return None, "kubectl is not on PATH"
        except Exception as exc:
            return None, f"could not read kubeconfig ({exc.__class__.__name__})"
        if proc.returncode != 0:
            return None, "kubectl could not read the kubeconfig"
        try:
            cfg = json.loads(proc.stdout)
        except ValueError:
            return None, "kubeconfig is not parseable"

        # A context's cluster entry is a kubeconfig-local alias; match on the cluster
        # ARN or its trailing name, both of which contain the real cluster name.
        matching_clusters = {
            c.get("name") for c in (cfg.get("clusters") or [])
            if c.get("name", "").endswith("/" + self.cluster) or c.get("name") == self.cluster
        }
        for ctx in (cfg.get("contexts") or []):
            if (ctx.get("context") or {}).get("cluster") in matching_clusters:
                return ctx.get("name"), ""
        current = cfg.get("current-context") or "<none>"
        return None, (
            f"no kubeconfig context for cluster '{self.cluster}' "
            f"(current context is '{current}'). Add one with: "
            f"aws eks update-kubeconfig --name {self.cluster} --region {self.region}"
        )

    def _kubectl(self, *args: str, timeout: int = 30) -> Optional[subprocess.CompletedProcess]:
        """Run kubectl against THIS cluster, or return None if that is impossible."""
        if not self.kube_context:
            return None
        try:
            return subprocess.run(
                ["kubectl", "--context", self.kube_context, *args],
                capture_output=True, text=True, timeout=timeout,
            )
        except Exception:
            return None

    # -- helpers ----------------------------------------------------------

    def _client(self, service: str):
        if not self._session:
            return None
        try:
            return self._session.client(service)
        except Exception:
            return None

    def _account(self) -> str:
        c = self._client("sts")
        if not c:
            return ""
        try:
            return c.get_caller_identity()["Account"]
        except Exception as exc:
            self.notes.append(f"cannot resolve AWS account ({exc.__class__.__name__}); "
                              "credentials may be missing or expired")
            return ""

    @staticmethod
    def _check(name: str, status: str, detail: str = "") -> Dict[str, Any]:
        return {"name": name, "status": status, "detail": detail}

    # -- Phase 1: models in S3 -------------------------------------------

    def phase1(self) -> List[Dict[str, Any]]:
        bucket = f"{self.stack}-triton-models-{self.uid}" if self.uid else ""
        if not bucket:
            return [self._check("model bucket", UNKNOWN, "account not resolved")]
        s3 = self._client("s3")
        if not s3:
            return [self._check("model bucket", UNKNOWN, "no s3 client")]
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchBucket"):
                return [self._check("model bucket", MISSING, bucket)]
            return [self._check("model bucket", UNKNOWN, f"{bucket}: {code}")]
        except Exception as exc:
            return [self._check("model bucket", UNKNOWN, f"{bucket}: {exc}")]

        # A bucket with no objects is not a completed Phase 1 -- it is a bucket.
        try:
            listing = s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
            count = listing.get("KeyCount", 0)
        except Exception as exc:
            return [self._check("model bucket", UNKNOWN, f"{bucket}: {exc}")]
        if count == 0:
            return [self._check("ONNX models in S3", MISSING, f"{bucket} is empty")]
        return [self._check("ONNX models in S3", OK, bucket)]

    # -- Phase 2: images, builds, cluster, nodes -------------------------

    def phase2(self) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []
        checks.extend(self._images())
        checks.extend(self._codebuild())
        checks.extend(self._cluster())
        return checks

    def _expected_images(self) -> Tuple[str, ...]:
        names = list(CORE_IMAGES)
        if self.retraining:
            names.extend(RETRAINING_IMAGES)
        return tuple(names)

    def _images(self) -> List[Dict[str, Any]]:
        ecr = self._client("ecr")
        if not ecr:
            return [self._check("container images", UNKNOWN, "no ecr client")]
        present: Dict[str, int] = {}
        try:
            paginator = ecr.get_paginator("describe_repositories")
            for page in paginator.paginate():
                for repo in page.get("repositories", []):
                    name = repo["repositoryName"]
                    if name.startswith(self.stack + "-"):
                        present[name[len(self.stack) + 1:]] = 0
        except Exception as exc:
            return [self._check("container images", UNKNOWN, str(exc))]

        # A repository with no images is not a built image. This distinction is the
        # difference between "Phase 2 ran" and "Phase 2 produced something".
        for short in list(present):
            try:
                imgs = ecr.list_images(repositoryName=f"{self.stack}-{short}", maxResults=1)
                present[short] = len(imgs.get("imageIds", []))
            except Exception:
                present[short] = 0

        expected = self._expected_images()
        missing = [n for n in expected if n not in present]
        empty = [n for n in expected if present.get(n, 0) == 0 and n in present]
        if missing:
            return [self._check("container images", MISSING,
                                f"{len(expected) - len(missing)}/{len(expected)} repos; "
                                f"absent: {', '.join(missing)}")]
        if empty:
            return [self._check("container images", MISSING,
                                f"repos exist but hold no image: {', '.join(empty)}")]
        return [self._check("container images", OK,
                            f"{len(expected)}/{len(expected)} repositories with images")]

    def _codebuild(self) -> List[Dict[str, Any]]:
        cb = self._client("codebuild")
        if not cb:
            return []
        project = f"{self.stack}-image-builder"
        try:
            ids = cb.list_builds_for_project(projectName=project, sortOrder="DESCENDING").get("ids", [])
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return [self._check("image build project", MISSING, project)]
            return [self._check("image build project", UNKNOWN, str(exc))]
        except Exception as exc:
            return [self._check("image build project", UNKNOWN, str(exc))]
        if not ids:
            return [self._check("image builds", MISSING, "project exists, never built")]
        try:
            builds = cb.batch_get_builds(ids=ids[:1]).get("builds", [])
        except Exception as exc:
            return [self._check("image builds", UNKNOWN, str(exc))]
        if not builds:
            return [self._check("image builds", UNKNOWN, "build record unavailable")]
        b = builds[0]
        status = b.get("buildStatus", "")
        if status == "IN_PROGRESS":
            # This is the whole reason IN_PROGRESS exists as a status: a caller can
            # WAIT for this build instead of starting a competing one.
            return [self._check("image build", IN_PROGRESS,
                                f"{b.get('currentPhase', '?')} — {b.get('id', '')}")]
        if status == "SUCCEEDED":
            return [self._check("image build", OK, "last build succeeded")]
        return [self._check("image build", FAILED, f"last build {status} — {b.get('id', '')}")]

    def _cluster(self) -> List[Dict[str, Any]]:
        eks = self._client("eks")
        if not eks:
            return [self._check("EKS cluster", UNKNOWN, "no eks client")]
        cluster = self.cluster
        try:
            info = eks.describe_cluster(name=cluster)["cluster"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return [self._check("EKS cluster", MISSING, cluster)]
            return [self._check("EKS cluster", UNKNOWN, str(exc))]
        except Exception as exc:
            return [self._check("EKS cluster", UNKNOWN, str(exc))]

        status = info.get("status", "")
        if status == "CREATING":
            return [self._check("EKS cluster", IN_PROGRESS, f"{cluster} CREATING")]
        if status != "ACTIVE":
            return [self._check("EKS cluster", FAILED, f"{cluster} {status}")]
        checks = [self._check("EKS cluster", OK, f"{cluster} ACTIVE")]
        checks.extend(self._nodegroups(eks, cluster))
        return checks

    def _nodegroups(self, eks, cluster: str) -> List[Dict[str, Any]]:
        """Nodegroup status, running instances, AND nodes registered in the cluster.

        All three, because each can disagree with the others and only the last one
        decides whether a pod can be scheduled:

          - `describe-nodegroup` said ACTIVE, desiredSize 1, health.issues []
          - EC2 said 1 instance running
          - the cluster had ZERO nodes from that group

        The Triton pod sat Pending for 20 hours against that combination. An earlier
        version of this probe checked only the first two and reported the nodegroup
        OK -- reproducing, in a tool built to prevent it, exactly the fault it was
        built to catch. A running instance is not a node; a node is what schedules.
        """
        out: List[Dict[str, Any]] = []
        try:
            names = eks.list_nodegroups(clusterName=cluster).get("nodegroups", [])
        except Exception as exc:
            return [self._check("node groups", UNKNOWN, str(exc))]
        if not names:
            return [self._check("node groups", MISSING, "cluster has no node groups")]
        ec2 = self._client("ec2")
        registered = self._registered_nodes()      # None when kubectl cannot answer
        for ng in names:
            try:
                d = eks.describe_nodegroup(clusterName=cluster, nodegroupName=ng)["nodegroup"]
            except Exception as exc:
                out.append(self._check(f"node group {ng}", UNKNOWN, str(exc)))
                continue
            status = d.get("status", "")
            desired = (d.get("scalingConfig") or {}).get("desiredSize", 0)
            running = self._running_instances(ec2, cluster, ng)
            issues = [i.get("code", "") for i in (d.get("health") or {}).get("issues", [])]
            joined = None if registered is None else registered.get(ng, 0)

            if status in ("CREATING", "UPDATING"):
                out.append(self._check(f"node group {ng}", IN_PROGRESS, status))
                continue
            if status != "ACTIVE":
                out.append(self._check(f"node group {ng}", FAILED,
                                       f"{status}{' — ' + ','.join(issues) if issues else ''}"))
                continue
            if running is None:
                out.append(self._check(f"node group {ng}", UNKNOWN,
                                       f"ACTIVE, desired {desired}, instance count unavailable"))
                continue
            if running < desired:
                out.append(self._check(
                    f"node group {ng}", FAILED,
                    f"ACTIVE but only {running}/{desired} instances running"
                    + (f" — {','.join(issues)}" if issues else
                       " — no health issue reported; suspect instance-type capacity in"
                       " this AZ or an EC2 service quota")))
                continue
            # desiredSize 0 is a DELIBERATE scale-to-zero, not a fault. This project
            # ships a nightly GPU shutdown to stop paying for idle g5 capacity, and a
            # probe that cried FAILED every morning would be noise. But it is not OK
            # either: nothing can schedule on an empty group, so the dependent
            # workloads really are down. Reported as absent capacity, with the reason.
            if desired == 0:
                action = self._scheduled_action(d)
                out.append(self._check(
                    f"node group {ng}", MISSING,
                    f"scaled to 0 on purpose"
                    + (f" by scheduled action '{action}'" if action else "")
                    + " — GPU workloads cannot schedule until it scales back up; "
                      "re-running deploy.sh restores it"))
                continue
            if joined is None:
                out.append(self._check(
                    f"node group {ng}", UNKNOWN,
                    f"{running}/{desired} instances running; cannot confirm they joined "
                    "the cluster (kubectl unavailable)"))
                continue
            if joined == 0:
                # Distinct from the scale-to-zero above: capacity is WANTED here.
                out.append(self._check(
                    f"node group {ng}", FAILED,
                    f"desired {desired}, {running} instance(s) running, but NO nodes "
                    f"registered in the cluster — nothing can schedule here. Check "
                    f"kubelet bootstrap on the instances and the cluster's "
                    f"aws-auth/access entries."))
                continue
            if joined < running:
                # EC2 is happy, EKS is happy, and the scheduler is short of nodes.
                out.append(self._check(
                    f"node group {ng}", FAILED,
                    f"{running} instance(s) running but only {joined} registered as "
                    f"cluster nodes — some pods needing this group cannot schedule."))
                continue
            out.append(self._check(
                f"node group {ng}", OK,
                f"{joined}/{desired} node(s) registered, {d.get('instanceTypes', [])}"))
        return out

    def _scheduled_action(self, nodegroup: Dict[str, Any]) -> str:
        """Name a scheduled scaling action on this nodegroup's ASG, if there is one.

        Turns an alarming "0 nodes" into "0 nodes, on purpose, because of this".
        Best-effort: an empty string simply omits the explanation.
        """
        try:
            asgs = (nodegroup.get("resources") or {}).get("autoScalingGroups") or []
            if not asgs:
                return ""
            name = asgs[0].get("name")
            if not name:
                return ""
            asg = self._client("autoscaling")
            if not asg:
                return ""
            resp = asg.describe_scheduled_actions(AutoScalingGroupName=name)
            actions = resp.get("ScheduledUpdateGroupActions") or []
            zeroing = [a for a in actions if a.get("DesiredCapacity") == 0]
            chosen = zeroing or actions
            return chosen[0].get("ScheduledActionName", "") if chosen else ""
        except Exception:
            return ""

    def _registered_nodes(self) -> Optional[Dict[str, int]]:
        """Nodes actually registered in THIS cluster, counted per nodegroup.

        Returns None (not an empty dict) when kubectl cannot answer for this specific
        cluster, so the caller reports UNKNOWN instead of mistaking "cannot see" for
        "none joined" -- or, worse, counting another cluster's nodes.
        """
        proc = self._kubectl("get", "nodes", "-o", "json")
        if proc is None or proc.returncode != 0:
            return None
        try:
            items = json.loads(proc.stdout).get("items", [])
        except ValueError:
            return None
        counts: Dict[str, int] = {}
        for node in items:
            labels = node.get("metadata", {}).get("labels", {}) or {}
            ng = labels.get("eks.amazonaws.com/nodegroup")
            if not ng:
                continue
            ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                        for c in node.get("status", {}).get("conditions", []))
            if ready:
                counts[ng] = counts.get(ng, 0) + 1
        return counts

    @staticmethod
    def _running_instances(ec2, cluster: str, nodegroup: str) -> Optional[int]:
        if not ec2:
            return None
        try:
            resp = ec2.describe_instances(Filters=[
                {"Name": "tag:eks:cluster-name", "Values": [cluster]},
                {"Name": "tag:eks:nodegroup-name", "Values": [nodegroup]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ])
        except Exception:
            return None
        return sum(len(r.get("Instances", [])) for r in resp.get("Reservations", []))

    # -- Phase 3: workloads ----------------------------------------------

    def phase3(self) -> List[Dict[str, Any]]:
        """Kubernetes readiness. Needs kubectl; says so when it cannot look.

        Reported as UNKNOWN rather than OK when kubectl is unavailable. The previous
        design's failure mode was exactly this: claiming a phase complete without
        having verified anything.
        """
        wanted = self._manifest_deployments()
        if not self.kube_context:
            # Deliberately UNKNOWN, not OK and not MISSING. Reporting a workload state
            # without being able to reach the right cluster is how the previous version
            # of this probe produced a confident description of a different cluster.
            return [self._check("workloads", UNKNOWN, self.kube_reason)]
        proc = self._kubectl("get", "deploy", "-o", "json")
        if proc is None:
            return [self._check("workloads", UNKNOWN, "kubectl could not be run")]
        if proc.returncode != 0:
            msg = (proc.stderr or "").strip().splitlines()
            return [self._check("workloads", UNKNOWN,
                                f"kubectl failed: {msg[-1] if msg else 'unknown error'}")]
        try:
            items = json.loads(proc.stdout).get("items", [])
        except ValueError:
            return [self._check("workloads", UNKNOWN, "kubectl returned unparseable JSON")]

        actual = {}
        for item in items:
            name = item.get("metadata", {}).get("name", "")
            spec = item.get("spec", {}).get("replicas", 0)
            ready = item.get("status", {}).get("readyReplicas", 0) or 0
            actual[name] = (ready, spec, item.get("status", {}).get("conditions", []) or [])

        checks = []
        if wanted is None:
            # The desired set could not be determined. Report that, rather than
            # falling back to "expect whatever happens to be running" -- which would
            # mark the phase OK while a deployment was genuinely absent.
            checks.append(self._check(
                "expected workloads", UNKNOWN,
                "could not read _APPLIED_MANIFESTS from deploy.sh — "
                "listing what is running, not what is missing"))
            wanted = []
        for name in sorted(wanted or actual):
            if name not in actual:
                checks.append(self._check(f"deployment {name}", MISSING, "not applied"))
                continue
            ready, spec, conds = actual[name]
            if ready >= spec and spec > 0:
                checks.append(self._check(f"deployment {name}", OK, f"{ready}/{spec} ready"))
                continue
            status, why = self._rollout_status(name, conds)
            checks.append(self._check(f"deployment {name}", status, f"{ready}/{spec} ready — {why}"))
        if not checks:
            checks.append(self._check("workloads", MISSING, "no deployments found"))
        checks.append(self._orchestrator_internal_nlb())
        checks.append(self._stack_check(prefixed(self.prefix, UI_API_PROXY_STACK), label="UI API proxy"))
        return checks

    def _orchestrator_internal_nlb(self) -> Dict[str, Any]:
        """The orchestrator-internal Service: the address the UI API proxy forwards to.

        OK once the in-tree controller has given it a hostname, IN_PROGRESS while the
        Service exists without one (NLB still provisioning), MISSING when not applied.
        Without this line a probe could call Phase 3 complete while every UI call 502s.
        """
        title = f"service {ORCHESTRATOR_INTERNAL_SVC}"
        proc = self._kubectl("get", "svc", ORCHESTRATOR_INTERNAL_SVC, "-o", "json")
        if proc is None:
            return self._check(title, UNKNOWN, "kubectl could not be run")
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            if "NotFound" in err or "not found" in err:
                return self._check(title, MISSING, "not applied")
            msg = err.splitlines()
            return self._check(title, UNKNOWN, f"kubectl failed: {msg[-1] if msg else 'unknown error'}")
        try:
            svc = json.loads(proc.stdout)
        except ValueError:
            return self._check(title, UNKNOWN, "kubectl returned unparseable JSON")
        ingress = ((svc.get("status") or {}).get("loadBalancer") or {}).get("ingress") or []
        host = (ingress[0].get("hostname") if ingress else None) or ""
        if host:
            return self._check(title, OK, f"internal NLB {host}")
        return self._check(title, IN_PROGRESS, "no load balancer hostname yet (NLB provisioning)")

    # Container waiting-reasons that will not fix themselves. Waiting out the
    # progress deadline to state the obvious would be its own kind of unhelpful.
    _HARD_POD_FAILURES = (
        "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "InvalidImageName",
        "CreateContainerConfigError", "CreateContainerError",
    )

    def _rollout_status(self, deployment: str, conds: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Not-ready: is it still starting, or is it stuck? Returns (status, reason).

        This used to report every `ready < spec` as FAILED, which meant a Triton pod
        80 seconds into loading TensorRT engines -- a completely normal rollout --
        was presented as a failed deployment. Over-reporting FAILED is the same
        defect as under-reporting it: both leave the reader unable to trust the line.

        FAILED is claimed on two kinds of evidence, never on "not ready yet":

          * ProgressDeadlineExceeded. This is Kubernetes' OWN verdict that a rollout
            is stuck (default 600s), so the judgement is time-based and cannot be
            fooled by something merely slow to start.
          * A container waiting on a reason that does not self-resolve
            (CrashLoopBackOff, ImagePullBackOff, ...).

        Anything else that has pods is IN_PROGRESS, which also makes a re-run WAIT
        for the rollout rather than re-applying manifests underneath it.
        """
        for cond in conds:
            if (cond.get("type") == "Progressing"
                    and cond.get("reason") == "ProgressDeadlineExceeded"):
                return FAILED, f"rollout stuck — {self._pod_reason(deployment)}"

        reason = self._pod_reason(deployment)
        if any(hard in reason for hard in self._HARD_POD_FAILURES):
            return FAILED, reason
        return IN_PROGRESS, reason or "starting"

    def _expected_manifest_files(self) -> Optional[List[str]]:
        """Manifest FILENAMES this deploy applies. None when it cannot be determined.

        Read from the scripts that do the applying, because "declared in eks/" and
        "applied by the deploy" are different sets and confusing them is a bug:

          * deploy.sh applies an explicit list, `_APPLIED_MANIFESTS`. Files in eks/
            that are not in it are never applied by a normal deploy.
          * deploy_prebid.sh (--with-prebid) applies the rest -- prebid-server AND
            amt-simulator.

        This used to glob eks/*.yaml and drop anything whose FILENAME contained
        "prebid". That heuristic missed amt-simulator-deployment.yaml, which only
        --with-prebid ever applies, so a normal deploy reported amt-simulator as
        permanently missing. That is not merely a wrong line: MISSING tells the gate
        to run the phase, so Phase 3 re-ran on every single deploy and could never
        converge, and --status could never exit 0.
        """
        deployment_dir = os.path.dirname(self.manifest_dir.rstrip(os.sep))
        files: List[str] = []

        main = os.path.join(deployment_dir, "deploy.sh")
        try:
            with open(main, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            return None
        m = re.search(r"^_APPLIED_MANIFESTS=\(([^)]*)\)", text, re.M)
        if not m:
            # The array was renamed or restructured. Say so rather than guess: the
            # fallback of "expect whatever is in eks/" is what caused the bug above.
            return None
        files.extend(re.findall(r"[\w.-]+\.ya?ml", m.group(1)))

        if self.prebid:
            prebid = os.path.join(deployment_dir, "deploy_prebid.sh")
            try:
                with open(prebid, encoding="utf-8") as handle:
                    files.extend(re.findall(r"eks/([\w.-]+\.ya?ml)", handle.read()))
            except OSError:
                return None

        return sorted(set(files))

    def _manifest_deployments(self) -> Optional[List[str]]:
        """Deployment names this deploy is expected to create -- the desired state.

        Read from the manifests rather than hardcoded so the two cannot drift. A
        line scanner, not a YAML parse: the files contain shell placeholders that a
        strict parser rejects, and all that is needed is the name under each
        `kind: Deployment`.
        """
        expected = self._expected_manifest_files()
        if expected is None:
            return None
        names: List[str] = []
        if not os.path.isdir(self.manifest_dir):
            return None
        for fname in expected:
            path = os.path.join(self.manifest_dir, fname)
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
            except OSError:
                continue
            for doc in text.split("\n---"):
                if not re.search(r"^kind:\s*Deployment\s*$", doc, re.M):
                    continue
                m = re.search(r"^metadata:\s*$\s*(?:^\s+.*$\s*)*?^\s+name:\s*(\S+)",
                              doc, re.M)
                if m:
                    names.append(m.group(1))
        return sorted(set(names))

    def _pod_reason(self, deployment: str) -> str:
        """Why a deployment is not ready, from its pods. Blank on any difficulty."""
        try:
            proc = self._kubectl("get", "pods", "-o", "json", timeout=20)
            if proc is None or proc.returncode != 0:
                return "see: kubectl describe deploy " + deployment
            for pod in json.loads(proc.stdout).get("items", []):
                name = pod.get("metadata", {}).get("name", "")
                if not name.startswith(deployment + "-"):
                    continue
                phase = pod.get("status", {}).get("phase", "")
                conds = pod.get("status", {}).get("conditions", []) or []
                ready = next((c for c in conds if c.get("type") == "Ready"), {})

                # A Running pod that is already Ready is not the problem -- keep
                # looking. A Running pod that is NOT Ready is the interesting case,
                # and skipping it (as this used to) left the caller with no reason
                # at all for the most common not-ready state there is.
                if phase == "Running" and ready.get("status") == "True":
                    continue

                # A waiting container names the actual fault (ImagePullBackOff,
                # CrashLoopBackOff), so prefer it over the generic Ready message.
                for cs in pod.get("status", {}).get("containerStatuses", []) or []:
                    w = (cs.get("state") or {}).get("waiting") or {}
                    if w.get("reason"):
                        return f"pod {phase}: {w['reason']} {w.get('message', '')[:100]}".strip()
                for cond in conds:
                    if cond.get("status") == "False" and cond.get("message"):
                        return f"pod {phase}: {cond['message'][:120]}"
                return f"pod {phase}"
        except Exception:
            pass
        return "see: kubectl describe deploy " + deployment

    # -- Phase 4: frontend + auth ----------------------------------------

    def phase4(self) -> List[Dict[str, Any]]:
        checks = []
        cf = self._client("cloudfront")
        if not cf:
            checks.append(self._check("frontend (CloudFront)", UNKNOWN, "no cloudfront client"))
        else:
            try:
                found = None
                paginator = cf.get_paginator("list_distributions")
                for page in paginator.paginate():
                    for d in (page.get("DistributionList", {}) or {}).get("Items", []) or []:
                        if d.get("Comment", "") == self.stack:
                            found = d
                            break
                    if found:
                        break
                if not found:
                    checks.append(self._check("frontend (CloudFront)", MISSING,
                                              f'no distribution with Comment "{self.stack}"'))
                elif found.get("Status") == "Deployed":
                    checks.append(self._check("frontend (CloudFront)", OK,
                                              f"https://{found.get('DomainName', '')}"))
                else:
                    checks.append(self._check("frontend (CloudFront)", IN_PROGRESS,
                                              f"{found.get('Id')} {found.get('Status')}"))
            except Exception as exc:
                checks.append(self._check("frontend (CloudFront)", UNKNOWN, str(exc)))

        idp = self._client("cognito-idp")
        pool_name = f"{self.stack}-users"
        if not idp:
            checks.append(self._check("login (Cognito)", UNKNOWN, "no cognito client"))
            return checks
        try:
            pool_id = None
            paginator = idp.get_paginator("list_user_pools")
            for page in paginator.paginate(MaxResults=60):
                for p in page.get("UserPools", []):
                    if p.get("Name") == pool_name:
                        pool_id = p.get("Id")
                        break
                if pool_id:
                    break
            if pool_id:
                checks.append(self._check("login (Cognito)", OK, pool_id))
            else:
                checks.append(self._check("login (Cognito)", MISSING, pool_name))
        except Exception as exc:
            checks.append(self._check("login (Cognito)", UNKNOWN, str(exc)))
        return checks

    # -- Phase 5: closed loop + agents -----------------------------------

    def phase5(self) -> List[Dict[str, Any]]:
        if not self.retraining:
            checks = self._agent_runtimes()
            return checks
        return self._closed_loop_stacks() + self._agent_runtimes()

    def _stack_check(self, name: str, label: Optional[str] = None) -> Dict[str, Any]:
        """One CloudFormation stack as a check line: OK / IN_PROGRESS / FAILED / MISSING."""
        title = f"{label} (stack {name})" if label else f"stack {name}"
        cfn = self._client("cloudformation")
        if not cfn:
            return self._check(title, UNKNOWN, "no cloudformation client")
        try:
            st = cfn.describe_stacks(StackName=name)["Stacks"][0]["StackStatus"]
        except ClientError as exc:
            msg = str(exc)
            if "does not exist" in msg:
                return self._check(title, MISSING, "not created")
            return self._check(title, UNKNOWN, msg[:100])
        except Exception as exc:
            return self._check(title, UNKNOWN, str(exc)[:100])
        if st.endswith("_IN_PROGRESS"):
            return self._check(title, IN_PROGRESS, st)
        if st.endswith("_COMPLETE") and not st.startswith("ROLLBACK") \
                and not st.startswith("DELETE"):
            return self._check(title, OK, st)
        return self._check(title, FAILED, st)

    def _closed_loop_stacks(self) -> List[Dict[str, Any]]:
        return [self._stack_check(prefixed(self.prefix, base)) for base in CLOSED_LOOP_STACKS]

    def _agent_runtimes(self) -> List[Dict[str, Any]]:
        rp = runtime_prefix(self.prefix)
        # Matched by suffix, not by a fully reconstructed name: the MCP runtime's name
        # is derived from the stack name with its own punctuation rules, and guessing
        # those exactly would make this probe report MISSING for a runtime that is
        # sitting right there.
        wanted_suffixes = ["_mcp"]
        if self.retraining:
            wanted_suffixes += ["AdaptiveBiddingStrategyAgent", "ModelPromotionGovernanceAgent"]
        if self.skip_agentcore_effective():
            return [self._check("agent runtimes", OK, "skipped (--skip-agentcore)")]

        c = self._client("bedrock-agentcore-control")
        if not c:
            return [self._check("agent runtimes", UNKNOWN,
                                "bedrock-agentcore-control unavailable in this botocore")]
        try:
            runtimes = []
            token = None
            while True:
                kwargs = {"nextToken": token} if token else {}
                resp = c.list_agent_runtimes(**kwargs)
                runtimes.extend(resp.get("agentRuntimes", []))
                token = resp.get("nextToken")
                if not token:
                    break
        except Exception as exc:
            return [self._check("agent runtimes", UNKNOWN, str(exc)[:120])]

        mine = {r.get("agentRuntimeName", ""): r.get("status", "")
                for r in runtimes
                if r.get("agentRuntimeName", "").startswith(rp or "")}
        checks = []
        for suffix in wanted_suffixes:
            match = next((n for n in mine if n.endswith(suffix)), None)
            if not match:
                checks.append(self._check(f"agent runtime *{suffix}", MISSING, "not created"))
                continue
            st = mine[match]
            if st == "READY":
                checks.append(self._check(f"agent runtime {match}", OK, st))
            elif st in ("CREATING", "UPDATING"):
                checks.append(self._check(f"agent runtime {match}", IN_PROGRESS, st))
            else:
                checks.append(self._check(f"agent runtime {match}", FAILED, st))
        return checks

    def skip_agentcore_effective(self) -> bool:
        return not self.agentcore

    # -- assemble ---------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        phases = {}
        for n, fn in ((1, self.phase1), (2, self.phase2), (3, self.phase3),
                      (4, self.phase4), (5, self.phase5)):
            try:
                checks = fn()
            except Exception as exc:  # a probe must never be the thing that breaks
                checks = [self._check(PHASE_LABELS[n], UNKNOWN, f"probe error: {exc}")]
            phases[n] = {"label": PHASE_LABELS[n], "status": phase_status(checks),
                         "checks": checks}
        if not self.kube_context and self.kube_reason:
            self.notes.append(self.kube_reason)
        return {
            "prefix": self.prefix,
            "stackName": self.stack,
            "cluster": self.cluster,
            "kubeContext": self.kube_context,
            "account": self.account,
            "region": self.region,
            "phases": phases,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render(result: Dict[str, Any]) -> str:
    lines = []
    head = f"Deployment status — prefix '{result['prefix'] or '<none>'}'"
    lines.append(head)
    lines.append(f"  account {result['account'] or '?'} · region {result['region']} "
                 f"· stack {result['stackName']}")
    lines.append("  (read live from AWS — safe to run any time, from anywhere, "
                 "while a deploy is running)")
    lines.append("")
    for n in sorted(result["phases"]):
        p = result["phases"][n]
        lines.append(f"  {MARKS.get(p['status'], '[?]')} Phase {n}/5  {p['label']}")
        for c in p["checks"]:
            if c["status"] == OK and p["status"] == OK:
                continue          # don't bury the problems in a wall of green
            detail = f" — {c['detail']}" if c["detail"] else ""
            lines.append(f"            {MARKS.get(c['status'], '[?]')} {c['name']}{detail}")
    for note in result.get("notes", []):
        lines.append(f"\n  note: {note}")
    return "\n".join(lines)


def overall(result: Dict[str, Any]) -> str:
    return phase_status([{"status": p["status"]} for p in result["phases"].values()])


def render_shell(result: Dict[str, Any]) -> str:
    """`eval`-able assignments, so deploy.sh can gate phases without jq.

    One probe, one interpreter start, and the shell gets everything it needs:

        DEPLOY_PHASE_1=ok
        DEPLOY_PHASE_2=in_progress
        ...
        DEPLOY_OVERALL=in_progress
        DEPLOY_BLOCKER_2='container images — repos exist but hold no image: ...'

    Values are single-quoted with embedded quotes escaped, because details carry
    arbitrary AWS message text.
    """
    def q(s: str) -> str:
        return "'" + str(s).replace("'", "'\\''") + "'"

    lines = []
    for n in sorted(result["phases"]):
        p = result["phases"][n]
        lines.append(f"DEPLOY_PHASE_{n}={p['status']}")
        blocker = next((c for c in p["checks"] if c["status"] != OK), None)
        if blocker:
            lines.append(
                f"DEPLOY_BLOCKER_{n}={q(blocker['name'] + (' — ' + blocker['detail'] if blocker['detail'] else ''))}")
        else:
            lines.append(f"DEPLOY_BLOCKER_{n}=''")
    lines.append(f"DEPLOY_OVERALL={overall(result)}")
    lines.append(f"DEPLOY_KUBE_CONTEXT={q(result.get('kubeContext') or '')}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE") or None,
                    help="AWS CLI profile for every call (default: AWS_PROFILE, else the SDK default chain)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--shell", action="store_true",
                    help="emit eval-able DEPLOY_PHASE_n=... assignments for deploy.sh")
    ap.add_argument("--no-retraining", dest="retraining", action="store_false", default=True)
    ap.add_argument("--with-prebid", dest="prebid", action="store_true", default=False)
    ap.add_argument("--skip-agentcore", dest="agentcore", action="store_false", default=True)
    ap.add_argument("--manifest-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eks"))
    args = ap.parse_args(argv)

    if boto3 is None:
        sys.stderr.write("deploy_status: boto3 is required\n")
        return 1

    probe = Probe(args.prefix, args.region, retraining=args.retraining,
                  prebid=args.prebid, agentcore=args.agentcore,
                  manifest_dir=args.manifest_dir, profile=args.profile)
    result = probe.run()
    if args.shell:
        sys.stdout.write(render_shell(result) + "\n")
    elif args.json:
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(render(result) + "\n")
    return 0 if overall(result) == OK else 1


if __name__ == "__main__":
    sys.exit(main())
