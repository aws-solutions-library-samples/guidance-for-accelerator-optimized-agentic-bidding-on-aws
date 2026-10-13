"""Deploy wiring for the Deal Scorer build and the demand Lambda's deals seed.

Static checks on the files deploy.sh and deploy_prebid.sh ship: the deal-scorer
Deployment receives NCF_MODEL_LABEL through a placeholder deploy.sh substitutes,
and prebid_cfn.yaml maps DealsSeedFingerprint to the demand function's
DEALS_SEED_FINGERPRINT, which deploy_prebid.sh fills with the seed's sha256.
The shell behaviour of the helpers is covered in
deployment/tests/test_deploy_phases.sh (test_ncf_build_wiring).
"""

import hashlib
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from shared import ncf_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = REPO_ROOT / "deployment"
DEPLOY_SH = DEPLOYMENT / "deploy.sh"
DEPLOY_PREBID_SH = DEPLOYMENT / "deploy_prebid.sh"
PREBID_CFN = DEPLOYMENT / "prebid_cfn.yaml"
ARTF_MANIFEST = DEPLOYMENT / "eks" / "artf-containers-deployment.yaml"
DEALS_SEED = DEPLOYMENT / "seed" / "deals_and_campaigns.json"

SAMPLE_LABEL = "ncf-neumf-sample-0123456789ab"


class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that keeps CloudFormation short-form intrinsics (!Ref, !Sub, ...)
    as plain {tag: value} dicts so the template can be inspected."""


def _cfn_tag(loader, tag_suffix, node):
    name = "Ref" if tag_suffix == "Ref" else f"Fn::{tag_suffix}"
    if isinstance(node, yaml.ScalarNode):
        return {name: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {name: loader.construct_sequence(node)}
    return {name: loader.construct_mapping(node)}


_CfnLoader.add_multi_constructor("!", _cfn_tag)


@pytest.fixture(scope="module")
def deploy_sh():
    return DEPLOY_SH.read_text()


@pytest.fixture(scope="module")
def prebid_cfn():
    # yaml.safe_load would reject the CloudFormation tags; _CfnLoader is a SafeLoader
    # with those tags registered, so drive it directly instead of via yaml.load.
    loader = _CfnLoader(PREBID_CFN.read_text())
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _sed_placeholders(script):
    """Placeholder names the deploy.sh manifest sed loop substitutes."""
    return set(re.findall(r'-e "s\|__([A-Z0-9_]+)__\|', script))


def _render_manifest(substitutions):
    text = ARTF_MANIFEST.read_text()
    for name, value in substitutions.items():
        text = text.replace(f"__{name}__", value)
    return text


def _deal_scorer_env(rendered):
    for doc in yaml.safe_load_all(rendered):
        if not doc or doc.get("kind") != "Deployment":
            continue
        if doc["metadata"]["name"] != "deal-scorer":
            continue
        containers = doc["spec"]["template"]["spec"]["containers"]
        return {e["name"]: e.get("value") for c in containers for e in c.get("env", [])}
    raise AssertionError("artf-containers-deployment.yaml has no deal-scorer Deployment")


class TestDealScorerLabel:
    def test_label_rule_matches_the_container(self, deploy_sh):
        match = re.search(r"^NCF_LABEL_RE='([^']+)'$", deploy_sh, re.MULTILINE)
        assert match, "deploy.sh defines no NCF_LABEL_RE"
        assert match.group(1) == ncf_ids.MODEL_LABEL_RE

    def test_every_manifest_placeholder_is_substituted(self, deploy_sh):
        in_manifest = set(re.findall(r"__([A-Z0-9_]+)__", ARTF_MANIFEST.read_text()))
        assert "NCF_MODEL_LABEL" in in_manifest
        assert in_manifest <= _sed_placeholders(deploy_sh)

    def test_rendered_manifest_carries_the_label(self):
        names = set(re.findall(r"__([A-Z0-9_]+)__", ARTF_MANIFEST.read_text()))
        values = {name: f"t-{name.lower()}" for name in names}
        values["NCF_MODEL_LABEL"] = SAMPLE_LABEL
        rendered = _render_manifest(values)
        assert "__NCF_MODEL_LABEL__" not in rendered
        env = _deal_scorer_env(rendered)
        assert env["NCF_MODEL_LABEL"] == SAMPLE_LABEL
        assert re.match(ncf_ids.MODEL_LABEL_RE, env["NCF_MODEL_LABEL"])

    def test_only_the_deal_scorer_gets_the_label(self):
        rendered = _render_manifest({"NCF_MODEL_LABEL": SAMPLE_LABEL})
        assert rendered.count(SAMPLE_LABEL) == 1

    def test_phase3_resolves_and_checks_the_label(self, deploy_sh):
        assert '-e "s|__NCF_MODEL_LABEL__|${NCF_MODEL_LABEL}|g"' in deploy_sh
        assert "grep -q '__NCF_MODEL_LABEL__'" in deploy_sh
        phase3_start = deploy_sh.index('step "Step 8')
        assert "resolve_ncf_model_label" in deploy_sh[phase3_start : phase3_start + 2000]

    def test_onnxruntime_is_a_required_package(self, deploy_sh):
        match = re.search(r'^REQUIRED_PY_PACKAGES="([^"]*)"', deploy_sh, re.MULTILINE)
        assert match
        assert "onnxruntime" in match.group(1).split()

    @pytest.mark.parametrize("missing", [("onnxruntime",), ("onnxruntime", "boto3"), ()])
    def test_a_missing_onnxruntime_is_installed_pinned(self, deploy_sh, tmp_path, missing):
        """The install block, run under /bin/bash with a stub interpreter."""
        start = deploy_sh.index('REQUIRED_PY_PACKAGES="')
        end = deploy_sh.index("\nfi\n", deploy_sh.index('pip install --quiet ${MISSING_PY_PACKAGES}', start)) + 4
        pip_log = tmp_path / "pip.log"
        stub = tmp_path / "python"
        stub.write_text(
            "#!/bin/bash\n"
            'if [[ "$1" == "-c" ]]; then\n'
            '  case " ${MISSING} " in *" ${2#import } "*) exit 1 ;; esac\n'
            "  exit 0\n"
            "fi\n"
            f'echo "$@" >> "{pip_log}"\n'
        )
        stub.chmod(0o755)
        script = tmp_path / "install.sh"
        script.write_text(
            "set -euo pipefail\n"
            "log() { :; }\n"
            'fail() { echo "fail: $*" >&2; exit 9; }\n'
            f'PYTHON="{stub}"\n' + deploy_sh[start:end]
        )
        env = {"PATH": "/usr/bin:/bin", "MISSING": " ".join(missing)}
        result = subprocess.run(["/bin/bash", str(script)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        if not missing:
            assert not pip_log.exists()
            return
        installed = pip_log.read_text().split()
        assert installed[:3] == ["-m", "pip", "install"]
        assert "onnxruntime==1.27.0" in installed
        assert "onnxruntime" not in installed
        assert ("boto3" in installed) == ("boto3" in missing)


class TestDemandServiceWiring:
    """The demand service runs in-cluster and refreshes its catalog itself, so the
    stack no longer carries a Lambda or a seed fingerprint to force a reload."""

    def test_the_lambda_and_gateway_are_gone(self, prebid_cfn):
        resources = prebid_cfn["Resources"]
        for name in ("DemandFunction", "DemandApi", "DemandAuthorizer", "DemandRoute",
                     "DemandStage", "DemandAccessLogGroup", "DemandInvokePermission",
                     "DemandFunctionRole"):
            assert name not in resources
        assert "DealsSeedFingerprint" not in prebid_cfn["Parameters"]
        assert "DemandEndpointUrl" not in prebid_cfn["Outputs"]

    def test_the_demand_role_is_read_only_on_the_deals_table(self, prebid_cfn):
        role = prebid_cfn["Resources"]["DemandServiceRole"]
        statements = [st for pol in role["Properties"]["Policies"]
                      for st in pol["PolicyDocument"]["Statement"]]
        actions = {a for st in statements for a in st["Action"]}
        assert actions == {"dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan"}
        assert "DealsTableName" in str(statements[0]["Resource"])
        assert "DemandServiceAccountName" in str(role["Properties"]["AssumeRolePolicyDocument"])

    def test_deploy_prebid_no_longer_packages_a_lambda(self):
        script = DEPLOY_PREBID_SH.read_text()
        assert "DealsSeedFingerprint" not in script
        assert "artfhouse-demand.zip" not in script
        assert 'DEMAND_ENDPOINT="http://artfhouse-demand.${NAMESPACE}.svc.cluster.local/openrtb2/auction"' in script

    def test_deploy_sh_marker_hashes_the_same_seed(self, deploy_sh):
        assert 'write_phase1_marker \\\n  "$(file_sha256 "${SCRIPT_DIR}/seed/deals_and_campaigns.json")"' in deploy_sh
