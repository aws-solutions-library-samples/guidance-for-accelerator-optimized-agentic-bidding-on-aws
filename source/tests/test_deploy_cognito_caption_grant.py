"""Tests for the Auction Theater caption grant in deploy_cognito.py.

No AWS call is made: the IAM client is a stub. The point of these tests is the
generated policy document and the teardown path, both of which are checkable
offline and neither of which is checkable by reading the script.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "deployment"
    / "scripts"
    / "deploy_cognito.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("deploy_cognito_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


class FakeIam:
    """Records what would have been sent to IAM."""

    def __init__(self, *, role_exists=True):
        self.role_exists = role_exists
        self.put_calls = []
        self.deleted_policies = []
        self.deleted_roles = []

    def get_role(self, RoleName):  # noqa: N803 - boto3 casing
        if not self.role_exists:
            raise RuntimeError(f"NoSuchEntity: {RoleName}")
        return {"Role": {"Arn": f"arn:aws:iam::111122223333:role/{RoleName}"}}

    def put_role_policy(self, RoleName, PolicyName, PolicyDocument):  # noqa: N803
        self.put_calls.append((RoleName, PolicyName, json.loads(PolicyDocument)))

    def delete_role_policy(self, RoleName, PolicyName):  # noqa: N803
        self.deleted_policies.append((RoleName, PolicyName))

    def delete_role(self, RoleName):  # noqa: N803
        self.deleted_roles.append(RoleName)


class TestCaptionPolicyDocument:
    def test_grants_every_invocation_form(self, mod):
        iam = FakeIam()
        mod._put_caption_policy(iam, role_name="stack-cl-auth-role")
        (_role, name, doc) = iam.put_calls[0]
        assert name == mod._CAPTION_POLICY_NAME
        statement = doc["Statement"][0]
        assert statement["Action"] == "bedrock:Invoke*"
        assert statement["Effect"] == "Allow"

    def test_resource_is_the_wildcard_the_design_specifies(self, mod):
        """A deliberate, documented exception -- asserted so a later 'tightening'
        that breaks the global inference profile fails loudly rather than silently.
        """
        iam = FakeIam()
        mod._put_caption_policy(iam, role_name="stack-cl-auth-role")
        (_role, _name, doc) = iam.put_calls[0]
        assert doc["Statement"][0]["Resource"] == "*"

    def test_document_is_static(self, mod):
        """No account id, no region, no model id -- so the action is region-independent
        and needs no sts or bedrock lookup.
        """
        iam = FakeIam()
        mod._put_caption_policy(iam, role_name="stack-cl-auth-role")
        (_role, _name, doc) = iam.put_calls[0]
        rendered = json.dumps(doc)
        assert "111122223333" not in rendered
        assert "us-east-1" not in rendered
        assert "haiku" not in rendered.lower()
        assert "inference-profile" not in rendered

    def test_identical_for_any_region(self, mod):
        docs = []
        for role in ("a-cl-auth-role", "b-cl-auth-role"):
            iam = FakeIam()
            mod._put_caption_policy(iam, role_name=role)
            docs.append(json.dumps(iam.put_calls[0][2], sort_keys=True))
        assert docs[0] == docs[1]


class TestGrantCaptionInvoke:
    def test_makes_no_sts_or_bedrock_call(self, mod, monkeypatch, tmp_path):
        created = []

        def fake_client(service, **_kwargs):
            created.append(service)
            if service == "iam":
                return FakeIam()
            raise AssertionError(f"unexpected client: {service}")

        monkeypatch.setattr(mod.boto3, "client", fake_client)
        monkeypatch.setattr(mod, "_OUTPUTS_PATH", str(tmp_path / "outputs.json"))

        mod.grant_caption_invoke(stack_name="stack", region="eu-west-1")

        assert created == ["iam"]
        assert "sts" not in created
        assert "bedrock" not in created

    def test_attaches_the_caption_policy_only(self, mod, monkeypatch, tmp_path):
        iam = FakeIam()
        monkeypatch.setattr(mod.boto3, "client", lambda *_a, **_k: iam)
        monkeypatch.setattr(mod, "_OUTPUTS_PATH", str(tmp_path / "outputs.json"))

        mod.grant_caption_invoke(stack_name="stack", region="us-east-1")

        names = [name for (_r, name, _d) in iam.put_calls]
        assert names == [mod._CAPTION_POLICY_NAME]
        assert mod._INVOKE_POLICY_NAME not in names

    def test_fails_loudly_when_the_role_does_not_exist(self, mod, monkeypatch, tmp_path):
        monkeypatch.setattr(mod.boto3, "client", lambda *_a, **_k: FakeIam(role_exists=False))
        monkeypatch.setattr(mod, "_OUTPUTS_PATH", str(tmp_path / "outputs.json"))
        with pytest.raises(RuntimeError, match="NoSuchEntity"):
            mod.grant_caption_invoke(stack_name="stack", region="us-east-1")

    def test_is_idempotent(self, mod, monkeypatch, tmp_path):
        iam = FakeIam()
        monkeypatch.setattr(mod.boto3, "client", lambda *_a, **_k: iam)
        monkeypatch.setattr(mod, "_OUTPUTS_PATH", str(tmp_path / "outputs.json"))

        mod.grant_caption_invoke(stack_name="stack", region="us-east-1")
        mod.grant_caption_invoke(stack_name="stack", region="us-east-1")

        assert len(iam.put_calls) == 2
        assert iam.put_calls[0][2] == iam.put_calls[1][2]

    def test_records_the_grant_in_outputs(self, mod, monkeypatch, tmp_path):
        outputs = tmp_path / "outputs.json"
        monkeypatch.setattr(mod.boto3, "client", lambda *_a, **_k: FakeIam())
        monkeypatch.setattr(mod, "_OUTPUTS_PATH", str(outputs))
        mod.grant_caption_invoke(stack_name="stack", region="us-east-1")
        assert json.loads(outputs.read_text())["CaptionInvokeGranted"] is True


class TestTeardownDeletesEveryInlinePolicy:
    """IAM refuses DeleteRole while any inline policy remains, and delete_role only
    warns on failure, so a policy missing from the teardown loop orphans the role
    silently. This is the regression that test guards.
    """

    def test_both_policies_are_registered(self, mod):
        assert mod._INVOKE_POLICY_NAME in mod._ROLE_INLINE_POLICIES
        assert mod._CAPTION_POLICY_NAME in mod._ROLE_INLINE_POLICIES

    def test_destroy_attempts_deletion_of_every_registered_policy(self, mod, monkeypatch, tmp_path):
        iam = FakeIam()

        def fake_client(service, **_kwargs):
            if service == "iam":
                return iam
            return _StubCognito()

        monkeypatch.setattr(mod.boto3, "client", fake_client)
        monkeypatch.setattr(mod, "_OUTPUTS_PATH", str(tmp_path / "outputs.json"))

        mod.destroy(stack_name="stack", region="us-east-1")

        deleted = {name for (_role, name) in iam.deleted_policies}
        assert deleted == set(mod._ROLE_INLINE_POLICIES)
        assert iam.deleted_roles == [mod._auth_role_name("stack")]

    def test_every_attached_policy_is_in_the_teardown_registry(self, mod):
        """Whatever _put_* functions attach must be deletable. Catches a third grant
        added without registering it.
        """
        iam = FakeIam()
        mod._put_caption_policy(iam, role_name="r")
        mod._put_invoke_policy(
            iam, role_name="r",
            runtime_arns=["arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/x"],
        )
        attached = {name for (_r, name, _d) in iam.put_calls}
        assert attached <= set(mod._ROLE_INLINE_POLICIES)


class _EmptyPaginator:
    def __init__(self, key):
        self._key = key

    def paginate(self, **_kwargs):
        return [{self._key: []}]


class _StubCognito:
    """Enough of cognito-idp / cognito-identity for destroy() to run.

    Both pool lookups go through boto3 paginators, so the stub has to provide
    get_paginator rather than the bare list_* operations.
    """

    _PAGE_KEYS = {
        "list_identity_pools": "IdentityPools",
        "list_user_pools": "UserPools",
    }

    def get_paginator(self, operation):
        return _EmptyPaginator(self._PAGE_KEYS[operation])

    def delete_identity_pool(self, **_kwargs):
        return {}

    def delete_user_pool(self, **_kwargs):
        return {}


class TestCliSurface:
    def test_grant_caption_invoke_is_an_action(self, mod):
        parser_source = _SCRIPT.read_text()
        assert '"grant-caption-invoke"' in parser_source

    def test_action_is_dispatched(self, mod, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            mod, "grant_caption_invoke",
            lambda **kwargs: calls.append(kwargs) or {},
        )
        monkeypatch.setattr(mod, "_OUTPUTS_PATH", str(tmp_path / "outputs.json"))
        mod.main([
            "--action", "grant-caption-invoke",
            "--stack-name", "stack",
            "--region", "us-west-2",
        ])
        assert calls == [{"stack_name": "stack", "region": "us-west-2"}]
