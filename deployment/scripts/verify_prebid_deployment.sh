#!/usr/bin/env bash
# =============================================================================
# verify_prebid_deployment.sh -- is the Prebid variant a DEPLOYED capability?
#
# Not a test of the auction's output; verify_auction.py does that. This checks the
# property that output cannot demonstrate: that everything the variant does is
# produced by the deploy scripts, and that nothing depends on a file, key or
# resource a human placed by hand.
#
# It exists because two defects in this feature were invisible to both unit tests
# and live inspection:
#
#   - the demand Lambda's S3 key was derived from the Prebid Java build-context
#     hash, so a Python-only change produced the same key, CloudFormation saw no
#     change, and the function kept serving the previous package. The fix looked
#     like it worked only because it had been applied by hand;
#   - the configuration overlay was uploaded only when absent, so a template
#     change reached a fresh account and silently never reached a cluster that had
#     already been deployed.
#
# Both are "the live environment is right, the deploy is not" faults. This script
# is the gate for that class.
#
# Usage:
#   deployment/scripts/verify_prebid_deployment.sh [--prefix nv5] [--url https://...]
#
# The live checks are skipped, not failed, when no URL or token is available, so
# this is runnable offline as a pre-commit gate.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STACK_PREFIX="${STACK_PREFIX:-}"
FRONTEND_URL="${FRONTEND_URL:-}"
# Both spellings, because the other scripts in this directory accept both and a
# silently ignored --url made the live checks report "skipped" while looking as
# though they had been requested.
_PREV=""
for arg in "$@"; do
  case "${arg}" in
    --prefix=*) STACK_PREFIX="${arg#--prefix=}" ;;
    --url=*)    FRONTEND_URL="${arg#--url=}" ;;
    --prefix|--url) ;;  # value arrives next
    -*)
      echo "warning: unrecognised option ${arg}" >&2
      ;;
    *)
      case "${_PREV}" in
        --prefix) STACK_PREFIX="${arg}" ;;
        --url)    FRONTEND_URL="${arg}" ;;
      esac
      ;;
  esac
  _PREV="${arg}"
done

PASS=0; FAIL=0; SKIP=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
skip() { echo "  SKIP  $1"; SKIP=$((SKIP+1)); }

demand_hash() {
  (cd source && find demand -type f ! -name '*.pyc' ! -path '*__pycache__*' -print0 \
    | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-12)
}

echo "=== 1. every packaged artifact is keyed on a hash of its own source ==="
BEFORE="$(demand_hash)"
printf '\n# transient probe\n' >> source/demand/artfhouse/catalog.py
CHANGED="$(demand_hash)"
# Restored by removing exactly what was appended. NOT with git checkout, which
# would discard any uncommitted work in that file.
python3 - <<'PY'
p = "source/demand/artfhouse/catalog.py"
s = open(p).read()
probe = "\n# transient probe\n"
assert s.endswith(probe), "unexpected file tail; leaving it alone"
open(p, "w").write(s[: -len(probe)])
PY
RESTORED="$(demand_hash)"

[[ "${BEFORE}" != "${CHANGED}" ]] \
  && ok "a one-line source change moves the demand key (${BEFORE} -> ${CHANGED})" \
  || bad "the demand key did not move, so a source change would not redeploy"
[[ "${BEFORE}" == "${RESTORED}" ]] \
  && ok "the key returns when the change is reverted" \
  || bad "the key did not return (${BEFORE} vs ${RESTORED})"
grep -q 'DEMAND_KEY="prebid-lambda/demand-\${DEMAND_HASH}' deployment/deploy_prebid.sh \
  && ok "DEMAND_KEY is derived from the source hash" \
  || bad "DEMAND_KEY is not derived from the source hash"
grep -q 'rendered-sha' deployment/deploy_prebid.sh \
  && ok "the config overlay is compared against its own recorded render" \
  || bad "the config overlay has no render stamp, so template changes may not apply"

echo
echo "=== 2. templates and scripts are valid ==="
for f in deployment/deploy_prebid.sh deployment/deploy.sh source/prebid/copy-bidder-files.sh \
         source/prebid/amt-simulator/server.py; do
  case "${f}" in
    *.py) python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "${f}" >/dev/null 2>&1 ;;
    *)    bash -n "${f}" >/dev/null 2>&1 ;;
  esac
  [[ $? -eq 0 ]] && ok "${f} parses" || bad "${f} does not parse"
done

if command -v aws >/dev/null 2>&1 && aws sts get-caller-identity >/dev/null 2>&1; then
  aws cloudformation validate-template \
    --template-body "file://${REPO_ROOT}/deployment/prebid_cfn.yaml" >/dev/null 2>&1 \
    && ok "prebid_cfn.yaml validates" || bad "prebid_cfn.yaml does not validate"
else
  skip "prebid_cfn.yaml validation (no AWS credentials)"
fi

python3 - <<'PY' >/dev/null 2>&1
import yaml
PLACEHOLDERS = (
    "__NAMESPACE__", "__PREBID_ROLE_ARN__", "__IMAGE__", "__CONFIG_BUCKET__", "__AWS_REGION__",
    "__TOKEN_ENDPOINT__", "__CREDENTIAL_SECRET__", "__DEMAND_ENDPOINT__", "__ORCHESTRATOR_URL__",
    "__ORCHESTRATOR_GRPC_TARGET__", "__ARTF_TRANSPORT__",
    "__ORCHESTRATOR_SCOPE__", "__ARTF_TOKEN_SCOPES__", "__AMT_SIMULATOR_ENDPOINT__",
)
for f in ("deployment/eks/prebid-server-deployment.yaml",
          "deployment/eks/amt-simulator-deployment.yaml"):
    raw = open(f).read()
    for k in PLACEHOLDERS:
        raw = raw.replace(k, "x")
    docs = [d for d in yaml.safe_load_all(raw) if d]
    assert docs, f
    # The simulator must never be publicly exposed: its caller cannot authenticate.
    for d in docs:
        if d.get("kind") == "Service" and d["metadata"]["name"] == "amt-simulator":
            assert d["spec"]["type"] == "ClusterIP", "amt-simulator Service is not ClusterIP"
PY
[[ $? -eq 0 ]] \
  && ok "both manifests parse, and the simulator Service is ClusterIP" \
  || bad "a manifest does not parse, or the simulator is publicly exposed"

echo
echo "=== 3. the destroy path removes what the deploy added ==="
for key in amt-simulator amt-simulator-code PREBID_AUCTION_URL- ARTF_MUTATIONS_REQUIRED_SCOPE-; do
  grep -q -- "${key}" deployment/deploy_prebid.sh \
    && ok "--destroy references ${key}" \
    || bad "--destroy does not reference ${key}"
done

echo
echo "=== 4. no capability depends on a hand-placed file ==="
[[ -f source/prebid/fixtures/contested-auction-request.json ]] \
  && ok "the verification request is a repo fixture, not a /tmp file" \
  || bad "the verification fixture is missing from the repo"
[[ -f source/prebid/amt-simulator/server.py ]] \
  && ok "the simulator wrapper is in the repo" \
  || bad "the simulator wrapper is missing"
if git ls-files source/prebid | grep -qE 'Amt(Bidder|Configuration|Test)\.java|ExtImpAmt\.java|amt\.(json|yaml)'; then
  bad "an upstream release file is TRACKED under source/prebid/ (Apache-2.0 in an MIT-0 tree)"
else
  ok "no upstream release file is tracked under source/prebid/"
fi

echo
echo "=== 5. the live deployment answers, and answers honestly ==="
TOKEN_FILE="${ARTF_TOKEN_FILE:-/tmp/tok2.txt}"
if [[ -z "${FRONTEND_URL}" ]]; then
  skip "live checks (pass --url https://<distribution> to enable)"
elif [[ ! -s "${TOKEN_FILE}" ]]; then
  skip "live checks (no token at ${TOKEN_FILE})"
else
  TOK="$(cat "${TOKEN_FILE}")"
  code=$(curl -sS -m 20 "${FRONTEND_URL}/api/v1/auction/status" \
    -H "Authorization: Bearer ${TOK}" -o /tmp/_auction_status.json -w '%{http_code}' 2>/dev/null)
  [[ "${code}" == "200" ]] && ok "GET /api/v1/auction/status -> 200" || bad "status -> ${code}"
  python3 - <<'PY' >/dev/null 2>&1
import json
d = json.load(open("/tmp/_auction_status.json"))
assert d.get("prebid") in ("configured", "not_configured"), d
# Whichever it is, the endpoint must say which rather than omitting it.
assert d.get("detail")
PY
  [[ $? -eq 0 ]] \
    && ok "the switch states its condition explicitly" \
    || bad "the switch response does not state its condition"
fi

echo
echo "=== ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ==="
[[ "${FAIL}" -eq 0 ]]
