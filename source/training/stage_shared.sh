#!/bin/sh
# Stage shared modules into the training container's Docker build context.
# =======================================================================
#
# The training image is built with `source/training/container/` as its build
# context, so its Dockerfile cannot COPY anything from `source/shared/`. The
# modules listed below are therefore copied into `container/shared/` right
# before the build, and the Dockerfile COPYs that directory.
#
# `container/shared/` is generated, gitignored, and never edited by hand. The
# single definition of each staged module lives in `source/shared/`, which the
# serving containers import directly. dlrm_features.py in particular defines
# the DLRM feature vector for both the trainer and the bid shader; a second
# hand-maintained copy under the build context is what let the two sides
# describe different vectors.
#
# Ordering requirement: callers MUST run this before hashing
# `source/training/container/` for the rebuild gate (deploy_closed_loop.sh's
# _nemo_source_hash). The staged files are inside the hashed directory, so
# running it first is what makes an edit to source/shared/dlrm_features.py
# change the src-<hash> tag and force a new training image. Stage after the
# hash and the trainer silently keeps running the previous feature spec.
#
# Idempotent: safe to run repeatedly, and on an already-staged tree.
#
# Usage:
#   source/training/stage_shared.sh

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_SRC="${SCRIPT_DIR}/../shared"
STAGE_DIR="${SCRIPT_DIR}/container/shared"

# Modules the trainer imports from `shared`. Keep this list minimal: every
# entry becomes part of the training image and part of its rebuild hash.
#   __init__.py       -- makes `shared` an importable package
#   dlrm_features.py  -- the DLRM feature vector contract (stdlib only)
#   onnx_compat.py    -- filters torch.onnx.export kwargs the installed torch
#                        does not accept (stdlib only: inspect, typing)
#   shading_policy.py -- the bid policy's three coefficients, which
#                        search_policy_parameters() searches (stdlib only:
#                        dataclasses, typing)
STAGED_MODULES="__init__.py dlrm_features.py onnx_compat.py shading_policy.py"

# Guard: refuse to build if the trainer imports a `shared` module that is not
# staged. Adding an import without adding it here produces an image that builds
# and pushes cleanly, then dies on `ImportError` inside SageMaker -- after a
# full image build, an instance provision and an image pull. That is a ~25
# minute round trip to learn about a one-word omission, and it has happened.
#
# Deliberately a grep rather than a real parser: it covers the three import
# forms actually used in Python, and a form it misses only returns the build to
# the behaviour it had before this check existed.
IMPORTED_MODULES="$(
  grep -rhE '^[[:space:]]*(from[[:space:]]+shared([[:space:]]+import|\.)|import[[:space:]]+shared\.)' \
    "${SCRIPT_DIR}/container" --include='*.py' 2>/dev/null \
    | sed -E 's/^[[:space:]]*from[[:space:]]+shared[[:space:]]+import[[:space:]]+//' \
    | sed -E 's/^[[:space:]]*from[[:space:]]+shared\.([A-Za-z0-9_]+).*$/\1/' \
    | sed -E 's/^[[:space:]]*import[[:space:]]+shared\.([A-Za-z0-9_]+).*$/\1/' \
    | tr ',' '\n' \
    | sed -E 's/[[:space:]]+as[[:space:]]+.*$//' \
    | sed -E 's/[[:space:]]//g' \
    | grep -vE '^$' \
    | sort -u
)"

MISSING_MODULES=""
for imported in ${IMPORTED_MODULES}; do
  case " ${STAGED_MODULES} " in
    *" ${imported}.py "*) ;;
    *) MISSING_MODULES="${MISSING_MODULES} ${imported}.py" ;;
  esac
done

if [ -n "${MISSING_MODULES}" ]; then
  echo "stage_shared.sh: the trainer imports shared modules that are not staged:" >&2
  for missing in ${MISSING_MODULES}; do
    echo "    ${missing}" >&2
  done
  echo "  Add them to STAGED_MODULES in this script. Refusing to build an image" >&2
  echo "  that would fail on ImportError inside SageMaker." >&2
  exit 1
fi

mkdir -p "${STAGE_DIR}"

for module in ${STAGED_MODULES}; do
  src="${SHARED_SRC}/${module}"
  if [ ! -f "${src}" ]; then
    echo "stage_shared.sh: missing ${src}" >&2
    echo "  The training image needs this module. Refusing to build an image" >&2
    echo "  without it rather than fail later inside SageMaker." >&2
    exit 1
  fi
  cp "${src}" "${STAGE_DIR}/${module}"
done

# Left in the staged directory so anyone who opens it in an editor, or finds it
# in a container, knows where the real file is.
cat > "${STAGE_DIR}/README.md" <<'STAGE_README'
# Generated directory — do not edit

These files are copied from `source/shared/` by `source/training/stage_shared.sh`
immediately before the training image is built, because the image's Docker build
context is `source/training/container/` and cannot reach outside it.

Edit `source/shared/` instead. Changes there are picked up on the next build.
STAGE_README

echo "stage_shared.sh: staged ${STAGED_MODULES} into ${STAGE_DIR}"
