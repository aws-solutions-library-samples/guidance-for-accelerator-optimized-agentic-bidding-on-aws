"""deploy.sh _source_hash: the image content-hash cache key.

The functions are extracted from the real deploy.sh and run with bash against a
throwaway git repository laid out like this one (deployment/ + source/), so the
test exercises the shipped shell code rather than a copy of it.

Regression: the hash used a plain `find`, so __pycache__/*.pyc written by a local
pytest run changed every Python image's hash and the next deploy rebuilt all of
them even when only one image's source had changed.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY_SH = Path(__file__).resolve().parents[2] / "deployment" / "deploy.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _extract(name: str, text: str) -> str:
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert m, f"{name}() not found in deploy.sh"
    return m.group(0)


@pytest.fixture()
def repo(tmp_path):
    text = DEPLOY_SH.read_text()
    fns = "".join(_extract(n, text) for n in ("_sha256", "_hashable_files", "_source_hash"))
    root = tmp_path / "repo"
    (root / "deployment").mkdir(parents=True)
    (root / "deployment" / "hashfns.sh").write_text(fns)
    src = root / "source"
    (src / "triton").mkdir(parents=True)
    (src / "triton" / "Dockerfile.triton-artf").write_text("FROM x\n")
    (src / "shared").mkdir()
    (src / "shared" / "artf.py").write_text("A = 1\n")
    c = src / "containers" / "yield_optimizer_floor"
    c.mkdir(parents=True)
    (c / "app.py").write_text("print('floor')\n")
    (root / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.pytest_cache/\n.hypothesis/\n**/.DS_Store\n**/node_modules/\n"
    )

    def git(*a):
        subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return root


def _hash(root: Path, key: str = "yield-optimizer-floor") -> str:
    script = (
        "set -u\n"
        f"SCRIPT_DIR='{root / 'deployment'}'\n"
        f". '{root / 'deployment' / 'hashfns.sh'}'\n"
        f"_source_hash {key}\n"
    )
    out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    h = out.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{16}", h), h
    return h


def test_gitignored_artifacts_do_not_change_the_hash(repo):
    before = _hash(repo)
    src = repo / "source"
    pc = src / "shared" / "__pycache__"
    pc.mkdir()
    (pc / "artf.cpython-313.pyc").write_bytes(b"\x00compiled")
    (src / "containers" / "yield_optimizer_floor" / "stray.pyc").write_bytes(b"\x01")
    (src / "shared" / ".DS_Store").write_bytes(b"\x02")
    (src / "shared" / ".hypothesis").mkdir()
    (src / "shared" / ".hypothesis" / "db").write_text("x")
    (src / "shared" / ".pytest_cache").mkdir()
    (src / "shared" / ".pytest_cache" / "v").write_text("x")
    assert _hash(repo) == before


def test_uncommitted_edit_to_tracked_file_changes_the_hash(repo):
    before = _hash(repo)
    (repo / "source" / "containers" / "yield_optimizer_floor" / "app.py").write_text("print('v2')\n")
    assert _hash(repo) != before


def test_new_untracked_source_file_changes_the_hash(repo):
    before = _hash(repo)
    (repo / "source" / "shared" / "new_module.py").write_text("B = 2\n")
    assert _hash(repo) != before


def test_deleted_tracked_file_changes_the_hash(repo):
    before = _hash(repo)
    (repo / "source" / "shared" / "artf.py").unlink()
    assert _hash(repo) != before


def test_other_images_source_does_not_change_this_hash(repo):
    before = _hash(repo)
    other = repo / "source" / "containers" / "yield_optimizer_margin"
    other.mkdir()
    (other / "app.py").write_text("print('margin')\n")
    assert _hash(repo) == before


def test_fallback_without_git_excludes_artifacts(repo):
    shutil.rmtree(repo / ".git")
    before = _hash(repo)
    pc = repo / "source" / "shared" / "__pycache__"
    pc.mkdir()
    (pc / "artf.cpython-313.pyc").write_bytes(b"\x00")
    (repo / "source" / "shared" / ".DS_Store").write_bytes(b"\x02")
    assert _hash(repo) == before
    (repo / "source" / "shared" / "artf.py").write_text("A = 3\n")
    assert _hash(repo) != before
