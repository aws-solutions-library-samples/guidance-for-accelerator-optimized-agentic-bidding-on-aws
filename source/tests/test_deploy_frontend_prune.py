"""The post-upload prune in deploy_frontend.py.

No AWS call is made: the S3 client is a stub. What matters here is WHICH keys the
prune selects, and that is not checkable by reading the script -- getting it wrong
in either direction is a real failure:

  pruning too little   a removed scenario's payload stays live at a stable URL and
                       goes on being served as if the app still ships it. That is
                       what happened: bid-shading.json was deleted from the repo
                       and stayed in the bucket, returning 200 from CloudFront.

  pruning too much     `assets/` filenames are content-hashed, so every past
                       deploy's bundle is still reachable by the index.html that
                       named it. Deleting old assets would 404 a client's own
                       bundle.
"""

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "deployment"
    / "scripts"
    / "deploy_frontend.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("deploy_frontend_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


class FakeS3:
    """Records deletes; serves a fixed listing through a paginator stub."""

    def __init__(self, keys):
        self._keys = list(keys)
        self.deleted: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        outer = self

        class Paginator:
            def paginate(self, Bucket, Prefix):  # noqa: N803 - boto3 casing
                matching = [k for k in outer._keys if k.startswith(Prefix)]
                # Two pages, so a single-page implementation would fail.
                mid = len(matching) // 2
                yield {"Contents": [{"Key": k} for k in matching[:mid]]}
                yield {"Contents": [{"Key": k} for k in matching[mid:]]}

        return Paginator()

    def delete_objects(self, Bucket, Delete):  # noqa: N803 - boto3 casing
        self.deleted.extend(o["Key"] for o in Delete["Objects"])


def _dist(tmp_path, relative_paths):
    for rel in relative_paths:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    return tmp_path


class TestWhatIsPruned:
    def test_a_sample_the_build_no_longer_contains_is_deleted(self, mod, tmp_path):
        dist = _dist(tmp_path, ["index.html", "samples/kept.json"])
        s3 = FakeS3(["index.html", "samples/kept.json", "samples/removed.json"])

        mod._prune_stale(s3, "bucket", dist)

        assert s3.deleted == ["samples/removed.json"]

    def test_nothing_is_deleted_when_the_build_matches_the_bucket(self, mod, tmp_path):
        dist = _dist(tmp_path, ["samples/a.json", "samples/b.json"])
        s3 = FakeS3(["samples/a.json", "samples/b.json"])

        mod._prune_stale(s3, "bucket", dist)

        assert s3.deleted == []

    def test_deletes_are_spread_across_pages_of_the_listing(self, mod, tmp_path):
        dist = _dist(tmp_path, ["samples/keep.json"])
        stale = [f"samples/gone-{i}.json" for i in range(8)]
        s3 = FakeS3([*stale[:4], "samples/keep.json", *stale[4:]])

        mod._prune_stale(s3, "bucket", dist)

        assert sorted(s3.deleted) == sorted(stale)


class TestWhatIsNotPruned:
    def test_historical_hashed_assets_are_left_alone(self, mod, tmp_path):
        # Only the current bundle is in dist/. Every earlier one must survive.
        dist = _dist(tmp_path, ["index.html", "assets/index-NEW.js"])
        s3 = FakeS3([
            "index.html",
            "assets/index-NEW.js",
            "assets/index-OLD.js",
            "assets/index-OLDER.css",
            "assets/loadCognitoIdentity-OLD.js",
        ])

        mod._prune_stale(s3, "bucket", dist)

        assert s3.deleted == []

    def test_only_declared_prefixes_are_considered(self, mod, tmp_path):
        dist = _dist(tmp_path, ["index.html"])
        s3 = FakeS3([
            "index.html",
            "fonts/AmazonEmber_Rg.ttf",
            "some-other-object.txt",
        ])

        mod._prune_stale(s3, "bucket", dist)

        # Neither fonts/ nor a root object is in _PRUNED_PREFIXES, so neither is
        # touched even though dist/ does not contain them.
        assert s3.deleted == []

    def test_the_pruned_prefix_list_does_not_include_assets(self, mod):
        # Stated explicitly so the guard cannot be removed by widening the tuple.
        assert "assets/" not in mod._PRUNED_PREFIXES
        assert mod._PRUNED_PREFIXES == ("samples/",)


class TestItIsWiredIntoTheUpload:
    def test_build_and_upload_calls_the_prune(self, mod):
        import inspect

        source = inspect.getsource(mod._build_and_upload_react)
        assert "_prune_stale(" in source
