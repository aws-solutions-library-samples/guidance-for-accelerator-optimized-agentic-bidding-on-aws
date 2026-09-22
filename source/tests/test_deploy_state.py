"""Tests for deployment/scripts/deploy_state.py.

The module under test is now much smaller than it was: it remembers the flags a
previous run was GIVEN, and nothing about progress. Progress comes from AWS (see
deploy_status.py), because a local file recording "what the script did" is not the
same thing as "what exists" -- it once reported Phase 3 complete for a deployment
whose Triton pod had never started.

So these tests cover two things only:
  1. the file can never break a deploy (corrupt input, unwritable path, odd types)
  2. remembered values are applied to the right prefix, and secrets never land on
     disk
"""

import json
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# deployment/scripts is not a package, so it goes on sys.path directly
# (same approach as test_prebid_release.py).
_SCRIPTS = Path(__file__).resolve().parents[2] / "deployment" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from deploy_state import (  # noqa: E402
    REDACTED,
    SCHEMA_VERSION,
    dump,
    empty_document,
    empty_record,
    identity_conflicts,
    load,
    main,
    record_for,
    redact,
    set_values,
)


# --------------------------------------------------------------------- load()
# Every case here is a file the user could plausibly end up with, and none of them
# may raise: a corrupt state file must cost only the remembered flags it held.


@pytest.mark.parametrize("text", [None, "", "   \n", "not json", "[1,2,3]", '"a string"',
                                  "{", '{"deployments": []}'])
def test_unusable_input_yields_a_fresh_document(text):
    assert load(text) == empty_document()


def test_a_valid_document_round_trips():
    doc = empty_document()
    doc["deployments"]["stg"] = dict(empty_record(), remembered={"region": "us-west-2"})
    assert load(dump(doc))["deployments"]["stg"]["remembered"]["region"] == "us-west-2"


def test_a_document_from_another_schema_version_is_discarded():
    """Including schema 1, which carried the progress tracking this version removed.

    Not migrated on purpose: that data was the unreliable part, and the remembered
    flags beside it are cheap to supply again. Silently reading a shape this code no
    longer understands could point a deploy at the wrong region or stack.
    """
    old = {"schemaVersion": 1, "deployments": {"stg": {"remembered": {"region": "eu-west-1"},
                                                      "phases": [{"n": 1, "status": "complete"}]}}}
    assert load(json.dumps(old)) == empty_document()


def test_non_scalar_values_are_dropped_rather_than_carried():
    raw = {"schemaVersion": SCHEMA_VERSION,
           "deployments": {"a": {"remembered": {"ok": "yes", "bad": {"nested": 1}}}}}
    remembered = load(json.dumps(raw))["deployments"]["a"]["remembered"]
    assert remembered == {"ok": "yes"}


def test_a_record_that_is_not_a_dict_is_skipped():
    raw = {"schemaVersion": SCHEMA_VERSION, "deployments": {"a": "nonsense", "b": {}}}
    doc = load(json.dumps(raw))
    assert "a" not in doc["deployments"]
    assert "b" in doc["deployments"]


# ---------------------------------------------------------------- record_for()


def test_an_absent_prefix_yields_a_fresh_record():
    assert record_for(empty_document(), "never-seen") == empty_record()


def test_the_empty_prefix_is_a_distinct_key_from_a_named_one():
    """A plain ./deploy.sh and --prefix stg must never read each other's values."""
    doc = set_values(empty_document(), "", "remembered", {"region": "us-east-1"})
    doc = set_values(doc, "stg", "remembered", {"region": "eu-west-1"})
    assert record_for(doc, "")["remembered"]["region"] == "us-east-1"
    assert record_for(doc, "stg")["remembered"]["region"] == "eu-west-1"


# --------------------------------------------------------------- set_values()


def test_values_are_written_into_the_named_section():
    doc = set_values(empty_document(), "stg", "remembered", {"maxGPUs": "5"})
    assert record_for(doc, "stg")["remembered"]["maxGPUs"] == "5"
    assert record_for(doc, "stg")["resolved"] == {}


@pytest.mark.parametrize("section", ["phases", "", "REMEMBERED", "other"])
def test_an_unknown_section_is_ignored(section):
    doc = set_values(empty_document(), "stg", section, {"k": "v"})
    assert doc == empty_document()


def test_an_empty_value_never_overwrites_a_stored_one():
    """This is what lets a run that skipped Phase 4 still print the frontend URL."""
    doc = set_values(empty_document(), "stg", "resolved", {"cloudFrontDomain": "d1.net"})
    doc = set_values(doc, "stg", "resolved", {"cloudFrontDomain": ""})
    assert record_for(doc, "stg")["resolved"]["cloudFrontDomain"] == "d1.net"


def test_an_empty_value_does_write_when_nothing_is_stored():
    doc = set_values(empty_document(), "stg", "resolved", {"nlbDns": ""})
    assert record_for(doc, "stg")["resolved"]["nlbDns"] == ""


def test_set_values_does_not_mutate_its_input():
    original = empty_document()
    set_values(original, "stg", "remembered", {"k": "v"})
    assert original == empty_document()


# -------------------------------------------------------- identity_conflicts()
# A conflict means the record describes a DIFFERENT environment under the same
# prefix key. Reusing its values would silently point the run at the wrong place.


def test_no_conflict_when_identity_matches():
    record = dict(empty_record(), identity={"account": "111", "region": "us-east-1"})
    assert identity_conflicts(record, account="111", region="us-east-1") == []


def test_a_differing_account_is_a_conflict():
    record = dict(empty_record(), identity={"account": "111"})
    assert identity_conflicts(record, account="222") == ["account"]


def test_every_differing_field_is_reported():
    record = dict(empty_record(),
                  identity={"account": "111", "region": "us-east-1",
                            "stackName": "a", "clusterName": "a-triton"})
    conflicts = identity_conflicts(record, account="222", region="eu-west-1",
                                   stackName="b", clusterName="b-triton")
    assert set(conflicts) == {"account", "region", "stackName", "clusterName"}


def test_an_absent_stored_value_is_not_a_conflict():
    """First run for a prefix: nothing stored yet, so nothing can disagree."""
    assert identity_conflicts(empty_record(), account="111") == []


def test_an_absent_incoming_value_is_not_a_conflict():
    record = dict(empty_record(), identity={"account": "111"})
    assert identity_conflicts(record, account="") == []


# ------------------------------------------------------------------- redact()
# The recorded argv must never contain a credential, in any position.


@pytest.mark.parametrize("argv,expected", [
    (["--ngc-key", "SECRET"], ["--ngc-key", REDACTED]),
    (["--ngc-key=SECRET"], [f"--ngc-key={REDACTED}"]),
    (["--ngc-key"], ["--ngc-key"]),                          # flag with no value
    (["--prefix", "stg"], ["--prefix", "stg"]),
    ([], []),
])
def test_redact_removes_secret_values_and_nothing_else(argv, expected):
    assert redact(argv) == expected


def test_redact_never_leaks_a_key_in_any_position():
    for argv in (["--ngc-key", "K", "--prefix", "p"],
                 ["--prefix", "p", "--ngc-key", "K"],
                 ["--ngc-key=K", "--verbose"]):
        assert "K" not in redact(argv)


# ----------------------------------------------------------------- CLI / I/O


def test_set_then_read_round_trips_through_a_file(tmp_path, capsys):
    path = str(tmp_path / "state.json")
    main(["--file", path, "--prefix", "stg", "set", "--section", "remembered",
          "--kv", "region=us-west-2"])
    capsys.readouterr()
    main(["--file", path, "--prefix", "stg", "read", "--field", "remembered.region"])
    assert capsys.readouterr().out == "us-west-2"


def test_reading_an_absent_field_prints_nothing(tmp_path, capsys):
    path = str(tmp_path / "state.json")
    main(["--file", path, "read", "--field", "remembered.nope"])
    assert capsys.readouterr().out == ""


def test_reading_a_missing_file_is_not_an_error(tmp_path, capsys):
    rc = main(["--file", str(tmp_path / "absent.json"), "read", "--field", "remembered.region"])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_an_unwritable_path_still_exits_zero(tmp_path):
    """A state write failure must never end a deployment."""
    unwritable = tmp_path / "nodir"
    unwritable.write_text("i am a file, not a directory")
    rc = main(["--file", str(unwritable / "state.json"), "set",
               "--section", "remembered", "--kv", "region=us-east-1"])
    assert rc == 0


def test_start_run_redacts_the_ngc_key_on_disk(tmp_path):
    path = tmp_path / "state.json"
    main(["--file", str(path), "start-run", "--account", "1", "--region", "us-east-1",
          "--argv", "--ngc-key", "SUPERSECRET", "--prefix", "stg"])
    assert "SUPERSECRET" not in path.read_text()


def test_start_run_records_identity_and_argv(tmp_path):
    path = tmp_path / "state.json"
    main(["--file", str(path), "--prefix", "stg", "start-run",
          "--account", "111", "--region", "us-east-1", "--stack", "s", "--cluster", "c",
          "--argv", "--prefix", "stg"])
    record = record_for(load(path.read_text()), "stg")
    assert record["identity"] == {"account": "111", "region": "us-east-1",
                                  "stackName": "s", "clusterName": "c"}
    assert record["lastRun"]["invokedWith"] == ["--prefix", "stg"]
    assert record["lastRun"]["startedAt"].endswith("Z")


def test_start_run_records_no_pid_host_or_outcome(tmp_path):
    """Those existed so a second invocation could find and follow the first.

    AWS is asked directly now, so they serve nothing and could only go stale. Their
    absence is asserted rather than assumed, because re-adding them would quietly
    reintroduce a second source of truth.
    """
    path = tmp_path / "state.json"
    main(["--file", str(path), "start-run", "--account", "1"])
    last = record_for(load(path.read_text()), "")["lastRun"]
    for gone in ("pid", "host", "logFile", "outcome", "waitingFor", "endedAt"):
        assert gone not in last


def test_start_run_reports_an_identity_conflict_on_stdout(tmp_path, capsys):
    path = str(tmp_path / "state.json")
    main(["--file", path, "start-run", "--account", "111", "--region", "us-east-1"])
    capsys.readouterr()
    main(["--file", path, "start-run", "--account", "222", "--region", "us-east-1"])
    assert "account" in capsys.readouterr().out


def test_clear_removes_only_the_named_prefix(tmp_path):
    path = tmp_path / "state.json"
    main(["--file", str(path), "--prefix", "a", "set", "--section", "remembered",
          "--kv", "region=us-east-1"])
    main(["--file", str(path), "--prefix", "b", "set", "--section", "remembered",
          "--kv", "region=eu-west-1"])
    main(["--file", str(path), "--prefix", "a", "clear"])
    doc = load(path.read_text())
    assert "a" not in doc["deployments"]
    assert doc["deployments"]["b"]["remembered"]["region"] == "eu-west-1"


def test_clear_all_empties_the_document(tmp_path):
    path = tmp_path / "state.json"
    main(["--file", str(path), "--prefix", "a", "set", "--section", "remembered",
          "--kv", "region=us-east-1"])
    main(["--file", str(path), "clear", "--all"])
    assert load(path.read_text()) == empty_document()


def test_read_record_emits_parseable_json(tmp_path, capsys):
    path = str(tmp_path / "state.json")
    main(["--file", path, "--prefix", "stg", "set", "--section", "remembered",
          "--kv", "region=us-east-1"])
    capsys.readouterr()
    main(["--file", path, "--prefix", "stg", "read-record"])
    assert json.loads(capsys.readouterr().out)["remembered"]["region"] == "us-east-1"


def test_a_written_file_is_valid_json_with_the_current_schema_version(tmp_path):
    path = tmp_path / "state.json"
    main(["--file", str(path), "set", "--section", "remembered", "--kv", "region=us-east-1"])
    assert json.loads(path.read_text())["schemaVersion"] == SCHEMA_VERSION


# ------------------------------------------------------- property-based tests

_prefixes = st.sampled_from(["", "stg", "v1", "prod-2"])
_keys = st.sampled_from(["region", "maxGPUs", "artfNodeRole", "ngcSecretName", "imageTag"])
_values = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=0, max_size=40
)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(_prefixes, _keys, _values), max_size=12))
def test_a_document_always_survives_a_dump_load_round_trip(writes):
    """Whatever sequence of writes happened, the file must still be readable."""
    doc = empty_document()
    for prefix, key, value in writes:
        doc = set_values(doc, prefix, "remembered", {key: value})
    assert load(dump(doc)) == doc


@settings(max_examples=150, deadline=None)
@given(_prefixes, _keys, _values.filter(lambda v: v != ""))
def test_a_nonempty_value_is_never_lost_to_a_later_empty_write(prefix, key, value):
    doc = set_values(empty_document(), prefix, "remembered", {key: value})
    doc = set_values(doc, prefix, "remembered", {key: ""})
    assert record_for(doc, prefix)["remembered"][key] == value


@settings(max_examples=150, deadline=None)
@given(st.lists(st.text(max_size=20), max_size=10))
def test_redact_preserves_argv_length(argv):
    """A redacted argv must stay a faithful record of the SHAPE of the invocation."""
    assert len(redact(argv)) == len(argv)


@settings(max_examples=100, deadline=None)
@given(st.text(min_size=1, max_size=30).filter(lambda s: s.strip() != ""))
def test_no_ngc_key_value_survives_redaction(secret):
    assert secret not in redact(["--ngc-key", secret]) or secret == REDACTED
