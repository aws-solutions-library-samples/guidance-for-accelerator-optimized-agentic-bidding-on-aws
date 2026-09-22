#!/usr/bin/env python3
"""Remembered INPUTS for deploy.sh. Deliberately not progress.

Tracks, per `--prefix`, the flags a previous run was GIVEN, so a re-run does not
have to re-ask for them. The NGC secret name is the case that justifies this file
existing at all: pass `--ngc-key` once and later runs of the same prefix find the
stored secret on their own.

**Progress is NOT here, and that is the point.** An earlier version of this file
recorded which phases had completed, and it was actively harmful: it recorded what
the *script* had done, which is not the same as what *exists*. It reported
"Phase 3 complete" for a deployment whose Triton pod had never started, it could
not be read from another machine or after a reboot, and it was wrong the moment
anything changed outside the script. Deployment state now comes from AWS --
see `scripts/deploy_status.py`, which asks the account.

What is left is exactly the set AWS cannot answer: "what did you type last time?"

State lives at `deployment/.deploy-state.json`, is gitignored, and is deleted by
`deploy.sh --destroy`.

Two properties the rest of the design rests on:

1. **The file is an optimisation, never a requirement.** Every command here exits
   0 even when the state layer fails, and every parse error yields a fresh empty
   document. Deleting the file costs you the remembered flags and nothing else.
   Callers run under `set -e`, so a non-zero exit from a state read would take down
   a 30-minute deployment over a cache miss.

2. **`remembered` and `resolved` are separate sections.** `remembered` holds inputs
   a later run may re-apply; `resolved` holds outputs it may only display. The
   merge path only ever reads `remembered`, so a resolved value cannot be fed back
   in as an input.

No secret material is written. `--ngc-key`'s value is replaced with a placeholder
in the recorded argv; the secret itself stays in Secrets Manager, and the temporary
Cognito password is never persisted at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 2

SECTIONS = ("remembered", "resolved")

REDACTED = "<redacted>"
SECRET_FLAGS = ("--ngc-key",)

_IDENTITY_FIELDS = ("account", "region", "stackName", "clusterName")


# ---------------------------------------------------------------------------
# Pure functions — no I/O, no globals. Everything below this line is unit
# testable without an AWS account or a filesystem.
# ---------------------------------------------------------------------------


def empty_document() -> Dict[str, Any]:
    return {"schemaVersion": SCHEMA_VERSION, "deployments": {}}


def empty_record() -> Dict[str, Any]:
    return {"identity": {}, "lastRun": {}, "remembered": {}, "resolved": {}}


def load(text: Optional[str]) -> Dict[str, Any]:
    """Parse a state document, tolerating anything.

    An empty file, truncated JSON, a JSON array, or a document from a different
    schema version all yield a usable document rather than an exception. A corrupt
    state file must cost the user nothing beyond the remembered values it held.

    Schema 1 documents (which carried a `phases` array) are intentionally NOT
    migrated: that data was the unreliable progress tracking this version removes,
    and the remembered flags it sat next to are cheap to supply again.
    """
    if not text or not text.strip():
        return empty_document()
    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        return empty_document()
    if not isinstance(raw, dict):
        return empty_document()
    if raw.get("schemaVersion") != SCHEMA_VERSION:
        return empty_document()

    deployments = raw.get("deployments")
    if not isinstance(deployments, dict):
        return empty_document()

    doc = empty_document()
    for prefix, record in deployments.items():
        if not isinstance(prefix, str) or not isinstance(record, dict):
            continue
        doc["deployments"][prefix] = _normalise_record(record)
    return doc


def _normalise_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce one record into the canonical shape, dropping anything unusable."""
    clean = empty_record()
    for key in ("identity", "lastRun", "remembered", "resolved"):
        value = record.get(key)
        if isinstance(value, dict):
            clean[key] = {
                k: v for k, v in value.items()
                if isinstance(k, str) and isinstance(v, (str, int, float, bool, list))
            }
    return clean


def dump(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def record_for(doc: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """The record for a prefix. An absent prefix yields a fresh record.

    The empty prefix is a real key, distinct from every named prefix, so a plain
    `./deploy.sh` and `--prefix stg` can never read each other's values.
    """
    record = doc.get("deployments", {}).get(prefix)
    if isinstance(record, dict):
        return record
    return empty_record()


def set_values(
    doc: Dict[str, Any], prefix: str, section: str, pairs: Dict[str, Any]
) -> Dict[str, Any]:
    """Write into `remembered` or `resolved`.

    An empty incoming value never overwrites a stored non-empty one. A phase that
    did not run this time resolves nothing, and must not blank what an earlier run
    did resolve -- that is what lets a run which skipped Phase 4 still print the
    frontend URL.
    """
    if section not in SECTIONS:
        return doc
    record = record_for(doc, prefix)
    merged = dict(record.get(section, {}))
    for key, value in pairs.items():
        if value in ("", None) and merged.get(key) not in ("", None):
            continue
        merged[key] = value
    record = dict(record)
    record[section] = merged
    doc = dict(doc)
    doc["deployments"] = dict(doc.get("deployments", {}))
    doc["deployments"][prefix] = record
    return doc


def identity_conflicts(record: Dict[str, Any], **current: str) -> List[str]:
    """Identity fields whose stored value disagrees with this invocation.

    A conflict means the record describes a different environment than the one
    being deployed -- a different account, or a `STACK_NAME` env override under the
    same prefix. Remembered values are not reused across one.
    """
    stored = record.get("identity", {})
    conflicts = []
    for field in _IDENTITY_FIELDS:
        incoming = current.get(field)
        if not incoming:
            continue
        existing = stored.get(field)
        if existing and existing != incoming:
            conflicts.append(field)
    return conflicts


def redact(argv: Iterable[str]) -> List[str]:
    """Replace secret flag values so the recorded argv holds no credentials.

    Handles `--ngc-key K`, `--ngc-key=K`, and `--ngc-key` as the final argument
    with no value following it.
    """
    out: List[str] = []
    expect_value_for = None
    for arg in argv:
        if expect_value_for is not None:
            out.append(REDACTED)
            expect_value_for = None
            continue
        matched = False
        for flag in SECRET_FLAGS:
            if arg == flag:
                out.append(arg)
                expect_value_for = flag
                matched = True
                break
            if arg.startswith(flag + "="):
                out.append(f"{flag}={REDACTED}")
                matched = True
                break
        if not matched:
            out.append(arg)
    return out


def _dotted(data: Dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# I/O boundary — the only place that touches the filesystem.
# ---------------------------------------------------------------------------


def read_document(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return load(handle.read())
    except FileNotFoundError:
        return empty_document()
    except OSError as exc:
        _warn(f"could not read {path}: {exc}")
        return empty_document()


def write_document(path: str, doc: Dict[str, Any]) -> bool:
    """Atomic write: temp file in the SAME directory, then os.replace.

    Same directory matters -- os.replace is only atomic within one filesystem,
    and the system temp dir is frequently a different one.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".deploy-state-", suffix=".tmp",
            delete=False,
        )
        try:
            handle.write(dump(doc))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, path)
        return True
    except OSError as exc:
        _warn(f"could not write {path}: {exc}")
        try:
            os.unlink(handle.name)
        except Exception:
            pass
        return False


def _warn(message: str) -> None:
    sys.stderr.write(f"\033[0;33m[state][warn]\033[0m {message}\n")


def _kv_pairs(items: Optional[List[str]]) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key:
            pairs[key] = value
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".deploy-state.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None, help="state file path")
    parser.add_argument("--prefix", default="", help="deployment prefix key ('' is valid)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("read", help="print one dotted field from the record")
    p.add_argument("--field", required=True)

    sub.add_parser("read-record", help="print the whole record as JSON")

    p = sub.add_parser("set")
    p.add_argument("--section", required=True, choices=SECTIONS)
    p.add_argument("--kv", action="append", default=[], help="key=value (repeatable)")

    p = sub.add_parser("start-run")
    p.add_argument("--account", default="")
    p.add_argument("--region", default="")
    p.add_argument("--stack", default="")
    p.add_argument("--cluster", default="")
    # REMAINDER must stay last: it swallows everything after it.
    p.add_argument("--argv", nargs=argparse.REMAINDER, default=[])

    p = sub.add_parser("clear")
    p.add_argument("--all", action="store_true")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.file or _default_path()
    prefix = args.prefix

    doc = read_document(path)
    record = record_for(doc, prefix)

    if args.command == "read":
        value = _dotted(record, args.field)
        if isinstance(value, bool):
            sys.stdout.write("1" if value else "0")
        elif value is not None and not isinstance(value, (dict, list)):
            sys.stdout.write(str(value))
        return 0

    if args.command == "read-record":
        sys.stdout.write(json.dumps(record, indent=2, sort_keys=True))
        return 0

    if args.command == "set":
        doc = set_values(doc, prefix, args.section, _kv_pairs(args.kv))
        write_document(path, doc)
        return 0

    if args.command == "start-run":
        conflicts = identity_conflicts(
            record,
            account=args.account,
            region=args.region,
            stackName=args.stack,
            clusterName=args.cluster,
        )
        if conflicts:
            _warn(
                "recorded state for prefix '%s' describes a different environment (%s differ); "
                "those values will not be reused" % (prefix or "<none>", ", ".join(conflicts))
            )
        identity = {
            k: v
            for k, v in (
                ("account", args.account),
                ("region", args.region),
                ("stackName", args.stack),
                ("clusterName", args.cluster),
            )
            if v
        }
        record = dict(record)
        record["identity"] = dict(record.get("identity", {}), **identity)
        # No pid, host, or outcome. Those existed so a second invocation could find
        # and follow the first; AWS is asked directly now, so there is nothing for
        # them to serve and they could only go stale.
        record["lastRun"] = {
            "startedAt": _now(),
            "invokedWith": redact(args.argv or []),
        }
        doc = dict(doc)
        doc["deployments"] = dict(doc.get("deployments", {}))
        doc["deployments"][prefix] = record
        write_document(path, doc)
        # Conflicts are reported on stderr and echoed on stdout so a shell caller
        # can act on them without re-parsing the warning text.
        if conflicts:
            sys.stdout.write(",".join(conflicts))
        return 0

    if args.command == "clear":
        if args.all:
            write_document(path, empty_document())
            return 0
        doc = dict(doc)
        deployments = dict(doc.get("deployments", {}))
        deployments.pop(prefix, None)
        doc["deployments"] = deployments
        write_document(path, doc)
        return 0

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        # Nothing this helper can fail at is worth ending a deployment over.
        _warn(f"unexpected error ({exc}); continuing without state")
        sys.exit(0)
