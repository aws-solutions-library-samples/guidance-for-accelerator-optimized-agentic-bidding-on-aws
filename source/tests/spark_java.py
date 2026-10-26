"""Locate a working JDK for the PySpark-backed tests.

Spark is a JVM program. With ``pyspark`` installed but no Java reachable, every test
that builds a SparkSession errors out at fixture setup with "Java gateway process
exited before sending its port number" — a message that names neither the cause nor
the fix. This module resolves a JDK when one is present but unadvertised, and lets
the fixtures skip cleanly when there genuinely isn't one.

Two things make the naive check wrong:

* macOS ships a ``/usr/bin/java`` stub that exists on PATH whether or not a JDK is
  installed, and only reports the absence when you run it. Presence of the file is
  not evidence of a runtime.
* Homebrew's ``openjdk`` formulae are keg-only, so an installed JDK is frequently
  neither on PATH nor named by ``JAVA_HOME``.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess

#: Tried in Spark's supported order rather than newest-first — a JDK too new for the
#: bundled Spark starts and then fails on internal module access.
_CANDIDATES = (
    "/opt/homebrew/opt/openjdk@17",
    "/opt/homebrew/opt/openjdk@21",
    "/usr/local/opt/openjdk@17",
    "/usr/local/opt/openjdk@21",
    "/opt/homebrew/opt/openjdk",
    "/usr/local/opt/openjdk",
)


def java_runs(java_binary: str) -> bool:
    """Whether this java binary is a working runtime, established by running it."""
    try:
        return (
            subprocess.run(
                [java_binary, "-version"], capture_output=True, timeout=30
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_java_home() -> str | None:
    """Point JAVA_HOME and PATH at a working JDK; return it, or None if there is none.

    Returns the empty-ish case as None so a caller can skip rather than fail. A JDK
    already reachable on PATH returns whatever JAVA_HOME happens to be — Spark does
    not need the variable when java is on PATH.
    """
    existing = os.environ.get("JAVA_HOME")
    if existing and java_runs(os.path.join(existing, "bin", "java")):
        return existing

    on_path = shutil.which("java")
    if on_path and java_runs(on_path):
        return existing or on_path

    candidates = list(_CANDIDATES) + sorted(
        glob.glob("/Library/Java/JavaVirtualMachines/*/Contents/Home")
    )
    for candidate in candidates:
        if java_runs(os.path.join(candidate, "bin", "java")):
            os.environ["JAVA_HOME"] = candidate
            os.environ["PATH"] = (
                os.path.join(candidate, "bin") + os.pathsep + os.environ.get("PATH", "")
            )
            return candidate
    return None


SKIP_REASON = (
    "no working Java runtime found; Spark cannot start. Install a JDK "
    "(e.g. `brew install openjdk@17`) or set JAVA_HOME."
)
