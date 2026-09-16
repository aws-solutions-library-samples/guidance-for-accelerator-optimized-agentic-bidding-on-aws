#!/usr/bin/env python3
"""Structural checks for the ARTF Java sources, without a Java toolchain.

There is no JDK or Maven on the machine this was written on, and the tech-stack
decision confines the Java toolchain to the container build. So these sources
cannot be compiled or unit-tested locally, and pretending otherwise would be the
kind of claim this project is careful not to make.

What CAN be checked mechanically is the error class that actually bites when
writing against an unfamiliar API: a type or package name that does not exist.
Every ``org.prebid.server.*`` import is resolved against the pinned
prebid-server-java checkout, so a typo or a moved class fails here rather than
twenty minutes into a CodeBuild run.

Also checked: that each file's ``package`` declaration matches its path, that
braces and parentheses balance, and that the module and hook codes in the Java
match the ones the hooks execution plan names -- a mismatch there fails Prebid's
startup validation, so it is worth catching statically.

Usage:
    python3 source/prebid/verify_symbols.py <path-to-prebid-server-java-checkout>

Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JAVA_ROOTS = [
    REPO_ROOT / "source" / "prebid" / "artf-hook" / "src" / "main" / "java",
    REPO_ROOT / "source" / "prebid" / "artf-hook" / "src" / "test" / "java",
    REPO_ROOT / "source" / "prebid" / "artfhouse-adapter" / "src" / "main" / "java",
    REPO_ROOT / "source" / "prebid" / "artfhouse-adapter" / "src" / "test" / "java",
]

#: Packages that belong to this feature and therefore resolve within our own tree
#: rather than in the upstream checkout.
#:
#: The adapter's own two packages are NOT listed here. They live at
#: org.prebid.server.bidder.artfhouse and org.prebid.server.spring.config.bidder --
#: the second of which is an UPSTREAM package we add a class to. So an import of
#: something under it must resolve against the upstream checkout, which is exactly
#: what we want checked.
OWN_PACKAGE_PREFIXES = (
    "org.prebid.server.hooks.modules.artf",
    "org.prebid.server.bidder.artfhouse",
)

#: Third-party packages that come from Maven dependencies rather than from the
#: prebid-server-java source tree, so they cannot be resolved by file lookup.
#: Listed explicitly rather than skipped by a catch-all, so a genuinely unknown
#: import is not quietly waved through.
EXTERNAL_PREFIXES = (
    "com.fasterxml.jackson.",
    "com.iab.openrtb.",
    "io.vertx.",
    "org.springframework.",
    "org.junit.",
    "org.assertj.",
    "org.mockito.",
    "lombok.",
    "java.",
    "javax.",
)

IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)
PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)


def java_files() -> list[Path]:
    files: list[Path] = []
    for root in JAVA_ROOTS:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.java")))
    return files


def resolves_in_checkout(fqcn: str, checkout: Path) -> bool:
    """Whether an org.prebid.server.* name exists in the pinned checkout.

    A nested type such as ``Imp.ImpBuilder`` is not imported, so only top-level
    names reach here. Both main and test trees are searched, and a trailing
    segment that is a nested class is tolerated by also trying the parent.
    """
    parts = fqcn.split(".")
    for trim in (0, 1):
        candidate = parts[: len(parts) - trim] if trim else parts
        if not candidate:
            continue
        relative = Path(*candidate).with_suffix(".java")
        for tree in ("src/main/java", "src/test/java"):
            if (checkout / tree / relative).is_file():
                return True
    return False


def check_imports(checkout: Path) -> list[str]:
    problems: list[str] = []
    checked = 0

    for path in java_files():
        text = path.read_text(encoding="utf-8")
        for fqcn in IMPORT_RE.findall(text):
            if fqcn.startswith(OWN_PACKAGE_PREFIXES):
                continue
            if fqcn.startswith(EXTERNAL_PREFIXES):
                continue
            if not fqcn.startswith("org.prebid.server."):
                problems.append(
                    f"{path.name}: import '{fqcn}' is neither ours, a known external "
                    f"package, nor org.prebid.server.* -- add it to EXTERNAL_PREFIXES "
                    f"deliberately or fix the import"
                )
                continue
            checked += 1
            if not resolves_in_checkout(fqcn, checkout):
                problems.append(f"{path.name}: import '{fqcn}' does not exist in the pinned checkout")

    print(f"  resolved {checked} org.prebid.server.* imports against {checkout.name}")
    return problems


def check_own_imports() -> list[str]:
    """Every import of our own package must name a file we actually wrote."""
    problems: list[str] = []
    own: set[str] = set()

    for path in java_files():
        text = path.read_text(encoding="utf-8")
        package = PACKAGE_RE.search(text)
        if package:
            own.add(f"{package.group(1)}.{path.stem}")

    for path in java_files():
        text = path.read_text(encoding="utf-8")
        for fqcn in IMPORT_RE.findall(text):
            if not fqcn.startswith(OWN_PACKAGE_PREFIXES):
                continue
            # Tolerate a nested type: Foo.Bar resolves if Foo does.
            if fqcn in own or fqcn.rsplit(".", 1)[0] in own:
                continue
            problems.append(f"{path.name}: import '{fqcn}' names no class in this unit")

    print(f"  {len(own)} classes declared in this unit")
    return problems


def check_package_matches_path() -> list[str]:
    problems: list[str] = []
    for path in java_files():
        text = path.read_text(encoding="utf-8")
        match = PACKAGE_RE.search(text)
        if not match:
            problems.append(f"{path.name}: no package declaration")
            continue
        expected = match.group(1).replace(".", "/")
        if not str(path.parent).endswith(expected):
            problems.append(
                f"{path.name}: package '{match.group(1)}' does not match its directory "
                f"{path.parent}"
            )
    return problems


def check_balanced() -> list[str]:
    """Braces and parentheses balance, ignoring strings, chars and comments.

    Not a parser. It catches the specific mistake a hand-written file is prone to --
    an unclosed block -- and says nothing about whether the code is valid Java.
    """
    problems: list[str] = []
    for path in java_files():
        text = path.read_text(encoding="utf-8")
        depth = {"{": 0, "(": 0}
        i = 0
        in_line_comment = in_block_comment = in_string = in_char = False
        while i < len(text):
            c = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if in_line_comment:
                if c == "\n":
                    in_line_comment = False
            elif in_block_comment:
                if c == "*" and nxt == "/":
                    in_block_comment = False
                    i += 1
            elif in_string:
                if c == "\\":
                    i += 1
                elif c == '"':
                    in_string = False
            elif in_char:
                if c == "\\":
                    i += 1
                elif c == "'":
                    in_char = False
            elif c == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif c == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif c == '"':
                in_string = True
            elif c == "'":
                in_char = True
            elif c in "{(":
                depth[c] += 1
            elif c == "}":
                depth["{"] -= 1
            elif c == ")":
                depth["("] -= 1
            i += 1
        if depth["{"] != 0:
            problems.append(f"{path.name}: unbalanced braces (net {depth['{']})")
        if depth["("] != 0:
            problems.append(f"{path.name}: unbalanced parentheses (net {depth['(']})")
    return problems


def check_codes_match_config() -> list[str]:
    """The module and hook codes in the Java must match the execution plan.

    Prebid throws at startup for a plan naming an unknown hook
    (HookStageExecutor.java:169-177), so this mismatch is fatal -- which makes it
    exactly the sort of thing worth catching without a deployment.
    """
    problems: list[str] = []
    base = REPO_ROOT / "source" / "prebid" / "artf-hook" / "src" / "main" / "java" / "org" / "prebid" / "server" / "hooks" / "modules" / "artf"

    module_src = (base / "v1" / "ArtfModule.java").read_text(encoding="utf-8")
    hook_src = (base / "v1" / "ArtfProcessedAuctionRequestHook.java").read_text(encoding="utf-8")

    module_code = re.search(r'String CODE\s*=\s*"([^"]+)"', module_src)
    hook_code = re.search(r'String CODE\s*=\s*"([^"]+)"', hook_src)
    if not module_code or not hook_code:
        return ["could not find a CODE constant in ArtfModule or the hook"]

    config_path = REPO_ROOT / "deployment" / "scripts" / "prebid_config_template.yaml"

    # PARSED, not string-matched. An earlier version looked for the literal
    # "hooks:\n  artf-orchestrator:" and failed on a comment line between the two --
    # reporting a missing block that was present. A check that can be defeated by a
    # comment is worse than no check, because it trains the reader to ignore it.
    import yaml  # local import: only this check needs it

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hooks = (config or {}).get("hooks") or {}

    module = module_code.group(1)
    hook = hook_code.group(1)

    if module not in hooks:
        problems.append(
            f"prebid_config_template.yaml has no 'hooks.{module}' block, so "
            f"@ConditionalOnProperty leaves every bean unregistered"
        )
    elif hooks[module].get("enabled") is not True:
        problems.append(
            f"prebid_config_template.yaml has hooks.{module}.enabled != true, so the "
            f"module's beans are never registered"
        )

    plan_raw = hooks.get("host-execution-plan")
    if not plan_raw:
        problems.append(
            "prebid_config_template.yaml has no hooks.host-execution-plan -- the module "
            "would be present and NEVER INVOKED, which is the one genuinely silent failure"
        )
        return problems

    # The plan is a JSON STRING, not YAML structure. Parsing it here is also the check
    # that it IS valid JSON: Prebid decodes it with Jackson and refuses to start
    # otherwise (HookStageExecutor.java:148-156).
    import json

    try:
        plan = json.loads(plan_raw)
    except json.JSONDecodeError as exc:
        problems.append(f"hooks.host-execution-plan is not valid JSON: {exc}")
        return problems

    sequences = [
        entry
        for endpoint in (plan.get("endpoints") or {}).values()
        for stage in (endpoint.get("stages") or {}).values()
        for group in (stage.get("groups") or [])
        for entry in (group.get("hook-sequence") or [])
    ]
    if not any(
        entry.get("module-code") == module and entry.get("hook-impl-code") == hook
        for entry in sequences
    ):
        problems.append(
            f"the execution plan names no hook matching module-code '{module}' and "
            f"hook-impl-code '{hook}'. Prebid throws at startup for an unknown hook, so a "
            f"mismatch here is fatal rather than silent"
        )

    stages = [
        stage_name
        for endpoint in (plan.get("endpoints") or {}).values()
        for stage_name in (endpoint.get("stages") or {})
    ]
    if "processed-auction-request" not in stages:
        problems.append(
            "the execution plan does not register the hook at processed-auction-request. "
            "After the bidder fan-out there is no request left to enrich (BR-1, FR-2)"
        )

    print(f"  module-code '{module}', hook-impl-code '{hook}', stages {stages}")
    return problems


def check_bidder_name_agreement() -> list[str]:
    """The bidder name must agree across all four registration artifacts.

    A mismatch is a STARTUP failure, not a runtime one, so it is worth catching without a
    deployment. The four places:

      1. BIDDER_NAME in ArtfhouseConfiguration
      2. the adapters.<name> key in bidder-config/<name>.yaml
      3. the @ConfigurationProperties prefix, adapters.<name>
      4. the filename static/bidder-params/<name>.json

    Skipped silently when the adapter is absent, so this checker still works for a
    hook-only tree.
    """
    import re as _re

    import yaml

    adapter = REPO_ROOT / "source" / "prebid" / "artfhouse-adapter"
    config_java = (
        adapter / "src/main/java/org/prebid/server/spring/config/bidder/ArtfhouseConfiguration.java"
    )
    if not config_java.is_file():
        print("  adapter absent, skipped")
        return []

    problems: list[str] = []
    source = config_java.read_text(encoding="utf-8")

    name_match = _re.search(r'BIDDER_NAME\s*=\s*"([^"]+)"', source)
    if not name_match:
        return ["ArtfhouseConfiguration has no BIDDER_NAME constant"]
    name = name_match.group(1)

    prefix_match = _re.search(r'@ConfigurationProperties\("adapters\.([^"]+)"\)', source)
    if not prefix_match:
        problems.append("ArtfhouseConfiguration has no @ConfigurationProperties(\"adapters.<name>\")")
    elif prefix_match.group(1) != name:
        problems.append(
            f"@ConfigurationProperties prefix 'adapters.{prefix_match.group(1)}' does not match "
            f"BIDDER_NAME '{name}'"
        )

    property_source = _re.search(r'classpath:/bidder-config/([^"]+)\.yaml', source)
    if not property_source:
        problems.append("ArtfhouseConfiguration has no @PropertySource for a bidder-config yaml")
    elif property_source.group(1) != name:
        problems.append(
            f"@PropertySource names bidder-config/{property_source.group(1)}.yaml but BIDDER_NAME "
            f"is '{name}'"
        )

    yaml_path = adapter / "src/main/resources/bidder-config" / f"{name}.yaml"
    if not yaml_path.is_file():
        problems.append(f"missing bidder-config/{name}.yaml -- the @PropertySource would fail at startup")
    else:
        adapters = (yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}).get("adapters") or {}
        if name not in adapters:
            problems.append(f"bidder-config/{name}.yaml has no 'adapters.{name}' key")
        else:
            endpoint = adapters[name].get("endpoint", "")
            # A build-time placeholder here would survive verbatim into the image: this
            # file is baked in by CodeBuild, which does not know the endpoint's URL.
            if "__" in str(endpoint):
                problems.append(
                    f"bidder-config/{name}.yaml endpoint '{endpoint}' contains a build-time "
                    f"placeholder, which nothing substitutes inside the image"
                )

    params_path = adapter / "src/main/resources/static/bidder-params" / f"{name}.json"
    if not params_path.is_file():
        problems.append(
            f"missing static/bidder-params/{name}.json -- BidderParamValidator requires one "
            f"per registered bidder, so the server would not start"
        )
    else:
        import json as _json

        try:
            _json.loads(params_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError as exc:
            problems.append(f"static/bidder-params/{name}.json is not valid JSON: {exc}")

    print(f"  bidder name '{name}' agrees across all four registration artifacts")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    checkout = Path(sys.argv[1])
    if not (checkout / "pom.xml").is_file():
        print(f"error: {checkout} is not a prebid-server-java checkout", file=sys.stderr)
        return 2

    files = java_files()
    print(f"Checking {len(files)} Java files against {checkout}")

    all_problems: list[str] = []
    for name, check in (
        ("package matches path", check_package_matches_path),
        ("braces balance", check_balanced),
        ("own imports resolve", check_own_imports),
        ("module/hook codes match the config", check_codes_match_config),
        ("bidder name agrees across registration artifacts", check_bidder_name_agreement),
    ):
        print(f"- {name}")
        all_problems.extend(check())

    print("- upstream imports resolve")
    all_problems.extend(check_imports(checkout))

    print()
    if all_problems:
        print(f"FAILED with {len(all_problems)} problem(s):")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1

    print("All structural checks passed.")
    print("This is NOT a compile. The Java is first compiled by the container build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
