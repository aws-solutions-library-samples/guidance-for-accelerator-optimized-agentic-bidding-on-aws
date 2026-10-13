"""Export segment identifier -> condensed name as JSON for the frontend.

The frontend needs display names for the Audience Taxonomy identifiers the Audience
Activator now emits, so it can show the taxonomy's own name rather than inventing one
(see the theater's BR-26). The names live in the bundled TSV, which Python reads and
JavaScript cannot, so they are exported here.

Generating rather than hand-maintaining keeps one source of truth: the bundled TSV.

Regenerate after changing the bundled taxonomy version:

    python3 source/shared/iab_taxonomy/export_names.py

Writes source/frontend-react/src/utils/audienceTaxonomyNames.json.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared import iab_taxonomy  # noqa: E402

_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "frontend-react", "src", "utils", "audienceTaxonomyNames.json",
)


def build() -> dict[str, object]:
    # Only the identifiers the container can emit. Purchase Intent nodes are
    # unreachable, so shipping their names to the browser is dead weight.
    names = iab_taxonomy.segment_names(emittable_only=True)
    return {
        "_comment": (
            "GENERATED FILE, do not edit. Produced by "
            "source/shared/iab_taxonomy/export_names.py from the bundled IAB Audience "
            "Taxonomy. IAB Tech Lab Audience Taxonomy, copyright IAB Technology "
            "Laboratory, licensed under CC BY 3.0. See "
            "source/shared/iab_taxonomy/data/PROVENANCE.md."
        ),
        "taxonomyVersion": iab_taxonomy.AUDIENCE_TAXONOMY_VERSION,
        "names": dict(sorted(names.items())),
    }


def main() -> None:
    payload = build()
    path = os.path.normpath(_OUTPUT)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=0, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    size = os.path.getsize(path)
    print(f"wrote {len(payload['names'])} names to {path} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
