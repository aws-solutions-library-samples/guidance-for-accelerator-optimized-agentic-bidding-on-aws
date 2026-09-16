"""IAB Tech Lab taxonomy resolution.

Reads the bundled Content Taxonomy 3.1 and Audience Taxonomy 1.1 files and answers
three questions:

1. Which taxonomy do a request's category codes belong to, from OpenRTB ``cattax``?
2. Does a content category carry the Special Category Data flag?
3. Which Audience Taxonomy Interest segment corresponds to a content category?

**On the mapping in (3).** IAB publishes mappings between several of its taxonomies
but NOT between Content and Audience -- the ``Taxonomy Mappings`` folder has
Content-to-Ad-Product, Content-to-Content, and genre mappings, and nothing joining
Content to Audience. So this mapping is OURS. It joins on tier path, which works
because the two taxonomies share tier vocabulary by design, and resolves 565 of the
705 Content 3.1 categories with no hand-curation. See

**On Special Category Data.** IAB's implementation guidance describes the SCD
extension as a control to reduce the risk that content categorisation gets used to
build sensitive profiles about a person, covering attributes like race, politics and
religion. It is a privacy marker, NOT a brand-safety or suitability signal. Deriving a
user interest segment from an SCD-flagged content category is the exact pattern it
warns against, so ``segments_for_categories`` withholds those.

The flag is set per row and is **not inherited**: ``186`` Family and Relationships
carries it, its children including ``192`` Parenting do not, and two grandchildren do.
Had it been inherited, flagging the grandchildren separately would be redundant. So
only the categories actually present on a request are tested, never their ancestors.

Tables are built once at import. Measured: 15.7ms to parse both files, ~0.15MB
resident, and 0.21 to 1.24 microseconds to resolve 1 to 8 categories -- roughly four
orders of magnitude below the containers' millisecond-scale latency budget.

Data provenance, checksums and licence attribution: see ``data/PROVENANCE.md``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CONTENT_TAXONOMY_VERSION = "3.1"
AUDIENCE_TAXONOMY_VERSION = "1.1"

# OpenRTB/AdCOM `cattax` values. From AdCOM's "List: Category Taxonomies", which is
# the enumeration OpenRTB references normatively. Note that IAB's Content Taxonomy
# implementation guidance says loosely to "set cattax to 2 for version 2.x and
# higher"; AdCOM's explicit enumeration is followed instead.
CATTAX_CONTENT_1_0 = 1
CATTAX_CONTENT_2_0 = 2
CATTAX_AUDIENCE_1_1 = 4
CATTAX_CONTENT_2_1 = 5
CATTAX_CONTENT_2_2 = 6
CATTAX_CONTENT_3_0 = 7
CATTAX_CONTENT_3_1 = 9

# Taxonomy selection outcomes.
TAXONOMY_CONTENT_1_0 = "content-1.0"
TAXONOMY_CONTENT_3_X = "content-3.x"
TAXONOMY_UNRECOGNISED = "unrecognised"

# Per-category dispositions.
MAPPED = "mapped"
WITHHELD = "withheld"
NO_EQUIVALENT = "no-equivalent"
UNRECOGNISED = "unrecognised"


class TaxonomyDataError(RuntimeError):
    """A bundled taxonomy file is missing or unparseable.

    Raised at import. These files ship inside the image, so their absence can only
    mean the image was built wrong -- a build defect, not a runtime condition.
    Failing here stops the deployment, where it is cheap to fix. Degrading silently
    would ship a container that cannot do half its job and whose only symptom is
    missing segments.
    """


@dataclass(frozen=True)
class ContentCategory:
    id: str
    parent_id: str | None
    name: str
    tier_path: tuple[str, ...]
    is_special_category_data: bool


@dataclass(frozen=True)
class AudienceSegment:
    id: str
    tier_path: tuple[str, ...]
    condensed_name: str


@dataclass(frozen=True)
class AgeBucket:
    id: str
    lower_age: int
    upper_age: int | None


@dataclass(frozen=True)
class CategoryOutcome:
    """One category's fate. Four outcomes, deliberately distinguishable.

    ``withheld`` is kept separate from ``no-equivalent`` because they mean different
    things: "we declined to infer from this" versus "there was nothing to infer".
    Collapsing them would destroy the only evidence the privacy gate did anything.
    """

    category_id: str
    disposition: str
    segment_id: str | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _read_tsv(filename: str) -> list[list[str]]:
    path = os.path.join(_DATA_DIR, filename)
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
    except OSError as exc:
        raise TaxonomyDataError(f"cannot read bundled taxonomy file {path}: {exc}") from exc
    if len(rows) < 2:
        raise TaxonomyDataError(f"bundled taxonomy file {path} has no data rows")
    return rows[1:]  # drop the header


def _load_content_taxonomy() -> dict[str, ContentCategory]:
    """Content Taxonomy 3.1: Unique ID, Parent ID, Name, Tier 1..4, Extension."""
    out: dict[str, ContentCategory] = {}
    for row in _read_tsv("content_taxonomy_3_1.tsv"):
        row = row + [""] * (8 - len(row))
        uid = row[0].strip()
        if not uid:
            continue
        tiers = tuple(t.strip() for t in row[3:7] if t.strip())
        out[uid] = ContentCategory(
            id=uid,
            parent_id=row[1].strip() or None,
            name=row[2].strip(),
            tier_path=tiers,
            # The Extension column carries "SCD" for Special Category Data rows.
            is_special_category_data="SCD" in row[7],
        )
    if not out:
        raise TaxonomyDataError("content taxonomy parsed to zero categories")
    return out


# Identifiers this container can actually emit: Interest and Demographic nodes.
# Purchase Intent nodes are named for completeness but unreachable, so exporting
# their names to the frontend would be dead weight.
_EMITTABLE_IDS: set[str] = set()


def normalise_asserted_name(name: str) -> str:
    """Normalise a third-party segment name for matching against taxonomy node names.

    A DMP asserts names in whatever form it likes -- "Parents with Children",
    "parents_with_children", "HH: PARENTS WITH CHILDREN". Lowercasing and reducing
    everything that is not alphanumeric to single spaces handles casing and
    separators without resorting to substring matching, which across ~700 node names
    would produce false positives.
    """
    if not isinstance(name, str):
        return ""
    reduced = "".join(ch.lower() if ch.isalnum() else " " for ch in name)
    return " ".join(reduced.split())


def _load_audience_taxonomy() -> tuple[dict[tuple[str, ...], AudienceSegment],
                                       dict[str, AudienceSegment],
                                       dict[str, AudienceSegment],
                                       tuple[AgeBucket, ...]]:
    """Audience Taxonomy 1.1: leading blank, Unique ID, Parent ID, Condensed Name, Tier 1..6.

    Returns four things:

    - Interest nodes keyed by tier path with the ``Interest`` root dropped, so they
      join directly against a content category's tier path.
    - Every node keyed by id, for name lookup. This deliberately includes Demographic
      nodes beyond Age Range: an earlier version registered only Age Range, which left
      real identifiers such as ``98`` Parents with Children nameless.
    - Demographic and Interest nodes keyed by their normalised leaf name, for matching
      segment names a request already asserts.
    - The Demographic age-range rows.

    Purchase Intent nodes are registered for names but excluded from leaf-name
    matching. The taxonomy marks that tier with an asterisk and refers to a separate
    Purchase Intent Classification extension, so treating a name match as an assertion
    of purchase intent would claim more than the taxonomy supports.
    """
    by_path: dict[tuple[str, ...], AudienceSegment] = {}
    by_id: dict[str, AudienceSegment] = {}
    by_leaf_name: dict[str, AudienceSegment] = {}
    age_rows: list[tuple[str, str]] = []
    _EMITTABLE_IDS.clear()

    for row in _read_tsv("audience_taxonomy_1_1.tsv"):
        row = row + [""] * (11 - len(row))
        uid = row[1].strip()
        if not uid:
            continue
        # The taxonomy's condensed names carry a trailing " |" on leaf rows, e.g.
        # "Interest | Family and Relationships | Parenting |". Trimming it is
        # presentation of the taxonomy's own value, not a rewrite of it.
        condensed = row[3].strip().rstrip("|").strip()
        tiers = [t.strip() for t in row[4:10] if t.strip()]
        if not tiers:
            continue

        root = tiers[0]
        segment = AudienceSegment(
            id=uid, tier_path=tuple(tiers[1:]), condensed_name=condensed
        )
        # Every node gets a name, whatever tier it sits under.
        by_id.setdefault(uid, segment)

        if root == "Interest" and len(tiers) > 1:
            # First occurrence wins, so a duplicate tier path cannot silently
            # reassign an already-registered segment.
            by_path.setdefault(segment.tier_path, segment)

        if root in ("Interest", "Demographic"):
            _EMITTABLE_IDS.add(uid)
            if len(tiers) > 1:
                leaf = normalise_asserted_name(tiers[-1])
                if leaf:
                    by_leaf_name.setdefault(leaf, segment)

        if root == "Demographic" and len(tiers) > 2 and tiers[1] == "Age Range":
            age_rows.append((uid, tiers[2]))

    if not by_path:
        raise TaxonomyDataError("audience taxonomy parsed to zero Interest nodes")
    return by_path, by_id, by_leaf_name, _parse_age_buckets(age_rows)


def _parse_age_buckets(rows: Iterable[tuple[str, str]]) -> tuple[AgeBucket, ...]:
    """Read the taxonomy's own age ranges, so the buckets cannot drift from the standard.

    Handles both "18-20" and open-ended "75+".
    """
    buckets: list[AgeBucket] = []
    for uid, label in rows:
        text = label.strip()
        try:
            if text.endswith("+"):
                buckets.append(AgeBucket(id=uid, lower_age=int(text[:-1]), upper_age=None))
            elif "-" in text:
                low, high = text.split("-", 1)
                buckets.append(AgeBucket(id=uid, lower_age=int(low), upper_age=int(high)))
        except ValueError:
            # A range this parser does not understand is skipped rather than guessed at.
            continue
    return tuple(sorted(buckets, key=lambda b: b.lower_age))


# Built once at import. See the module docstring for the measured cost.
CONTENT_CATEGORIES: dict[str, ContentCategory] = _load_content_taxonomy()
_INTEREST_BY_PATH, _AUDIENCE_BY_ID, _AUDIENCE_BY_LEAF_NAME, AGE_BUCKETS = _load_audience_taxonomy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_taxonomy(cattax: object) -> str:
    """Which taxonomy a request's category codes belong to.

    Absent or 1 selects Content Taxonomy 1.0, preserving this project's existing
    behaviour. 7 and 9 select Content Taxonomy 3.0/3.1.

    Everything else, including Content Taxonomy 2.x, is unrecognised. 2.x identifiers
    are not interchangeable with 3.x -- IAB documents 3.0 as a breaking change -- so
    resolving them against a 3.1 table would mis-resolve silently. An unrecognised
    taxonomy derives nothing and says so, rather than falling back to a default that
    would consult the wrong table.
    """
    if cattax is None:
        return TAXONOMY_CONTENT_1_0
    if isinstance(cattax, bool) or not isinstance(cattax, int):
        return TAXONOMY_UNRECOGNISED
    if cattax == CATTAX_CONTENT_1_0:
        return TAXONOMY_CONTENT_1_0
    if cattax in (CATTAX_CONTENT_3_0, CATTAX_CONTENT_3_1):
        return TAXONOMY_CONTENT_3_X
    return TAXONOMY_UNRECOGNISED


def is_special_category_data(category_id: str) -> bool:
    """Whether this specific category carries the flag. Ancestors are not consulted."""
    entry = CONTENT_CATEGORIES.get(category_id)
    return bool(entry and entry.is_special_category_data)


def segment_for_category(category_id: str) -> str | None:
    """The Audience Taxonomy Interest segment for a content category, or None.

    Matches the category's tier path against Interest tier paths, most specific first,
    falling back to shorter paths. Returns None when no Interest node matches, which is
    a normal outcome: categories such as Crime, Disasters, Politics and Religion &
    Spirituality describe content rather than a durable user interest and should not
    become interest segments.

    Does NOT apply the Special Category Data gate -- that is
    ``segments_for_categories``' job, so this function stays usable for inspection.
    """
    entry = CONTENT_CATEGORIES.get(category_id)
    if entry is None:
        return None
    tiers = entry.tier_path
    for length in range(len(tiers), 0, -1):
        match = _INTEREST_BY_PATH.get(tiers[:length])
        if match is not None:
            return match.id
    return None


def content_tier1(category_id: str) -> str | None:
    """The tier-1 name a content category sits under, or None if unknown.

    Content Taxonomy 3.1 has 37 tier-1 categories, several of which name their own
    unsuitability directly: Crime, Disasters, Law, Politics, Religion & Spirituality,
    Sensitive Topics, War and Conflicts. Callers judging content suitability should
    read the tier-1 name rather than the Special Category Data flag, which is a
    privacy control about inference and says nothing about suitability.
    """
    entry = CONTENT_CATEGORIES.get(category_id)
    if entry is None or not entry.tier_path:
        return None
    return entry.tier_path[0]


def segments_for_categories(category_ids: Iterable[str]) -> list[CategoryOutcome]:
    """Resolve categories to segments, withholding Special Category Data.

    One outcome per input category, in input order. A malformed or unknown identifier
    never raises and never prevents the others from resolving.
    """
    outcomes: list[CategoryOutcome] = []
    for raw in category_ids:
        if not isinstance(raw, str) or not raw:
            continue
        category_id = raw.strip()
        if category_id not in CONTENT_CATEGORIES:
            outcomes.append(CategoryOutcome(category_id, UNRECOGNISED))
            continue
        if is_special_category_data(category_id):
            outcomes.append(CategoryOutcome(category_id, WITHHELD))
            continue
        segment_id = segment_for_category(category_id)
        outcomes.append(
            CategoryOutcome(category_id, MAPPED, segment_id) if segment_id
            else CategoryOutcome(category_id, NO_EQUIVALENT)
        )
    return outcomes


def segment_for_asserted_name(name: str) -> str | None:
    """Match a segment name a request already asserts to an Audience Taxonomy node.

    This is for ``user.data[].segment[].name`` -- audience data a first or third party
    has already asserted about the person. Reclassifying someone else's assertion into
    the standard taxonomy is not an inference; it is a translation.

    That distinction is why this exists as a separate entry point from
    ``segment_for_category``. A life-stage segment such as ``98`` Parents with Children
    is only reachable this way. Deriving it from page content would be inferring a
    personal circumstance from what someone happened to read, which is precisely the
    inference IAB's Special Category Data flag exists to discourage.

    Matches the node's leaf name exactly after normalisation, never as a substring.
    Returns None when nothing matches, which is the common case.
    """
    key = normalise_asserted_name(name)
    if not key:
        return None
    match = _AUDIENCE_BY_LEAF_NAME.get(key)
    return match.id if match else None


def age_bucket_id(year_of_birth: int, *, now: datetime | None = None) -> str | None:
    """The Audience Taxonomy age-range identifier for a year of birth, or None.

    Uses the taxonomy's own ranges rather than hand-written buckets.
    """
    if isinstance(year_of_birth, bool) or not isinstance(year_of_birth, int):
        return None
    current_year = (now or datetime.now(timezone.utc)).year
    if not (1900 < year_of_birth <= current_year):
        return None
    age = current_year - year_of_birth
    for bucket in AGE_BUCKETS:
        if age >= bucket.lower_age and (bucket.upper_age is None or age <= bucket.upper_age):
            return bucket.id
    return None


def condensed_name(segment_id: str) -> str | None:
    """The taxonomy's own display name for a segment, so callers need not invent one."""
    entry = _AUDIENCE_BY_ID.get(segment_id)
    return entry.condensed_name if entry else None


def segment_names(*, emittable_only: bool = False) -> dict[str, str]:
    """Segment identifier to condensed name, for UI label lookup.

    ``emittable_only`` restricts the result to Interest and Demographic nodes, which
    are the only ones this container can produce. The frontend uses that form so its
    bundle does not carry names for the 864 Purchase Intent nodes it can never show.
    """
    items = _AUDIENCE_BY_ID.items()
    if emittable_only:
        return {sid: seg.condensed_name for sid, seg in items if sid in _EMITTABLE_IDS}
    return {sid: seg.condensed_name for sid, seg in items}


def stats() -> dict[str, int | str]:
    """Table sizes and versions, for a readiness probe or a log line at start."""
    return {
        "content_taxonomy_version": CONTENT_TAXONOMY_VERSION,
        "audience_taxonomy_version": AUDIENCE_TAXONOMY_VERSION,
        "content_categories": len(CONTENT_CATEGORIES),
        "interest_segments": len(_INTEREST_BY_PATH),
        "named_segments": len(_AUDIENCE_BY_ID),
        "emittable_segments": len(_EMITTABLE_IDS),
        "matchable_leaf_names": len(_AUDIENCE_BY_LEAF_NAME),
        "age_buckets": len(AGE_BUCKETS),
        "special_category_data_rows": sum(
            1 for c in CONTENT_CATEGORIES.values() if c.is_special_category_data
        ),
    }
