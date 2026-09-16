// segmentLabels.js — resolve a display label for a segment identifier.
//
// The Audience Activator's identifier space is changing from this project's own
// `int-*` / `demo-*` / `ctx-*` names to real IAB Audience Taxonomy 1.1 numeric
// identifiers (see the audience-taxonomy-upgrade unit). This module renders
// either era without special casing, which matters because the theater ships
// before that upgrade lands.
//
// Rule (BR-26, BR-27): a segment renders as the taxonomy's own condensed name
// with its identifier beside it. Where no name is known for an identifier, the
// identifier renders alone. This module never invents a display name.

import audienceTaxonomy from "./audienceTaxonomyNames.json";

/**
 * Names for this project's own pre-taxonomy identifiers. These are our labels
 * for our own identifiers, not invented names for standard ones.
 */
const VENDOR_SEGMENT_NAMES = Object.freeze({
  "int-auto": "Automotive interest",
  "int-food": "Food and drink interest",
  "int-gaming": "Gaming interest",
  "int-finance": "Finance interest",
  "int-sports": "Sports interest",
  "int-fashion": "Fashion interest",
  "int-tech": "Technology interest",
  "int-travel": "Travel interest",
  "demo-18-24": "Age 18 to 24",
  "demo-25-34": "Age 25 to 34",
  "demo-35-44": "Age 35 to 44",
  "demo-45-54": "Age 45 to 54",
  "ctx-premium": "Premium inventory",
  "ctx-video": "Video inventory",
  "ctx-mobile": "Mobile inventory",
});

/**
 * IAB Audience Taxonomy 1.1 names, generated from the bundled taxonomy file by
 * `source/shared/iab_taxonomy/export_names.py`. One source of truth: the TSV the
 * container reads.
 */
const TAXONOMY_NAMES = audienceTaxonomy.names || {};

export const AUDIENCE_TAXONOMY_VERSION = audienceTaxonomy.taxonomyVersion;

/**
 * The taxonomy's condensed names are full tier paths, e.g.
 * "Interest | Family and Relationships | Parenting". The last tier is what a reader
 * needs; the full path is kept so it can be shown on hover.
 */
function shortenCondensedName(condensed) {
  const parts = condensed.split("|").map((p) => p.trim()).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : condensed;
}

/**
 * Resolve a segment identifier to `{ id, name, fullName }`.
 *
 * `taxonomyNames` overrides the bundled lookup, which exists for tests. `name` is
 * null when no name is known, in which case the identifier renders alone rather
 * than beside an invented label.
 */
export function resolveSegmentLabel(id, taxonomyNames) {
  if (typeof id !== "string" || id.length === 0) return null;
  const source = taxonomyNames || TAXONOMY_NAMES;
  const condensed = typeof source[id] === "string" ? source[id] : null;
  if (condensed) {
    return { id, name: shortenCondensedName(condensed), fullName: condensed };
  }
  const vendorName = VENDOR_SEGMENT_NAMES[id] ?? null;
  return { id, name: vendorName, fullName: vendorName };
}

export function resolveSegmentLabels(ids, taxonomyNames) {
  if (!Array.isArray(ids)) return [];
  return ids
    .map((id) => resolveSegmentLabel(id, taxonomyNames))
    .filter((label) => label !== null);
}

/**
 * True when this identifier belongs to our own vendor space rather than a
 * published IAB taxonomy. The UI uses this to mark vendor segments distinctly,
 * so a viewer is never led to believe a local identifier is a standard one.
 */
export function isVendorSegment(id) {
  return typeof id === "string" && (
    id.startsWith("int-") || id.startsWith("demo-") || id.startsWith("ctx-")
  );
}

export { VENDOR_SEGMENT_NAMES };
