import { describe, it, expect } from "vitest";
import {
  resolveSegmentLabel,
  resolveSegmentLabels,
  isVendorSegment,
  AUDIENCE_TAXONOMY_VERSION,
} from "./segmentLabels.js";

// These assertions changed with the audience-taxonomy-upgrade unit. Before it, no
// name was known for a numeric identifier such as "350" and it rendered alone.
// Now the bundled IAB Audience Taxonomy supplies the name, so the same identifier
// resolves. The rule is unchanged: show the taxonomy's own name where known, the
// bare identifier where not, and never an invented label.

describe("bundled taxonomy", () => {
  it("reports the version it was generated from", () => {
    expect(AUDIENCE_TAXONOMY_VERSION).toBe("1.1");
  });
});

describe("resolveSegmentLabel", () => {
  it("uses the taxonomy's own name for a taxonomy identifier", () => {
    expect(resolveSegmentLabel("350")).toEqual({
      id: "350",
      name: "Parenting",
      fullName: "Interest | Family and Relationships | Parenting",
    });
  });

  it("shortens a tier path to its last tier, keeping the full path available", () => {
    const label = resolveSegmentLabel("354");
    expect(label.name).toBe("Parenting Babies and Toddlers");
    expect(label.fullName).toContain("Family and Relationships");
  });

  it("resolves an age-range identifier", () => {
    const label = resolveSegmentLabel("7");
    expect(label.name).toBe("35-39");
    expect(label.fullName).toBe("Demographic | Age Range | 35-39");
  });

  it("names this project's own vendor identifiers", () => {
    expect(resolveSegmentLabel("ctx-video")).toEqual({
      id: "ctx-video",
      name: "Video inventory",
      fullName: "Video inventory",
    });
  });

  it("returns a null name for an identifier it does not know, rather than inventing one", () => {
    expect(resolveSegmentLabel("no-such-id")).toEqual({
      id: "no-such-id",
      name: null,
      fullName: null,
    });
  });

  it("lets an explicit lookup override the bundled one, for tests", () => {
    const label = resolveSegmentLabel("350", { 350: "Something | Else" });
    expect(label.name).toBe("Else");
  });

  it("rejects non-string and empty identifiers", () => {
    expect(resolveSegmentLabel("")).toBeNull();
    expect(resolveSegmentLabel(undefined)).toBeNull();
    expect(resolveSegmentLabel(350)).toBeNull();
  });
});

describe("resolveSegmentLabels", () => {
  it("maps a list and drops unusable entries", () => {
    const labels = resolveSegmentLabels(["int-tech", "", null, "350"]);
    expect(labels.map((l) => l.id)).toEqual(["int-tech", "350"]);
    expect(labels.map((l) => l.name)).toEqual(["Technology interest", "Parenting"]);
  });

  it("returns an empty list for a non-array", () => {
    expect(resolveSegmentLabels(undefined)).toEqual([]);
  });
});

describe("isVendorSegment", () => {
  it("recognises this project's own identifier prefixes", () => {
    expect(isVendorSegment("int-auto")).toBe(true);
    expect(isVendorSegment("demo-25-34")).toBe(true);
    expect(isVendorSegment("ctx-video")).toBe(true);
  });

  it("does not claim a numeric taxonomy identifier as ours", () => {
    expect(isVendorSegment("350")).toBe(false);
    expect(isVendorSegment("7")).toBe(false);
  });
});
