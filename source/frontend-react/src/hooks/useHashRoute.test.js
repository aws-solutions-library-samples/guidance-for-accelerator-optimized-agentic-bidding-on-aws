import { describe, it, expect } from "vitest";
import { normaliseHash, THEATER_ROUTE } from "./useHashRoute.js";

describe("normaliseHash", () => {
  it("treats an empty hash as the root route", () => {
    expect(normaliseHash("")).toBe("/");
    expect(normaliseHash("#")).toBe("/");
  });

  it("strips the leading hash", () => {
    expect(normaliseHash("#/theater")).toBe("/theater");
  });

  it("adds a leading slash when the hash omits one", () => {
    expect(normaliseHash("#theater")).toBe("/theater");
  });

  it("falls back to the root route for a non-string", () => {
    expect(normaliseHash(undefined)).toBe("/");
    expect(normaliseHash(null)).toBe("/");
  });

  it("matches the exported theater route", () => {
    expect(normaliseHash("#/theater")).toBe(THEATER_ROUTE);
  });
});
