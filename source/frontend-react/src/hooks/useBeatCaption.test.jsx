/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { useBeatCaption, CAPTION_DEADLINE_MS } from "./useBeatCaption.js";
import { factualCaption } from "../utils/theaterCaptions.js";
import {
  SEGMENTS_BEAT, SEGMENTS_CONTEXT, FLOOR_BEAT, ACCEPTED,
} from "../utils/captionFixtures.js";

let container;
let root;
let latest;

function Probe({ beat, context, generate, deadlineMs }) {
  latest = useBeatCaption({ beat, context, generate, deadlineMs });
  return null;
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  latest = null;
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.useRealTimers();
});

function render(props) {
  act(() => root.render(<Probe {...props} />));
}

/** Let queued microtasks and promise callbacks run. */
async function settle() {
  await act(async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); });
}

const SEG_TEXT = ACCEPTED.find((f) => f.beat === SEGMENTS_BEAT).text;
const never = () => new Promise(() => {});

describe("fallback-first render", () => {
  it("shows the factual caption before any generation resolves", () => {
    render({ beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, generate: never });
    expect(latest.text).toBe(factualCaption(SEGMENTS_BEAT, SEGMENTS_CONTEXT));
    expect(latest.text.length).toBeGreaterThan(0);
  });

  it("never exposes an empty caption, even with no beat", () => {
    render({ beat: null, context: SEGMENTS_CONTEXT, generate: never });
    expect(latest.text).toBe("");
    expect(latest.beatIndex).toBeNull();
  });

  it("exposes no loading or generated flag", () => {
    // BR3-28 and BR3-30: a field that exists is a field someone will render.
    render({ beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, generate: never });
    expect(Object.keys(latest).sort()).toEqual(["beatIndex", "text"]);
  });
});

describe("replacement on success", () => {
  it("replaces the factual caption with the generated one", async () => {
    render({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      generate: async () => ({ ok: true, text: SEG_TEXT }),
    });
    await settle();
    expect(latest.text).toBe(SEG_TEXT);
    expect(latest.beatIndex).toBe(SEGMENTS_BEAT.index);
  });

  it("keeps the factual caption for every failure kind", async () => {
    const factual = factualCaption(SEGMENTS_BEAT, SEGMENTS_CONTEXT);
    for (const kind of ["not_configured", "not_authenticated", "access_denied", "throttled",
                        "timeout", "cancelled", "invalid", "unreachable", "unknown"]) {
      // eslint-disable-next-line no-await-in-loop
      await act(async () => {
        root.render(<Probe beat={SEGMENTS_BEAT} context={SEGMENTS_CONTEXT}
          generate={async () => ({ ok: false, kind, detail: "x" })} />);
      });
      // eslint-disable-next-line no-await-in-loop
      await settle();
      expect(latest.text, kind).toBe(factual);
    }
  });

  it("keeps the factual caption when the generator resolves ok with empty text", async () => {
    const factual = factualCaption(SEGMENTS_BEAT, SEGMENTS_CONTEXT);
    render({ beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, generate: async () => ({ ok: true, text: "" }) });
    await settle();
    expect(latest.text).toBe(factual);
  });

  it("keeps the factual caption when the generator throws", async () => {
    const factual = factualCaption(SEGMENTS_BEAT, SEGMENTS_CONTEXT);
    render({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      generate: async () => { throw new Error("unexpected"); },
    });
    await settle();
    expect(latest.text).toBe(factual);
  });
});

describe("BR3-23: the epoch guard", () => {
  it("discards a beat-3 result that arrives after the reader reached beat 4", async () => {
    // The test that stands between this feature and a true caption attached to
    // the wrong beat.
    const beat3 = { ...SEGMENTS_BEAT, index: 3 };
    const beat4 = { ...FLOOR_BEAT, index: 4 };

    let resolveBeat3;
    const generate = ({ beat }) => {
      if (beat.index === 3) return new Promise((resolve) => { resolveBeat3 = resolve; });
      return new Promise(() => {}); // beat 4's request never settles
    };

    render({ beat: beat3, context: SEGMENTS_CONTEXT, generate });
    expect(latest.beatIndex).toBe(3);

    // Reader advances before beat 3's caption comes back.
    render({ beat: beat4, context: SEGMENTS_CONTEXT, generate });
    expect(latest.beatIndex).toBe(4);
    const beat4Factual = factualCaption(beat4, SEGMENTS_CONTEXT);
    expect(latest.text).toBe(beat4Factual);

    // Beat 3's request now succeeds. Its prose is real and correct for beat 3.
    await act(async () => {
      resolveBeat3({ ok: true, text: "BEAT THREE PROSE that must never be shown here." });
      await Promise.resolve();
    });
    await settle();

    expect(latest.beatIndex).toBe(4);
    expect(latest.text).toBe(beat4Factual);
    expect(latest.text).not.toContain("BEAT THREE PROSE");
  });

  it("applies a result for the beat that is still current", async () => {
    const beat3 = { ...SEGMENTS_BEAT, index: 3 };
    let resolve;
    const generate = () => new Promise((r) => { resolve = r; });

    render({ beat: beat3, context: SEGMENTS_CONTEXT, generate });
    await act(async () => {
      resolve({ ok: true, text: SEG_TEXT });
      await Promise.resolve();
    });
    await settle();
    expect(latest.text).toBe(SEG_TEXT);
  });

  it("discards a result that arrives after the reader returned to the same index", async () => {
    // Re-landing on a beat starts a NEW request; the earlier one is still stale.
    const beat3 = { ...SEGMENTS_BEAT, index: 3 };
    const beat4 = { ...FLOOR_BEAT, index: 4 };
    const resolvers = [];
    const generate = () => new Promise((r) => resolvers.push(r));

    render({ beat: beat3, context: SEGMENTS_CONTEXT, generate });
    render({ beat: beat4, context: SEGMENTS_CONTEXT, generate });
    render({ beat: beat3, context: SEGMENTS_CONTEXT, generate });
    expect(resolvers.length).toBe(3);

    // The FIRST beat-3 request resolves. Its index matches the current beat, and
    // the text is legitimately beat 3's, so this one is allowed through. What must
    // not happen is beat 4's result landing.
    await act(async () => { resolvers[1]({ ok: true, text: "BEAT FOUR PROSE" }); await Promise.resolve(); });
    await settle();
    expect(latest.text).not.toContain("BEAT FOUR PROSE");
  });
});

describe("BR3-24: abort on beat change and unmount", () => {
  it("aborts the in-flight request when the beat changes", async () => {
    const signals = [];
    const generate = ({ signal }) => { signals.push(signal); return new Promise(() => {}); };

    render({ beat: { ...SEGMENTS_BEAT, index: 3 }, context: SEGMENTS_CONTEXT, generate });
    expect(signals[0].aborted).toBe(false);

    render({ beat: { ...FLOOR_BEAT, index: 4 }, context: SEGMENTS_CONTEXT, generate });
    expect(signals[0].aborted).toBe(true);
    expect(signals[1].aborted).toBe(false);
  });

  it("aborts on unmount", async () => {
    const signals = [];
    const generate = ({ signal }) => { signals.push(signal); return new Promise(() => {}); };
    render({ beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, generate });
    act(() => root.unmount());
    expect(signals[0].aborted).toBe(true);
    // Re-create so afterEach's unmount does not double-unmount.
    root = createRoot(container);
  });

  it("does not apply a result after unmount", async () => {
    let resolve;
    const generate = () => new Promise((r) => { resolve = r; });
    render({ beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, generate });
    const before = latest.text;
    act(() => root.unmount());
    await act(async () => { resolve({ ok: true, text: SEG_TEXT }); await Promise.resolve(); });
    expect(latest.text).toBe(before);
    root = createRoot(container);
  });
});

describe("BR3-26: the deadline", () => {
  it("aborts the request once the deadline passes", async () => {
    vi.useFakeTimers();
    const signals = [];
    const generate = ({ signal }) => { signals.push(signal); return new Promise(() => {}); };
    render({ beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, generate, deadlineMs: 50 });
    expect(signals[0].aborted).toBe(false);
    await act(async () => { vi.advanceTimersByTime(60); });
    expect(signals[0].aborted).toBe(true);
  });

  it("leaves the factual caption in place when the deadline is hit", async () => {
    vi.useFakeTimers();
    const factual = factualCaption(SEGMENTS_BEAT, SEGMENTS_CONTEXT);
    render({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, deadlineMs: 50,
      generate: ({ signal }) => new Promise((resolve) => {
        signal.addEventListener("abort", () => resolve({ ok: false, kind: "cancelled", detail: "" }));
      }),
    });
    await act(async () => { vi.advanceTimersByTime(60); await Promise.resolve(); });
    expect(latest.text).toBe(factual);
  });

  it("defaults to the value NFR3-3 fixed", () => {
    expect(CAPTION_DEADLINE_MS).toBe(2500);
  });
});

describe("BR3-27: one request in flight per beat", () => {
  it("issues exactly one request per beat change", () => {
    const calls = [];
    const generate = ({ beat }) => { calls.push(beat.index); return new Promise(() => {}); };
    render({ beat: { ...SEGMENTS_BEAT, index: 1 }, context: SEGMENTS_CONTEXT, generate });
    render({ beat: { ...SEGMENTS_BEAT, index: 2 }, context: SEGMENTS_CONTEXT, generate });
    render({ beat: { ...SEGMENTS_BEAT, index: 3 }, context: SEGMENTS_CONTEXT, generate });
    expect(calls).toEqual([1, 2, 3]);
  });

  it("does not re-issue when nothing changed", () => {
    const calls = [];
    const generate = ({ beat }) => { calls.push(beat.index); return new Promise(() => {}); };
    const props = { beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, generate };
    render(props);
    render(props);
    render(props);
    expect(calls.length).toBe(1);
  });

  it("issues nothing at all when there is no beat", () => {
    const calls = [];
    const generate = () => { calls.push(1); return new Promise(() => {}); };
    render({ beat: null, context: SEGMENTS_CONTEXT, generate });
    expect(calls.length).toBe(0);
  });
});
