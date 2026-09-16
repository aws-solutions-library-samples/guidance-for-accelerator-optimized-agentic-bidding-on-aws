import { describe, it, expect, vi, beforeEach } from "vitest";
import { generateCaption, mapError, DEFAULT_PROFILE_ID } from "./bedrockCaptionClient.js";
import {
  SEGMENTS_BEAT, SEGMENTS_CONTEXT, ACCEPTED,
} from "./utils/captionFixtures.js";

// No test calls Bedrock (NFR3-24). The client factory and the trace are injected.

const GOOD = ACCEPTED.find((f) => f.beat === SEGMENTS_BEAT).text;

function stubClient(sendImpl) {
  return { getClient: async () => ({ client: { send: sendImpl } }) };
}

function converseReply(text) {
  return { output: { message: { content: [{ text }] } } };
}

let traced;
const captureTrace = (entry) => { traced = entry; };

beforeEach(() => {
  traced = undefined;
});

describe("success", () => {
  it("returns the validated caption", async () => {
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: { ...stubClient(async () => converseReply(GOOD)), trace: captureTrace },
    });
    expect(result).toEqual({ ok: true, text: GOOD });
  });

  it("sends the profile id, the system prompt and the beat message", async () => {
    let command;
    await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: {
        ...stubClient(async (cmd) => { command = cmd; return converseReply(GOOD); }),
        trace: captureTrace,
      },
    });
    const input = command.input;
    expect(input.modelId).toBe(DEFAULT_PROFILE_ID);
    expect(input.modelId).toMatch(/^global\./);
    expect(input.system[0].text).toMatch(/45 words/);
    expect(input.messages[0].content[0].text).toContain("Parents with Children");
    expect(input.inferenceConfig).toEqual({ maxTokens: 120, temperature: 0.3 });
  });
});

describe("validation gates the response", () => {
  it("returns invalid with the violated rules when the model invents a number", async () => {
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: {
        ...stubClient(async () => converseReply("Audience Activator matched segment 411 in 41 milliseconds.")),
        trace: captureTrace,
      },
    });
    expect(result.ok).toBe(false);
    expect(result.kind).toBe("invalid");
    expect(result.violations.map((v) => v.rule)).toContain("BR3-13");
  });

  it("returns invalid for an empty response", async () => {
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: { ...stubClient(async () => converseReply("")), trace: captureTrace },
    });
    expect(result.ok).toBe(false);
    expect(result.kind).toBe("invalid");
  });

  it("returns invalid for a response with no content at all", async () => {
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: { ...stubClient(async () => ({})), trace: captureTrace },
    });
    expect(result.ok).toBe(false);
    expect(result.kind).toBe("invalid");
  });

  it("never repairs a rejected caption", async () => {
    const bad = "Audience Activator matched segment 411.";
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: { ...stubClient(async () => converseReply(bad)), trace: captureTrace },
    });
    expect(result.text).toBeUndefined();
  });
});

describe("configuration", () => {
  it("reports not_configured when no profile id is available", async () => {
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: { profileId: "", trace: captureTrace },
    });
    expect(result).toMatchObject({ ok: false, kind: "not_configured" });
  });

  it("reports the client factory's own configuration failure", async () => {
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: {
        getClient: async () => ({ error: { kind: "not_configured", detail: "no pool" } }),
        trace: captureTrace,
      },
    });
    expect(result).toMatchObject({ ok: false, kind: "not_configured", detail: "no pool" });
  });

  it("reports not_authenticated when there is no token", async () => {
    const result = await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: {
        getClient: async () => ({ error: { kind: "not_authenticated", detail: "no token" } }),
        trace: captureTrace,
      },
    });
    expect(result).toMatchObject({ ok: false, kind: "not_authenticated" });
  });

  it("does not fall back to the bare model id", async () => {
    // BR3-2: the bare id is unusable, so a missing profile id must not substitute it.
    let command;
    await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: {
        ...stubClient(async (cmd) => { command = cmd; return converseReply(GOOD); }),
        trace: captureTrace,
      },
    });
    expect(command.input.modelId).not.toBe("anthropic.claude-haiku-4-5-20251001-v1:0");
  });
});

describe("error mapping", () => {
  const cases = [
    ["AccessDeniedException", "User is not authorized to perform bedrock:InvokeModel", "access_denied"],
    ["ThrottlingException", "Too many requests", "throttled"],
    ["AbortError", "The operation was aborted", "cancelled"],
    ["TypeError", "Failed to fetch", "unreachable"],
    ["ValidationException", "Invocation of model ID ... isn't supported", "unknown"],
    ["SomethingElse", "who knows", "unknown"],
  ];

  for (const [name, message, expected] of cases) {
    it(`maps ${name} to ${expected}`, () => {
      const err = new Error(message);
      err.name = name;
      expect(mapError(err).kind).toBe(expected);
    });
  }

  for (const [name, message, expected] of cases) {
    it(`surfaces ${expected} from a failed send (${name})`, async () => {
      const result = await generateCaption({
        beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
        deps: {
          ...stubClient(async () => {
            const err = new Error(message);
            err.name = name;
            throw err;
          }),
          trace: captureTrace,
        },
      });
      expect(result).toMatchObject({ ok: false, kind: expected });
    });
  }

  it("never throws, whatever the SDK does", async () => {
    for (const thrown of [new Error("x"), "a string", null, undefined, 42, { weird: true }]) {
      // eslint-disable-next-line no-await-in-loop
      const result = await generateCaption({
        beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
        deps: { ...stubClient(async () => { throw thrown; }), trace: captureTrace },
      });
      expect(result.ok).toBe(false);
      expect(typeof result.kind).toBe("string");
    }
  });
});

describe("cancellation", () => {
  it("passes the abort signal through to the SDK", async () => {
    const controller = new AbortController();
    let options;
    await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, signal: controller.signal,
      deps: {
        ...stubClient(async (_cmd, opts) => { options = opts; return converseReply(GOOD); }),
        trace: captureTrace,
      },
    });
    expect(options.abortSignal).toBe(controller.signal);
  });
});

describe("the trace carries the full exchange (NFR3-16)", () => {
  it("includes the prompt and the raw response on success", async () => {
    await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: { ...stubClient(async () => converseReply(GOOD)), trace: captureTrace },
    });
    expect(traced.prompt.message).toContain("Parents with Children");
    expect(traced.raw).toBe(GOOD);
    expect(traced.outcome.ok).toBe(true);
    expect(typeof traced.elapsedMs).toBe("number");
  });

  it("includes the raw response and the violated rules on rejection", async () => {
    const bad = "Audience Activator matched segment 411 in 41 milliseconds.";
    await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: { ...stubClient(async () => converseReply(bad)), trace: captureTrace },
    });
    expect(traced.raw).toBe(bad);
    expect(traced.outcome.violations.map((v) => v.rule)).toContain("BR3-13");
  });

  it("traces the prompt even when the call fails outright", async () => {
    await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: {
        ...stubClient(async () => {
          const e = new Error("denied"); e.name = "AccessDeniedException"; throw e;
        }),
        trace: captureTrace,
      },
    });
    expect(traced.prompt).not.toBeNull();
    expect(traced.outcome.kind).toBe("access_denied");
    expect(traced.raw).toBeUndefined();
  });

  it("the real trace writes to the console without throwing", async () => {
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const grouped = vi.spyOn(console, "groupCollapsed").mockImplementation(() => {});
    await generateCaption({
      beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
      deps: stubClient(async () => converseReply(GOOD)),
    });
    expect(grouped).toHaveBeenCalled();
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
    grouped.mockRestore();
  });
});
