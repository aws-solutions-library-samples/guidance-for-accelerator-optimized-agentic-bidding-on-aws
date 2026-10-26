/**
 * bedrockCaptionClient.js — generate one caption for one beat, from the browser.
 *
 * Modelled on agentCoreClient.js, with one deliberate divergence: the client and
 * its credential provider are MEMOISED, keyed on the Cognito ID token.
 * agentCoreClient builds both on every call, which is fine there because agent
 * invocation is user-initiated and infrequent. Captions fire once per beat landed
 * on, so a per-call provider would put the credential exchange in front of every
 * caption instead of the first.
 *
 * Every outcome RESOLVES; nothing throws. The caller's handling is identical for
 * all failures: keep the factual caption.
 *
 * Auth: Cognito Identity Pool authenticated role, exchanged from the signed-in
 * user's ID token. There is no unauthenticated path.
 *
 * Refs:
 * - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
 * - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html
 */

import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { COGNITO_REGION, getSigV4Credentials, isIdentityConfigured } from "./awsCredentials";
import { buildCaptionPrompt } from "./utils/captionPrompt.js";
import { validateCaption } from "./utils/captionValidation.js";
import { buildRunSummaryPrompt } from "./utils/runSummaryPrompt.js";
import { validateRunSummary } from "./utils/runSummaryValidation.js";
import { resolveSegmentLabel } from "./utils/segmentLabels.js";

const REGION = import.meta.env.VITE_BEDROCK_REGION || COGNITO_REGION;

/**
 * The global inference profile, in every region. Haiku 4.5 reports
 * INFERENCE_PROFILE as its only supported inference type, so the bare model id
 * is not an option; and `global.` is available everywhere, so nothing has to
 * derive a geography from the deployment region.
 */
const DEFAULT_PROFILE_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0";
const PROFILE_ID = import.meta.env.VITE_CAPTION_INFERENCE_PROFILE_ID || DEFAULT_PROFILE_ID;

/** @type {{token: string, client: BedrockRuntimeClient} | null} */
let _cached = null;

/**
 * A client whose credentials derive from the current ID token.
 *
 * Keyed on the token so a refresh produces a new provider rather than reusing
 * credentials derived from a stale one.
 *
 * Note it gates on the presence of a real token, NOT on `isAuthenticated()`.
 * That helper returns true when no user pool is configured ("no auth configured
 * = always authed") while `getIdToken()` returns null in the same case, so
 * gating on it would proceed tokenless and fail at the credential exchange.
 */
async function getClient() {
  const { token, credentials, error } = await getSigV4Credentials();
  if (error) {
    return { error };
  }
  if (_cached && _cached.token === token) {
    return { client: _cached.client };
  }
  const client = new BedrockRuntimeClient({ region: REGION, credentials });
  _cached = { token, client };
  return { client };
}

/** Exposed for tests; a token change is otherwise the only thing that resets it. */
export function _resetClientCache() {
  _cached = null;
}

export function mapError(err) {
  const name = err?.name || "";
  const message = String(err?.message || err);
  const both = `${name} ${message}`;
  if (name === "AbortError" || /aborted|abort/i.test(both)) return { kind: "cancelled", detail: message };
  if (/Throttl|TooManyRequests|ServiceQuotaExceeded/i.test(both)) return { kind: "throttled", detail: message };
  if (/AccessDenied|not authorized|Forbidden|UnrecognizedClient/i.test(both)) {
    return { kind: "access_denied", detail: message };
  }
  if (/ResourceNotFound|NotFound/i.test(both)) return { kind: "unknown", detail: message };
  if (/NetworkError|Failed to fetch|CORS/i.test(both)) return { kind: "unreachable", detail: message };
  return { kind: "unknown", detail: message };
}

/**
 * Log the whole exchange: profile, prompt, raw response, verdict, elapsed time.
 *
 * The scenarios are synthetic sample requests authored for this demo, so there is
 * no personal data to withhold, and withholding the prompt would leave a rejected
 * caption indistinguishable from an ungenerated one. Because the fallback is
 * correct either way, this trace is the only thing that distinguishes a missing
 * IAM grant from a wrong profile id from a prompt regression (NFR3-16).
 *
 * Tokens and credentials are never logged.
 */
function trace({ beat, prompt, elapsedMs, outcome, raw }) {
  const label = `[caption] beat ${beat?.index ?? "?"} ${beat?.intent ?? beat?.kind ?? "-"}` +
    ` ${Math.round(elapsedMs)}ms ${outcome.ok ? "ACCEPTED" : outcome.kind.toUpperCase()}`;
  /* eslint-disable no-console */
  const group = console.groupCollapsed ?? console.log;
  group.call(console, label);
  console.log("profile ", PROFILE_ID, "region", REGION);
  if (prompt) console.log("prompt  ", prompt.message);
  if (raw !== undefined) console.log("response", raw);
  if (outcome.ok) console.log("verdict  PASS");
  else if (outcome.violations) {
    console.log("verdict  FAIL", outcome.violations.map((v) => `${v.rule} ${v.detail}`).join("; "));
  } else {
    console.log("verdict ", outcome.kind, outcome.detail);
  }
  if (console.groupEnd) console.groupEnd();
  /* eslint-enable no-console */
}

/**
 * Generate a caption for one beat.
 *
 * @param {object}       args
 * @param {object}       args.beat
 * @param {object}       args.context
 * @param {AbortSignal} [args.signal]
 * @param {object}      [args.deps] injection seam for tests
 * @returns {Promise<{ok: true, text: string} | {ok: false, kind: string, detail: string, violations?: object[]}>}
 */
export async function generateCaption({ beat, context, signal, deps = {} }) {
  const started = (globalThis.performance ?? Date).now();
  const buildPrompt = deps.buildCaptionPrompt ?? buildCaptionPrompt;
  const validate = deps.validateCaption ?? validateCaption;
  const clientFactory = deps.getClient ?? getClient;
  const profileId = deps.profileId ?? PROFILE_ID;
  const emit = deps.trace ?? trace;

  const elapsed = () => (globalThis.performance ?? Date).now() - started;

  if (!profileId) {
    const outcome = { ok: false, kind: "not_configured", detail: "No inference profile id configured." };
    emit({ beat, prompt: null, elapsedMs: elapsed(), outcome });
    return outcome;
  }

  const prompt = buildPrompt(beat, context, { resolveSegmentLabel: deps.resolveSegmentLabel ?? resolveSegmentLabel });

  const { client, error } = await clientFactory();
  if (error) {
    const outcome = { ok: false, ...error };
    emit({ beat, prompt, elapsedMs: elapsed(), outcome });
    return outcome;
  }

  let raw;
  try {
    // Converse, not ConverseStream: a partially streamed caption cannot be
    // validated, because a fabricated number may still be arriving. The IAM
    // grant permits streaming, so this is a validatability choice.
    const response = await client.send(
      new ConverseCommand({
        modelId: profileId,
        system: [{ text: prompt.system }],
        messages: [{ role: "user", content: [{ text: prompt.message }] }],
        inferenceConfig: { maxTokens: prompt.maxTokens, temperature: prompt.temperature },
      }),
      signal ? { abortSignal: signal } : undefined,
    );
    raw = response?.output?.message?.content?.[0]?.text ?? "";
  } catch (err) {
    const outcome = { ok: false, ...mapError(err) };
    emit({ beat, prompt, elapsedMs: elapsed(), outcome });
    return outcome;
  }

  const verdict = validate(raw, beat, context);
  if (!verdict.ok) {
    const outcome = {
      ok: false,
      kind: "invalid",
      detail: verdict.violations.map((v) => v.rule).join(","),
      violations: verdict.violations,
    };
    emit({ beat, prompt, elapsedMs: elapsed(), outcome, raw });
    return outcome;
  }

  const outcome = { ok: true, text: verdict.text };
  emit({ beat, prompt, elapsedMs: elapsed(), outcome, raw });
  return outcome;
}

/**
 * Generate ONE summary for a completed run.
 *
 * Shares this module's client, credential cache, error mapping and trace with
 * `generateCaption`, and differs only in which prompt and which validator it
 * uses. The per-beat path is left exactly as it was: `captionPrompt.js` still
 * never receives the sequence, so BR3-7 is still enforced by construction for
 * captions — this function is the one place that is *supposed* to see the whole
 * run, and it says so in its signature.
 *
 * Fires once per run rather than once per beat, so it is strictly fewer model
 * invocations than the per-beat captions it replaces.
 *
 * @param {object}       args
 * @param {object[]}     args.beats
 * @param {object}       args.context
 * @param {object}       args.viewModel  offers view model, for the outcome
 * @param {AbortSignal} [args.signal]
 * @param {object}      [args.deps]      injection seam for tests
 * @returns {Promise<{ok: true, text: string} | {ok: false, kind: string, detail: string, violations?: object[]}>}
 */
export async function generateRunSummary({ beats, context, viewModel, baseline = null, signal, deps = {} }) {
  const started = (globalThis.performance ?? Date).now();
  const buildPrompt = deps.buildRunSummaryPrompt ?? buildRunSummaryPrompt;
  const validate = deps.validateRunSummary ?? validateRunSummary;
  const clientFactory = deps.getClient ?? getClient;
  const profileId = deps.profileId ?? PROFILE_ID;
  const emit = deps.trace ?? trace;

  const elapsed = () => (globalThis.performance ?? Date).now() - started;
  // The trace is keyed on beats, not on one beat. A synthetic marker keeps the
  // log line's shape identical so the console output stays scannable.
  const traceBeat = { index: "run", intent: null, kind: "run-summary" };

  if (!profileId) {
    const outcome = { ok: false, kind: "not_configured", detail: "No inference profile id configured." };
    emit({ beat: traceBeat, prompt: null, elapsedMs: elapsed(), outcome });
    return outcome;
  }

  const prompt = buildPrompt(beats, context, viewModel, {
    resolveSegmentLabel: deps.resolveSegmentLabel ?? resolveSegmentLabel,
    baseline,
  });

  const { client, error } = await clientFactory();
  if (error) {
    const outcome = { ok: false, ...error };
    emit({ beat: traceBeat, prompt, elapsedMs: elapsed(), outcome });
    return outcome;
  }

  let raw;
  try {
    // Converse, not ConverseStream, for the same reason as the per-beat path: a
    // partially streamed summary cannot be validated, because a fabricated number
    // may still be arriving.
    const response = await client.send(
      new ConverseCommand({
        modelId: profileId,
        system: [{ text: prompt.system }],
        messages: [{ role: "user", content: [{ text: prompt.message }] }],
        inferenceConfig: { maxTokens: prompt.maxTokens, temperature: prompt.temperature },
      }),
      signal ? { abortSignal: signal } : undefined,
    );
    raw = response?.output?.message?.content?.[0]?.text ?? "";
  } catch (err) {
    const outcome = { ok: false, ...mapError(err) };
    emit({ beat: traceBeat, prompt, elapsedMs: elapsed(), outcome });
    return outcome;
  }

  const verdict = validate(raw, beats, context, viewModel);
  if (!verdict.ok) {
    const outcome = {
      ok: false,
      kind: "invalid",
      detail: verdict.violations.map((v) => v.rule).join(","),
      violations: verdict.violations,
    };
    emit({ beat: traceBeat, prompt, elapsedMs: elapsed(), outcome, raw });
    return outcome;
  }

  const outcome = { ok: true, text: verdict.text };
  emit({ beat: traceBeat, prompt, elapsedMs: elapsed(), outcome, raw });
  return outcome;
}

/** True when the build carries what caption generation needs. */
export function isCaptionConfigured() {
  return isIdentityConfigured() && !!PROFILE_ID;
}

export { PROFILE_ID, DEFAULT_PROFILE_ID, REGION };
