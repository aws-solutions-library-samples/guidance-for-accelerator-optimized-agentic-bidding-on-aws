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
// From the specific provider package, not the `@aws-sdk/credential-providers`
// umbrella. The umbrella re-exports Node-only providers -- `fromTokenFile` does
// `import { readFileSync } from "node:fs"` -- which breaks Vite's dependency
// pre-bundling and made `npm run dev` unusable while production builds still
// succeeded. Same reason as in agentCoreClient.js.
import { fromCognitoIdentityPool } from "@aws-sdk/credential-provider-cognito-identity";
import { getIdToken } from "./auth";
import { buildCaptionPrompt } from "./utils/captionPrompt.js";
import { validateCaption } from "./utils/captionValidation.js";
import { resolveSegmentLabel } from "./utils/segmentLabels.js";

const COGNITO_REGION = import.meta.env.VITE_COGNITO_REGION || "us-east-1";
const REGION = import.meta.env.VITE_BEDROCK_REGION || COGNITO_REGION;
const IDENTITY_POOL_ID = import.meta.env.VITE_IDENTITY_POOL_ID || "";
const USER_POOL_ID = import.meta.env.VITE_COGNITO_USER_POOL_ID || "";

/**
 * The global inference profile, in every region. Haiku 4.5 reports
 * INFERENCE_PROFILE as its only supported inference type, so the bare model id
 * is not an option; and `global.` is available everywhere, so nothing has to
 * derive a geography from the deployment region.
 */
const DEFAULT_PROFILE_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0";
const PROFILE_ID = import.meta.env.VITE_CAPTION_INFERENCE_PROFILE_ID || DEFAULT_PROFILE_ID;

const COGNITO_PROVIDER = `cognito-idp.${COGNITO_REGION}.amazonaws.com/${USER_POOL_ID}`;

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
  if (!IDENTITY_POOL_ID || !USER_POOL_ID) {
    return { error: { kind: "not_configured", detail: "Identity Pool or User Pool not in this build." } };
  }
  const token = await getIdToken();
  if (!token) {
    return { error: { kind: "not_authenticated", detail: "No Cognito ID token." } };
  }
  if (_cached && _cached.token === token) {
    return { client: _cached.client };
  }
  const client = new BedrockRuntimeClient({
    region: REGION,
    credentials: fromCognitoIdentityPool({
      clientConfig: { region: COGNITO_REGION },
      identityPoolId: IDENTITY_POOL_ID,
      logins: { [COGNITO_PROVIDER]: token },
    }),
  });
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

/** True when the build carries what caption generation needs. */
export function isCaptionConfigured() {
  return !!(IDENTITY_POOL_ID && USER_POOL_ID && PROFILE_ID);
}

export { PROFILE_ID, DEFAULT_PROFILE_ID, REGION };
