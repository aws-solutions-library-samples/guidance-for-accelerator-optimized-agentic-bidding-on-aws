/**
 * AgentCore client — invokes the closed-loop agents DIRECTLY from the browser
 * (never through the real-time bidding orchestrator).
 *
 * Auth: IAM SigV4 with temporary credentials from the Cognito Identity Pool.
 * The runtimes stay on SigV4 inbound auth (the same mechanism the production
 * EventBridge invokers use), so we can invoke them with the AWS SDK. See
 * SigV4 (not OAuth/JWT) is used here.
 *
 * Refs:
 * - https://docs.aws.amazon.com/sdk-for-javascript/v3/developer-guide/loading-browser-credentials-cognito.html
 * - https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_InvokeAgentRuntime.html
 */

import {
  BedrockAgentCoreClient,
  InvokeAgentRuntimeCommand,
} from "@aws-sdk/client-bedrock-agentcore";
import { COGNITO_REGION as REGION, getSigV4Credentials, isIdentityConfigured } from "./awsCredentials";

// Runtime ARNs. Accept the legacy VITE_BID_SHADING_RUNTIME_ARN name as a
// fallback for the adaptive runtime so older deploys keep working.
const ADAPTIVE_BIDDING_RUNTIME_ARN =
  import.meta.env.VITE_ADAPTIVE_BIDDING_RUNTIME_ARN ||
  import.meta.env.VITE_BID_SHADING_RUNTIME_ARN ||
  "";
const GOVERNANCE_RUNTIME_ARN = import.meta.env.VITE_GOVERNANCE_RUNTIME_ARN || "";

/**
 * Error thrown when an agent is not deployed/configured or is unreachable.
 * Carries a machine-readable `kind` so the UI can render an honest state
 * (never a fabricated decision).
 */
export class AgentInvokeError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = "AgentInvokeError";
    this.kind = kind; // "not_configured" | "not_authenticated" | "access_denied" | "not_found" | "unreachable" | "unknown"
  }
}

/**
 * BedrockAgentCoreClient authenticated via the Cognito Identity Pool.
 * The user's ID token is exchanged for temporary SigV4 credentials.
 */
async function getClient() {
  const { credentials, error } = await getSigV4Credentials();
  if (error) {
    throw new AgentInvokeError(error.kind, error.detail);
  }
  return new BedrockAgentCoreClient({ region: REGION, credentials });
}

function _mapError(err) {
  const name = err?.name || "";
  const msg = String(err?.message || err);
  if (err instanceof AgentInvokeError) return err;
  if (/AccessDenied|not authorized|Forbidden/i.test(name + msg)) {
    return new AgentInvokeError(
      "access_denied",
      "Access denied invoking the agent. The Identity Pool authenticated role needs bedrock-agentcore:InvokeAgentRuntime on this runtime ARN."
    );
  }
  if (/ResourceNotFound|NotFound/i.test(name + msg)) {
    return new AgentInvokeError("not_found", "Agent runtime not found. It may not be deployed yet.");
  }
  if (/NetworkError|Failed to fetch|CORS|TypeError/i.test(name + msg)) {
    return new AgentInvokeError(
      "unreachable",
      "Could not reach the agent runtime from the browser (possible CORS/network issue)."
    );
  }
  return new AgentInvokeError("unknown", msg);
}

/**
 * Invoke an AgentCore runtime (SigV4) and return the parsed JSON response.
 * @param {string} runtimeArn - full runtime ARN
 * @param {object} payload - JSON-serializable invocation payload
 */
async function invokeRuntime(runtimeArn, payload) {
  if (!runtimeArn) {
    throw new AgentInvokeError(
      "not_configured",
      "Agent runtime ARN not configured. Deploy the agents with: ./deploy.sh --with-retraining"
    );
  }
  let client;
  try {
    client = await getClient();
  } catch (e) {
    throw _mapError(e);
  }

  // Session id must be >= 33 chars.
  const sessionId = `ui-${Date.now()}-${crypto.randomUUID()}`;

  try {
    const command = new InvokeAgentRuntimeCommand({
      agentRuntimeArn: runtimeArn,
      runtimeSessionId: sessionId,
      payload: JSON.stringify(payload),
      contentType: "application/json",
      accept: "application/json",
      qualifier: "DEFAULT",
    });
    const response = await client.send(command);
    const body = await response.response?.transformToString();
    if (!body) {
      throw new AgentInvokeError("unknown", "Empty response from the agent runtime.");
    }
    return JSON.parse(body);
  } catch (e) {
    throw _mapError(e);
  }
}

/** Invoke the Adaptive Bidding Strategy Agent. */
export function invokeAdaptive(payload) {
  return invokeRuntime(ADAPTIVE_BIDDING_RUNTIME_ARN, payload);
}

/** Invoke the Model Promotion Governance Agent. */
export function invokeGovernance(payload) {
  return invokeRuntime(GOVERNANCE_RUNTIME_ARN, payload);
}

/** True when the shared prerequisites (identity pool + user pool) are present. */
const _identityConfigured = isIdentityConfigured;

export function isAdaptiveConfigured() {
  return _identityConfigured() && !!ADAPTIVE_BIDDING_RUNTIME_ARN;
}

export function isGovernanceConfigured() {
  return _identityConfigured() && !!GOVERNANCE_RUNTIME_ARN;
}

/** Back-compat: prior code called isAgentCoreConfigured() for the adaptive agent. */
export function isAgentCoreConfigured() {
  return isAdaptiveConfigured();
}

/** Human-readable reason an agent isn't invokable (for honest UI states), or null if it is. */
export function agentUnavailableReason(agent /* "adaptive" | "governance" */) {
  if (!_identityConfigured()) {
    return "Sign-in / Identity Pool not configured — re-deploy with deploy.sh.";
  }
  const arn = agent === "governance" ? GOVERNANCE_RUNTIME_ARN : ADAPTIVE_BIDDING_RUNTIME_ARN;
  if (!arn) {
    return "Agent not deployed — run ./deploy.sh --with-retraining, then redeploy the frontend.";
  }
  return null;
}
