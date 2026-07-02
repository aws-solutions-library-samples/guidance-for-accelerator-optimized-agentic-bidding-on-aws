/**
 * AgentCore client — invokes the Bid Shading Strategy Agent directly from the
 * browser using AWS SDK for JS v3 with Cognito Identity Pool credentials.
 *
 * Pattern from:
 * https://docs.aws.amazon.com/sdk-for-javascript/v3/developer-guide/loading-browser-credentials-cognito.html
 * https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-cli-typescript.html
 */

import {
  BedrockAgentCoreClient,
  InvokeAgentRuntimeCommand,
} from "@aws-sdk/client-bedrock-agentcore";
import { fromCognitoIdentityPool } from "@aws-sdk/credential-providers";
import { getIdToken } from "./auth";

// Configuration from Vite env vars (injected at build time by deploy.sh)
const REGION = import.meta.env.VITE_COGNITO_REGION || "us-east-1";
const IDENTITY_POOL_ID = import.meta.env.VITE_IDENTITY_POOL_ID || "";
const USER_POOL_ID = import.meta.env.VITE_COGNITO_USER_POOL_ID || "";
const BID_SHADING_RUNTIME_ARN = import.meta.env.VITE_BID_SHADING_RUNTIME_ARN || "";

// Cognito Identity provider key: "cognito-idp.REGION.amazonaws.com/USER_POOL_ID"
const COGNITO_PROVIDER = `cognito-idp.${REGION}.amazonaws.com/${USER_POOL_ID}`;

/**
 * Get a BedrockAgentCoreClient authenticated via Cognito Identity Pool.
 * The user's ID token is exchanged for temporary AWS credentials.
 */
async function getClient() {
  const idToken = await getIdToken();
  if (!idToken) {
    throw new Error("Not authenticated — no Cognito ID token available");
  }
  if (!IDENTITY_POOL_ID) {
    throw new Error(
      "VITE_IDENTITY_POOL_ID not configured. Re-deploy with deploy.sh to create the Identity Pool."
    );
  }

  return new BedrockAgentCoreClient({
    region: REGION,
    credentials: fromCognitoIdentityPool({
      clientConfig: { region: REGION },
      identityPoolId: IDENTITY_POOL_ID,
      logins: {
        [COGNITO_PROVIDER]: idToken,
      },
    }),
  });
}

/**
 * Invoke the Bid Shading Strategy Agent via AgentCore.
 *
 * @param {object} payload - JSON payload to send to the agent
 * @returns {object} Parsed JSON response from the agent
 */
export async function invokeAgentCore(payload) {
  if (!BID_SHADING_RUNTIME_ARN) {
    throw new Error(
      "VITE_BID_SHADING_RUNTIME_ARN not configured. Deploy the agent with --with-retraining."
    );
  }

  const client = await getClient();

  // Generate a session ID >= 33 chars
  const sessionId = `ui-${Date.now()}-${crypto.randomUUID()}`;

  const command = new InvokeAgentRuntimeCommand({
    agentRuntimeArn: BID_SHADING_RUNTIME_ARN,
    runtimeSessionId: sessionId,
    payload: JSON.stringify(payload),
    contentType: "application/json",
    qualifier: "DEFAULT",
  });

  const response = await client.send(command);

  // Read the streaming response body
  const body = await response.response?.transformToString();
  if (!body) {
    throw new Error("Empty response from AgentCore runtime");
  }

  return JSON.parse(body);
}

/**
 * Check if AgentCore direct invocation is configured.
 */
export function isAgentCoreConfigured() {
  return !!(IDENTITY_POOL_ID && BID_SHADING_RUNTIME_ARN && USER_POOL_ID);
}
