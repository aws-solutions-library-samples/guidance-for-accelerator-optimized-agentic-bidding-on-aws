/**
 * Shared SigV4 credential provider for browser-side AWS calls.
 *
 * Every direct-to-AWS call the UI makes (AgentCore invoke, Bedrock captions,
 * the UI API proxy Lambda) exchanges the user's Cognito ID token for temporary
 * credentials through the same Identity Pool. This module owns that exchange so
 * the three clients share one configuration and one set of error kinds.
 *
 * Imported from the sub-package rather than the `@aws-sdk/credential-providers`
 * umbrella, which re-exports Node-only providers that break `vite dev`.
 */

import { fromCognitoIdentityPool } from "@aws-sdk/credential-provider-cognito-identity";
import { getIdToken } from "./auth";

export const COGNITO_REGION = import.meta.env.VITE_COGNITO_REGION || "us-east-1";
export const IDENTITY_POOL_ID = import.meta.env.VITE_IDENTITY_POOL_ID || "";
export const USER_POOL_ID = import.meta.env.VITE_COGNITO_USER_POOL_ID || "";

// Cognito Identity provider key: "cognito-idp.REGION.amazonaws.com/USER_POOL_ID"
export const COGNITO_PROVIDER = `cognito-idp.${COGNITO_REGION}.amazonaws.com/${USER_POOL_ID}`;

/** True when both pools are present in this build. */
export function isIdentityConfigured() {
  return !!(IDENTITY_POOL_ID && USER_POOL_ID);
}

/**
 * Resolve the current ID token and build a credential provider for it.
 *
 * Returns `{ token, credentials }` on success or `{ error: { kind, detail } }`,
 * mirroring bedrockCaptionClient's shape so callers can decide whether to throw.
 * Gates on the presence of a real token, not on `isAuthenticated()`, which is
 * true when no user pool is configured.
 */
export async function getSigV4Credentials() {
  if (!isIdentityConfigured()) {
    return {
      error: {
        kind: "not_configured",
        detail: "Identity Pool or User Pool not in this build (VITE_IDENTITY_POOL_ID / VITE_COGNITO_USER_POOL_ID). Re-deploy with deploy.sh.",
      },
    };
  }
  const token = await getIdToken();
  if (!token) {
    return { error: { kind: "not_authenticated", detail: "Not signed in; no Cognito ID token available." } };
  }
  return {
    token,
    credentials: fromCognitoIdentityPool({
      clientConfig: { region: COGNITO_REGION },
      identityPoolId: IDENTITY_POOL_ID,
      logins: { [COGNITO_PROVIDER]: token },
    }),
  };
}
