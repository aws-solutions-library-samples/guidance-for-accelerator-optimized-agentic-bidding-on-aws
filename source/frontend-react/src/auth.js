/**
 * Cognito authentication module.
 *
 * Uses amazon-cognito-identity-js for SRP-based login against the
 * Cognito User Pool provisioned by deploy.sh.
 *
 * Config is injected at build time via Vite's env replacement:
 *   VITE_COGNITO_USER_POOL_ID
 *   VITE_COGNITO_CLIENT_ID
 *   VITE_COGNITO_REGION
 */

import {
  CognitoUserPool,
  CognitoUser,
  AuthenticationDetails,
} from "amazon-cognito-identity-js";

const POOL_ID = import.meta.env.VITE_COGNITO_USER_POOL_ID || "";
const CLIENT_ID = import.meta.env.VITE_COGNITO_CLIENT_ID || "";

const poolData = POOL_ID && CLIENT_ID ? { UserPoolId: POOL_ID, ClientId: CLIENT_ID } : null;
const userPool = poolData ? new CognitoUserPool(poolData) : null;

/**
 * Returns true if Cognito is configured (pool + client IDs present).
 */
export function isAuthConfigured() {
  return !!userPool;
}

/**
 * Get the current authenticated user's JWT access token, or null.
 */
export function getAccessToken() {
  if (!userPool) return null;
  const user = userPool.getCurrentUser();
  if (!user) return null;

  return new Promise((resolve) => {
    user.getSession((err, session) => {
      if (err || !session || !session.isValid()) {
        resolve(null);
      } else {
        resolve(session.getAccessToken().getJwtToken());
      }
    });
  });
}

/**
 * Get the current user's ID token (contains email, name claims).
 */
export function getIdToken() {
  if (!userPool) return null;
  const user = userPool.getCurrentUser();
  if (!user) return null;

  return new Promise((resolve) => {
    user.getSession((err, session) => {
      if (err || !session || !session.isValid()) {
        resolve(null);
      } else {
        resolve(session.getIdToken().getJwtToken());
      }
    });
  });
}

/**
 * Check if a user is currently authenticated with a valid session.
 */
export function isAuthenticated() {
  if (!userPool) return Promise.resolve(true); // No auth configured = always authed
  const user = userPool.getCurrentUser();
  if (!user) return Promise.resolve(false);

  return new Promise((resolve) => {
    user.getSession((err, session) => {
      resolve(!err && session && session.isValid());
    });
  });
}

/**
 * Sign in with email + password. Returns the session on success.
 * Throws on failure with a message string.
 */
export function signIn(email, password) {
  if (!userPool) return Promise.reject(new Error("Auth not configured"));

  const user = new CognitoUser({ Username: email, Pool: userPool });
  const authDetails = new AuthenticationDetails({ Username: email, Password: password });

  return new Promise((resolve, reject) => {
    user.authenticateUser(authDetails, {
      onSuccess: (session) => resolve(session),
      onFailure: (err) => reject(err),
      newPasswordRequired: (userAttributes) => {
        // First login — Cognito requires password change
        // For demo purposes, resolve with a special marker
        resolve({ newPasswordRequired: true, user, userAttributes });
      },
    });
  });
}

/**
 * Complete new password challenge (first login after admin-created account).
 */
export function completeNewPassword(cognitoUser, newPassword) {
  return new Promise((resolve, reject) => {
    cognitoUser.completeNewPasswordChallenge(newPassword, {}, {
      onSuccess: (session) => resolve(session),
      onFailure: (err) => reject(err),
    });
  });
}

/**
 * Sign out the current user.
 */
export function signOut() {
  if (!userPool) return;
  const user = userPool.getCurrentUser();
  if (user) user.signOut();
}

/**
 * Get the current user's email from the ID token.
 */
export async function getCurrentUserEmail() {
  if (!userPool) return null;
  const token = await getIdToken();
  if (!token) return null;
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload.email || payload["cognito:username"] || null;
  } catch {
    return null;
  }
}
