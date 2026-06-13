/**
 * Authenticated fetch wrapper.
 *
 * Injects the Cognito access token as a Bearer header on every request
 * to the backend API. Falls through to regular fetch if auth is not configured.
 */

import { isAuthConfigured, getAccessToken } from "./auth";

/**
 * Drop-in replacement for fetch() that adds Authorization header.
 * Use this for all backend API calls.
 */
export async function authFetch(url, init = {}) {
  if (!isAuthConfigured()) {
    return fetch(url, init);
  }

  const token = await getAccessToken();
  if (!token) {
    // Session expired — force reload to show login
    window.location.reload();
    throw new Error("Session expired");
  }

  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${token}`);

  return fetch(url, { ...init, headers });
}
