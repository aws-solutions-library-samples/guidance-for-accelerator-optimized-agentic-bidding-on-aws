import { useState, useEffect } from "react";
import { isAuthConfigured, isAuthenticated } from "../auth";
import LoginScreen from "./LoginScreen";

/**
 * AuthGate — wraps the app and shows a login screen if Cognito is configured
 * and the user is not authenticated. If auth is not configured (local dev),
 * renders children immediately.
 */
export default function AuthGate({ children }) {
  const [authed, setAuthed] = useState(null); // null = checking, true/false = known

  useEffect(() => {
    if (!isAuthConfigured()) {
      setAuthed(true);
      return;
    }
    isAuthenticated().then((ok) => setAuthed(ok));
  }, []);

  if (authed === null) {
    // Still checking session
    return (
      <div className="login-screen">
        <div className="login-card">
          <p style={{ textAlign: "center", opacity: 0.6 }}>Checking session...</p>
        </div>
      </div>
    );
  }

  if (!authed) {
    return <LoginScreen onAuthenticated={() => setAuthed(true)} />;
  }

  return (
    <>
      {children}
    </>
  );
}

/**
 * Helper to get auth headers for fetch calls.
 * Returns {} if auth is not configured.
 */
export async function getAuthHeaders() {
  if (!isAuthConfigured()) return {};
  const token = await getAccessToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}
