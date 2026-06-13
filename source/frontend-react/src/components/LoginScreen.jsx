import { useState } from "react";
import { signIn, completeNewPassword } from "../auth";

export default function LoginScreen({ onAuthenticated }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [challengeState, setChallengeState] = useState(null); // { user, userAttributes }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const result = await signIn(email, password);
      if (result.newPasswordRequired) {
        setChallengeState({ user: result.user, userAttributes: result.userAttributes });
      } else {
        onAuthenticated();
      }
    } catch (err) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  const handleNewPassword = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await completeNewPassword(challengeState.user, newPassword);
      onAuthenticated();
    } catch (err) {
      setError(err.message || "Password change failed");
    } finally {
      setLoading(false);
    }
  };

  if (challengeState) {
    return (
      <div className="login-screen">
        <div className="login-card">
          <div className="login-header">
            <h1>Set New Password</h1>
            <p className="login-subtitle">Your account requires a password change</p>
          </div>
          <form onSubmit={handleNewPassword}>
            <div className="login-field">
              <label htmlFor="new-password">New Password</label>
              <input
                id="new-password"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="Enter new password"
                required
                minLength={8}
                autoFocus
              />
            </div>
            {error && <div className="login-error" role="alert">{error}</div>}
            <button type="submit" className="login-button" disabled={loading}>
              {loading ? "Updating..." : "Set Password"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="login-screen">
      <div className="login-card">
        <div className="login-header">
          <h1>Accelerator-optimized Agentic Bidding</h1>
          <p className="login-subtitle">Sign in to access the demo</p>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="login-field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              autoFocus
              autoComplete="username"
            />
          </div>
          <div className="login-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              required
              autoComplete="current-password"
            />
          </div>
          {error && <div className="login-error" role="alert">{error}</div>}
          <button type="submit" className="login-button" disabled={loading}>
            {loading ? "Signing in..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
}
