import NvidiaLogo from "../logos/Nvidia_logo.svg";
import { isAuthConfigured, signOut, getCurrentUserEmail } from "../auth";
import { useState, useEffect } from "react";

export default function Header({ loading, error, onContainersClick }) {
  const statusClass = error ? "err" : loading ? "warn" : "ok";
  const statusLabel = error ? "Error" : loading ? "Processing…" : "Connected";
  const [userEmail, setUserEmail] = useState(null);

  useEffect(() => {
    if (isAuthConfigured()) {
      getCurrentUserEmail().then(setUserEmail);
    }
  }, []);

  return (
    <header className="header">
      <div className="header-brand">
        <img src={NvidiaLogo} alt="NVIDIA" className="header-logo" />
        <div>
          <h1>ARTF Containers on AWS</h1>
          <div className="subtitle">Low Latency Agentic Bidstream Mutations</div>
        </div>
      </div>
      <div className="header-actions">
        <div className="status-bar">
          <span className={`status-dot ${statusClass}`} aria-hidden="true" />
          <span>{statusLabel}</span>
        </div>
        <button className="btn btn-secondary" onClick={onContainersClick}>
          Containers
        </button>
        {userEmail && (
          <div className="header-auth">
            <span className="auth-user">{userEmail}</span>
            <button
              className="auth-signout"
              onClick={() => { signOut(); window.location.reload(); }}
            >
              Sign Out
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
