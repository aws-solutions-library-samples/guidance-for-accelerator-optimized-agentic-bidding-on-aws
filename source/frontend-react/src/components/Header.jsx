import { isAuthConfigured, signOut, getCurrentUserEmail } from "../auth";
import { useState, useEffect } from "react";
import { LEARNER_COMPLIANCE_SENTENCE } from "./LearnerModeNotice";

const GOVERNANCE_UNAVAILABLE =
  "Unavailable in learner mode: training, retraining and promotion require GPUs.";

export default function Header({
  loading,
  error,
  onContainersClick,
  view,
  onViewChange,
  deploymentMode = "standard",
  deploymentModeReason = null,
}) {
  const learner = deploymentMode === "learner";
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
        {learner && (
          <span
            className="header-mode-badge header-mode-badge--learner"
            data-testid="header-learner-mode-badge"
            title={`Deployed in learner mode: inference runs on CPUs instead of GPUs. ${LEARNER_COMPLIANCE_SENTENCE}`}
          >
            Learner mode
          </span>
        )}
        {deploymentMode === "unknown" && (
          <span
            className="header-mode-badge header-mode-badge--unknown"
            data-testid="header-mode-unknown-badge"
            title={deploymentModeReason || "The orchestrator did not report a deployment mode."}
          >
            Deployment mode unknown
          </span>
        )}
        <button className="btn btn-secondary" onClick={onContainersClick}>
          Containers
        </button>
        <nav className="header-nav">
          <button
            className={`header-nav-link${view === "scenarios" ? " active" : ""}`}
            onClick={() => onViewChange("scenarios")}
          >
            Scenarios
          </button>
          <button
            className={`header-nav-link${view === "adaptive" ? " active" : ""}`}
            onClick={() => onViewChange("adaptive")}
          >
            Adaptive Bidding
          </button>
          {learner ? (
            // aria-disabled rather than disabled, so the button stays focusable
            // and the tooltip is reachable from the keyboard.
            <button
              type="button"
              className="header-nav-link header-nav-link--disabled"
              aria-disabled="true"
              title={GOVERNANCE_UNAVAILABLE}
              data-testid="header-governance-nav-link"
              onClick={(e) => e.preventDefault()}
            >
              Governance{" "}
              <span className="header-nav-link-note">(unavailable in learner mode)</span>
            </button>
          ) : (
            <button
              className={`header-nav-link${view === "governance" ? " active" : ""}`}
              data-testid="header-governance-nav-link"
              onClick={() => onViewChange("governance")}
            >
              Governance
            </button>
          )}
          {/* No Auction Theater entry here any more. The Theater is opened from a
              scenario card, so it always arrives with a scenario and that card's
              tuner values in hand — a nav button could only ever open it empty. */}
        </nav>
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
