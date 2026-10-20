import { useEffect, useRef } from "react";

/**
 * Modal shown on every page load of a learner-mode deployment.
 *
 * Acknowledgement lives only in the caller's React state, never in storage, so
 * the notice comes back on the next load. Escape does not dismiss it: the
 * compliance statement has to be acknowledged with the button. Focus starts on
 * the button and stays inside the dialog while it is open.
 */
export const LEARNER_COMPLIANCE_SENTENCE =
  "For this implementation to remain ARTF-compliant, learner mode must not be used for production workloads.";

export default function LearnerModeNotice({ onAcknowledge }) {
  const buttonRef = useRef(null);

  useEffect(() => {
    if (buttonRef.current) buttonRef.current.focus();
  }, []);

  const onKeyDown = (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (event.key === "Tab") {
      // The button is the only focusable element, so Tab and Shift+Tab both
      // return to it.
      event.preventDefault();
      if (buttonRef.current) buttonRef.current.focus();
    }
  };

  return (
    <div className="learner-notice-backdrop">
      <div
        className="learner-notice"
        role="dialog"
        aria-modal="true"
        aria-labelledby="learner-notice-title"
        aria-describedby="learner-notice-body"
        data-testid="learner-mode-notice-dialog"
        onKeyDown={onKeyDown}
      >
        <h2 id="learner-notice-title" className="learner-notice-title">
          Deployed in learner mode
        </h2>
        <div id="learner-notice-body" className="learner-notice-body">
          <p>
            This workload was deployed in learner mode. {LEARNER_COMPLIANCE_SENTENCE}
          </p>
          <p>
            Inference runs on CPUs instead of GPUs: Triton Inference Server serves the
            models on the cluster&apos;s CPU nodes. The Governance tab is unavailable
            because training, retraining and promotion require GPUs.
          </p>
        </div>
        <div className="learner-notice-actions">
          <button
            ref={buttonRef}
            type="button"
            className="btn btn-primary"
            data-testid="learner-mode-notice-acknowledge-button"
            onClick={onAcknowledge}
          >
            I understand
          </button>
        </div>
      </div>
    </div>
  );
}
