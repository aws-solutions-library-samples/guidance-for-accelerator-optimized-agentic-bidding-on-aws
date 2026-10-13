// GovernanceStepper.jsx — the three-step journey bar on the Governance page.
//
// The steps are NOT gated. Nothing here is truly sequential — a user who wants to
// look at a past comparison should not have to run a load test first — and locking
// later steps would turn a navigation aid into an obstacle. The numbers say what
// order the journey runs in; they do not enforce it.
//
// The outcome pipeline (capture, Firehose flush, Glue sweep, trainable) is part of
// the Train step: one run picker, with the pipeline for the selected run under it.

export const GOVERNANCE_STEPS = [
  {
    key: "load-test",
    number: 1,
    title: "Run Load Test",
    blurb: "Send synthetic traffic through the containers and watch the result.",
  },
  {
    key: "train",
    number: 2,
    title: "Train from Load Test",
    blurb: "Pick a run, follow its outcomes into the training bucket, and train on them.",
  },
  {
    key: "test-versions",
    number: 3,
    title: "Test Model Versions",
    blurb: "Compare two runs' outcomes and decide whether to promote.",
  },
];

export default function GovernanceStepper({ activeKey, onSelect }) {
  return (
    <nav className="gov-stepper" aria-label="Governance journey" data-testid="governance-stepper">
      <ol className="gov-stepper-list">
        {GOVERNANCE_STEPS.map((step) => {
          const isOn = step.key === activeKey;
          return (
            <li key={step.key} className="gov-stepper-item">
              <button
                type="button"
                className={`gov-step${isOn ? " is-on" : ""}`}
                aria-current={isOn ? "step" : undefined}
                onClick={() => onSelect(step.key)}
                data-testid={`governance-step-${step.key}`}
              >
                <span className="gov-step-number" aria-hidden="true">{step.number}</span>
                <span className="gov-step-text">
                  <span className="gov-step-title">{step.title}</span>
                  <span className="gov-step-blurb">{step.blurb}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
