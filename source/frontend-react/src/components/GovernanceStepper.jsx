// GovernanceStepper.jsx — the four-step journey bar on the Governance page.
//
// The steps are NOT gated. Nothing here is truly sequential — a user who wants to
// look at a past comparison should not have to run a load test first — and locking
// later steps would turn a navigation aid into an obstacle. The numbers say what
// order the journey runs in; they do not enforce it.

export const GOVERNANCE_STEPS = [
  {
    key: "load-test",
    number: 1,
    title: "Run Load Test",
    blurb: "Send synthetic traffic through the containers and watch the result.",
  },
  {
    key: "outcome-pipeline",
    number: 2,
    title: "Load Test Outcome Pipeline",
    blurb: "Follow a run from capture through the Glue sweep into the training bucket.",
  },
  {
    key: "train",
    number: 3,
    title: "Train from Load Test",
    blurb: "Train a model on the outcomes one of those runs produced.",
  },
  {
    key: "test-versions",
    number: 4,
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
