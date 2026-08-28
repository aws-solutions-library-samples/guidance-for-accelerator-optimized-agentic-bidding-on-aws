import { useState, useRef } from "react";

const AGE_RANGES = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"];

// Each scenario's `models` lists the real backend container(s)/model(s)
// this scenario actually invokes, in the same identifier form the
// Governance panel's model selectors use (dlrm_bid_shader,
// deal_yield_manager_floor/margin, ncf_deal_manager) -- so it's always
// clear which model a card is demonstrating, not just which ARTF intent.
// Rule-based containers (Audience Activator, Signals Enricher) are labeled
// "rules" rather than a model_type, since they have no trained model or
// Triton dependency (see containers/widedeep_segment_activator/app.py).
export const SCENARIOS = [
  {
    id: "yield-optimizer",
    name: "PMP Deals — Yield Optimizer",
    desc: "Sports video impression with 3 private marketplace deals (guaranteed, open mid-tier, open remnant). The yield optimizer predicts a floor-price multiplier and margin adjustment per deal via an XGBoost model on Triton's FIL backend.",
    models: [
      { key: "deal_yield_manager_floor", label: "Yield Optimizer — Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer — Margin" },
    ],
    tags: [
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
    ],
    file: "yield-optimizer.json",
    controls: ["bidFloor", "explore"],
  },
  {
    id: "banner-basic",
    name: "Banner Ad — Segment Activation",
    desc: "ESPN sports page with a 300×250 banner. The audience activator activates audience segments via rules, the signals enricher adds viewability scores.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "banner-basic.json",
    controls: ["bidFloor", "ageRange", "segThreshold"],
  },
  {
    id: "bid-shading",
    name: "Bid Shading — Bid Pricer Optimization",
    desc: "Nike DSP bid response at $7.50. The bid pricer predicts CTR and shades the bid down to save budget without losing win rate.",
    models: [{ key: "dlrm_bid_shader", label: "DLRM Bid Shader" }],
    tags: [{ cls: "shade", label: "BID_SHADE" }],
    file: "bid-shading.json",
    controls: ["shadeFactor", "convValue"],
  },
  {
    id: "video-deals",
    name: "Video + PMP Deals — Deal Scorer",
    desc: "Video impression with 3 private marketplace deals. The deal scorer scores user-deal relevance, activates matches, suppresses poor fits.",
    models: [
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "deal", label: "SUPPRESS_DEALS" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "video-deals.json",
    controls: ["bidFloor", "ageRange", "numDeals", "segThreshold"],
  },
  {
    id: "full-pipeline",
    name: "SSP Enrichment — 3 Containers",
    desc: "CNN sports page triggering the SSP-side enrichment containers: segment activation, deal scoring, and signal enrichment in one fan-out. Bid shading is a DSP-side decision made downstream, not something the SSP would request.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "isv-ecosystem.json",
    controls: ["bidFloor"],
  },
];

export default function ScenarioCard({ scenario, isActive, isLoading, disabled, onSelect, onSend }) {
  const [bidFloor, setBidFloor] = useState(1.5);
  const [ageRange, setAgeRange] = useState(1);
  const [numDeals, setNumDeals] = useState(3);
  const [shadeFactor, setShadeFactor] = useState(0.65);
  const [convValue, setConvValue] = useState(12);
  const [segThreshold, setSegThreshold] = useState(0.55);
  // Yield Optimizer's bounded exploration (see shared/yield_exploration.py's
  // resolve_effective_epsilon) -- on by default so a scenario Send
  // against the still-untrained genesis model can produce a real mutation
  // instead of always 0. Once a real model has been trained at least once,
  // the user can flip this off to see that model's unperturbed prediction.
  const [explore, setExplore] = useState(true);

  const controls = scenario.controls || [];

  const getParams = () => ({
    bidFloor,
    ageRange,
    numDeals,
    shadeFactor,
    convValue,
    segThreshold,
    explore,
  });

  const handleClick = (e) => {
    if (disabled) return;
    // Don't trigger select when clicking sliders
    if (e.target.closest(".scenario-tuner")) return;
    if (isActive) {
      onSend(scenario, getParams());
    } else {
      onSelect(scenario);
    }
  };

  const handleSend = (e) => {
    e.stopPropagation();
    onSend(scenario, getParams());
  };

  return (
    <div
      className={`scenario ${isActive ? "active" : ""} ${isLoading ? "loading" : ""} ${disabled ? "disabled" : ""}`}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label={`Run scenario: ${scenario.name}`}
      aria-disabled={disabled}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (disabled) return;
        if (e.target.closest(".scenario-tuner")) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (isActive) onSend(scenario, getParams());
          else onSelect(scenario);
        }
      }}
    >
      <h3>{scenario.name}</h3>
      {scenario.models?.length > 0 && (
        <div className="scenario-models" data-testid="scenario-model-labels">
          {scenario.models.map((m) => (
            <span key={m.key} className={`scenario-model-chip${m.rulesBased ? " rules-based" : ""}`}>
              {m.label}{m.rulesBased ? " (rules)" : ""}
            </span>
          ))}
        </div>
      )}
      <p>{scenario.desc}</p>
      <div className="tags">
        {scenario.tags.map((tag, i) => (
          <span key={i} className={`tag ${tag.cls}`}>{tag.label}</span>
        ))}
      </div>

      {/* Inline tuner — only visible when active */}
      {isActive && (
        <div className="scenario-tuner" onClick={(e) => e.stopPropagation()}>
          {controls.includes("bidFloor") && (
            <div className="tuner-row">
              <label>Bid Floor</label>
              <input type="range" min="0.5" max="15" step="0.5" value={bidFloor}
                onChange={(e) => setBidFloor(parseFloat(e.target.value))} />
              <span className="tuner-value">${bidFloor.toFixed(2)}</span>
            </div>
          )}
          {controls.includes("ageRange") && (
            <div className="tuner-row">
              <label>Age Range</label>
              <input type="range" min="0" max="5" step="1" value={ageRange}
                onChange={(e) => setAgeRange(parseInt(e.target.value))} />
              <span className="tuner-value">{AGE_RANGES[ageRange]}</span>
            </div>
          )}
          {controls.includes("numDeals") && (
            <div className="tuner-row">
              <label>Num Deals</label>
              <input type="range" min="0" max="5" step="1" value={numDeals}
                onChange={(e) => setNumDeals(parseInt(e.target.value))} />
              <span className="tuner-value">{numDeals}</span>
            </div>
          )}
          {controls.includes("shadeFactor") && (
            <div className="tuner-row">
              <label>Shade Factor</label>
              <input type="range" min="0.5" max="0.95" step="0.05" value={shadeFactor}
                onChange={(e) => setShadeFactor(parseFloat(e.target.value))} />
              <span className="tuner-value">{shadeFactor.toFixed(2)}</span>
            </div>
          )}
          {controls.includes("convValue") && (
            <div className="tuner-row">
              <label>Conv Value</label>
              <input type="range" min="5" max="100" step="5" value={convValue}
                onChange={(e) => setConvValue(parseInt(e.target.value))} />
              <span className="tuner-value">${convValue}</span>
            </div>
          )}
          {controls.includes("segThreshold") && (
            <div className="tuner-row">
              <label>Seg Threshold</label>
              <input type="range" min="0.3" max="0.8" step="0.05" value={segThreshold}
                onChange={(e) => setSegThreshold(parseFloat(e.target.value))} />
              <span className="tuner-value">{segThreshold.toFixed(2)}</span>
            </div>
          )}
          {controls.includes("explore") && (
            <div className="tuner-row tuner-row-toggle">
              <label htmlFor={`explore-toggle-${scenario.id}`}>
                Explore
                <span className="tuner-hint">
                  {" "}(perturbs the prediction so an untrained model can still produce a mutation — turn off once trained)
                </span>
              </label>
              <input
                id={`explore-toggle-${scenario.id}`}
                type="checkbox"
                checked={explore}
                onChange={(e) => setExplore(e.target.checked)}
                data-testid="yield-explore-toggle"
              />
            </div>
          )}
          <button className="tuner-send" onClick={handleSend} disabled={disabled}>▶ Send</button>
        </div>
      )}
    </div>
  );
}
