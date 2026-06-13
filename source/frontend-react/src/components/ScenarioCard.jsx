import { useState, useRef } from "react";

const AGE_RANGES = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"];

export const SCENARIOS = [
  {
    id: "banner-basic",
    name: "Banner Ad — Segment Activation",
    desc: "ESPN sports page with a 300×250 banner. Wide & Deep activates audience segments, Metrics adds viewability scores.",
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "banner-basic.json",
    controls: ["bidFloor", "ageRange", "segThreshold"],
  },
  {
    id: "bid-shading",
    name: "Bid Shading — DLRM Price Optimization",
    desc: "Nike DSP bid response at $7.50. DLRM predicts CTR and shades the bid down to save budget without losing win rate.",
    tags: [{ cls: "shade", label: "BID_SHADE" }],
    file: "bid-shading.json",
    controls: ["shadeFactor", "convValue"],
  },
  {
    id: "video-deals",
    name: "Video + PMP Deals — NCF Scoring",
    desc: "Video impression with 3 private marketplace deals. NCF scores user-deal relevance, activates matches, suppresses poor fits.",
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
    name: "Full Pipeline — All 4 Containers",
    desc: "CNN sports page triggering all containers: segments, deals, bid shading, and metrics in one fan-out.",
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "shade", label: "BID_SHADE" },
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

  const controls = scenario.controls || [];

  const getParams = () => ({
    bidFloor,
    ageRange,
    numDeals,
    shadeFactor,
    convValue,
    segThreshold,
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
          <button className="tuner-send" onClick={handleSend} disabled={disabled}>▶ Send</button>
        </div>
      )}
    </div>
  );
}
