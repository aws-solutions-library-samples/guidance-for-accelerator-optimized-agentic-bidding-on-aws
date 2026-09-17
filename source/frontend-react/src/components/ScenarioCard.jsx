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
  // Every scenario is a PUBLISHER BID REQUEST, and each is framed the same way:
  // the page it came from, the audience the exchange asserted on it, and the demand
  // that is eligible to compete for it. The three together are what the sell side
  // actually decides on, so they are the three things a reader needs.
  //
  // The demand line is not decoration. A scenario is only interesting if its deal
  // ids, sizes, media type and floors line up with the campaign catalog in
  // source/demand/artfhouse/catalog.py -- a page whose category matches nothing
  // produces one open-market bid and a folded list of ineligible campaigns. Each
  // `demand` line below was verified against the live exchange, and the contest it
  // states is the contest that runs.
  {
    id: "home-lifestyle",
    name: "Home & Lifestyle — Four-Way Deal Contest",
    page: "A small-space living room guide, declaring Content Taxonomy 3.1 category 283 Interior Decorating. 300x250, and the impression carries home and lifestyle targeting categories.",
    audience: "Interior Decorating and Home Improvement, plus First Time Homeowner — a household life stage, so it is reachable only from the data the exchange asserted, never from the page.",
    demand: "Four deals on the impression. Cedar & Co takes it at $6.35 on the premium home deal; Northlake and the remnant pool bid and lose. Vantage Motorsport holds an auto deal here and is turned away by targeting rather than by price, and the open-market campaign falls under the $2.50 floor.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "deal", label: "SUPPRESS_DEALS" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "home-lifestyle.json",
    controls: ["bidFloor", "segThreshold"],
  },
  {
    id: "finance-news",
    name: "Finance Vertical — One Endemic Buyer",
    page: "A rate-decision analysis declaring Content Taxonomy 3.1 category 410 Personal Investing. 300x250, and the impression carries finance targeting.",
    audience: "Personal Investing and Retirement Planning, a reader in the 50-54 bracket in Illinois.",
    demand: "The scenario where page context decides the auction. Harbour Financial wins at $4.20 on its PMP deal — but Cedar & Co also holds a deal on this impression at $6.35, the highest declared price in the catalog, and targeting is the only thing that stops it. Change the impression's category and the outcome changes. The remnant pool and an outside buyer bid and lose; the open-market campaign falls under the $2.20 floor.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer — Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer — Margin" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "finance-news.json",
    controls: ["bidFloor", "explore"],
  },
  {
    id: "parenting-narrative",
    name: "Parenting Article — Full Sell-Side",
    page: "A first-year sleep guide on a parenting title declaring Content Taxonomy 3.1 category 192, 300x250.",
    audience: "Parents with Children and Parenting Babies and Toddlers. Both are reachable only from the data the exchange asserted, never inferred from what the reader was reading.",
    demand: "Three household deals. Brightstart Family wins at $4.10 on the premium parenting deal, the family network and remnant pools bid and lose, and one campaign is turned away by the floor the yield optimizer set.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer — Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer — Margin" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "deal", label: "SUPPRESS_DEALS" },
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "parenting-narrative.json",
    controls: ["bidFloor", "segThreshold", "explore"],
  },
  {
    id: "yield-optimizer",
    name: "CTV Guaranteed — PMP Yield Optimizer",
    page: "A 1280x720 connected-TV slot on a sports property, sold through three private marketplace deals.",
    audience: "A household segment on a large-format living-room device.",
    demand: "Guaranteed, open mid-tier and open remnant deals. Meridian Guaranteed wins at $13.40; the mid-tier and remnant tiers bid and lose, as does an outside buyer. The yield optimizer predicts a floor multiplier and a margin adjustment per deal on Triton's FIL backend.",
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
    id: "video-deals",
    name: "Mid-roll Video — Outside Bidder Wins",
    page: "A 640x480 mid-roll on a streaming property, offered through three private marketplace deals at premium, standard and remnant tiers.",
    audience: "A viewer segment resolved from the request, then scored against each deal in turn by the deal scorer. All three video campaigns take any category, so here the audience colours the story rather than deciding it.",
    demand: "The scenario an SSP least wants to see: all three house deals bid — $11.50, $8.75 and $8.10 — and an outside buyer takes the impression at $12.50 over the top of them. It is also the scenario that declares the most intents, so the walkthrough is the longest.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer — Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer — Margin" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "deal", label: "SUPPRESS_DEALS" },
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "video-deals.json",
    controls: ["bidFloor", "ageRange", "numDeals", "segThreshold", "explore"],
  },
  {
    id: "banner-basic",
    name: "Open Market — No Premium Demand",
    page: "An NBA article on a sports title, 300x250 on an iPhone. It still declares the legacy Content Taxonomy 1.0, so it is also the scenario that exercises the old category map.",
    audience: "Sports Enthusiast, asserted by the publisher's data provider, on a reader in Illinois.",
    demand: "The thin end of the market: one remnant deal and open-market buyers. An outside buyer takes it at $3.25 over both house offers — what an impression looks like when no premium deal applies to it.",
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
    file: "banner-basic.json",
    controls: ["bidFloor", "ageRange", "segThreshold"],
  },
  {
    id: "full-pipeline",
    name: "Two Impressions — SSP Enrichment Fan-out",
    page: "An automotive review page offering two slots at once: a 970x250 leaderboard and a 300x600 rail.",
    audience: "An in-market automotive segment, resolved once and applied to both impressions.",
    demand: "The leaderboard is a deal-only impression and Autoline Premium takes it at $7.20; Autoline Standard bids $4.60 and loses. The rail admits nobody: every candidate is turned away, and the outside buyer's price falls under both floors.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "metric", label: "ADD_METRICS" },
      { cls: "metric", label: "ADD_CIDS" },
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
      <dl className="scenario-brief" data-testid="scenario-brief">
        <dt>Page</dt>
        <dd>{scenario.page}</dd>
        <dt>Audience</dt>
        <dd>{scenario.audience}</dd>
        <dt>Demand</dt>
        <dd>{scenario.demand}</dd>
      </dl>
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
