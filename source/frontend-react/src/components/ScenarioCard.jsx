import { useState, useRef } from "react";
import { DEAL_SCORER_TRAINING_DISCLOSURE, scenarioUsesDealScorer } from "../utils/dealScorerTraining.js";

const AGE_RANGES = ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"];

// Each scenario's `models` lists the real backend container(s)/model(s)
// this scenario actually invokes, in the same identifier form the
// Governance panel's model selectors use (dlrm_bid_shader,
// deal_yield_manager_floor/margin, ncf_deal_manager) -- so it's always
// clear which model a card is demonstrating, not just which ARTF intent.
// Rule-based containers (Audience Activator, Signals Enricher) are labeled
// "rules" rather than a model_type, since they have no trained model or
// Triton dependency (see containers/widedeep_segment_activator/app.py).
/**
 * The two ARTF surfaces a scenario can mutate.
 *
 * This is a DECLARATION on each scenario, not something derived at render time,
 * and it is asserted against the payload's `lifecycle` in
 * test_scenario_card_fixture_wiring.py — a card filed under the wrong surface would
 * appear under the wrong toggle with nothing to catch it.
 *
 * The asymmetry between them is a fact about the system, not a gap in the
 * scenarios: `orchestrator/app.py`'s CONTAINERS list gives exactly ONE container a
 * response-side intent (Bid Pricer / BID_SHADE). The other five all mutate the
 * request. So response scenarios vary by payload content, not by intent count.
 */
export const SURFACE_REQUEST = "request";
export const SURFACE_RESPONSE = "response";

export const SURFACES = [
  { key: SURFACE_REQUEST, label: "Bid Request Mutations" },
  { key: SURFACE_RESPONSE, label: "Bid Response Mutations" },
];

export const SCENARIOS = [
  // Every scenario is a PUBLISHER BID REQUEST, and each is framed the same way:
  // the page it came from, the audience the exchange asserted on it, and the demand
  // that is eligible to compete for it. The three together are what the sell side
  // actually decides on, so they are the three things a reader needs.
  //
  // The demand line is not decoration. A scenario is only interesting if its deal
  // ids, sizes, media type and floors line up with the campaign and deal records
  // in deployment/seed/deals_and_campaigns.json (loaded into the deals table) --
  // a page whose category matches nothing
  // produces one open-market bid and a folded list of ineligible campaigns.
  //
  // WHAT THESE LINES MAY AND MAY NOT CLAIM.
  //
  // An earlier version of each line named a winner and a price, and two went stale
  // within a day: the genesis Deal Scorer's scores sat around 0.5, right on its
  // thresholds, so the same payload flipped a deal between live and suppressed.
  //
  // The Deal Scorer is now trained at deploy time on sample interactions built from
  // the deal book and the cohort spec (source/training/ncf_sample/cohorts.json), and
  // the export gate fails the deploy unless each scenario user's activations and
  // suppressions come out as the spec says. The spec also names each scenario's
  // target winner, and test_seed_price_invariants.py checks that every target
  // winner's CPM clears the highest floor the Yield Optimizer can set on its deal
  // (1.5f + 0.5 on a first-price deal, 2.25f otherwise) and beats the simulator.
  //
  // A card's demand line is at most two short sentences: the baseline winner and
  // the with-ARTF winner, with prices. Those prices come from what the code, the
  // seed and the cohort spec fix (and agree with the live run in
  // .agents/tasks/ncf-scenario-training-2026-10-07/report.md):
  //   - which deals the publisher offers on the request (only its open/remnant
  //     deals, see below) and which campaign holds each
  //   - which deals the Deal Scorer activates from its publisher deal library
  //     (the deals table's deal records, matched by site.domain or app.bundle) and
  //     the floor each arrives with
  //   - each campaign's declared CPM (its campaign record in the deals seed;
  //     Courtside 4.25, Harbour 5.80, Brightstart 5.65, Cedar 6.80, Autoline
  //     Premium 9.55, Skyline 13.75, Meridian Guaranteed 13.40 for the winners)
  //   - who targeting turns away (deterministic from imp.ext.artf.categories)
  //   - who the impression floor turns away (deterministic arithmetic)
  //   - the bidder simulator's fixed prices ($3.25 banner, $12.50 video)
  // A price the Yield Optimizer sets at run time is not quoted; the walkthrough
  // shows the decisions that were actually made.
  //
  // WHY THE FIXTURES OFFER ONLY OPEN/REMNANT DEALS.
  //
  // The Theater runs every request scenario twice, with and without ARTF, and
  // compares the outcomes. When a fixture pre-declared the premium deal the Deal
  // Scorer would have activated, both passes carried the same deals and both
  // passes produced the same winner at the same price. So each request fixture now
  // offers only the deal(s) whose id contains "open" or "remnant"; the premium and
  // mid-tier deals live in the Deal Scorer's library and reach the auction only
  // when it activates them, floor included. The baseline pass therefore sees the
  // thin book, and the with-ARTF pass sees whatever the scorer added to it.
  {
    id: "home-lifestyle",
    surface: SURFACE_REQUEST,
    name: "Home & Lifestyle: Four-Way Deal Contest",
    page: "A small-space living room guide, 300x250.",
    audience: "Home decor readers and first-time homeowners.",
    demand: "Without ARTF, an outside buyer wins at $3.25. With ARTF, the Deal Scorer adds the premium home deal and Cedar & Co wins at $6.80.",
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
    defaults: { bidFloor: 2.5 },
  },
  {
    id: "finance-news",
    surface: SURFACE_REQUEST,
    name: "Finance Vertical: One Endemic Buyer",
    page: "A rate-decision analysis on a finance title, 300x250.",
    audience: "Investing and retirement readers, age 50 to 54.",
    demand: "Without ARTF, an outside buyer wins at $3.25. With ARTF, the Deal Scorer adds the finance deal and Harbour Financial wins at $5.80.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer: Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer: Margin" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "finance-news.json",
    controls: ["bidFloor", "explore"],
    defaults: { bidFloor: 2.2 },
  },
  {
    id: "parenting-narrative",
    surface: SURFACE_REQUEST,
    name: "Parenting Article: Full Sell-Side",
    page: "A first-year sleep guide on a parenting title, 300x250.",
    audience: "Parents of babies and toddlers.",
    demand: "Without ARTF, an outside buyer wins at $3.25. With ARTF, the premium parenting deal and the parenting segments let Brightstart Family win at $6.05.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer: Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer: Margin" },
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
    defaults: { bidFloor: 2.6 },
  },
  {
    id: "yield-optimizer",
    surface: SURFACE_REQUEST,
    name: "CTV Guaranteed: PMP Yield Optimizer",
    page: "A connected-TV slot on a sports property, 1280x720.",
    audience: "A household on a living-room screen.",
    demand: "Without ARTF, an outside buyer wins at $12.50. With ARTF, the Deal Scorer adds the guaranteed deal and Meridian Guaranteed wins at $13.40.",
    models: [
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer: Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer: Margin" },
    ],
    tags: [
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
    ],
    file: "yield-optimizer.json",
    controls: ["bidFloor", "explore"],
    defaults: { bidFloor: 6.0 },
  },
  {
    id: "video-deals",
    surface: SURFACE_REQUEST,
    name: "Mid-roll Video: The Activated Deal Wins",
    page: "A mid-roll on a streaming property, 640x480.",
    audience: "A viewer watching streamed video.",
    demand: "Without ARTF, an outside buyer wins at $12.50. With ARTF, the Deal Scorer adds the premium video deal and Skyline wins at $13.75.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer: Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer: Margin" },
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
    // numDeals was dropped from this card: it sliced the pre-declared deal list,
    // and the fixture now offers a single remnant deal (the rest come from the
    // Deal Scorer's library at run time, which a request-side slider cannot reach).
    controls: ["bidFloor", "ageRange", "segThreshold", "explore"],
    defaults: { bidFloor: 8.0 },
  },
  {
    id: "banner-basic",
    surface: SURFACE_REQUEST,
    name: "Sports Banner: One Library Deal",
    page: "An NBA article on a sports title, 300x250 on iPhone.",
    audience: "Sports enthusiasts in Illinois.",
    demand: "Without ARTF, an outside buyer wins at $3.25. With ARTF, the Deal Scorer adds the sports deal and Courtside Outfitters wins at $4.25.",
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
    defaults: { bidFloor: 1.9 },
  },
  {
    id: "full-pipeline",
    surface: SURFACE_REQUEST,
    name: "Two Impressions: SSP Enrichment Fan-out",
    page: "An automotive review with a leaderboard and a rail.",
    audience: "In-market auto shoppers.",
    demand: "Without ARTF, neither slot fills. With ARTF, Autoline Premium fills the leaderboard at $9.55 and the rail stays empty.",
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
    defaults: { bidFloor: 4.0 },
  },
  // The two RideSharer entries put both deals on the request, because the Yield
  // Optimizer is what differs between the passes, and their copy quotes only the
  // request's own floors: no price, winner or latency the live run decides.
  {
    id: "ridesharer-airport",
    surface: SURFACE_REQUEST,
    name: "RideSharer: Airport Run, Raise the Floor",
    page: "An in-app full-screen banner on a 45-minute ride to the airport.",
    audience: "Business travellers on a weekday evening.",
    demand: "Altimara Travel and Ivory Row Luggage hold deals at a $12.00 floor. Without ARTF Altimara Travel wins at $12.70; with ARTF the travel segments lift it to $13.30 and the Yield Optimizer sets the floor at run time.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer: Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer: Margin" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "ridesharer-airport-evening.json",
    controls: ["bidFloor", "segThreshold", "explore"],
    defaults: { bidFloor: 5.0 },
  },
  {
    id: "ridesharer-midday",
    surface: SURFACE_REQUEST,
    name: "RideSharer: Midday Ride, Lower the Floor",
    page: "An in-app full-screen banner on a short midday ride.",
    audience: "Local riders near RideSharer Eats restaurants.",
    demand: "Basil & Brine Kitchen and Maple Corner Noodle House bid under the $12.00 floor, so nothing sells without ARTF. With ARTF and Explore on, the Yield Optimizer lowers the floor at run time so one can clear.",
    models: [
      { key: "widedeep_segment_activator", label: "Audience Activator", rulesBased: true },
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer: Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer: Margin" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "seg", label: "ACTIVATE_SEGMENTS" },
      { cls: "deal", label: "ACTIVATE_DEALS" },
      { cls: "yield", label: "ADJUST_DEAL_FLOOR" },
      { cls: "yield", label: "ADJUST_DEAL_MARGIN" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "ridesharer-suburban-midday.json",
    controls: ["bidFloor", "segThreshold", "explore"],
    defaults: { bidFloor: 5.0 },
  },

  // -------------------------------------------------------------------------
  // BID RESPONSE MUTATIONS
  //
  // One container mutates a bid response: the Bid Pricer (BID_SHADE). So these
  // four differ by payload content, not by intent count. What separates them is
  // which term of the shader's own arithmetic ends up binding:
  //
  //   ceiling = predicted_ctr * conversion_value * shade_factor
  //   shaded  = max(impression_floor, min(original_price, ceiling))
  //   ... and a mutation is emitted only when |shaded - original_price| > $0.01
  //
  // `predicted_ctr` is a live DLRM prediction whose features include the HOUR OF
  // DAY (containers/dlrm_bid_shader/app.py's _extract_features_np), so the exact
  // shaded price is not stable across a day. Each scenario is therefore built so
  // its OUTCOME CLASS holds for any plausible ctr, using `defaults` to place the
  // ceiling — and the card text describes the mechanism rather than quoting a
  // price the walkthrough will show for real.
  //
  // The Theater's offers column resolves the BID REQUEST's own auction. On a
  // response scenario that auction is the sell side's, not this DSP's response,
  // which is why none of the text below implies the shaded price competed.
  // -------------------------------------------------------------------------
  {
    id: "response-shade-headroom",
    surface: SURFACE_RESPONSE,
    name: "Bid Response — Priced Down To Its Worth",
    page: "A 728x90 leaderboard with a $0.10 floor.",
    audience: "Reader age, device and hashed ids, as the DLRM reads them.",
    demand: "The DSP bids $25.00. The Bid Pricer prices it down to the model's expected value.",
    models: [
      { key: "dlrm_bid_shader", label: "Bid Pricer" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "shade", label: "BID_SHADE" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "response-shade-headroom.json",
    controls: ["bidFloor", "shadeFactor", "convValue"],
    // Places the ceiling between the $0.10 floor and the $25.00 bid for any ctr
    // in roughly (0.003, 0.78), which covers the whole plausible range.
    defaults: { bidFloor: 0.1, convValue: 40, shadeFactor: 0.8 },
  },
  {
    id: "response-floor-clamped",
    surface: SURFACE_RESPONSE,
    name: "Bid Response — The Floor Overrides The Model",
    page: "A 300x250 on a finance title with a $3.00 floor.",
    audience: "A 55-plus reader on desktop.",
    demand: "The DSP bids $12.00 and the model values it under $3.00. The Bid Pricer clamps the price at the $3.00 floor.",
    models: [
      { key: "dlrm_bid_shader", label: "Bid Pricer" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "shade", label: "BID_SHADE" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "response-floor-clamped.json",
    controls: ["bidFloor", "shadeFactor", "convValue"],
    // Ceiling = ctr * 2.5, under the $3.00 floor for any ctr below 1.2 — i.e.
    // always. Raise conversion value on the slider and the floor stops binding.
    defaults: { bidFloor: 3.0, convValue: 5, shadeFactor: 0.5 },
  },
  {
    id: "response-no-shade",
    surface: SURFACE_RESPONSE,
    name: "Bid Response — The Model Declines To Act",
    page: "A 320x50 mobile banner on a recipe page, $0.50 floor.",
    audience: "A young reader on Android.",
    demand: "The DSP bids $1.20, already at or under the model's value. The Bid Pricer returns no mutation.",
    models: [
      { key: "dlrm_bid_shader", label: "Bid Pricer" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "shade", label: "BID_SHADE" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "response-no-shade.json",
    controls: ["bidFloor", "shadeFactor", "convValue"],
    // Ceiling = ctr * 95, above the $1.20 bid for any ctr over 0.013. Drop the
    // conversion value and the model starts pricing it down.
    defaults: { bidFloor: 0.5, convValue: 100, shadeFactor: 0.95 },
  },
  {
    id: "response-multi-seat",
    surface: SURFACE_RESPONSE,
    name: "Bid Response — Four Bids, Two Seats, Priced Independently",
    page: "A 1280x720 video slot on a sports property, $0.25 floor.",
    audience: "A connected-TV viewer.",
    demand: "Two seats bid $22.00, $9.50, $1.80 and $0.60. The two above the model's estimate are priced down; the two below are left alone.",
    models: [
      { key: "dlrm_bid_shader", label: "Bid Pricer" },
      { key: "metrics_enricher", label: "Signals Enricher", rulesBased: true },
    ],
    tags: [
      { cls: "shade", label: "BID_SHADE" },
      { cls: "metric", label: "ADD_METRICS" },
    ],
    file: "response-multi-seat.json",
    controls: ["bidFloor", "shadeFactor", "convValue"],
    // Ceiling = ctr * 12, which lands between $1.80 and $9.50 for any ctr in
    // (0.15, 0.79) -- so the top two bids shade and the bottom two do not. The
    // measured ctr on the deployed model sits near the middle of that window,
    // and the window is a 5x span, which is what makes the split hold as the
    // hour-of-day feature moves the prediction.
    defaults: { bidFloor: 0.25, convValue: 24, shadeFactor: 0.5 },
  },
];

/**
 * Tuner starting values when a scenario does not override them.
 *
 * `bidFloor` has no useful global default and every scenario overrides it, because
 * the tuner WRITES its value onto `imp[0].bidfloor` on submit. A shared 1.5 meant
 * every scenario's declared floor was silently replaced the moment it ran: the
 * $2.50 floor that turns Openfield away on home-lifestyle, the $8.50 that turns
 * away the remnant video pool, the $0.10 that gives the shader room to work. The
 * card said one thing and the request carried another. A test pins each scenario's
 * default to its own payload so that cannot come back.
 */
export const TUNER_DEFAULTS = {
  bidFloor: 1.5,
  ageRange: 1,
  numDeals: 3,
  shadeFactor: 0.65,
  convValue: 12,
  segThreshold: 0.55,
  // Yield Optimizer's bounded exploration (see shared/yield_exploration.py's
  // resolve_effective_epsilon) -- on by default so a scenario Send against the
  // still-untrained genesis model can produce a real mutation instead of always
  // 0. Once a real model has been trained at least once, the user can flip this
  // off to see that model's unperturbed prediction.
  explore: true,
};

/**
 * Starting tuner value for one control.
 *
 * A scenario may override any of them via `defaults`. The response-side scenarios
 * rely on this: each one places the shader's price ceiling relative to its own
 * floor and bid so that its outcome class holds whatever the live model predicts.
 * Sharing one set of defaults across them would collapse three distinct outcomes
 * into whichever one the shared numbers happened to produce.
 */
function initialTunerValue(scenario, key) {
  const override = scenario.defaults?.[key];
  return override === undefined ? TUNER_DEFAULTS[key] : override;
}

/**
 * The full detail and controls for ONE scenario — the one the picker has selected.
 *
 * The card is not itself a control any more. It used to be a `role="button"` div
 * whose click meant "select me" on the first press and "submit me" on the second,
 * with the tuner hidden until that first press. That made sense when the sidebar
 * listed every scenario and you had to choose among cards. With a dropdown above
 * it there is only ever one card, and it is already the selection — so the hidden
 * state was a click that existed for no reason, and click-to-submit was an
 * accidental submit waiting to happen on a panel full of sliders.
 *
 * Now: the dropdown selects, the card shows, and the only things that submit are
 * the two buttons that say so. It also stops nesting buttons and range inputs
 * inside a `role="button"`, which was never valid.
 */
export default function ScenarioCard({
  scenario,
  isActive,
  isLoading,
  disabled,
  onSend,
  onOpenTheater,
}) {
  const [bidFloor, setBidFloor] = useState(() => initialTunerValue(scenario, "bidFloor")); // nosemgrep: react-props-in-state seed value only; ScenarioPicker keys the card by scenario id so a new scenario remounts it
  const [ageRange, setAgeRange] = useState(() => initialTunerValue(scenario, "ageRange")); // nosemgrep: react-props-in-state seed value only; ScenarioPicker keys the card by scenario id so a new scenario remounts it
  const [numDeals, setNumDeals] = useState(() => initialTunerValue(scenario, "numDeals")); // nosemgrep: react-props-in-state seed value only; ScenarioPicker keys the card by scenario id so a new scenario remounts it
  const [shadeFactor, setShadeFactor] = useState(() => initialTunerValue(scenario, "shadeFactor")); // nosemgrep: react-props-in-state seed value only; ScenarioPicker keys the card by scenario id so a new scenario remounts it
  const [convValue, setConvValue] = useState(() => initialTunerValue(scenario, "convValue")); // nosemgrep: react-props-in-state seed value only; ScenarioPicker keys the card by scenario id so a new scenario remounts it
  const [segThreshold, setSegThreshold] = useState(() => initialTunerValue(scenario, "segThreshold")); // nosemgrep: react-props-in-state seed value only; ScenarioPicker keys the card by scenario id so a new scenario remounts it
  const [explore, setExplore] = useState(() => initialTunerValue(scenario, "explore")); // nosemgrep: react-props-in-state seed value only; ScenarioPicker keys the card by scenario id so a new scenario remounts it

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

  const handleSend = (e) => {
    e.stopPropagation();
    onSend(scenario, getParams());
  };

  // Same scenario, same tuner values, different destination: the mutation
  // timeline or the stepped walkthrough. The params are read from the SAME
  // getParams() so the two cannot disagree about what was submitted.
  const handleOpenTheater = (e) => {
    e.stopPropagation();
    onOpenTheater?.(scenario, getParams());
  };

  return (
    <section
      className={`scenario ${isActive ? "active" : ""} ${isLoading ? "loading" : ""} ${disabled ? "disabled" : ""}`}
      aria-label={`Scenario: ${scenario.name}`}
      data-testid="scenario-card"
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
      {scenarioUsesDealScorer(scenario) ? (
        <p className="scenario-disclosure" data-testid="scenario-deal-scorer-disclosure">
          {DEAL_SCORER_TRAINING_DISCLOSURE}
        </p>
      ) : null}
      <div className="tags">
        {scenario.tags.map((tag, i) => (
          <span key={i} className={`tag ${tag.cls}`}>{tag.label}</span>
        ))}
      </div>

      {/* The tuner is always here. It used to be revealed by clicking the card,
          which was a click with nothing behind it once the dropdown became the
          thing that selects. */}
      <div className="scenario-tuner">
          {controls.includes("bidFloor") && (
            <div className="tuner-row">
              <label>Bid Floor</label>
              {/* 0.05 steps from 0.10, not 0.5 steps from 0.50: a range input
                  CLAMPS a value outside [min,max] and SNAPS one off the step grid,
                  so the old bounds could not represent $0.10, $0.25, $1.90 or
                  $2.20 — it would have quietly moved each scenario's own floor to
                  the nearest representable one. */}
              <input type="range" min="0.1" max="15" step="0.05" value={bidFloor}
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
          <div className="tuner-actions">
            <button
              className="tuner-send"
              onClick={handleSend}
              disabled={disabled}
              data-testid="scenario-send"
            >
              ▶ Send
            </button>
            <button
              className="tuner-send tuner-send-theater"
              onClick={handleOpenTheater}
              disabled={disabled}
              data-testid="scenario-open-theater"
            >
              Step through in Auction Theater
            </button>
          </div>
      </div>
    </section>
  );
}
