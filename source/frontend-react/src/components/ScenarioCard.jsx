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
  // ids, sizes, media type and floors line up with the campaign catalog in
  // source/demand/artfhouse/catalog.py -- a page whose category matches nothing
  // produces one open-market bid and a folded list of ineligible campaigns.
  //
  // WHAT THESE LINES MAY AND MAY NOT CLAIM.
  //
  // An earlier version of each line named a winner and a price. Two of those went
  // stale within a day: home-lifestyle said "Cedar & Co takes it at $6.35" and the
  // Deal Scorer suppressed deal-home-premium on a later run of the identical
  // payload, handing the impression to the outside buyer at $3.25.
  //
  // The cause is that the Deal Scorer's decision is marginal BY CONSTRUCTION.
  // containers/ncf_deal_manager/app.py sets ACTIVATE_THRESHOLD = 0.499 and
  // SUPPRESS_THRESHOLD = 0.497, and the genesis NCF's scores cluster around 0.5 --
  // so a few ten-thousandths of numerical drift between runs, or between the two
  // deal-scorer replicas, flips a deal from live to suppressed. Its features are
  // user and deal hashes with no time term, so this is not an hour-of-day effect;
  // it is a threshold sitting inside the noise.
  //
  // So a demand line states only what the code and the catalog fix:
  //   - which deals the publisher offers on the request (only its open/remnant
  //     deals -- see below) and which campaign holds each
  //   - which deals the Deal Scorer can ACTIVATE from its publisher deal library
  //     (containers/ncf_deal_manager/deal_library.py, keyed by site.domain) and
  //     the floor each arrives with
  //   - each campaign's declared CPM (constants in catalog.py)
  //   - who targeting turns away (deterministic from imp.ext.artf.categories)
  //   - who the impression floor turns away (deterministic arithmetic)
  //   - the bidder simulator's fixed prices ($3.25 banner, $12.50 video)
  // and it names the Deal Scorer as the run-time variable rather than pretending
  // the outcome is fixed. The walkthrough shows the decision that was actually made.
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
    name: "Home & Lifestyle — Four-Way Deal Contest",
    page: "A small-space living room guide, declaring Content Taxonomy 3.1 category 283 Interior Decorating. 300x250, and the impression carries home and lifestyle targeting categories.",
    audience: "Interior Decorating and Home Improvement, plus First Time Homeowner — a household life stage, so it is reachable only from the data the exchange asserted, never from the page.",
    demand: "The publisher offers one deal on the request: the remnant pool at $2.75 on its open remnant deal. An outside buyer bids $3.25, and the $2.05 open-market campaign falls under the $2.50 floor — so without ARTF the outside buyer takes it. Three more deals sit in the Deal Scorer's library for this publisher: the premium home deal at a $3.00 floor, which Cedar & Co holds at $6.35; a retail deal at $2.60, Northlake at $3.10; and an auto deal at $1.50, Vantage Motorsport at $1.80. Vantage is turned away by TARGETING either way — a home page is not automotive. Whether the Deal Scorer activates the $6.35 deal onto the impression is its call at run time, and it is the whole difference between the two passes.",
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
    name: "Finance Vertical — One Endemic Buyer",
    page: "A rate-decision analysis declaring Content Taxonomy 3.1 category 410 Personal Investing. 300x250, and the impression carries finance targeting.",
    audience: "Personal Investing and Retirement Planning, a reader in the 50-54 bracket in Illinois.",
    demand: "The scenario where page context, not price, decides who competes. The publisher offers only its open remnant deal, where the remnant pool bids $2.75; an outside buyer bids $3.25 and the $2.05 open-market campaign is under the $2.20 floor — so the baseline goes to the outside buyer. The Deal Scorer's library for this publisher holds two more: a finance PMP deal at a $3.50 floor, which Harbour Financial holds at $4.20, and the premium home deal at $3.00, which Cedar & Co holds at $6.35 — the highest declared CPM in the whole catalog. Activate both and TARGETING still stops Cedar, because a finance page is not home or lifestyle; Harbour is the one activation that changes the price. The Yield Optimizer then adjusts floor and margin on whatever deals are live.",
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
    name: "Parenting Article — Full Sell-Side",
    page: "A first-year sleep guide on a parenting title declaring Content Taxonomy 3.1 category 192, 300x250.",
    audience: "Parents with Children and Parenting Babies and Toddlers. Both are reachable only from the data the exchange asserted, never inferred from what the reader was reading.",
    demand: "The publisher offers one deal: the remnant pool at $2.75 on its open remnant deal, against an outside buyer at $3.25, with the $2.05 open-market campaign under the $2.60 floor — so the baseline goes to the outside buyer. Two household deals wait in the Deal Scorer's library: the premium parenting deal at a $3.40 floor, held by Brightstart Family at $4.10, and the family network deal at $2.10, held at $3.05. Both of those campaigns also pay for the audience ARTF activates — Brightstart adds $0.40 when the parenting segments are on the request, the family network $0.25 for parents with children — so the with-ARTF pass can move the price twice: once by putting the deal on the impression, once by the segments the buyer is paying for. The deepest sell-side chain of any scenario.",
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
    defaults: { bidFloor: 2.6 },
  },
  {
    id: "yield-optimizer",
    surface: SURFACE_REQUEST,
    name: "CTV Guaranteed — PMP Yield Optimizer",
    page: "A 1280x720 connected-TV slot on a sports property. The publisher offers its two open private marketplace deals on the request; the guaranteed tier is the Deal Scorer's to add.",
    audience: "A household segment on a large-format living-room device.",
    demand: "Two open tiers on the request: Meridian mid-tier at $6.80 on a $3.25 deal floor and Meridian remnant at $6.20 on a $0.75 floor, both clearing the $6.00 impression floor, against an outside buyer at $12.50 — so without ARTF the outside buyer wins by a wide margin. The guaranteed tier is not offered: it sits in the Deal Scorer's library at a $12.00 floor, and Meridian Guaranteed holds it at $13.40. If the scorer activates it, the publisher's own demand tops the outside buyer; if not, the outside buyer keeps the slot. The Yield Optimizer then predicts a floor multiplier and a margin adjustment per live deal on Triton's FIL backend.",
    models: [
      { key: "ncf_deal_manager", label: "NCF Deal Manager" },
      { key: "deal_yield_manager_floor", label: "Yield Optimizer — Floor" },
      { key: "deal_yield_manager_margin", label: "Yield Optimizer — Margin" },
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
    name: "Mid-roll Video — Outside Bidder Wins",
    page: "A 640x480 mid-roll on a streaming property. The publisher offers only its remnant private marketplace deal on the request; the premium and standard tiers are in the Deal Scorer's library.",
    audience: "A viewer segment resolved from the request, then scored against each deal in turn by the deal scorer. All three video campaigns take any category, so here the audience colours the story rather than deciding it.",
    demand: "The scenario an SSP least wants to see. On the request: one house deal, the video remnant at $8.10 on a $0.50 deal floor, clearing the $8.00 impression floor, and an outside buyer at $12.50. The Deal Scorer can add two more from its library — the premium video deal at a $10.00 floor, held by Skyline at $11.50, and the standard deal at $5.00, Skyline Standard at $8.75. Even with both activated, the outside buyer at $12.50 is over the top of all three. The publisher's own demand is healthy and still loses, which is the case a yield team actually has to answer for — and the comparison shows how much the activations closed the gap.",
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
    // numDeals was dropped from this card: it sliced the pre-declared deal list,
    // and the fixture now offers a single remnant deal (the rest come from the
    // Deal Scorer's library at run time, which a request-side slider cannot reach).
    controls: ["bidFloor", "ageRange", "segThreshold", "explore"],
    defaults: { bidFloor: 8.0 },
  },
  {
    id: "banner-basic",
    surface: SURFACE_REQUEST,
    name: "Open Market — No Premium Demand",
    page: "An NBA article on a sports title, 300x250 on an iPhone. It still declares the legacy Content Taxonomy 1.0, so it is also the scenario that exercises the old category map.",
    audience: "Sports Enthusiast, asserted by the publisher's data provider, on a reader in Illinois.",
    demand: "The thin end of the market: one remnant deal at $2.75 and one open-market campaign at $2.05, both clearing the $1.90 floor, against an outside buyer at $3.25. No premium deal applies to this impression at all, so there is no deal for the sell side to defend — this is what the floor alone gets you.",
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
    name: "Two Impressions — SSP Enrichment Fan-out",
    page: "An automotive review page offering two slots at once: a 970x250 leaderboard and a 300x600 rail.",
    audience: "An in-market automotive segment, resolved once and applied to both impressions.",
    demand: "Two impressions, and the publisher offers no deals on either. The house seat's only deal-free campaign bids $2.05 open-market — under the leaderboard's $4.00 floor and the rail's $3.00 — so without ARTF neither slot fills from this seat. The leaderboard's demand is deal-only and lives in the Deal Scorer's library: Autoline Premium at $7.20 on a $6.00 deal floor and Autoline Standard at $4.60 on $3.50. If the scorer activates them, the leaderboard fills; the 300x600 rail has no deal in the library at all, so it returns NOTHING either way. An enrichment fan-out across two impressions where one of them was never going to fill.",
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
    page: "A 728x90 leaderboard on a markets brief, floor $0.10 — almost no floor at all, so nothing stops the price falling.",
    audience: "What the DLRM actually reads: the impression floor, the reader's age, whether the slot is video, and hashed ids for the user, domain and device. No segment or category reaches this model.",
    demand: "The DSP bid $25.00. That is far above what the model thinks the impression is worth, so the Bid Pricer prices it down to its predicted expected value and the floor never enters into it. The base case: headroom, and the model takes it.",
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
    page: "A 300x250 on a finance title with a hard $3.00 floor.",
    audience: "A 55-plus reader on desktop. Age is one of the four dense features the DLRM reads, so it moves the prediction directly.",
    demand: "The DSP bid $12.00 and the model wants far less than $3.00 for it — but the publisher's floor is $3.00, and a bid under a floor cannot transact. So the shader clamps at the floor instead of its own estimate. The case where the model is overruled by the market, and the walkthrough shows which term won.",
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
    page: "A 320x50 mobile banner on a recipe page, floor $0.50.",
    audience: "A young reader on an Android phone, on a low-floor impression — the combination the model scores most generously.",
    demand: "The DSP bid $1.20, which is already at or under what the model thinks the impression is worth. There is nothing to price down, so the Bid Pricer returns NO mutation. An empty mutation list here is the correct answer, not a failure — and it is the one outcome nothing else in this UI demonstrates.",
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
    page: "A 1280x720 skippable video slot on a sports property, floor $0.25.",
    audience: "A connected-TV viewer. The video flag is a dense feature in its own right, so a video slot scores differently from a banner with everything else held equal.",
    demand: "Two DSP seats bidding four creatives at $22.00, $9.50, $1.80 and $0.60. The shader walks every bid in every seat and prices each against the SAME estimate, and the spread straddles it: the two above it come down, the two below it are left untouched. One prediction, four bids, two different decisions — and the cheap bids are not \"missed\", they are already at or under what the model thinks the impression is worth.",
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
  const [bidFloor, setBidFloor] = useState(() => initialTunerValue(scenario, "bidFloor"));
  const [ageRange, setAgeRange] = useState(() => initialTunerValue(scenario, "ageRange"));
  const [numDeals, setNumDeals] = useState(() => initialTunerValue(scenario, "numDeals"));
  const [shadeFactor, setShadeFactor] = useState(() => initialTunerValue(scenario, "shadeFactor"));
  const [convValue, setConvValue] = useState(() => initialTunerValue(scenario, "convValue"));
  const [segThreshold, setSegThreshold] = useState(() => initialTunerValue(scenario, "segThreshold"));
  const [explore, setExplore] = useState(() => initialTunerValue(scenario, "explore"));

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
