// captionFixtures.js — real Haiku 4.5 output, captured during U3 NFR Requirements.
//
// These are not invented test strings. They are what the caption model actually
// wrote when given the beat shapes below, recorded verbatim.
//
// They exist because a validator written against imagined output would have
// passed its own tests and rejected every real caption: the model writes
// "1.2 USD" rather than "$1.20", and "41 milliseconds" rather than "41ms".
//
// NEGATIVE fixtures come from round one of the measurement, where the prompt
// constrained characters rather than words. They must be rejected.

/** The parenting ACTIVATE_SEGMENTS beat of FR-32. */
export const SEGMENTS_BEAT = Object.freeze({
  index: 3,
  kind: "container",
  containerName: "audience-activator",
  displayLabel: "Audience Activator",
  intent: "ACTIVATE_SEGMENTS",
  operation: "add",
  modelVersion: "segment-rules-v2-iab-taxonomy",
  explored: false,
  latencyMs: 41,
  path: "/user/data",
  values: [{
    kind: "ids",
    role: "segments",
    ids: ["350", "354", "98", "7"],
    names: [
      "Interest | Family and Relationships | Parenting",
      "Interest | Family and Relationships | Parenting Babies and Toddlers",
      "Demographic | Household Data | Parents with Children",
      "Demographic | Age Range | 35-39",
    ],
  }],
});

export const SEGMENTS_CONTEXT = Object.freeze({
  requestId: "req-parenting-001",
  publisher: "parenting-weekly.example",
  page: "/guides/first-year-sleep-routines",
  contentCategories: ["192"],
  categoryTaxonomy: 9,
  impressionFormat: "banner",
  bidFloor: 1.2,
  deals: [],
  userSignals: [
    { label: "year of birth", value: "1989", provenance: "request" },
    { label: "data segment", value: "Parents with Children", provenance: "request" },
  ],
});

export const FLOOR_BEAT = Object.freeze({
  index: 4,
  kind: "container",
  containerName: "yield-optimizer-floor",
  displayLabel: "Yield Optimizer",
  intent: "ADJUST_DEAL_FLOOR",
  operation: "replace",
  modelVersion: "deal-yield-v1:explore",
  explored: true,
  latencyMs: 23,
  path: "/imp/1/deals/deal-premium-002",
  values: [{ kind: "floor", dealId: "deal-premium-002", before: 1.2, after: 1.68 }],
});

export const FLOOR_CONTEXT = Object.freeze({
  publisher: "parenting-weekly.example",
  impressionFormat: "banner",
  bidFloor: 1.2,
  deals: [{ id: "deal-premium-002", bidFloor: 1.2, auctionType: 1 }],
  userSignals: [],
});

export const METRICS_BEAT = Object.freeze({
  index: 1,
  kind: "container",
  containerName: "signals-enricher",
  displayLabel: "Signals Enricher",
  intent: "ADD_METRICS",
  operation: "add",
  modelVersion: "metrics-rules-v1",
  explored: false,
  latencyMs: 8,
  path: "/imp/1/metric",
  values: [
    { kind: "metric", type: "viewability", value: 0.72 },
    { kind: "metric", type: "brand_safety", value: 0.95 },
  ],
});

export const METRICS_CONTEXT = Object.freeze({
  publisher: "parenting-weekly.example",
  impressionFormat: "banner",
  bidFloor: 1.2,
  deals: [],
  userSignals: [],
});

/**
 * Round 2, word-budget prompt. 24 of 24 passed at measurement time; these are
 * the distinct forms observed.
 */
export const ACCEPTED = Object.freeze([
  {
    beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
    text: "Audience Activator added four segments to the request, including parenting interests and a demographic segment for parents with children aged 35-39, completing in 41 milliseconds.",
    note: "carries the NFR3-27 compound-segment limitation: passes, but 'children aged 35-39' is the more natural misreading",
  },
  {
    beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
    text: "Audience Activator added four segments to the request: two interest-based segments about parenting, one demographic segment for parents with children, and one for ages 35-39. The service completed this in 41 milliseconds.",
  },
  {
    beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
    text: "Audience Activator added four segments to the request, including parenting interests and a demographic segment for parents with children aged 35-39, based on signals from the exchange indicating year of birth 1989 and parent status.",
  },
  {
    beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
    text: "Audience Activator added four segments to the request: two interest-based segments about parenting, one demographic segment for parents with children, and one for ages 35-39.",
  },
  {
    beat: FLOOR_BEAT, context: FLOOR_CONTEXT,
    text: "The Yield Optimizer raised the price floor for deal-premium-002 from 1.2 USD to 1.68 USD in 23 milliseconds as part of an exploration test. This adjustment applies to the banner impression from parenting-weekly.example.",
  },
  {
    beat: FLOOR_BEAT, context: FLOOR_CONTEXT,
    text: "The Yield Optimizer raised the price floor for deal-premium-002 from 1.2 USD to 1.68 USD as part of an exploration test, completing the adjustment in 23 milliseconds.",
  },
  {
    beat: FLOOR_BEAT, context: FLOOR_CONTEXT,
    text: "The Yield Optimizer raised the price floor for deal-premium-002 from 1.2 USD to 1.68 USD as part of an exploration test, responding in 23 milliseconds.",
  },
  {
    beat: METRICS_BEAT, context: METRICS_CONTEXT,
    text: "Signals Enricher added viewability and brand safety metrics of 0.72 and 0.95 to the request for a banner impression on parenting-weekly.example with a floor price of 1.2 USD.",
  },
  {
    beat: METRICS_BEAT, context: METRICS_CONTEXT,
    text: "Signals Enricher added viewability and brand safety metrics of 0.72 and 0.95 to the request for a banner impression on parenting-weekly.example with a floor of 1.2 USD.",
  },
]);

/**
 * Round 1, character-budget prompt. Real output that must be rejected.
 * `expectRule` is the rule the rejection must cite.
 */
export const REJECTED = Object.freeze([
  {
    beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
    expectRule: "BR3-20",
    text: 'The Audience Activator service matched the ad request to four audience segments, including "Interest | Family and Relationships | Parenting" and "Demographic | Age Range | 35-39", completing its analysis in 41 milliseconds. These segments were added to help identify relevant audiences for the banner ad on parenting-weekly.example\'s sleep routines guide.',
    note: "355 characters",
  },
  {
    beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT,
    expectRule: "BR3-20",
    text: "The Audience Activator service added four audience segments to this ad request in 41 milliseconds, identifying the user as someone aged 35-39 with interests in parenting and babies and toddlers. These segments were matched against an impression on parenting-weekly.example where the publisher had already indicated the user's year of birth as 1989 and their status as a parent with children.",
    note: "391 characters, and the provenance error NFR3-8 addresses: the data said the exchange sent that signal, not the publisher",
  },
]);

/** Synthetic cases that pin specific rules. Not model output, and labelled so. */
export const SYNTHETIC_REJECTIONS = Object.freeze([
  {
    beat: SEGMENTS_BEAT, context: SEGMENTS_CONTEXT, expectRule: "BR3-13",
    text: "Audience Activator added four segments to the request in 41 milliseconds, matching segment 411 for parenting interests.",
    note: "411 is not on the beat",
  },
  {
    beat: FLOOR_BEAT, context: FLOOR_CONTEXT, expectRule: "BR3-13",
    text: "The Yield Optimizer raised the price floor for deal-premium-002 from 1.2 USD to 1.68 USD in 47 milliseconds.",
    note: "measured latency was 23ms; 47 appears nowhere",
  },
  {
    beat: METRICS_BEAT, context: METRICS_CONTEXT, expectRule: "BR3-17",
    text: "Signals Enricher added viewability 0.72 and brand safety 0.95 \u2014 both above target.",
    note: "em-dash",
  },
  {
    beat: METRICS_BEAT, context: METRICS_CONTEXT, expectRule: "BR3-18",
    text: "Signals Enricher added viewability 0.72 and brand safety 0.95 to your request.",
    note: "second person",
  },
  {
    beat: METRICS_BEAT, context: METRICS_CONTEXT, expectRule: "BR3-19",
    text: "Signals Enricher seamlessly added viewability 0.72 and brand safety 0.95 to the request.",
    note: "promotional register",
  },
  {
    beat: METRICS_BEAT, context: METRICS_CONTEXT, expectRule: "BR3-21",
    text: "   ",
    note: "whitespace only",
  },
]);
