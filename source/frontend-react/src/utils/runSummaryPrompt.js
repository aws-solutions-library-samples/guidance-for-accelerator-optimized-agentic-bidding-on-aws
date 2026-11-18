// runSummaryPrompt.js — build the model prompt for ONE WHOLE RUN.
//
// Sibling to captionPrompt.js, not a replacement. That module is per-beat by
// construction (BR3-7: it never receives the sequence) and stays that way. This
// one deliberately receives the sequence, because the thing being described is
// the run: which publisher, which audience, which advertisement was served, and
// why that campaign won against the others.
//
// The same register constraints apply and for the same measured reasons — see
// captionPrompt.js for why each exists. Two differ:
//
//   - the WORD budget is larger, because four facts cannot be stated in 45 words
//     the way one mutation's values can;
//   - the prompt carries the AUCTION OUTCOME, which a per-beat prompt never sees.
//
// Pure. No I/O, no clock, no randomness.

import { BEAT_CONTAINER } from "./theaterBeats.js";

export const MAX_TOKENS = 260;
export const TEMPERATURE = 0.3;

/** NFR: four facts, plainly. Measured captions ran 46-61 tokens for one beat. */
export const SUMMARY_WORD_CEILING = 110;
export const SUMMARY_CHAR_CEILING = 720;

export const SUMMARY_SYSTEM_PROMPT = [
  "You write one short summary of a completed advertising auction for a reader with no",
  "advertising background. Describe, in this order: the publisher and page the",
  "impression came from, the audience the request described, which advertisement was",
  "served, and why that campaign won rather than the others.",
  "",
  "Length: at most five sentences and at most 110 words. Shorter is better.",
  "",
  "Numbers: use only numbers present in the data. Never introduce, infer, round or",
  "combine numbers. If a number is not in the data, do not state it.",
  "",
  "Attribution: data under signals_from_exchange arrived with the request from the",
  "exchange. Never attribute it to the publisher or to a service. Only describe a named",
  "service as having produced what is listed under that service.",
  "",
  "Segments: describe each segment separately. Never merge two segments into one",
  "phrase. For example, a household segment and an age segment are two independent",
  "facts about the same person and must not be written as one combined description.",
  "",
  "Causation: state why the winner won only from what the data gives you, such as the",
  "reasons recorded against the campaigns that did not win. Do not invent a reason.",
  "If the data records no reason, say the response records none.",
  "",
  "Comparison: if auction_without_artf is present, end with one sentence stating how",
  "the outcome of the same auction run without the services differed, using only the",
  "winner and prices given there. If it says unavailable, say the comparison could not",
  "be made. If it is absent, say nothing about it.",
  "",
  "Register: plain and factual. No promotional adjectives. No emojis. No em-dashes or",
  "en-dashes. No exclamation marks. No questions. No first person or second person.",
  "Never mention how this text was produced.",
].join("\n");

function omitEmpty(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined) continue;
    if (Array.isArray(v) && v.length === 0) continue;
    out[k] = v;
  }
  return out;
}

const ACTION_BY_INTENT = Object.freeze({
  ACTIVATE_SEGMENTS: "added audience segments to the request",
  ACTIVATE_DEALS: "activated private deals on the request",
  SUPPRESS_DEALS: "suppressed private deals on the request",
  ADD_METRICS: "attached quality metrics to the request",
  ADD_CIDS: "attached content identifiers to the request",
  ADJUST_DEAL_FLOOR: "changed the price floor on a private deal",
  ADJUST_DEAL_MARGIN: "changed the margin on a private deal",
  BID_SHADE: "changed the bid price",
});

/** One entry per container beat, resolving segment ids to taxonomy names. */
function describeChanges(beats, resolveSegmentLabel) {
  const out = [];
  for (const beat of beats ?? []) {
    if (beat.kind !== BEAT_CONTAINER) continue;
    const detail = {};
    for (const v of beat.values ?? []) {
      switch (v.kind) {
        case "metric":
          detail.metrics = detail.metrics ?? [];
          detail.metrics.push({ type: v.type, value: v.value });
          break;
        case "ids": {
          const resolved = v.ids.map((id) => {
            const label = resolveSegmentLabel ? resolveSegmentLabel(id) : null;
            return label?.fullName ? { id, name: label.fullName } : { id };
          });
          if (v.role === "segments") detail.segments = resolved;
          else if (v.role === "deals-activated") detail.deals_activated = resolved;
          else if (v.role === "deals-suppressed") detail.deals_suppressed = resolved;
          else detail[v.role] = resolved;
          break;
        }
        case "floor":
          detail.deal_id = v.dealId ?? undefined;
          // A null baseline is "no prior floor was recorded", not zero (BR3-12).
          detail.floor_before_usd = v.before === null ? "not recorded" : v.before;
          detail.floor_after_usd = v.after;
          break;
        case "margin":
          detail.deal_id = v.dealId ?? undefined;
          detail.margin_value = v.value;
          detail.margin_kind = v.calculationType === "PERCENT" ? "fraction of price" : "USD CPM";
          break;
        case "price":
          detail.price_before_usd = v.before === null ? "not recorded" : v.before;
          detail.price_after_usd = v.after;
          break;
        default:
          break;
      }
    }
    out.push(omitEmpty({
      service: beat.displayLabel ?? beat.containerName,
      action: ACTION_BY_INTENT[beat.intent] ?? `applied ${beat.intent ?? "a change"}`,
      ...detail,
    }));
  }
  return out;
}

function describeImpression(context) {
  if (!context) return {};
  return omitEmpty({
    publisher: context.publisher,
    page: context.page,
    content_categories: context.contentCategories,
    content_taxonomy_version: context.categoryTaxonomy,
    format: context.impressionFormat === "unknown" ? null : context.impressionFormat,
    floor_usd: context.bidFloor,
    deals: (context.deals ?? []).map((d) => omitEmpty({
      id: d.id, floor_usd: d.bidFloor, auction_type: d.auctionType,
    })),
    // Named this way on purpose (NFR3-8): a vaguer name plus no explicit
    // instruction produced a caption crediting the publisher.
    signals_from_exchange: (context.userSignals ?? []).map((s) => ({
      label: s.label, value: s.value,
    })),
  });
}

/**
 * The auction, as the model may describe it.
 *
 * `illustrative` is passed through so the model is never asked to narrate a
 * fixture as a measured outcome. `losing_campaigns` carries the recorded reasons
 * and nothing else — the "why it won" sentence has to come from those or not at
 * all.
 */
function describeAuction(viewModel) {
  if (!viewModel) return {};
  const winner = viewModel.winner;
  const bids = (viewModel.bidRows ?? []).map((o) => omitEmpty({
    campaign: o.campaignName ?? o.campaignId,
    deal_id: o.dealId,
    price_usd: o.price,
    outcome: o.outcome?.outcome,
    reason: o.outcome?.reason,
  }));
  const groups = (viewModel.groups ?? []).map((g) => ({
    summary: g.label,
    breakdown: g.breakdown.map((b) => `${b.count} ${b.label}`),
  }));

  return omitEmpty({
    illustrative: viewModel.notice === "illustrative" ? true : undefined,
    currency: viewModel.currency,
    winner: winner
      ? omitEmpty({
        campaign: winner.campaignName ?? winner.campaignId,
        deal_id: winner.dealId,
        cleared_price_usd: winner.clearedPrice,
      })
      : "the response records no winner",
    campaigns_that_bid: bids,
    campaigns_that_did_not_bid: groups,
  });
}

/** The baseline auction, as the model may describe it. Null when there is none. */
function describeBaseline(baseline) {
  if (!baseline) return null;
  if (baseline.unavailable) return { unavailable: baseline.unavailable };
  return omitEmpty({
    winner: baseline.sold
      ? omitEmpty({
        campaign: baseline.campaign,
        deal_id: baseline.dealId,
        cleared_price_usd: baseline.clearedPrice,
      })
      : "the response records no winner",
  });
}

/**
 * Build the prompt for one completed run.
 *
 * @param {object[]} beats     the whole sequence
 * @param {object}   context   ScenarioContext
 * @param {object}   viewModel the offers view model, for the auction outcome
 * @param {object}  [deps]     { resolveSegmentLabel, baseline }
 */
export function buildRunSummaryPrompt(beats, context, viewModel, deps = {}) {
  const { resolveSegmentLabel, baseline = null } = deps;

  const message = JSON.stringify(omitEmpty({
    impression: describeImpression(context),
    changes_made_by_services: describeChanges(beats, resolveSegmentLabel),
    auction: describeAuction(viewModel),
    // The same request auctioned with no service consulted. Present only when the
    // Theater ran that pass; its winner and price are the only numbers the model
    // may use for the comparison sentence.
    auction_without_artf: describeBaseline(baseline),
  }), null, 2);

  return {
    system: SUMMARY_SYSTEM_PROMPT,
    message,
    maxTokens: MAX_TOKENS,
    temperature: TEMPERATURE,
  };
}
