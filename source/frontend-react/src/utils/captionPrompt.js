// captionPrompt.js — build the model prompt for one beat.
//
// Pure. No I/O, no clock, no randomness.
//
// BR3-7 is enforced by construction rather than by inspection: this module
// receives a single beat, never the sequence, so there is no path by which a
// later beat's values could reach the prompt.
//
// The system prompt is static and carries the constraints that measurement
// proved necessary. Each one is here because the model got it wrong without it,
// not because it seemed prudent:
//
//   - a 45-WORD budget (NFR3-5). Instructed in characters, the model honoured
//     the limit in 1 of 6 samples. Instructed in words, 24 of 24.
//   - an explicit ATTRIBUTION rule (NFR3-8). Without it the model wrote "the
//     publisher had already indicated the user's year of birth as 1989" for a
//     field the payload had labelled as sent by the exchange.
//   - a NO-COMPOUND-SEGMENTS rule (BR3-20a, NFR3-27). Without it, 3 of 8 samples
//     merged segment 98 (Parents with Children) with segment 7 (Age Range 35-39)
//     into "parents with children aged 35-39", whose more natural reading is
//     false. Validation cannot detect this, so the prompt is the only control.
//
// for the output these constraints were derived from.

import { BEAT_ORIGIN, BEAT_RECAP } from "./theaterBeats.js";

export const MAX_TOKENS = 120;      // NFR3-10: roughly double the measured 46-61
export const TEMPERATURE = 0.3;     // NFR3-9: measured to be largely stable

export const SYSTEM_PROMPT = [
  "You write one caption for one step of a guided walkthrough of an advertising bid",
  "request moving through a chain of decision services. A reader with no advertising",
  "background must be able to follow it.",
  "",
  "Length: at most two sentences and at most 45 words. Shorter is better. Do not list",
  "every item; describe the picture.",
  "",
  "Numbers: use only numbers present in the data. Never introduce, infer, round or",
  "combine numbers. If a number is not in the data, do not state it.",
  "",
  "Attribution: data under signals_from_exchange arrived with the request from the",
  "exchange. Never attribute it to the publisher or to a service. Only describe the",
  "named service as having produced what is under step.",
  "",
  "Segments: describe each segment separately. Never merge two segments into one",
  "phrase. For example, a household segment and an age segment are two independent",
  "facts about the same person and must not be written as one combined description.",
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

/**
 * Describe the beat's typed values in terms a model can narrate, resolving
 * segment identifiers to their taxonomy names.
 *
 * BR3-10: handing the model bare numeric identifiers and asking for plain
 * language about who a user is invites it to guess, and a guess here is a
 * fabricated claim about a person.
 */
function describeValues(values, resolveSegmentLabel) {
  const out = {};
  for (const v of values ?? []) {
    switch (v.kind) {
      case "metric": {
        out.metrics = out.metrics ?? [];
        out.metrics.push({ type: v.type, value: v.value });
        break;
      }
      case "ids": {
        const resolved = v.ids.map((id) => {
          const label = resolveSegmentLabel ? resolveSegmentLabel(id) : null;
          return label?.fullName ? { id, name: label.fullName } : { id };
        });
        if (v.role === "segments") out.segments = resolved;
        else if (v.role === "deals-activated") out.deals_activated = resolved;
        else if (v.role === "deals-suppressed") out.deals_suppressed = resolved;
        else out[v.role] = resolved;
        break;
      }
      case "floor": {
        out.deal_id = v.dealId ?? undefined;
        // BR3-12: a null baseline is "no prior floor was recorded", not zero.
        out.floor_before_usd = v.before === null ? "not recorded" : v.before;
        out.floor_after_usd = v.after;
        break;
      }
      case "margin": {
        out.deal_id = v.dealId ?? undefined;
        out.margin_value = v.value;
        out.margin_kind = v.calculationType === "PERCENT" ? "fraction of price" : "USD CPM";
        break;
      }
      case "price": {
        out.price_before_usd = v.before === null ? "not recorded" : v.before;
        out.price_after_usd = v.after;
        break;
      }
      default:
        break;
    }
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
    // Named this way on purpose. NFR3-8: the earlier name plus no explicit
    // instruction produced a caption crediting the publisher.
    signals_from_exchange: (context.userSignals ?? []).map((s) => ({
      label: s.label, value: s.value,
    })),
  });
}

/**
 * Build the prompt for one beat.
 *
 * @param {object} beat    a single Beat, never the sequence (BR3-7)
 * @param {object} context ScenarioContext
 * @param {object} [deps]  { resolveSegmentLabel }
 * @returns {{system: string, message: string, maxTokens: number, temperature: number}}
 */
export function buildCaptionPrompt(beat, context, deps = {}) {
  const { resolveSegmentLabel } = deps;

  let step;
  if (!beat) {
    step = { stage: "unknown" };
  } else if (beat.kind === BEAT_ORIGIN) {
    step = {
      stage: "the bid request arrives from the exchange",
      nothing_has_been_changed_yet: true,
    };
  } else if (beat.kind === BEAT_RECAP) {
    step = omitEmpty({
      stage: "summary of what changed",
      services_that_changed_the_request: (beat.contributors ?? []).map((c) => c.displayLabel ?? c.containerName),
    });
  } else {
    step = omitEmpty({
      service: beat.displayLabel ?? beat.containerName,
      action: ACTION_BY_INTENT[beat.intent] ?? `applied ${beat.intent ?? "a change"}`,
      ...describeValues(beat.values, resolveSegmentLabel),
      responded_in_ms: beat.latencyMs,
      from_exploration_arm: beat.explored === true ? true : undefined,
    });
  }

  const message = JSON.stringify({ step, impression: describeImpression(context) }, null, 2);

  return {
    system: SYSTEM_PROMPT,
    message,
    maxTokens: MAX_TOKENS,
    temperature: TEMPERATURE,
  };
}
