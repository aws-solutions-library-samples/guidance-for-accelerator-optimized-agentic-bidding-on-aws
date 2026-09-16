// captionValidation.js — decide whether a generated caption may be displayed.
//
// Pure. No I/O, no clock, no randomness.
//
// A generated caption is the only prose in the theater that is not derived from
// data by construction, so it is the only place a number could appear that
// nothing measured. Validation therefore has two tiers, and the first matters
// more than the second:
//
//   1. Factual admissibility. Every number in the caption must trace to the
//      beat's real values, its measured latency, or the scenario context.
//   2. Register. Characters, person, promotional language, length.
//
// A failing caption is never repaired, truncated or partially accepted (BR3-15).
// Editing model output to make it pass would display prose that neither the
// model nor the data authored.
//
// (BR3-13 to BR3-21) and nfr-requirements/nfr-requirements.md (NFR3-6, NFR3-7).

const CHAR_CEILING = 260;   // NFR3-6: the caption strip holds ~3 lines of ~84 chars
const WORD_CEILING = 45;    // NFR3-5: the constraint the model actually respects
// "At most two sentences" is a PROMPT instruction, not a validation rule. The
// model honoured it in 6 of 6 measured samples, so enforcing it buys nothing —
// and enforcing it would reject our own factual fallback, which legitimately
// runs to three short sentences for an explored beat with a latency. The real
// constraint on strip fit is the character and word ceilings above.
const SENTENCE_ADVISORY = 2;

const EPSILON = 1e-9;

/** Units a numeric claim can carry. `bare` means no unit was written. */
const UNIT = Object.freeze({
  CURRENCY: "currency",
  MS: "ms",
  PERCENT: "percent",
  RATIO: "ratio",     // a 0..1 value the model may render as a percentage
  COUNT: "count",
  BARE: "bare",
});

const NUMBER_WORDS = Object.freeze({
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6,
  seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12,
});

/**
 * Promotional register and the familiar LLM tics. Deliberately narrow: a term
 * with real ad-tech meaning is not banned for sounding corporate, so "yield",
 * "premium" and "optimise" are absent from this list.
 */
const BANNED_PHRASES = Object.freeze([
  "seamless", "seamlessly", "robust", "powerful", "cutting-edge", "cutting edge",
  "revolutionary", "game-changing", "game changing", "supercharge", "unlock",
  "delve", "tapestry", "testament to", "it's worth noting", "it is worth noting",
  "in today's", "in todays", "navigate the landscape", "realm of", "elevate",
  "harness the power",
]);

/** Emoji and other pictographic ranges, plus dashes used as sentence punctuation. */
const BANNED_CHARS = /[\u2014\u2013]|[\u{1F300}-\u{1FAFF}]|[\u{2600}-\u{27BF}]|[\u{FE0F}]|[\u{1F000}-\u{1F0FF}]/u;
const PERSON = /\b(I|I'm|I've|we|we're|we've|our|ours|us|you|you're|your|yours)\b/i;

function violation(rule, detail) {
  return { rule, detail };
}

function near(a, b) {
  return Math.abs(a - b) < EPSILON || Math.abs(a - b) < Math.max(Math.abs(a), Math.abs(b)) * 1e-9;
}

/* ------------------------------------------------------- numeric extraction */

/**
 * Literal strings the beat and context legitimately contribute, which may
 * themselves contain digits: deal ids like `deal-premium-002`, taxonomy names
 * like `Demographic | Age Range | 35-39`, domains, page paths, model versions.
 *
 * These are masked out of the caption before numbers are extracted, so a real
 * identifier is never mistaken for an invented figure (BR3-16).
 */
export function sourcedLiterals(beat, context) {
  const out = [];
  const push = (s) => {
    if (typeof s === "string" && s.length > 0) out.push(s);
  };

  push(beat?.displayLabel);
  push(beat?.containerName);
  push(beat?.intent);
  push(beat?.modelVersion);
  push(beat?.path);
  for (const v of beat?.values ?? []) {
    push(v.dealId);
    push(v.type);
    push(v.role);
    for (const id of v.ids ?? []) push(id);
    for (const name of v.names ?? []) push(name);
  }
  for (const c of beat?.contributors ?? []) {
    push(c.displayLabel);
    push(c.containerName);
  }

  push(context?.publisher);
  push(context?.page);
  push(context?.requestId);
  push(context?.deviceType);
  push(context?.geo);
  for (const cat of context?.contentCategories ?? []) push(cat);
  for (const d of context?.deals ?? []) push(d.id);
  for (const s of context?.userSignals ?? []) push(s.value);

  // Longest first, so masking a long name cannot be pre-empted by a substring.
  return out.sort((a, b) => b.length - a.length);
}

/**
 * True for a literal that is safe to mask: it carries a digit AND at least one
 * non-digit, and is long enough not to appear incidentally.
 *
 * Bare numeric literals are deliberately excluded. Masking the segment id "7"
 * would strip every 7 from the caption, turning an invented "7777" into
 * whitespace and letting it through. Purely numeric identifiers are handled by
 * the allowed-number set instead.
 */
function isMaskable(literal) {
  return literal.length >= 4 && /\d/.test(literal) && /[^\d]/.test(literal);
}

function maskLiterals(text, literals) {
  let masked = text;
  for (const literal of literals) {
    if (!isMaskable(literal)) continue;
    masked = masked.split(literal).join(" ");
  }
  return masked;
}

/**
 * Numbers the caption asserts, with the unit each was written in.
 * Extracted from text that has already had sourced literals masked out.
 */
export function extractNumericClaims(text) {
  const claims = [];
  const re = /(\$\s*)?(\d+(?:\.\d+)?)\s*(%|percent|ms\b|millisecond(?:s)?\b|USD\b|dollars?\b|cents?\b)?/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    const [, dollar, digits, suffixRaw] = m;
    const value = Number.parseFloat(digits);
    if (!Number.isFinite(value)) continue;
    const suffix = (suffixRaw || "").toLowerCase();

    let unit = UNIT.BARE;
    if (dollar || suffix.startsWith("usd") || suffix.startsWith("dollar")) unit = UNIT.CURRENCY;
    else if (suffix.startsWith("cent")) unit = "cents";
    else if (suffix === "%" || suffix.startsWith("percent")) unit = UNIT.PERCENT;
    else if (suffix.startsWith("ms") || suffix.startsWith("millisecond")) unit = UNIT.MS;

    claims.push({ raw: m[0].trim(), normalised: value, unit });
  }

  for (const [word, value] of Object.entries(NUMBER_WORDS)) {
    const wordRe = new RegExp(`\\b${word}\\b`, "i");
    if (wordRe.test(text)) claims.push({ raw: word, normalised: value, unit: UNIT.COUNT });
  }

  return claims;
}

/** Numbers the beat and context legitimately support, with their units. */
export function allowedNumericClaims(beat, context) {
  const allowed = [];
  const add = (value, unit) => {
    if (typeof value === "number" && Number.isFinite(value)) allowed.push({ value, unit });
  };

  if (beat?.latencyMs != null) add(beat.latencyMs, UNIT.MS);

  const values = beat?.values ?? [];
  add(values.length, UNIT.COUNT);

  for (const v of values) {
    switch (v.kind) {
      case "metric":
        add(v.value, v.value >= 0 && v.value <= 1 ? UNIT.RATIO : UNIT.BARE);
        break;
      case "ids":
        add(v.ids.length, UNIT.COUNT);
        for (const id of v.ids) {
          const n = Number.parseFloat(id);
          if (Number.isFinite(n)) add(n, UNIT.BARE);
        }
        break;
      case "floor":
        add(v.after, UNIT.CURRENCY);
        add(v.before, UNIT.CURRENCY);
        break;
      case "margin":
        add(v.value, v.calculationType === "PERCENT" ? UNIT.RATIO : UNIT.CURRENCY);
        break;
      case "price":
        add(v.after, UNIT.CURRENCY);
        add(v.before, UNIT.CURRENCY);
        break;
      default:
        break;
    }
  }

  if (beat?.contributors) add(beat.contributors.length, UNIT.COUNT);

  // A caption may partition a collection: "two interest segments, one
  // demographic segment, and one for ages 35-39" describes four real ids with
  // three real subset counts. Every count from 1 to the largest collection on
  // the beat is therefore supportable. This still rejects 411 or 7777.
  const collectionSizes = [values.length, beat?.contributors?.length ?? 0];
  for (const v of values) if (Array.isArray(v.ids)) collectionSizes.push(v.ids.length);
  const largest = Math.max(0, ...collectionSizes);
  for (let n = 1; n <= largest; n += 1) add(n, UNIT.COUNT);

  // Numbers that appear inside identifiers and taxonomy names the beat really
  // carries. "Demographic | Age Range | 35-39" makes 35 and 39 sourced, because
  // a caption may quote the range without the full tier path.
  for (const literal of sourcedLiterals(beat, context)) {
    for (const match of literal.match(/\d+(?:\.\d+)?/g) ?? []) {
      const n = Number.parseFloat(match);
      if (Number.isFinite(n)) add(n, UNIT.BARE);
    }
  }

  add(context?.bidFloor, UNIT.CURRENCY);
  add(context?.categoryTaxonomy, UNIT.BARE);
  for (const d of context?.deals ?? []) {
    add(d.bidFloor, UNIT.CURRENCY);
    add(d.auctionType, UNIT.BARE);
  }
  for (const s of context?.userSignals ?? []) {
    const n = Number.parseFloat(s.value);
    if (Number.isFinite(n)) add(n, UNIT.BARE);
  }

  return allowed;
}

/**
 * True when a caption's claim is supported by one of the beat's.
 *
 * Units gate the comparison. A bare number may match any real value directly,
 * but a `ms` claim may not be satisfied by a currency value and vice versa —
 * which is what stops a caption asserting "47 milliseconds" from being accepted
 * because the request happened to carry a $0.47 floor.
 */
function isSupported(claim, allowed) {
  for (const { value, unit } of allowed) {
    switch (claim.unit) {
      case UNIT.BARE:
      case UNIT.COUNT:
        if (near(claim.normalised, value)) return true;
        break;
      case UNIT.CURRENCY:
        if ((unit === UNIT.CURRENCY || unit === UNIT.BARE) && near(claim.normalised, value)) return true;
        break;
      case "cents":
        if (unit === UNIT.CURRENCY && near(claim.normalised, value * 100)) return true;
        break;
      case UNIT.MS:
        if ((unit === UNIT.MS || unit === UNIT.BARE) && near(claim.normalised, value)) return true;
        break;
      case UNIT.PERCENT:
        if (unit === UNIT.RATIO && near(claim.normalised, value * 100)) return true;
        if (unit === UNIT.PERCENT && near(claim.normalised, value)) return true;
        if (unit === UNIT.BARE && near(claim.normalised, value)) return true;
        break;
      default:
        break;
    }
  }
  return false;
}

/* -------------------------------------------------------------- the two tiers */

function factualViolations(text, beat, context) {
  const out = [];
  const literals = sourcedLiterals(beat, context);
  const masked = maskLiterals(text, literals);
  const allowed = allowedNumericClaims(beat, context);

  for (const claim of extractNumericClaims(masked)) {
    if (!isSupported(claim, allowed)) {
      out.push(violation("BR3-13", `unsourced number "${claim.raw}" (${claim.unit})`));
    }
  }
  return out;
}

function countSentences(text) {
  // Split on terminal punctuation followed by whitespace or end of string, so a
  // decimal point ("1.2 USD") and a domain ("parenting-weekly.example") do not
  // read as sentence boundaries.
  return text.split(/[.!?](?=\s|$)/).filter((s) => s.trim().length > 0).length;
}

function registerViolations(text) {
  const out = [];
  const lower = text.toLowerCase();

  if (text.trim().length === 0) {
    out.push(violation("BR3-21", "empty or whitespace-only"));
    return out;
  }
  if (text.length > CHAR_CEILING) {
    out.push(violation("BR3-20", `${text.length} characters exceeds ${CHAR_CEILING}`));
  }
  const words = text.trim().split(/\s+/).length;
  if (words > WORD_CEILING) {
    out.push(violation("BR3-20", `${words} words exceeds ${WORD_CEILING}`));
  }
  const bannedChar = BANNED_CHARS.exec(text);
  if (bannedChar) {
    out.push(violation("BR3-17", `banned character ${JSON.stringify(bannedChar[0])}`));
  }
  if (/!/.test(text)) out.push(violation("BR3-17", "exclamation mark"));
  if (/\?/.test(text)) out.push(violation("BR3-17", "question mark"));

  const person = PERSON.exec(text);
  if (person) out.push(violation("BR3-18", `first or second person "${person[0]}"`));

  for (const phrase of BANNED_PHRASES) {
    if (lower.includes(phrase)) {
      out.push(violation("BR3-19", `promotional or LLM register "${phrase}"`));
    }
  }
  return out;
}

/**
 * Validate a caption against its beat.
 *
 * Returns the text unchanged on success. On failure returns every violation
 * rather than the first, so a test asserting "rejected for inventing a number"
 * is not satisfied by the caption also being too long. Each violation names the
 * business rule it broke.
 *
 * @param {string} text
 * @param {object} beat
 * @param {object} [context]
 * @returns {{ok: true, text: string} | {ok: false, violations: {rule: string, detail: string}[]}}
 */
export function validateCaption(text, beat, context) {
  if (typeof text !== "string") {
    return { ok: false, violations: [violation("BR3-21", `not a string (${typeof text})`)] };
  }
  const trimmed = text.trim();
  const violations = [
    ...factualViolations(trimmed, beat, context),
    ...registerViolations(trimmed),
  ];
  return violations.length === 0 ? { ok: true, text: trimmed } : { ok: false, violations };
}

export { CHAR_CEILING, WORD_CEILING, SENTENCE_ADVISORY, countSentences, UNIT, BANNED_PHRASES };
