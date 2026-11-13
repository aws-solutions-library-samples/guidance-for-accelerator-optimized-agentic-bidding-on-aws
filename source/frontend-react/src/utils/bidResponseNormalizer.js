// bidResponseNormalizer.js — a bid response becomes offer rows.
//
// Distinct from normalizer.js, which normalises *mutations* — what the ARTF
// containers proposed. This module normalises what the auction *returned*. The
// two are not merged because they answer different questions.
//
// Rows come from THREE places, and all three are needed:
//   - seatbid[].bid[]              what offered
//   - ext.seatnonbid[].nonbid[]    what Prebid itself rejected or lost
//   - ext.artf.excluded[]          what the demand endpoint never offered for
//
// The third exists because ext.seatnonbid cannot carry it. seatnonbid is a Prebid
// Server construct recording why a SEAT produced no bid for an impression, so if
// our single seat bids at all, Prebid records no seatnonbid entry for it — and
// campaigns considered but never offered would be invisible. The demand endpoint
// therefore reports those itself, and this module reads both sources.
//
// They are genuinely different facts:
//   seatnonbid       a campaign bid, and Prebid rejected or lost it (301 below floor)
//   artf.excluded    a campaign was considered and made no offer at all
//
// Reading only seatbid would show winners and losers but never the candidates
// that never offered, which is the distinction the sell-side view exists to
// make (BR-3, BR-14).
//
// Nothing absent from the response is invented. A field the response does not
// carry is reported as unknown (BR-2).

/** Marker for a value the response does not carry. */
export const UNKNOWN_FIELD = null;

/**
 * Campaign identity for a non-offering candidate travels in the nonbid's `ext`,
 * since OpenRTB's seatnonbid carries only impression and status code.
 *
 * COORDINATION POINT: this must match what the demand endpoint (U2) emits. The
 * reader is deliberately tolerant — a missing block yields unknown identity
 * rather than a thrown error — so a contract mismatch degrades to a visibly
 * incomplete row instead of an empty column.
 */
function artfExtOf(entry) {
  return entry?.ext?.artf ?? entry?.ext?.prebid?.artf ?? null;
}

/**
 * Prebid's own winner marker: the UNPREFIXED targeting keys.
 *
 * Two request flags produce targeting, and they mean different things
 * (docs.prebid.org, openrtb2 auction endpoint, "Ad Server Targeting"):
 *   includewinners     → `hb_pb`, `hb_bidder`, `hb_size` on the top bid per imp
 *   includebidderkeys  → `hb_pb_BIDDER`, `hb_size_BIDDER` on each bidder's top bid
 *
 * Both default to true, so in a two-seat auction EVERY seat's top bid carries a
 * targeting object and "targeting is non-empty" marks every seat as the winner.
 * The documented winner signal is the absence of a bidder suffix — "values without
 * prefixes on the winning bids only".
 *
 * This module reads that marker; it never ranks bids itself (BR-18, FR-17).
 */
const WINNER_KEYS = Object.freeze(["hb_bidder", "hb_pb"]);

function isMarkedWinner(bid) {
  const targeting = bid?.ext?.prebid?.targeting;
  if (targeting == null || typeof targeting !== "object") return false;
  return WINNER_KEYS.some((key) => targeting[key] != null);
}

function offerFromBid(bid, seat) {
  const artf = artfExtOf(bid);
  return {
    key: `bid:${seat ?? "seat"}:${bid?.id ?? bid?.impid ?? "unknown"}:${bid?.dealid ?? "nodeal"}`,
    campaignId: artf?.campaignId ?? bid?.crid ?? UNKNOWN_FIELD,
    campaignName: artf?.campaignName ?? bid?.adomain?.[0] ?? UNKNOWN_FIELD,
    seat: seat ?? UNKNOWN_FIELD,
    dealId: bid?.dealid ?? UNKNOWN_FIELD,
    impId: bid?.impid ?? UNKNOWN_FIELD,
    price: typeof bid?.price === "number" ? bid.price : UNKNOWN_FIELD,
    offered: true,
    markedWinner: isMarkedWinner(bid),
    statusCode: UNKNOWN_FIELD,
    exclusionReason: UNKNOWN_FIELD,
    observed: null,
  };
}

function offerFromNonBid(entry, seat) {
  const artf = artfExtOf(entry);
  return {
    key: `nonbid:${seat ?? "seat"}:${entry?.impid ?? "unknown"}:${artf?.campaignId ?? "unknown"}`,
    campaignId: artf?.campaignId ?? UNKNOWN_FIELD,
    campaignName: artf?.campaignName ?? UNKNOWN_FIELD,
    // The seat that did not bid. Read because a seatnonbid entry frequently carries
    // NO campaign identity — OpenRTB's seatnonbid is seat-level, and only this
    // repo's own demand endpoint adds an ext.artf campaign to it. Dropping the seat
    // rendered those rows "unknown" while the response named them plainly.
    seat: seat ?? UNKNOWN_FIELD,
    dealId: artf?.dealId ?? UNKNOWN_FIELD,
    impId: entry?.impid ?? UNKNOWN_FIELD,
    price: UNKNOWN_FIELD,
    offered: false,
    markedWinner: false,
    // Prebid's seatnonbid status code. 0 no bid, 101 timeout, 301 below floor.
    statusCode: typeof entry?.statuscode === "number" ? entry.statuscode : UNKNOWN_FIELD,
    // The ARTF exclusion reason from the demand endpoint, when present.
    exclusionReason: artf?.exclusionReason ?? UNKNOWN_FIELD,
    // The price this seat returned and the floor it faced, lifted by the
    // orchestrator from Prebid's debug httpcalls. Evidence, not a verdict: Prebid
    // said NO_BID and did not say the floor was the cause.
    observed: normalizeObserved(artf?.observed),
  };
}

/**
 * The observed block, or null.
 *
 * Both numbers are required. A half-populated block would render as "returned 2.75
 * against a floor of —", which reads as a measurement that was taken and came back
 * empty rather than one that was never available.
 */
function normalizeObserved(observed) {
  const price = observed?.returnedPrice;
  const floor = observed?.impFloor;
  if (typeof price !== "number" || typeof floor !== "number") return null;
  return {
    returnedPrice: price,
    impFloor: floor,
    currency: observed?.currency ?? UNKNOWN_FIELD,
    source: observed?.source ?? UNKNOWN_FIELD,
  };
}

/**
 * A campaign the demand endpoint considered and made no offer for.
 *
 * Shape emitted by the endpoint:
 *   { campaignId, campaignName, dealId, exclusionReason }
 */
function offerFromExcluded(entry, index) {
  return {
    // The index is part of the key, not decoration. A campaign is excluded once per
    // impression, so the same campaign and deal legitimately appear more than once;
    // impId distinguishes them when the endpoint stamps it, and the index keeps the
    // key unique even when it does not. Two rows sharing a key is a React
    // reconciliation fault, not a cosmetic one.
    key: `excluded:${index}:${entry?.campaignId ?? "unknown"}:${entry?.impId ?? "noimp"}:${
      entry?.dealId ?? "nodeal"
    }`,
    campaignId: entry?.campaignId ?? UNKNOWN_FIELD,
    campaignName: entry?.campaignName ?? UNKNOWN_FIELD,
    // No seat: this is the demand endpoint's own account of a campaign it did not
    // offer, which happened before any seat was involved.
    seat: UNKNOWN_FIELD,
    dealId: entry?.dealId ?? UNKNOWN_FIELD,
    impId: entry?.impId ?? UNKNOWN_FIELD,
    price: UNKNOWN_FIELD,
    offered: false,
    markedWinner: false,
    statusCode: UNKNOWN_FIELD,
    exclusionReason: entry?.exclusionReason ?? UNKNOWN_FIELD,
    observed: null,
  };
}

/**
 * One row per bid in `seatbid`, per entry in `ext.seatnonbid`, and per entry in
 * `ext.artf.excluded`.
 *
 * Order is stable: offers first in response order, then Prebid's non-bids, then the
 * endpoint's exclusions. Stability matters because NFR-5 requires the same response
 * to render the same view, and an unstable order would break that without failing.
 */
export function normalizeOffers(bidResponse) {
  const offers = [];

  for (const seatbid of bidResponse?.seatbid ?? []) {
    for (const bid of seatbid?.bid ?? []) {
      offers.push(offerFromBid(bid, seatbid?.seat));
    }
  }

  for (const seatnonbid of bidResponse?.ext?.seatnonbid ?? []) {
    for (const entry of seatnonbid?.nonbid ?? []) {
      offers.push(offerFromNonBid(entry, seatnonbid?.seat));
    }
  }

  const excluded = bidResponse?.ext?.artf?.excluded ?? [];
  for (let i = 0; i < excluded.length; i += 1) {
    offers.push(offerFromExcluded(excluded[i], i));
  }

  return offers;
}

/**
 * The winning deal and the campaign holding it, or null.
 *
 * `null` rather than an empty object: an unsold impression is a real outcome and
 * must be distinguishable from a missing one (BR-17). An empty object renders
 * like a failed lookup.
 *
 * The cleared price is carried as supporting detail, not as the headline — the
 * decision that produced the win is the point (BR-16).
 */
export function resolveWinner(offers) {
  const winner = (offers ?? []).find((o) => o.markedWinner === true);
  if (!winner) return null;
  return {
    dealId: winner.dealId,
    campaignId: winner.campaignId,
    campaignName: winner.campaignName,
    clearedPrice: winner.price,
    offerKey: winner.key,
  };
}
