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
 * Prebid marks the bid it resolved as winning by attaching targeting to it.
 * This module reads that marker; it never ranks bids itself (BR-18, FR-17).
 */
function isMarkedWinner(bid) {
  const targeting = bid?.ext?.prebid?.targeting;
  return targeting != null && Object.keys(targeting).length > 0;
}

function offerFromBid(bid, seat) {
  const artf = artfExtOf(bid);
  return {
    key: `bid:${seat ?? "seat"}:${bid?.id ?? bid?.impid ?? "unknown"}:${bid?.dealid ?? "nodeal"}`,
    campaignId: artf?.campaignId ?? bid?.crid ?? UNKNOWN_FIELD,
    campaignName: artf?.campaignName ?? bid?.adomain?.[0] ?? UNKNOWN_FIELD,
    dealId: bid?.dealid ?? UNKNOWN_FIELD,
    impId: bid?.impid ?? UNKNOWN_FIELD,
    price: typeof bid?.price === "number" ? bid.price : UNKNOWN_FIELD,
    offered: true,
    markedWinner: isMarkedWinner(bid),
    statusCode: UNKNOWN_FIELD,
    exclusionReason: UNKNOWN_FIELD,
  };
}

function offerFromNonBid(entry, seat) {
  const artf = artfExtOf(entry);
  return {
    key: `nonbid:${seat ?? "seat"}:${entry?.impid ?? "unknown"}:${artf?.campaignId ?? "unknown"}`,
    campaignId: artf?.campaignId ?? UNKNOWN_FIELD,
    campaignName: artf?.campaignName ?? UNKNOWN_FIELD,
    dealId: artf?.dealId ?? UNKNOWN_FIELD,
    impId: entry?.impid ?? UNKNOWN_FIELD,
    price: UNKNOWN_FIELD,
    offered: false,
    markedWinner: false,
    // Prebid's seatnonbid status code. 301 below floor, 101 timeout.
    statusCode: typeof entry?.statuscode === "number" ? entry.statuscode : UNKNOWN_FIELD,
    // The ARTF exclusion reason from the demand endpoint, when present.
    exclusionReason: artf?.exclusionReason ?? UNKNOWN_FIELD,
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
    key: `excluded:${entry?.campaignId ?? `unknown-${index}`}:${entry?.dealId ?? "nodeal"}`,
    campaignId: entry?.campaignId ?? UNKNOWN_FIELD,
    campaignName: entry?.campaignName ?? UNKNOWN_FIELD,
    dealId: entry?.dealId ?? UNKNOWN_FIELD,
    impId: entry?.impId ?? UNKNOWN_FIELD,
    price: UNKNOWN_FIELD,
    offered: false,
    markedWinner: false,
    statusCode: UNKNOWN_FIELD,
    exclusionReason: entry?.exclusionReason ?? UNKNOWN_FIELD,
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
