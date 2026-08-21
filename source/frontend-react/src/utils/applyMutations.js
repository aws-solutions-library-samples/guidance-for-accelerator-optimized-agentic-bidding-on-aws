// applyMutations.js — apply mutations to a bid request to produce a diff view

/**
 * Given a normalized result, compute the before/after diff rows for the
 * enriched bid request output node.
 *
 * Returns an array of { path, before, after } objects representing each
 * mutation's effect on the original payload.
 */
/**
 * Look up a deal's original bidfloor from the submitted bid request by
 * parsing a mutation path of the shape "/imp/{imp_id}/deals/{deal_id}"
 * (ARTF proto convention -- see the deal-yield-model unit's functional
 * design). Returns null (a real "unknown", not a fabricated value) if the
 * path doesn't match this shape or the deal can't be found.
 */
function _findOriginalDealBidfloor(result, path) {
  if (!path) return null;
  const match = /^\/imp\/([^/]+)\/deals\/([^/]+)$/.exec(path);
  if (!match) return null;
  const [, impId, dealId] = match;
  const imps = result?.submittedPayload?.bid_request?.imp || [];
  const imp = imps.find((i) => i.id === impId);
  const deal = imp?.pmp?.deals?.find((d) => d.id === dealId);
  return deal?.bidfloor ?? null;
}

export function computeDiffRows(result) {
  if (!result || !result.stops) return [];

  const rows = [];
  const allMutations = result.stops.flatMap((stop) => stop.mutations || []);

  for (const m of allMutations) {
    const path = m.path || m.intent;
    const intent = m.intent;
    const op = m.op;

    if (intent === "BID_SHADE" && m.payload) {
      const adj = m.payload;
      // The payload is { price: <shaded_value> } from AdjustBidPayload
      const shadedPrice = adj.price ?? adj.adjusted_price ?? null;
      // Original price comes from the submitted payload's bid_response
      const originalPrice = result.submittedPayload?.bid_response?.seatbid?.[0]?.bid?.[0]?.price ?? null;
      rows.push({
        path: "bid.price",
        before: originalPrice != null ? `$${Number(originalPrice).toFixed(2)}` : "—",
        after: shadedPrice != null ? `$${Number(shadedPrice).toFixed(2)}` : "—",
        type: "shade",
      });
    } else if (intent === "ACTIVATE_SEGMENTS" && m.payload) {
      const p = m.payload;
      const ids = Array.isArray(p) ? p : Array.isArray(p?.id) ? p.id : [p];
      const display = ids.slice(0, 3).join(", ") + (ids.length > 3 ? ` +${ids.length - 3}` : "");
      rows.push({
        path: "segments",
        before: "(none)",
        after: display,
        type: "seg",
      });
    } else if ((intent === "ACTIVATE_DEALS" || intent === "SUPPRESS_DEALS") && m.payload) {
      const ids = Array.isArray(m.payload) ? m.payload : [m.payload];
      const action = intent === "ACTIVATE_DEALS" ? "activated" : "suppressed";
      rows.push({
        path: "imp.pmp.deals",
        before: "—",
        after: `${ids.length} ${action}`,
        type: "deal",
      });
    } else if (intent === "ADD_METRICS" && m.payload) {
      const p = m.payload;
      const metrics = Array.isArray(p?.metric) ? p.metric : Array.isArray(p) ? p : [];
      const display = metrics.map((mt) => `${mt.type}: ${Number(mt.value).toFixed(2)}`).join(", ");
      rows.push({
        path: "metrics",
        before: "(none)",
        after: display || "+metrics",
        type: "metric",
      });
    } else if (intent === "ADJUST_DEAL_FLOOR" && m.payload) {
      // Real AdjustDealPayload shape is { bidfloor, margin } (ARTF proto),
      // not { original, adjusted }. The "before" value comes from the
      // deal's original bidfloor in the submitted bid request, matched by
      // the mutation's path (/imp/{imp_id}/deals/{deal_id}).
      const original = _findOriginalDealBidfloor(result, path);
      const adjusted = m.payload.bidfloor;
      rows.push({
        path: path || "deal.bidfloor",
        before: original != null ? `$${Number(original).toFixed(2)}` : "—",
        after: adjusted != null ? `$${Number(adjusted).toFixed(2)}` : "—",
        type: "deal",
      });
    } else if (intent === "ADJUST_DEAL_MARGIN" && m.payload) {
      const margin = m.payload.margin;
      const calcType = margin?.calculation_type === 0 ? "CPM" : "PERCENT";
      const display = margin?.value != null
        ? (calcType === "CPM" ? `$${Number(margin.value).toFixed(2)} CPM` : `${(Number(margin.value) * 100).toFixed(1)}%`)
        : "—";
      rows.push({
        path: path || "deal.margin",
        before: "—",
        after: display,
        type: "deal",
      });
    } else {
      // Generic mutation row
      rows.push({
        path: path || "unknown",
        before: "—",
        after: `${op} via ${intent}`,
        type: "metric",
      });
    }
  }

  return rows;
}
