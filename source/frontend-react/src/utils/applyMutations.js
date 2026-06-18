// applyMutations.js — apply mutations to a bid request to produce a diff view

/**
 * Given a normalized result, compute the before/after diff rows for the
 * enriched bid request output node.
 *
 * Returns an array of { path, before, after } objects representing each
 * mutation's effect on the original payload.
 */
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
      rows.push({
        path: "deal.bidfloor",
        before: m.payload.original != null ? `$${m.payload.original.toFixed(2)}` : "—",
        after: m.payload.adjusted != null ? `$${m.payload.adjusted.toFixed(2)}` : "—",
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
