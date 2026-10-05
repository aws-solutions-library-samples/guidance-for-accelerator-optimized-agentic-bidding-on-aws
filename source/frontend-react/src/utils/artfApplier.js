// artfApplier.js — apply ARTF mutations to the envelope the frontend holds.
//
// The JavaScript twin of shared/artf_applier.py (orchestrator, between stages)
// and ArtfMutationApplier.java (the Prebid hook). All three write to the same
// five locations with the same rules; artfApplier.test.js carries the vectors
// the other two assert.
//
//   /user/data/segment          ACTIVATE_SEGMENTS   append { name: "artf", segment: [...] } to user.data
//   /imp/{imp}                  ACTIVATE_DEALS      add { id } to imp.pmp.deals (no duplicate)
//                               SUPPRESS_DEALS      set deal.ext.artf.suppressed = true
//   /imp/{imp}/deals/{deal}     ADJUST_DEAL_FLOOR   deal.bidfloor (+bidfloorcur), imp.bidfloor = max(existing, new)
//                               ADJUST_DEAL_MARGIN  (PERCENT value is a fraction)
//   /imp/{imp}/metric           ADD_METRICS/ADD_CIDS append to imp.metric, value in [0, 1]
//   /seatbid/{seat}/bid/{bid}   BID_SHADE           bid.price on bid_response
//
// Input mutations are the frontend's normalised shape (utils/normalizer.js
// toMutationModel): { intent: "ACTIVATE_DEALS", path, payload }, where payload is
// the one populated ARTF payload (ids / adjust_deal / adjust_bid / add_metrics).
//
// Every mutation gets a disposition { intent, path, applied, reason, written },
// where `written` is the list of JSON pointers (from the envelope root) of the
// nodes the mutation wrote. A renderer highlights those; the mutation's path is a
// semantic address, not a location in the document.
//
// Each mutation is all-or-nothing: applied on a copy, adopted only on success.
// Inputs are never mutated. Pure.

const DEFAULT_CURRENCY = "USD";
const METRIC_MIN = 0;
const METRIC_MAX = 1;

class Reject extends Error {}

function clone(v) {
  return v === undefined ? undefined : JSON.parse(JSON.stringify(v));
}

function isNum(n) {
  return typeof n === "number" && Number.isFinite(n);
}

function idsOf(payload) {
  const raw = Array.isArray(payload) ? payload : Array.isArray(payload?.id) ? payload.id : [];
  return raw.filter((s) => typeof s === "string" && s.trim() !== "");
}

function resolvePath(path) {
  if (typeof path !== "string" || path.trim() === "") throw new Reject("mutation carries no path");
  const parts = path.split("/").filter((p) => p !== "");
  if (parts.length === 3 && parts[0] === "user" && parts[1] === "data" && parts[2] === "segment") {
    return { kind: "user_segments" };
  }
  if (parts[0] === "seatbid") {
    if (parts.length === 4 && parts[2] === "bid" && parts[1] && parts[3]) {
      return { kind: "bid_price", seat: parts[1], bidId: parts[3] };
    }
    throw new Reject(`path '${path}' addresses the auction response but does not name /seatbid/{seat}/bid/{bidId}`);
  }
  if (parts[0] === "imp") {
    if (parts.length === 2 && parts[1]) return { kind: "imp_deals", impId: parts[1] };
    if (parts.length === 3 && parts[2] === "metric" && parts[1]) return { kind: "imp_metrics", impId: parts[1] };
    if (parts.length === 4 && parts[2] === "deals" && parts[1] && parts[3]) {
      return { kind: "deal_floor", impId: parts[1], dealId: parts[3] };
    }
  }
  throw new Reject(
    `path '${path}' does not name a location this applier may write. Permitted: ` +
      "/user/data/segment, /imp/{impId}, /imp/{impId}/metric, /imp/{impId}/deals/{dealId}, /seatbid/{seat}/bid/{bidId}",
  );
}

function impIndex(req, impId) {
  const imps = Array.isArray(req?.imp) ? req.imp : [];
  return imps.findIndex((imp) => imp && imp.id === impId);
}

function dealIndex(deals, dealId) {
  return deals.findIndex((d) => d && d.id === dealId);
}

function currencyFor(req, deal, imp) {
  for (const c of [deal.bidfloorcur, imp.bidfloorcur]) {
    if (typeof c === "string" && c.trim() !== "") return c;
  }
  if (Array.isArray(req.cur) && typeof req.cur[0] === "string" && req.cur[0].trim() !== "") return req.cur[0];
  return DEFAULT_CURRENCY;
}

function applyUserSegments(req, intent, payload) {
  if (intent !== "ACTIVATE_SEGMENTS") throw new Reject(`intent ${intent} does not write user segments`);
  const ids = idsOf(payload);
  if (ids.length === 0) throw new Reject("no segment ids supplied");
  if (!req.user || typeof req.user !== "object") req.user = {};
  if (!Array.isArray(req.user.data)) req.user.data = [];
  req.user.data.push({ name: "artf", segment: ids.map((id) => ({ id })) });
  return [`/bid_request/user/data/${req.user.data.length - 1}`];
}

function applyImpDeals(req, intent, payload, impId) {
  const idx = impIndex(req, impId);
  if (idx < 0) throw new Reject(`no impression with id '${impId}'`);
  const ids = idsOf(payload);
  if (ids.length === 0) throw new Reject("no deal ids supplied");
  const imp = req.imp[idx];
  if (!imp.pmp || typeof imp.pmp !== "object") imp.pmp = {};
  if (!Array.isArray(imp.pmp.deals)) imp.pmp.deals = [];
  const deals = imp.pmp.deals;
  const written = [];
  if (intent === "ACTIVATE_DEALS") {
    const present = new Set(deals.map((d) => d?.id));
    for (const id of ids) {
      if (!present.has(id)) {
        deals.push({ id });
        present.add(id);
        written.push(`/bid_request/imp/${idx}/pmp/deals/${deals.length - 1}`);
      }
    }
    return written;
  }
  if (intent === "SUPPRESS_DEALS") {
    deals.forEach((d, di) => {
      if (d && ids.includes(d.id)) {
        if (!d.ext || typeof d.ext !== "object") d.ext = {};
        if (!d.ext.artf || typeof d.ext.artf !== "object") d.ext.artf = {};
        d.ext.artf.suppressed = true;
        written.push(`/bid_request/imp/${idx}/pmp/deals/${di}/ext`);
      }
    });
    return written;
  }
  throw new Reject(`intent ${intent} does not write deals`);
}

function resolveFloor(payload, deal, intent) {
  if (!payload || typeof payload !== "object") return null;
  if (intent === "ADJUST_DEAL_FLOOR") return isNum(payload.bidfloor) ? payload.bidfloor : null;
  const margin = payload.margin;
  if (!margin || !isNum(margin.value)) return isNum(payload.bidfloor) ? payload.bidfloor : null;
  const base = isNum(deal.bidfloor) ? deal.bidfloor : 0;
  if (margin.calculation_type === 0 || margin.calculation_type === undefined) return base + margin.value;
  if (margin.calculation_type === 1) return base + base * margin.value;
  return null;
}

function applyDealFloor(req, intent, payload, impId, dealId) {
  if (intent !== "ADJUST_DEAL_FLOOR" && intent !== "ADJUST_DEAL_MARGIN") {
    throw new Reject(`intent ${intent} does not write a deal floor`);
  }
  const idx = impIndex(req, impId);
  if (idx < 0) throw new Reject(`no impression with id '${impId}'`);
  const imp = req.imp[idx];
  const deals = imp.pmp && Array.isArray(imp.pmp.deals) ? imp.pmp.deals : null;
  if (!deals || deals.length === 0) throw new Reject(`impression '${impId}' carries no deals`);
  const di = dealIndex(deals, dealId);
  if (di < 0) throw new Reject(`impression '${impId}' has no deal '${dealId}'`);
  const deal = deals[di];
  const newFloor = resolveFloor(payload, deal, intent);
  if (newFloor === null) throw new Reject("mutation supplied no usable floor or margin value");
  if (newFloor <= 0) throw new Reject(`resulting floor ${newFloor} is not greater than zero, which Prebid ignores`);
  const currency = currencyFor(req, deal, imp);
  deal.bidfloor = newFloor;
  deal.bidfloorcur = currency;
  imp.bidfloor = isNum(imp.bidfloor) ? Math.max(imp.bidfloor, newFloor) : newFloor;
  imp.bidfloorcur = currency;
  return [`/bid_request/imp/${idx}/pmp/deals/${di}/bidfloor`, `/bid_request/imp/${idx}/bidfloor`];
}

function applyImpMetrics(req, intent, payload, impId) {
  if (intent !== "ADD_METRICS" && intent !== "ADD_CIDS") throw new Reject(`intent ${intent} does not write metrics`);
  const idx = impIndex(req, impId);
  if (idx < 0) throw new Reject(`no impression with id '${impId}'`);
  const metrics = Array.isArray(payload?.metric) ? payload.metric : Array.isArray(payload) ? payload : [];
  if (metrics.length === 0) throw new Reject("no metrics supplied");
  const incoming = [];
  for (const m of metrics) {
    if (!m || typeof m.type !== "string" || m.type.trim() === "" || !isNum(m.value)) {
      throw new Reject("metric requires a non-blank type and a value");
    }
    if (m.value < METRIC_MIN || m.value > METRIC_MAX) {
      throw new Reject(
        `metric '${m.type}' value ${m.value} is outside the [${METRIC_MIN}, ${METRIC_MAX}] range OpenRTB defines for imp.metric`,
      );
    }
    const entry = { type: m.type, value: m.value };
    if (m.vendor !== undefined && m.vendor !== null) entry.vendor = m.vendor;
    incoming.push(entry);
  }
  const imp = req.imp[idx];
  if (!Array.isArray(imp.metric)) imp.metric = [];
  const start = imp.metric.length;
  imp.metric.push(...incoming);
  return incoming.map((_, i) => `/bid_request/imp/${idx}/metric/${start + i}`);
}

function applyBidPrice(resp, intent, payload, seat, bidId) {
  if (intent !== "BID_SHADE") throw new Reject(`intent ${intent} does not write a bid price`);
  if (!resp || typeof resp !== "object") {
    throw new Reject(`path '/seatbid/${seat}/bid/${bidId}' addresses the auction response, and this envelope carries no bid_response`);
  }
  const price = isNum(payload?.price) ? payload.price : null;
  if (price === null) throw new Reject("mutation supplied no price");
  if (price <= 0) throw new Reject(`shaded price ${price} is not greater than zero`);
  const seatbids = Array.isArray(resp.seatbid) ? resp.seatbid : [];
  for (let si = 0; si < seatbids.length; si++) {
    const sb = seatbids[si];
    if (!sb || sb.seat !== seat) continue;
    const bids = Array.isArray(sb.bid) ? sb.bid : [];
    for (let bi = 0; bi < bids.length; bi++) {
      if (bids[bi] && bids[bi].id === bidId) {
        bids[bi].price = price;
        return [`/bid_response/seatbid/${si}/bid/${bi}/price`];
      }
    }
  }
  throw new Reject(`no bid '${bidId}' under seat '${seat}' in the bid response`);
}

/**
 * Apply `mutations` in order to an ARTF envelope `{ bid_request, bid_response? }`.
 *
 * @returns {{ envelope: object, dispositions: {intent, path, applied, reason, written}[] }}
 */
export function applyMutationsToEnvelope(envelope, mutations) {
  let current = clone(envelope && typeof envelope === "object" ? envelope : {});
  if (!current.bid_request || typeof current.bid_request !== "object") current.bid_request = {};
  const dispositions = [];

  for (const m of Array.isArray(mutations) ? mutations : []) {
    if (!m) {
      dispositions.push({ intent: "UNSPECIFIED", path: "", applied: false, reason: "mutation was null", written: [] });
      continue;
    }
    const intent = typeof m.intent === "string" ? m.intent : String(m.intent ?? "UNSPECIFIED");
    const path = typeof m.path === "string" ? m.path : "";
    const candidate = clone(current);
    try {
      const target = resolvePath(path);
      let written;
      switch (target.kind) {
        case "user_segments":
          written = applyUserSegments(candidate.bid_request, intent, m.payload);
          break;
        case "imp_deals":
          written = applyImpDeals(candidate.bid_request, intent, m.payload, target.impId);
          break;
        case "deal_floor":
          written = applyDealFloor(candidate.bid_request, intent, m.payload, target.impId, target.dealId);
          break;
        case "imp_metrics":
          written = applyImpMetrics(candidate.bid_request, intent, m.payload, target.impId);
          break;
        default:
          written = applyBidPrice(candidate.bid_response, intent, m.payload, target.seat, target.bidId);
      }
      current = candidate;
      dispositions.push({ intent, path, applied: true, reason: null, written });
    } catch (e) {
      const reason = e instanceof Reject ? e.message : `mutation could not be applied: ${e?.name ?? "Error"}: ${e?.message ?? ""}`;
      dispositions.push({ intent, path, applied: false, reason, written: [] });
    }
  }

  return { envelope: current, dispositions };
}

/**
 * Serialize `value` exactly as `JSON.stringify(value, null, 2)` would, and
 * return, for every node, the line range it occupies, keyed by JSON pointer.
 *
 * The highlighter needs to know which lines a written node spans. Searching the
 * text for a key name finds the first key with that name, which for `deals` or
 * `data` is the wrong one as often as not. Producing the text and the line map
 * from one walk makes them agree by construction.
 *
 * @returns {{ text: string, ranges: Map<string, {start:number, end:number}> }}
 */
export function stringifyWithPointers(value) {
  const lines = [];
  const ranges = new Map();

  function emit(node, pointer, indent, prefix, suffix) {
    const pad = " ".repeat(indent);
    const start = lines.length;
    if (Array.isArray(node)) {
      if (node.length === 0) {
        lines.push(`${pad}${prefix}[]${suffix}`);
      } else {
        lines.push(`${pad}${prefix}[`);
        node.forEach((item, i) => {
          emit(item === undefined ? null : item, `${pointer}/${i}`, indent + 2, "", i < node.length - 1 ? "," : "");
        });
        lines.push(`${pad}]${suffix}`);
      }
    } else if (node !== null && typeof node === "object") {
      const keys = Object.keys(node).filter((k) => node[k] !== undefined && typeof node[k] !== "function");
      if (keys.length === 0) {
        lines.push(`${pad}${prefix}{}${suffix}`);
      } else {
        lines.push(`${pad}${prefix}{`);
        keys.forEach((k, i) => {
          emit(node[k], `${pointer}/${k}`, indent + 2, `${JSON.stringify(k)}: `, i < keys.length - 1 ? "," : "");
        });
        lines.push(`${pad}}${suffix}`);
      }
    } else {
      lines.push(`${pad}${prefix}${JSON.stringify(node === undefined ? null : node)}${suffix}`);
    }
    ranges.set(pointer, { start, end: lines.length - 1 });
  }

  emit(value, "", 0, "", "");
  return { text: lines.join("\n"), ranges };
}
