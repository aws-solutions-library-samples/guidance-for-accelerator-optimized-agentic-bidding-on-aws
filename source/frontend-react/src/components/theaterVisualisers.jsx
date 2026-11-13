// theaterVisualisers.jsx — one visual per value kind.
//
// Each accepts a `landing` flag rather than reading the stepper, so a
// visualiser has no knowledge of where it sits in the walkthrough.
//
// Absent values render as an explicit absence, never as zero and never as a
// dash that could read as a measured value of nothing (BR-29, FR-18).

import { resolveSegmentLabels, isVendorSegment } from "../utils/segmentLabels.js";

const UNKNOWN = "not reported";

function fmtMoney(n) {
  return typeof n === "number" && Number.isFinite(n) ? `$${n.toFixed(2)}` : null;
}

function fmtPercent(n) {
  return typeof n === "number" && Number.isFinite(n) ? `${Math.round(n * 100)}%` : null;
}

/** Semicircular arc, animated by stroke-dashoffset. For a 0..1 metric. */
export function ArcGauge({ label, value, landing }) {
  const pct = typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : null;
  const r = 13;
  const arc = Math.PI * r;
  const offset = pct === null ? arc : arc * (1 - pct);
  const text = pct === null ? UNKNOWN : fmtPercent(pct);

  return (
    <div className={`thv thv-arc${landing ? " thv-landing" : ""}`}>
      <svg className="thv-svg" viewBox="0 0 34 24" aria-hidden="true" focusable="false">
        <path d="M4 20 A13 13 0 0 1 30 20" fill="none" stroke="var(--thv-track)"
          strokeWidth="4" strokeLinecap="round" />
        <path d="M4 20 A13 13 0 0 1 30 20" fill="none" stroke="var(--thv-accent)"
          strokeWidth="4" strokeLinecap="round"
          strokeDasharray={arc.toFixed(2)} strokeDashoffset={offset.toFixed(2)} />
      </svg>
      <span className="thv-meta">
        <span className="thv-label">{label}</span>
        <span className="thv-value">{text}</span>
      </span>
    </div>
  );
}

/** Shield for a brand-safety reading. The number is shown, not a pass/fail verdict. */
export function ShieldBadge({ label, value, landing }) {
  const text = fmtPercent(value) ?? UNKNOWN;
  return (
    <div className={`thv thv-shield${landing ? " thv-landing" : ""}`}>
      <svg className="thv-svg" viewBox="0 0 24 24" fill="none" stroke="var(--thv-safe)"
        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        aria-hidden="true" focusable="false">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="m9 12 2 2 4-4" />
      </svg>
      <span className="thv-meta">
        <span className="thv-label">{label}</span>
        <span className="thv-value">{text}</span>
      </span>
    </div>
  );
}

/**
 * Horizontal bar for a margin. Renders as CPM or percent according to the real
 * calculation type from the mutation, never assuming percent (BR-9).
 */
export function BarGauge({ label, value, calculationType, landing }) {
  const isPercent = calculationType === "PERCENT";
  const text = typeof value === "number" && Number.isFinite(value)
    ? (isPercent ? `${(value * 100).toFixed(1)}%` : `${fmtMoney(value)} CPM`)
    : UNKNOWN;
  // Percent margins are scaled against 30% and CPM against $5, both chosen only
  // to give the bar a readable range. The number beside it is the real value.
  const fraction = typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, isPercent ? value / 0.3 : value / 5))
    : 0;

  return (
    <div className={`thv thv-bar${landing ? " thv-landing" : ""}`}>
      <span className="thv-meta thv-meta-stacked">
        <span className="thv-label">{label}</span>
        <span className="thv-track-h" aria-hidden="true">
          <i style={{ width: `${(fraction * 100).toFixed(1)}%` }} />
        </span>
        <span className="thv-value">{text}</span>
      </span>
    </div>
  );
}

/**
 * Token row for an identifier set: deals or segments. `highlightId` marks the
 * deal a floor applies to; `highlightPrice` is that floor.
 */
export function DealTokenRow({ label, ids, role, highlightId, highlightPrice, taxonomyNames, landing }) {
  const isSegments = role === "segments";
  const labels = isSegments ? resolveSegmentLabels(ids, taxonomyNames) : null;

  return (
    <div className={`thv thv-tokens${landing ? " thv-landing" : ""}`}>
      <span className="thv-label">{label}</span>
      <span className="thv-token-list">
        {isSegments
          ? labels.map(({ id, name, fullName }) => (
            <span
              key={id}
              className={`thv-token thv-token-seg${isVendorSegment(id) ? " thv-token-vendor" : ""}`}
              title={fullName || undefined}
            >
              {/* The taxonomy's own name where known, the bare identifier where
                  not. Never an invented label (BR-26). */}
              {name ? <span className="thv-token-name">{name}</span> : null}
              <span className="thv-token-id">{id}</span>
            </span>
          ))
          : (ids ?? []).map((id) => {
            const isHighlight = id === highlightId;
            return (
              <span key={id} className={`thv-token${isHighlight ? " thv-token-floor" : ""}`}>
                <span className="thv-token-id">{id}</span>
                {isHighlight && fmtMoney(highlightPrice) ? (
                  <span className="thv-token-badge">{fmtMoney(highlightPrice)}</span>
                ) : null}
              </span>
            );
          })}
      </span>
    </div>
  );
}

/**
 * Bid bar with an optional floor marker. `before` may be null, which is a real
 * unknown baseline and renders as such rather than as zero (BR-10).
 */
export function BidBar({ label, before, after, floor, landing }) {
  const SCALE = 4;
  const pct = (n) => typeof n === "number" && Number.isFinite(n)
    ? Math.max(0, Math.min(100, (n / SCALE) * 100))
    : null;
  const afterPct = pct(after);
  const floorPct = pct(floor);
  const crossed = typeof after === "number" && typeof floor === "number" && after >= floor;

  return (
    <div className={`thv thv-bid${landing ? " thv-landing" : ""}`}>
      <span className="thv-label">{label}</span>
      <span className={`thv-bidbar${crossed ? " thv-crossed" : ""}`} aria-hidden="true">
        <i style={{ width: `${afterPct ?? 0}%` }} />
        {floorPct !== null ? (
          <span className="thv-floorline" style={{ left: `${floorPct}%` }} />
        ) : null}
      </span>
      <span className="thv-value">
        {before !== null && before !== undefined ? (
          <>
            <span className="thv-before">{fmtMoney(before) ?? UNKNOWN}</span>
            <span className="thv-arrow" aria-hidden="true">to</span>
          </>
        ) : (
          <span className="thv-before thv-unknown">{UNKNOWN}</span>
        )}
        <span className="thv-after">{fmtMoney(after) ?? UNKNOWN}</span>
      </span>
    </div>
  );
}

/** Dispatch a typed value to its visualiser. */
export function ValueVisual({ value, landing, taxonomyNames }) {
  if (!value) return null;
  switch (value.kind) {
    case "metric":
      return value.type === "brand_safety"
        ? <ShieldBadge label="brand safety" value={value.value} landing={landing} />
        : <ArcGauge label={value.type.replace(/_/g, " ")} value={value.value} landing={landing} />;
    case "ids":
      return (
        <DealTokenRow
          label={value.role === "segments" ? "audience segments"
            : value.role === "deals-activated" ? "deals activated" : "deals suppressed"}
          ids={value.ids}
          role={value.role}
          taxonomyNames={taxonomyNames}
          landing={landing}
        />
      );
    case "floor":
      return (
        <BidBar label={`deal floor · ${value.dealId ?? "unknown deal"}`}
          before={value.before} after={value.after} floor={value.after} landing={landing} />
      );
    case "margin":
      return (
        <BarGauge label={`publisher margin · ${value.dealId ?? "unknown deal"}`}
          value={value.value} calculationType={value.calculationType} landing={landing} />
      );
    case "price":
      return (
        <BidBar label="bid price" before={value.before} after={value.after}
          floor={null} landing={landing} />
      );
    default:
      return null;
  }
}

export { UNKNOWN };
