// TheaterPanels.jsx — the presentational pieces of the Auction Theater.
//
// Grouped in one module because each is small and they share the same value
// vocabulary. All are pure functions of their props; none reads the stepper.

import { ValueVisual, UNKNOWN } from "./theaterVisualisers.jsx";

/* ------------------------------------------------------------------ ribbon */

/**
 * Names of audience segments the request itself asserted, in request order.
 *
 * Only the `data segment` signals qualify. Year of birth and gender are also
 * request-borne, but they are raw demographic fields the request card already
 * shows, not audience assertions.
 */
function assertedSegmentNames(context) {
  return (context?.userSignals ?? [])
    .filter((s) => s?.label === "data segment" && typeof s.value === "string" && s.value)
    .map((s) => s.value);
}

/**
 * Publisher, page, content, audience and deals, entirely from real request
 * fields (FR-13).
 */
export function TheaterSceneRibbon({ context, revealed, stepLabel }) {
  const items = [
    { key: "publisher", label: "Publisher", value: context?.publisher },
    { key: "page", label: "Page", value: context?.page },
    {
      key: "content",
      label: "Content",
      value: context?.contentCategories?.length
        ? context.contentCategories.join(", ")
        : null,
    },
    // Audience and Deals complete the three things a bid request is read for:
    // the page, the audience asserted on it, and the demand eligible to compete.
    // Both come from the SUBMITTED request, so this is what the exchange sent --
    // never a segment a container went on to contribute (BR-23).
    {
      key: "audience",
      label: "Audience",
      value: assertedSegmentNames(context).join(", ") || null,
    },
    {
      key: "deals",
      label: "Deals",
      value: context?.deals?.length ? String(context.deals.length) : null,
    },
  ];

  return (
    <div className="th-ribbon">
      <div className="th-ribbon-title">ARTF Auction Theater</div>
      <div className="th-ribbon-scene">
        {items.map(({ key, label, value }) => (
          <div key={key} className={`th-scene-item${revealed ? " is-on" : ""}`}>
            <span className="th-scene-label">{label}</span>
            <span className="th-scene-value">{value ?? UNKNOWN}</span>
          </div>
        ))}
      </div>
      <div className="th-ribbon-step" data-testid="theater-progress-label">{stepLabel}</div>
    </div>
  );
}

/* ------------------------------------------------------------- request card */

/**
 * The centre column. Base fields come only from the submitted request; user
 * data from the request is shown in the base group labelled as already present,
 * never in the contributed group (BR-22, BR-23).
 */
export function TheaterRequestCard({ context, visible, landingValues, cardState, contributors, taxonomyNames }) {
  const baseFields = [
    ["request", context?.requestId],
    ["format", context?.impressionFormat],
    ["geo", context?.geo],
    ["floor", context?.bidFloor != null ? `$${context.bidFloor.toFixed(2)}` : null],
    ["taxonomy", context?.categoryTaxonomy != null ? `cattax ${context.categoryTaxonomy}` : null],
  ].filter(([, v]) => v !== null && v !== undefined);

  return (
    <div className={`th-card th-card-${cardState}`} data-testid="theater-request-card">
      <div className="th-card-head">Bid Request</div>
      <div className="th-card-body">
        <div className="th-group">
          <div className="th-group-label">As sent by the exchange</div>
          <div className="th-chips">
            {baseFields.map(([k, v]) => (
              <span key={k} className="th-chip">
                <span className="th-chip-key">{k}</span>{v}
              </span>
            ))}
          </div>
          {context?.userSignals?.length ? (
            <div className="th-chips th-chips-user">
              {context.userSignals.map((s, i) => (
                <span key={`${s.label}-${i}`} className="th-chip th-chip-request">
                  <span className="th-chip-key">{s.label}</span>{s.value}
                </span>
              ))}
            </div>
          ) : null}
          {context?.deals?.length ? (
            <div className="th-chips">
              {context.deals.map((d) => (
                <span key={d.id} className="th-chip">
                  <span className="th-chip-key">deal</span>
                  {d.id}
                  {d.bidFloor != null ? ` · $${d.bidFloor.toFixed(2)}` : ""}
                </span>
              ))}
            </div>
          ) : null}
        </div>

        <div className="th-group">
          <div className="th-group-label">Added by ARTF containers</div>
          {visible.length === 0 ? (
            <div className="th-empty">Nothing yet</div>
          ) : (
            <div className="th-values">
              {visible.map((value, i) => (
                <ValueVisual
                  key={`${value.kind}-${i}`}
                  value={value}
                  landing={landingValues.includes(value)}
                  taxonomyNames={taxonomyNames}
                />
              ))}
            </div>
          )}
        </div>

        {contributors?.length ? (
          <div className="th-contrib">
            <div className="th-group-label">Container contributions</div>
            {contributors.map((c) => (
              <div key={c.containerName} className="th-contrib-row">
                <span className="th-contrib-name">{c.displayLabel}</span>
                <span className="th-contrib-count">
                  {c.beatIndexes.length} {c.beatIndexes.length === 1 ? "mutation" : "mutations"}
                </span>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------- sell side panel */

/**
 * Real yield decisions only. A decision that was not made produces no row
 * (BR-28). Where a baseline is unknown it is shown as unknown rather than
 * omitted, so a change against an unknown baseline is still visible (BR-29).
 */
export function TheaterSellSidePanel({ values, revealed }) {
  const floors = values.filter((v) => v.kind === "floor");
  const margins = values.filter((v) => v.kind === "margin");
  const activated = values.filter((v) => v.kind === "ids" && v.role === "deals-activated");
  const suppressed = values.filter((v) => v.kind === "ids" && v.role === "deals-suppressed");
  const nothingYet = floors.length + margins.length + activated.length + suppressed.length === 0;

  return (
    <div className={`th-col th-col-sell${revealed ? " is-live" : ""}`}>
      <div className="th-col-head">
        <span className="th-col-title">Sell side</span>
        <span className="th-col-sub">Publisher yield decisions</span>
      </div>

      {nothingYet ? (
        <div className="th-empty">No yield decisions yet</div>
      ) : (
        <div className="th-rows">
          {floors.map((v, i) => (
            <div key={`floor-${i}`} className="th-row">
              <span className="th-row-key">Floor · {v.dealId ?? "unknown deal"}</span>
              <span className="th-row-val">
                {v.before != null ? `$${v.before.toFixed(2)}` : UNKNOWN}
                <span className="th-row-to">to</span>
                {`$${v.after.toFixed(2)}`}
              </span>
            </div>
          ))}
          {margins.map((v, i) => (
            <div key={`margin-${i}`} className="th-row">
              <span className="th-row-key">Margin · {v.dealId ?? "unknown deal"}</span>
              <span className="th-row-val">
                {v.calculationType === "PERCENT"
                  ? `${(v.value * 100).toFixed(1)}%`
                  : `$${v.value.toFixed(2)} CPM`}
              </span>
            </div>
          ))}
          {activated.map((v, i) => (
            <div key={`act-${i}`} className="th-row">
              <span className="th-row-key">Deals activated</span>
              <span className="th-row-val">{v.ids.join(", ")}</span>
            </div>
          ))}
          {suppressed.map((v, i) => (
            <div key={`sup-${i}`} className="th-row">
              <span className="th-row-key">Deals suppressed</span>
              <span className="th-row-val">{v.ids.join(", ")}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------- caption */

/**
 * Polite live region: a change is announced without interrupting a viewer
 * mid-sentence (BR-38).
 */
export function TheaterCaption({ text, stepLabel, recap }) {
  return (
    <div className="th-caption-wrap">
      <div className="th-caption" data-testid="theater-caption" aria-live="polite">
        <span className="th-caption-step">{stepLabel}</span>
        <span className="th-caption-text">{text}</span>
        {recap?.length ? (
          <ul className="th-recap">
            {recap.map((r) => (
              <li key={r.containerName}>
                <strong>{r.displayLabel}</strong>
                {` — ${r.beatIndexes.length} ${r.beatIndexes.length === 1 ? "mutation" : "mutations"}`}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- seam arrow */

/** Drawn only when attention crosses between columns (BR-20). */
export function TheaterSeamArrow({ movement, active }) {
  if (!active || !movement || movement === "none") return null;
  return (
    <div className={`th-seam th-seam-${movement}`} aria-hidden="true">
      <span className="th-seam-trail" />
      <span className="th-seam-head" />
    </div>
  );
}

/* ---------------------------------------------------------------- controls */

export function TheaterControls({
  index, total, isIndeterminate, isPlaying, canGoBack, canGoNext,
  onNext, onBack, onRestart, onTogglePlay, onExit,
}) {
  return (
    <div className="th-controls">
      <span className="th-controls-status">
        {isIndeterminate
          ? "Preparing"
          : `Step ${index + 1} of ${total}`}
      </span>
      <button type="button" className="th-btn th-btn-ghost" onClick={onBack}
        disabled={!canGoBack} data-testid="theater-back">Back</button>
      <button type="button" className="th-btn th-btn-primary" onClick={onTogglePlay}
        disabled={isIndeterminate} data-testid="theater-play-toggle">
        {isPlaying ? "Pause" : "Play"}
      </button>
      <button type="button" className="th-btn th-btn-ghost" onClick={onNext}
        disabled={!canGoNext} data-testid="theater-next">Next</button>
      <button type="button" className="th-btn th-btn-ghost" onClick={onRestart}
        disabled={isIndeterminate} data-testid="theater-restart">Restart</button>
      <span className="th-progress" data-testid="theater-progress" aria-hidden="true">
        {isIndeterminate
          ? <i className="is-indeterminate" />
          : Array.from({ length: total }, (_, i) => (
            <i key={i} className={i < index ? "is-done" : i === index ? "is-current" : ""} />
          ))}
      </span>
      <button type="button" className="th-btn th-btn-ghost" onClick={onExit}
        data-testid="theater-exit">Exit</button>
    </div>
  );
}
