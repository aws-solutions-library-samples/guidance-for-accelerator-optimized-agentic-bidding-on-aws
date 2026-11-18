// TheaterComparisonPanel.jsx — the two auctions, side by side.
//
// Renders the output of compareOutcomes(): one row per fact, a "Without ARTF"
// cell, a "With ARTF" cell, and a delta where the arithmetic is meaningful. Two
// surfaces use it with different row sets: the left column at the recap shows the
// headline set (SHORT_ROWS), and the run summary shows everything (FULL_ROWS).
//
// When the baseline did not run there is no "Without ARTF" column to fill, and
// nothing is put in its place. The panel says why, in the orchestrator's words,
// and shows the with-ARTF side alone. A comparison against an invented baseline
// would be the one thing this surface must never show.

import { selectRows } from "../utils/outcomeComparison.js";

/**
 * @param comparison  from compareOutcomes()
 * @param rows        the row ids to show, in order (SHORT_ROWS or FULL_ROWS)
 * @param title       heading
 * @param compact     tighter layout for the left column
 */
export function TheaterComparisonPanel({ comparison, rows, title = "With ARTF vs without", compact = false }) {
  if (!comparison) return null;
  const visible = selectRows(comparison, rows);
  const baselineMissing = !comparison.available;

  return (
    <div
      className={`th-compare${compact ? " th-compare-compact" : ""}`}
      data-testid="theater-comparison"
      data-available={comparison.available ? "true" : "false"}
    >
      <div className="th-col-head th-compare-head">
        <span className="th-col-title">{title}</span>
        <span className="th-col-sub">This impression only</span>
      </div>

      {comparison.illustrative ? (
        <p className="th-kpis-notice" data-testid="theater-comparison-notice">
          The with-ARTF side is the captured fixture, not a measured auction.
        </p>
      ) : null}

      {baselineMissing ? (
        <p className="th-compare-missing" data-testid="theater-comparison-missing">
          {comparison.reason ?? "baseline unavailable"}
        </p>
      ) : null}

      <div className="th-compare-grid" role="table" aria-label={title}>
        <div className="th-compare-row th-compare-labels" role="row">
          <span className="th-compare-key" role="columnheader" />
          <span className="th-compare-cell" role="columnheader">Without ARTF</span>
          <span className="th-compare-cell" role="columnheader">With ARTF</span>
          <span className="th-compare-delta" role="columnheader">Δ</span>
        </div>
        {visible.map((row) => (
          <div
            key={row.id}
            className="th-compare-row"
            role="row"
            data-testid={`theater-comparison-row-${row.id}`}
          >
            <span className="th-compare-key" role="rowheader">
              {row.label}
              {row.detail ? <span className="th-kpi-detail">{row.detail}</span> : null}
            </span>
            <span className="th-compare-cell th-compare-without" role="cell">
              {baselineMissing ? "—" : row.without}
            </span>
            <span className="th-compare-cell th-compare-with" role="cell">{row.with}</span>
            <span className="th-compare-delta" role="cell">
              {baselineMissing ? "" : (row.delta ?? "")}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
