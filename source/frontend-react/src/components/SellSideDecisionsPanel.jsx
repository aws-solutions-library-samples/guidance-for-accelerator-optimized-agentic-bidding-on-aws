// SellSideDecisionsPanel.jsx — the left column: what the publisher decided, and
// what that decision produced.
//
// Extends the existing sell-side column rather than replacing it:
// TheaterSellSidePanel keeps its `values`/`revealed` contract intact and renders
// as before, and this component appends the auction RESULT for the deal that
// transacted (FR-20).
//
// The result block replaces a `consequences` prop that was never supplied — it
// was read from `run.sellSideConsequences`, which `useTheaterRun` has never
// returned, so the block it guarded had never rendered once. Wired to derived
// KPIs instead, which is data that genuinely exists.
//
// Scoped to ONE IMPRESSION in its own copy. No rate, aggregate or time series
// appears here, because no data source for one exists (see sellSideKpis.js).

import { TheaterSellSidePanel } from "./TheaterPanels.jsx";
import { TheaterComparisonPanel } from "./TheaterComparisonPanel.jsx";
import { SHORT_ROWS } from "../utils/outcomeComparison.js";

/**
 * @param values      existing sell-side values, passed through untouched
 * @param kpis        from deriveSellSideKpis; null before the auction settles
 * @param comparison  from compareOutcomes; null until the recap. Rendered at the
 *                    TOP of the column with the headline rows, above the yield
 *                    decisions that caused the difference and the KPI block that
 *                    details the with-ARTF result. Both are kept.
 * @param revealed    existing reveal convention
 */
export function SellSideDecisionsPanel({ values, kpis, comparison = null, revealed }) {
  return (
    <div className="th-col-sell-wrap" data-testid="sell-side-decisions-panel">
      {comparison ? (
        <TheaterComparisonPanel comparison={comparison} rows={SHORT_ROWS} compact />
      ) : null}
      <TheaterSellSidePanel values={values} revealed={revealed} />

      {kpis ? (
        <div className="th-kpis" data-testid="sell-side-kpis">
          <div className="th-col-head th-kpis-head">
            <span className="th-col-title">Deal package result</span>
            {/* The scope is part of the claim, not a footnote. Without it a
                reader can reasonably read these as period figures. */}
            <span className="th-col-sub">This impression only</span>
          </div>

          {kpis.illustrative ? (
            <p className="th-kpis-notice" data-testid="sell-side-kpis-notice">
              Fixture values captured for building this view, not a measured
              auction.
            </p>
          ) : null}

          {kpis.sold ? (
            <div className="th-rows">
              {kpis.rows.map((row) => (
                <div key={row.id} className="th-row" data-testid={`sell-side-kpi-${row.id}`}>
                  <span className="th-row-key">
                    {row.label}
                    {row.detail ? (
                      <span className="th-kpi-detail">{row.detail}</span>
                    ) : null}
                  </span>
                  <span className="th-row-val">{row.value}</span>
                </div>
              ))}
            </div>
          ) : (
            /* An unsold impression is a real outcome. Rendering nothing would
               read as a failed lookup. */
            <div className="th-row" data-testid="sell-side-kpis-unsold">
              <span className="th-row-key">Nothing transacted</span>
              <span className="th-row-val">no deal cleared</span>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
