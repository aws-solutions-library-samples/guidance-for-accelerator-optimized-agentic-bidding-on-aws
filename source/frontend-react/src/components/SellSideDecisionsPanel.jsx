// SellSideDecisionsPanel.jsx — the left column, with consequences attached.
//
// Extends the existing sell-side column rather than replacing it:
// TheaterSellSidePanel keeps its `values`/`revealed` contract intact and renders
// as before, and this component adds the auction consequence beside each decision
// (FR-25, BR-26).
//
// A decision is never shown without its consequence. A decision alone asserts that
// something mattered with no evidence attached; the consequence is the evidence.

import { TheaterSellSidePanel } from "./TheaterPanels.jsx";

const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };

/**
 * @param values       existing sell-side values, passed through untouched
 * @param consequences [{ id, decision, consequence }] — may be empty
 * @param revealed     existing reveal convention
 */
export function SellSideDecisionsPanel({ values, consequences, revealed }) {
  const rows = consequences ?? [];

  return (
    <div className="th-col-sell-wrap" data-testid="sell-side-decisions-panel">
      <TheaterSellSidePanel values={values} revealed={revealed} />

      {rows.length > 0 ? (
        <div className="th-consequences">
          <div className="th-col-head">
            <span className="th-col-sub">Auction consequence</span>
          </div>
          {rows.map((row) => (
            <div
              key={row.id}
              className="th-row"
              data-testid={`sell-side-decision-${row.id}`}
            >
              <span className="th-row-key">{row.decision}</span>
              <span
                className="th-row-val"
                data-testid={`sell-side-consequence-${row.id}`}
              >
                {/*
                  An unknown consequence is stated as unknown rather than omitted,
                  so a decision whose effect was not reported is still visible as a
                  decision that happened.
                */}
                {row.consequence ?? (
                  <span style={SUBTLE}>consequence not reported</span>
                )}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
