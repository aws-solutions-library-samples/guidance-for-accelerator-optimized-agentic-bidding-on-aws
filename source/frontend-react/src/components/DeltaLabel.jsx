import { computeDelta } from "../utils/computeDelta";

/**
 * Inline label showing the baseline value and the delta between current and baseline.
 * Format: "baselineValue (arrow delta)"
 * Renders with color-coded delta styling.
 *
 * @param {{ current: number, baseline: number, unit?: string, lowerIsBetter?: boolean }} props
 * @returns {JSX.Element|null} Returns null if delta cannot be computed
 */
export default function DeltaLabel({ current, baseline, unit = "", lowerIsBetter = true }) {
  const result = computeDelta(current, baseline, lowerIsBetter);
  if (result === null) return null;

  const { value, improved, arrow } = result;

  let colorStyle;
  if (value === 0) {
    colorStyle = { color: "var(--delta-neutral, #888)" };
  } else if (improved) {
    colorStyle = { color: "var(--delta-improved, #22c55e)" };
  } else {
    colorStyle = { color: "var(--delta-regression, #ef4444)" };
  }

  // Round to 1 decimal place to avoid floating point noise
  const roundedDelta = Math.round(value * 10) / 10;
  const sign = roundedDelta > 0 ? "+" : "";
  const deltaDisplay = `${sign}${roundedDelta}${unit}`;

  // Format the baseline value
  const roundedBaseline = Math.round(baseline * 10) / 10;
  const baselineDisplay = `${roundedBaseline}${unit}`;

  return (
    <span className="delta-label">
      <span className="delta-label-baseline">{baselineDisplay}</span>
      {" "}
      <span style={colorStyle}>({arrow} {deltaDisplay})</span>
    </span>
  );
}
