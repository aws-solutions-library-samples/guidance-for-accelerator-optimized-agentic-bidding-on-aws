/**
 * Compute the delta between a current metric value and a baseline value.
 *
 * @param {number|null|undefined} current - Current metric value
 * @param {number|null|undefined} baseline - Baseline metric value
 * @param {boolean} [lowerIsBetter=true] - Whether a decrease is an improvement
 * @returns {{ value: number, improved: boolean, arrow: string } | null}
 *   Returns null if either current or baseline is null/undefined.
 *   Otherwise returns an object with:
 *   - value: the signed delta (current - baseline)
 *   - improved: true if the delta represents an improvement
 *   - arrow: "↓" for negative, "↑" for positive, "—" for zero
 */
export function computeDelta(current, baseline, lowerIsBetter = true) {
  if (current == null || baseline == null) return null;
  const delta = current - baseline;
  const improved = lowerIsBetter ? delta < 0 : delta > 0;
  const arrow = delta < 0 ? "↓" : delta > 0 ? "↑" : "—";
  return { value: delta, improved, arrow };
}
