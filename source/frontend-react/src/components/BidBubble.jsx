import { useEffect, useRef } from "react";

/**
 * Mutation intent → acronym + color (matching scenario pill colors)
 */
const MUTATION_STYLES = {
  BID_SHADE:          { acronym: "BS", color: "#16a34a" }, // green
  ACTIVATE_SEGMENTS:  { acronym: "AS", color: "#6366f1" }, // indigo
  ACTIVATE_DEALS:     { acronym: "AD", color: "#d97706" }, // amber
  SUPPRESS_DEALS:     { acronym: "SD", color: "#d97706" }, // amber
  ADD_METRICS:        { acronym: "AM", color: "#0891b2" }, // cyan
  ADD_CIDS:           { acronym: "AC", color: "#0891b2" }, // cyan
};

/**
 * BidBubble — A single floating bid value representing a processed bid.
 *
 * Displays the bid's dollar value. If the bid has mutations:
 * - BID_SHADE: the bid value text fades to green halfway up
 * - Other mutations: acronym labels appear below the value, fading in halfway
 *
 * Props:
 *   startX      — horizontal start position (0–100, percentage of container width)
 *   bidValue    — the dollar value to display (e.g. "$7.50")
 *   mutations   — array of intent strings applied to this bid (e.g. ["BID_SHADE", "ADD_METRICS"])
 *   delay       — stagger delay in ms before animation starts
 *   duration    — animation duration in ms (default 6500)
 *   waveAmp     — amplitude of side-to-side wave in px
 *   waveSpeed   — wave oscillation speed factor
 *   onComplete  — callback when animation finishes (for cleanup)
 */
export default function BidBubble({
  startX,
  bidValue,
  mutations = [],
  delay = 0,
  duration = 6500,
  waveAmp = 30,
  waveSpeed = 1,
  onComplete,
}) {
  const bubbleRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (onComplete) onComplete();
    }, duration + delay);
    return () => clearTimeout(timer);
  }, [duration, delay, onComplete]);

  const hasBidShade = mutations.includes("BID_SHADE");
  const otherMutations = mutations.filter((m) => m !== "BID_SHADE");

  const animStyle = {
    "--bubble-start-x": `${startX}%`,
    "--bubble-delay": `${delay}ms`,
    "--bubble-duration": `${duration}ms`,
    "--bubble-wave-amp": `${waveAmp}px`,
    "--bubble-wave-speed": `${waveSpeed}`,
  };

  return (
    <div
      ref={bubbleRef}
      className={`bid-bubble ${hasBidShade ? "bid-bubble--shade" : ""}`}
      style={animStyle}
      aria-hidden="true"
    >
      <span className="bid-bubble-value">{bidValue}</span>
      {otherMutations.length > 0 && (
        <span className="bid-bubble-mutations">
          {otherMutations.map((m) => {
            const style = MUTATION_STYLES[m];
            if (!style) return null;
            return (
              <span
                key={m}
                className="bid-bubble-mutation-tag"
                style={{ color: style.color }}
              >
                {style.acronym}
              </span>
            );
          })}
        </span>
      )}
    </div>
  );
}
