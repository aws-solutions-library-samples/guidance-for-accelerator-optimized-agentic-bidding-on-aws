import { useRef, useEffect } from "react";
import { gsap } from "gsap";
import { ScrambleTextPlugin } from "gsap/ScrambleTextPlugin";
import { useGSAP } from "@gsap/react";
import { computeDiffRows } from "../utils/applyMutations.js";

gsap.registerPlugin(ScrambleTextPlugin);

/**
 * EnrichedBidRequest — the output node showing progressive diff of mutations
 * applied to the original bid request. "After" values scramble in using
 * GSAP ScrambleTextPlugin for a terminal-decode effect.
 */
export default function EnrichedBidRequest({ result, visible }) {
  const containerRef = useRef(null);
  const prevResultIdRef = useRef(null);

  const diffRows = result ? computeDiffRows(result) : [];

  // Trigger scramble animation when result changes
  useGSAP(() => {
    if (!visible || !result || diffRows.length === 0) return;
    if (prevResultIdRef.current === result.id) return;
    prevResultIdRef.current = result.id;

    // Animate each .diff-after element with scramble text
    const afterEls = containerRef.current?.querySelectorAll(".diff-after");
    if (!afterEls || afterEls.length === 0) return;

    afterEls.forEach((el, i) => {
      const finalText = el.getAttribute("data-final") || el.textContent;
      // Set the final text first, then scramble FROM random chars TO it
      el.textContent = finalText;
      gsap.from(el, {
        duration: 1.0 + i * 0.2,
        scrambleText: {
          text: finalText,
          chars: "0123456789.$,",
          revealDelay: 0.3,
          speed: 0.5,
        },
        ease: "none",
        delay: i * 0.15,
      });
    });
  }, { scope: containerRef, dependencies: [result?.id, visible] });

  if (!result) {
    return (
      <div className="flow-output-node" ref={containerRef}>
        <div className="flow-output-label">Enriched Bid Request</div>
        <div className="placeholder" style={{ padding: "12px", fontSize: "12px" }}>
          Run a scenario to see mutations applied
        </div>
      </div>
    );
  }

  return (
    <div
      className="flow-output-node"
      ref={containerRef}
      style={{ opacity: visible ? 1 : 0.5, transition: "opacity 0.3s" }}
    >
      <div className="flow-output-label">Enriched Bid Request</div>
      {diffRows.length === 0 ? (
        <div className="placeholder" style={{ padding: "8px", fontSize: "12px" }}>
          No mutations applied
        </div>
      ) : (
        <div className="flow-outcome">
          <div className="diff-header">
            <span className="diff-label diff-label-before">Before</span>
            <span className="diff-label diff-label-after">After</span>
          </div>
          {diffRows.map((row, i) => (
            <div key={i} className="diff-row">
              <span className="diff-path">{row.path}</span>
              <span className="diff-values">
                <span className="diff-before">{row.before}</span>
                <span className="diff-arrow">→</span>
                <span className="diff-after" data-final={row.after}>
                  {row.after}
                </span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
