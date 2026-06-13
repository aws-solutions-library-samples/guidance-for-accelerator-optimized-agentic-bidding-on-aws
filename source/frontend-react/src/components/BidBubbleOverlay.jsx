import { useState, useEffect, useRef, useCallback } from "react";
import BidBubble from "./BidBubble";

/**
 * Scenario templates matching the server-side load test payload generator.
 * Each template defines which mutations are applied to that bid.
 */
const SCENARIO_TEMPLATES = [
  // Full fan-out (all intents)
  ["ACTIVATE_SEGMENTS", "ACTIVATE_DEALS", "BID_SHADE", "ADD_METRICS", "ADD_CIDS"],
  // Bid shading + metrics
  ["BID_SHADE", "ADD_METRICS"],
  // Segment activation + metrics
  ["ACTIVATE_SEGMENTS", "ADD_METRICS"],
  // Identity + deals + metrics
  ["ADD_CIDS", "ACTIVATE_DEALS", "ADD_METRICS"],
];

/**
 * BidBubbleOverlay — Renders floating bid values during an active load test.
 *
 * Each bubble shows a bid dollar value and the mutations applied to it.
 * Mutations fade in halfway through the float animation.
 *
 * Props:
 *   running   — whether the load test is currently active
 *   progress  — current progress object from SSE stream { completed, total, rps, ... }
 */
export default function BidBubbleOverlay({ running, progress }) {
  const [bubbles, setBubbles] = useState([]);
  const prevCompletedRef = useRef(0);
  const bubbleIdRef = useRef(0);

  // Reset when test starts
  useEffect(() => {
    if (running) {
      prevCompletedRef.current = 0;
      setBubbles([]);
      bubbleIdRef.current = 0;
    }
  }, [running]);

  // Spawn bubbles when progress.completed increases
  useEffect(() => {
    if (!running || !progress) return;

    const prevCompleted = prevCompletedRef.current;
    const currentCompleted = progress.completed || 0;
    const delta = currentCompleted - prevCompleted;

    if (delta <= 0) return;

    prevCompletedRef.current = currentCompleted;

    // Cap visual bubbles per tick
    const maxBubblesPerTick = 15;
    const bubblesToSpawn = Math.min(delta, maxBubblesPerTick);

    // Stagger interval
    const staggerInterval = 450 / bubblesToSpawn;

    const newBubbles = [];
    for (let i = 0; i < bubblesToSpawn; i++) {
      const id = bubbleIdRef.current++;
      const bidIndex = prevCompleted + i;

      // Pseudo-random horizontal spread
      const hash1 = ((bidIndex * 2654435761) >>> 0) % 1000;
      const hash2 = ((bidIndex * 340573 + i * 982451) >>> 0) % 1000;
      const startX = 8 + ((hash1 + hash2) % 840) / 10; // 8–92%

      // Generate a realistic bid value ($0.50–$25.00)
      const cents = 50 + ((hash1 * 7 + hash2) % 2450);
      const bidValue = `$${(cents / 100).toFixed(2)}`;

      // Assign mutations based on scenario template (round-robin)
      const templateIndex = bidIndex % SCENARIO_TEMPLATES.length;
      const mutations = SCENARIO_TEMPLATES[templateIndex];

      // Stagger delay with jitter
      const baseDelay = Math.round(i * staggerInterval);
      const jitter = (hash1 % 80);
      const delay = baseDelay + jitter;

      // Vary duration (5.5–8s)
      const duration = 5500 + ((hash2) % 2500);

      // Vary wave amplitude (20–45px) and speed (0.7–1.2)
      const waveAmp = 20 + (hash2 % 26);
      const waveSpeed = 0.7 + (hash1 % 6) / 10;

      newBubbles.push({ id, startX, bidValue, mutations, delay, duration, waveAmp, waveSpeed });
    }

    setBubbles((prev) => [...prev, ...newBubbles]);
  }, [running, progress]);

  // Remove bubble after its animation completes
  const handleBubbleComplete = useCallback((id) => {
    setBubbles((prev) => prev.filter((b) => b.id !== id));
  }, []);

  if (!running && bubbles.length === 0) return null;

  return (
    <div className="bid-bubble-overlay" aria-hidden="true">
      {bubbles.map((b) => (
        <BidBubble
          key={b.id}
          startX={b.startX}
          bidValue={b.bidValue}
          mutations={b.mutations}
          delay={b.delay}
          duration={b.duration}
          waveAmp={b.waveAmp}
          waveSpeed={b.waveSpeed}
          onComplete={() => handleBubbleComplete(b.id)}
        />
      ))}
    </div>
  );
}
