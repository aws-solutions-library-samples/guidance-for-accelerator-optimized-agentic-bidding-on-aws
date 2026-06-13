// useFlowAnimation.js — GSAP animation hook for the flow dot
// Computes waypoints from DOM node positions and drives the timeline.

import { useState, useRef, useCallback, useEffect } from "react";

/**
 * Hook that manages the flow animation state.
 * Returns controls for starting/stopping the animation and tracking which node is active.
 *
 * @param {Object} nodeRefs - Map of node id → DOM ref
 * @param {Object} containerRef - Ref to the pipeline container
 */
export function useFlowAnimation(nodeRefs, containerRef) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeNode, setActiveNode] = useState(null);
  const [waypoints, setWaypoints] = useState(null);

  const computeWaypoints = useCallback(() => {
    if (!containerRef?.current) return null;

    const containerRect = containerRef.current.getBoundingClientRect();
    const nodeOrder = ["ssp", "orchestrator", "dlrm", "widedeep", "ncf", "metrics", "aggregator", "output", "dsp"];
    const points = [];

    for (const id of nodeOrder) {
      const ref = nodeRefs[id];
      const el = ref?.current;
      if (!el) continue;

      const rect = el.getBoundingClientRect();
      points.push({
        id,
        x: rect.left + rect.width / 2 - containerRect.left,
        y: rect.top + rect.height / 2 - containerRect.top,
        duration: id === "orchestrator" ? 0.3 : 0.5,
      });
    }

    return points.length >= 2 ? points : null;
  }, [nodeRefs, containerRef]);

  const play = useCallback(() => {
    const wp = computeWaypoints();
    if (wp) {
      setWaypoints(wp);
      setActiveNode(null);
      setIsPlaying(true);
    }
  }, [computeWaypoints]);

  const stop = useCallback(() => {
    setIsPlaying(false);
    setActiveNode(null);
    setWaypoints(null);
  }, []);

  const onNodeReached = useCallback((nodeId) => {
    setActiveNode(nodeId);
    if (nodeId === "complete") {
      setIsPlaying(false);
    }
  }, []);

  return {
    isPlaying,
    activeNode,
    waypoints,
    play,
    stop,
    onNodeReached,
  };
}
