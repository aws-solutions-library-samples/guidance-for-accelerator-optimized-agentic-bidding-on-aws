import { useRef } from "react";
import { gsap } from "gsap";
import { MotionPathPlugin } from "gsap/MotionPathPlugin";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(MotionPathPlugin);

export default function FlowDot({ waypoints, isPlaying, onNodeReached }) {
  const dotRef = useRef(null);
  const tlRef = useRef(null);

  useGSAP(() => {
    if (!isPlaying || !waypoints || waypoints.length < 2) return;

    // Reset position to first waypoint
    gsap.set(dotRef.current, {
      x: waypoints[0].x - 8,
      y: waypoints[0].y - 8,
      opacity: 1,
    });

    const tl = gsap.timeline({
      onComplete: () => onNodeReached?.("complete"),
    });

    // Build the path values array for MotionPathPlugin
    const pathValues = waypoints.map((wp) => ({ x: wp.x - 8, y: wp.y - 8 }));

    // Total duration based on individual waypoint durations
    const totalDuration = waypoints.slice(1).reduce((sum, wp) => sum + (wp.duration || 0.5), 0);

    tl.to(dotRef.current, {
      duration: totalDuration,
      motionPath: {
        path: pathValues,
        curviness: 1.5,
        autoRotate: false,
      },
      ease: "power1.inOut",
      onUpdate: function () {
        // Fire onNodeReached callbacks at approximate progress points
        const progress = this.progress();
        const segmentCount = waypoints.length - 1;
        const currentSegment = Math.floor(progress * segmentCount);
        const prevSegment = Math.floor((progress - 0.01) * segmentCount);
        if (currentSegment > prevSegment && currentSegment < waypoints.length) {
          onNodeReached?.(waypoints[currentSegment].id);
        }
      },
    });

    tlRef.current = tl;

    return () => {
      tl.kill();
      tlRef.current = null;
    };
  }, [isPlaying, waypoints]);

  return (
    <div
      ref={dotRef}
      className="flow-dot"
      style={{
        position: "absolute",
        width: 16,
        height: 16,
        borderRadius: "50%",
        background: "var(--accent)",
        boxShadow: "0 0 12px var(--accent)",
        opacity: isPlaying ? 1 : 0,
        zIndex: 50,
        pointerEvents: "none",
        top: 0,
        left: 0,
      }}
    />
  );
}
