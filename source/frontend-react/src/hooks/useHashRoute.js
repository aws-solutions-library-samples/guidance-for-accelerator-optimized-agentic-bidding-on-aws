// useHashRoute.js — minimal hash routing.
//
// A routing library would be the wrong trade here: CloudFront serves this
// bundle from S3 with no 403/404 rewrite to index.html, so a real path like
// /theater would 404 at the origin. A hash route works as deployed today with
// no infrastructure change (FR-1).

import { useState, useEffect, useCallback } from "react";

/** Normalise `location.hash` to a leading-slash path. "" becomes "/". */
export function normaliseHash(hash) {
  if (typeof hash !== "string") return "/";
  const stripped = hash.replace(/^#/, "");
  if (stripped.length === 0) return "/";
  return stripped.startsWith("/") ? stripped : `/${stripped}`;
}

export function useHashRoute() {
  const [route, setRoute] = useState(() =>
    normaliseHash(typeof window !== "undefined" ? window.location.hash : "")
  );

  useEffect(() => {
    const onChange = () => setRoute(normaliseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = useCallback((path) => {
    const target = path === "/" ? "" : path;
    // Assigning to location.hash fires hashchange, which drives the state
    // update, so there is one source of truth for the current route.
    window.location.hash = target;
  }, []);

  return { route, navigate };
}

export const THEATER_ROUTE = "/theater";
