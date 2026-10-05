"""Drive POST /v1/mutations the way the Prebid hook does and report where time went.

The in-process load test (``orchestrator/loadtest.py``) calls the containers
directly, so its numbers exclude the orchestrator's own auth, parse, merge and
response build. The hook's figure includes all of that plus the network hop.
This script measures from the hook's seat: an authenticated HTTP POST carrying the
ARTF envelope, N times at concurrency C, reading back ``metadata.timing`` from the
orchestrator and from every container (shared/hop_timing.py segments).

Run it from inside the cluster to get a figure comparable to the hook's::

    kubectl exec -n <ns> deploy/orchestrator -- \\
      python -m orchestrator.measure_hops \\
        --url http://orchestrator.<ns>.svc.cluster.local/v1/mutations \\
        --token "$TOKEN" --payload - --requests 500 --concurrency 1 < fixture.json

or from a workstation against the orchestrator's load balancer (adds WAN latency
to the client round trip only; the server-side segments are unaffected)::

    python -m orchestrator.measure_hops --stack-prefix dv1 --region us-east-1 \\
      --url http://<lb-hostname>/v1/mutations --payload source/prebid/fixtures/contested-auction-request.json

Token sources, in precedence order: ``--token``; ``--client-id``/``--client-secret``/
``--token-url`` (client_credentials, scope ``artf-orchestrator/mutations:write``);
``--stack-prefix`` (reads the Prebid stack's outputs and its credential secret with
the caller's AWS credentials -- the same values deploy_prebid.sh gives the hook).

Output: a markdown table on stdout (``--json FILE`` also writes the raw figures).
Nothing is estimated: a segment a server did not report is printed as ``n/a``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import hop_timing  # noqa: E402

SCOPE = "artf-orchestrator/mutations:write"
DEFAULT_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "prebid", "fixtures", "contested-auction-request.json"
)


# ------------------------------------------------------------------ inputs

def _read_payload(path: str) -> dict:
    raw = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    data = json.loads(raw)
    # Accept either a bare OpenRTB BidRequest (wrap it as the hook would) or a
    # ready ARTF envelope (has bid_request).
    if "bid_request" in data:
        return data
    return {
        "id": data.get("id", "measure-hops"),
        "lifecycle": "LIFECYCLE_PUBLISHER_BID_REQUEST",
        "tmax": 100,
        "bid_request": data,
        "originator": {"type": "TYPE_EXCHANGE", "id": "measure-hops"},
    }


def _token_from_client_credentials(token_url: str, client_id: str, client_secret: str, scope: str) -> str:
    resp = httpx.post(
        token_url,
        data={"grant_type": "client_credentials", "scope": scope},
        auth=(client_id, client_secret),
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _token_from_stack(prefix: str, region: str) -> tuple[str, str]:
    """Return (token, token_url) using the Prebid stack's outputs and secret."""
    import boto3  # the orchestrator image carries boto3; a workstation needs it installed

    stack = f"{prefix}-prebid-artf" if prefix else "prebid-artf"
    cfn = boto3.client("cloudformation", region_name=region)
    outputs = {
        o["OutputKey"]: o["OutputValue"]
        for o in cfn.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    }
    token_url = outputs["TokenEndpoint"]
    scope = outputs.get("OrchestratorScope", SCOPE)
    secret = boto3.client("secretsmanager", region_name=region).get_secret_value(
        SecretId=outputs["CredentialSecretArn"]
    )
    cred = json.loads(secret["SecretString"])
    return _token_from_client_credentials(token_url, cred["client_id"], cred["client_secret"], scope), token_url


# ------------------------------------------------------------------ driving

async def _drive(url: str, token: str, payload: dict, n: int, concurrency: int, warmup: int, timeout_s: float) -> list[dict]:
    """Return one record per request: client_ms, status, and the parsed body (or error)."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
    sem = asyncio.Semaphore(concurrency)
    records: list[dict] = []

    async def one(i: int, client: httpx.AsyncClient) -> None:
        body = dict(payload, id=f"{payload.get('id', 'measure')}-{i}")
        async with sem:
            start = time.perf_counter()
            try:
                r = await client.post(url, json=body, headers=headers, timeout=timeout_s)
                client_ms = hop_timing.ms(start)
                rec = {"i": i, "client_ms": client_ms, "status": r.status_code}
                try:
                    rec["body"] = r.json()
                except Exception:
                    rec["body"] = None
            except Exception as exc:
                rec = {"i": i, "client_ms": hop_timing.ms(start), "status": None, "error": type(exc).__name__}
            records.append(rec)

    # One client, keep-alive: the hook also holds a pooled connection.
    async with httpx.AsyncClient() as client:
        # Warm-up requests are sent but not recorded, as the plan's protocol says.
        await asyncio.gather(*(one(-1 - i, client) for i in range(warmup)))
        records.clear()
        await asyncio.gather(*(one(i, client) for i in range(n)))
    records.sort(key=lambda r: r["i"])
    return records


# ------------------------------------------------------------------ summary

def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(int(len(s) * q), len(s) - 1)], 2)


def _stats(vals: list[float]) -> dict[str, float | None]:
    return {"p50": _pct(vals, 0.5), "p95": _pct(vals, 0.95), "p99": _pct(vals, 0.99), "n": len(vals)}


def summarize(records: list[dict]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == 200 and isinstance(r.get("body"), dict)]
    summary: dict[str, Any] = {
        "requests": len(records),
        "http_200": len(ok),
        "statuses": dict(Counter(str(r.get("status") or r.get("error")) for r in records)),
        "client_ms": _stats([r["client_ms"] for r in records]),
        "orchestrator": {},
        "containers": {},
    }
    orch_segments: dict[str, list[float]] = defaultdict(list)
    orch_total_latency: list[float] = []
    # client round trip minus everything the orchestrator accounted for = the
    # network hop + both HTTP stacks on hop A.
    hop_a_transport: list[float] = []
    cont_latency: dict[str, list[float]] = defaultdict(list)
    cont_segments: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cont_transport: dict[str, list[float]] = defaultdict(list)
    cont_status: dict[str, Counter] = defaultdict(Counter)

    for r in ok:
        md = r["body"].get("metadata") or {}
        timing = md.get("timing") or {}
        for k, v in timing.items():
            if isinstance(v, (int, float)):
                orch_segments[k].append(float(v))
        if isinstance(md.get("total_latency_ms"), (int, float)):
            orch_total_latency.append(float(md["total_latency_ms"]))
        if isinstance(timing.get("total"), (int, float)):
            accounted = float(timing["total"]) + float(timing.get("auth", 0.0) or 0.0)
            hop_a_transport.append(max(0.0, r["client_ms"] - accounted))
        for inv in md.get("containers") or []:
            name = inv.get("name", "?")
            cont_status[name][inv.get("status", "?")] += 1
            lat = inv.get("latency_ms")
            if isinstance(lat, (int, float)) and inv.get("status") not in ("skipped", "disabled"):
                cont_latency[name].append(float(lat))
                ct = inv.get("timing") or {}
                for k, v in ct.items():
                    if isinstance(v, (int, float)):
                        cont_segments[name][k].append(float(v))
                if isinstance(ct.get("total"), (int, float)):
                    cont_transport[name].append(max(0.0, float(lat) - float(ct["total"])))

    summary["orchestrator"] = {
        "segments": {k: _stats(v) for k, v in orch_segments.items()},
        "total_latency_ms": _stats(orch_total_latency),
        "hop_a_transport_ms": _stats(hop_a_transport),
    }
    for name in sorted(set(cont_status) | set(cont_latency)):
        summary["containers"][name] = {
            "statuses": dict(cont_status[name]),
            "latency_ms": _stats(cont_latency[name]),
            "segments": {k: _stats(v) for k, v in cont_segments[name].items()},
            "hop_b_transport_ms": _stats(cont_transport[name]),
        }
    return summary


def _fmt(st: dict | None, key: str = "p50") -> str:
    if not st or st.get(key) is None:
        return "n/a"
    return f"{st[key]:.2f}"


def render_markdown(summary: dict[str, Any], *, label: str) -> str:
    lines = [f"### {label}", ""]
    lines.append(f"Requests: {summary['requests']} (HTTP 200: {summary['http_200']}; statuses: {summary['statuses']})")
    lines.append("")
    lines.append("| Span | p50 ms | p95 ms | p99 ms |")
    lines.append("|---|---|---|---|")
    c = summary["client_ms"]
    lines.append(f"| client round trip (hook's seat) | {_fmt(c)} | {_fmt(c, 'p95')} | {_fmt(c, 'p99')} |")
    o = summary["orchestrator"]
    lines.append(f"| hop A transport (client - orchestrator auth+total) | {_fmt(o['hop_a_transport_ms'])} | {_fmt(o['hop_a_transport_ms'], 'p95')} | {_fmt(o['hop_a_transport_ms'], 'p99')} |")
    for seg in hop_timing.ORCHESTRATOR_SEGMENTS:
        st = o["segments"].get(seg)
        lines.append(f"| orchestrator.{seg} | {_fmt(st)} | {_fmt(st, 'p95')} | {_fmt(st, 'p99')} |")
    lines.append("")
    lines.append("| Container | statuses | latency p50 | latency p95 | hop B transport p50 | parse | queue | mutate | triton | build | total |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, cs in summary["containers"].items():
        segs = cs["segments"]
        row = [
            name,
            ", ".join(f"{k}={v}" for k, v in sorted(cs["statuses"].items())),
            _fmt(cs["latency_ms"]), _fmt(cs["latency_ms"], "p95"), _fmt(cs["hop_b_transport_ms"]),
        ] + [_fmt(segs.get(s)) for s in ("parse", "queue", "mutate", "triton", "build", "total")]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="POST /v1/mutations URL")
    ap.add_argument("--payload", default=DEFAULT_FIXTURE, help="OpenRTB BidRequest or ARTF envelope JSON file, or - for stdin")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timeout-s", type=float, default=2.0)
    ap.add_argument("--label", default="", help="Table heading (e.g. 'Phase 0 baseline, c=1')")
    ap.add_argument("--json", dest="json_out", default=None, help="Write raw summary JSON here")
    g = ap.add_argument_group("token")
    g.add_argument("--token")
    g.add_argument("--token-url")
    g.add_argument("--client-id")
    g.add_argument("--client-secret")
    g.add_argument("--scope", default=SCOPE)
    g.add_argument("--stack-prefix", default=None, help="Prebid stack prefix (reads outputs + secret via AWS)")
    g.add_argument("--region", default=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")))
    args = ap.parse_args(argv)

    if args.token:
        token = args.token
    elif args.client_id and args.client_secret and args.token_url:
        token = _token_from_client_credentials(args.token_url, args.client_id, args.client_secret, args.scope)
    elif args.stack_prefix is not None:
        token, _ = _token_from_stack(args.stack_prefix, args.region)
    else:
        ap.error("provide --token, or --client-id/--client-secret/--token-url, or --stack-prefix")

    payload = _read_payload(args.payload)
    records = asyncio.run(_drive(args.url, token, payload, args.requests, args.concurrency, args.warmup, args.timeout_s))
    summary = summarize(records)
    summary["run"] = {
        "url": args.url, "requests": args.requests, "concurrency": args.concurrency,
        "warmup": args.warmup, "payload_bytes": len(json.dumps(payload).encode()),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    label = args.label or f"{args.url} c={args.concurrency} n={args.requests}"
    print(render_markdown(summary, label=label))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return 0 if summary["http_200"] == summary["requests"] else 1


if __name__ == "__main__":
    sys.exit(main())
