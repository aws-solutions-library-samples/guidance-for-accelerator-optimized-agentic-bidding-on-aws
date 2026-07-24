"""Concrete integrations that wire the Governance Agent to the REAL deployment
components in ``source/deployment/`` (Model Optimizer, Triton loader, canary
deployer, guardrail monitor).

These are the production adapters the deployed governance runtime uses — there are
no stubs here. The pipeline runs model optimization (TensorRT), canary deploy on
Triton, A/B evaluation, guardrail monitoring, and promote/rollback for real.

Provides:
- ``HttpxClient``: an ``HttpClient`` (async get/post) backed by httpx, with S3
  support so the Triton loader can fetch the optimized ``s3://`` engine artifact.
- ``CloudWatchStatsAdapter``: async ``get_metric_statistics`` over boto3 for the
  GuardrailMonitor (maps percentile stats to CloudWatch ExtendedStatistics).
- ``GuardrailCheckAdapter``: exposes ``check(model_name) -> list[str]`` (the shape
  the governance pipeline calls) on top of ``GuardrailMonitor.check_guardrails``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from deployment.model_deployer import HttpResponse

logger = logging.getLogger("agentcore.governance.integrations")


# ---------------------------------------------------------------------------
# Concrete HttpClient (httpx + S3)
# ---------------------------------------------------------------------------


class HttpxClient:
    """Async HttpClient implementation for ModelOptimizer + TritonModelLoader.

    - ``http(s)://`` URLs go through httpx.
    - ``s3://`` URLs are fetched via boto3 (the NIM optimizer returns an S3 URI
      for the optimized engine, and the Triton loader downloads it here).
    """

    def __init__(self, region: str, timeout: float = 300.0):
        self._region = region
        self._timeout = timeout
        self._client: Any = None
        self._s3: Any = None

    def _httpx(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _s3_client(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3", region_name=self._region)
        return self._s3

    async def get(self, url: str) -> HttpResponse:
        parsed = urlparse(url)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            try:
                obj = await asyncio.to_thread(
                    self._s3_client().get_object, Bucket=bucket, Key=key
                )
                body = await asyncio.to_thread(obj["Body"].read)
                return HttpResponse(status=200, body=body, headers={})
            except Exception as exc:  # noqa: BLE001 - map to a non-200 response
                logger.error("S3 GET failed for %s: %s", url, exc)
                return HttpResponse(status=502, body=str(exc).encode(), headers={})

        resp = await self._httpx().get(url)
        return HttpResponse(status=resp.status_code, body=resp.content, headers=dict(resp.headers))

    async def post(self, url: str, json_body: dict | None = None) -> HttpResponse:
        resp = await self._httpx().post(url, json=json_body or {})
        return HttpResponse(status=resp.status_code, body=resp.content, headers=dict(resp.headers))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Lambda-proxy HttpClient (PUBLIC runtime -> VPC-attached Lambda -> internal NLBs)
# ---------------------------------------------------------------------------


class LambdaProxyHttpClient:
    """HttpClient that tunnels http(s) calls through a VPC-attached Lambda.

    The Governance AgentCore runtime runs in PUBLIC network mode, so it cannot
    reach the cluster-internal Model Optimizer / Triton NLBs directly. Instead of
    putting the runtime in VPC mode (which requires subnets in AgentCore-supported
    AZs), this client forwards each ``http(s)://`` request to a small Lambda that
    IS attached to the cluster VPC. The Lambda performs the request against the
    internal NLB and returns the response.

    - ``http(s)://`` -> ``lambda:InvokeFunction`` (RequestResponse) on the proxy.
    - ``s3://`` -> fetched directly via boto3 (S3 is reachable from PUBLIC), so the
      Triton loader's engine download does not need the proxy.

    The proxy contract (request payload / response shape) matches the proxy Lambda
    in ``deployment/vpc_proxy_cfn.yaml``:
        request:  {"method": "GET|POST", "url": "...", "headers": {..}, "json": {..}}
        response: {"status": int, "headers": {..}, "body_b64": "<base64>"}
    """

    def __init__(self, function_arn: str, region: str, timeout: float = 600.0):
        self._function_arn = function_arn
        self._region = region
        self._timeout = timeout
        self._lambda: Any = None
        self._s3: Any = None

    def _lambda_client(self):
        if self._lambda is None:
            import boto3
            from botocore.config import Config

            # RequestResponse holds the connection until the function returns, so the
            # botocore read timeout must exceed the proxy Lambda's max duration
            # (a TensorRT optimize call can take minutes). Disable retries so a slow
            # call is not duplicated.
            self._lambda = boto3.client(
                "lambda",
                region_name=self._region,
                config=Config(
                    read_timeout=int(self._timeout) + 30,
                    connect_timeout=10,
                    retries={"max_attempts": 0},
                ),
            )
        return self._lambda

    def _s3_client(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3", region_name=self._region)
        return self._s3

    async def _s3_get(self, url: str) -> HttpResponse:
        parsed = urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        try:
            obj = await asyncio.to_thread(
                self._s3_client().get_object, Bucket=bucket, Key=key
            )
            body = await asyncio.to_thread(obj["Body"].read)
            return HttpResponse(status=200, body=body, headers={})
        except Exception as exc:  # noqa: BLE001 - map to a non-200 response
            logger.error("S3 GET failed for %s: %s", url, exc)
            return HttpResponse(status=502, body=str(exc).encode(), headers={})

    async def _invoke(self, method: str, url: str, json_body: dict | None = None) -> HttpResponse:
        import base64
        import json as _json

        payload: dict[str, Any] = {"method": method, "url": url}
        if json_body is not None:
            payload["json"] = json_body

        def _call():
            return self._lambda_client().invoke(
                FunctionName=self._function_arn,
                InvocationType="RequestResponse",
                Payload=_json.dumps(payload).encode("utf-8"),
            )

        try:
            resp = await asyncio.to_thread(_call)
        except Exception as exc:  # noqa: BLE001 - network/permission error reaching the proxy
            logger.error("VPC proxy invoke failed for %s %s: %s", method, url, exc)
            return HttpResponse(status=502, body=f"proxy invoke failed: {exc}".encode(), headers={})

        # A Lambda function error (unhandled exception in the proxy) surfaces honestly.
        if resp.get("FunctionError"):
            raw = resp["Payload"].read()
            logger.error("VPC proxy function error for %s %s: %s", method, url, raw[:512])
            return HttpResponse(status=502, body=raw, headers={})

        result = _json.loads(resp["Payload"].read())
        body_b64 = result.get("body_b64", "")
        body = base64.b64decode(body_b64) if body_b64 else b""
        return HttpResponse(
            status=int(result.get("status", 502)),
            body=body,
            headers=result.get("headers", {}) or {},
        )

    async def get(self, url: str) -> HttpResponse:
        if urlparse(url).scheme == "s3":
            return await self._s3_get(url)
        return await self._invoke("GET", url)

    async def post(self, url: str, json_body: dict | None = None) -> HttpResponse:
        return await self._invoke("POST", url, json_body=json_body)

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# CloudWatch stats adapter for GuardrailMonitor
# ---------------------------------------------------------------------------


class CloudWatchStatsAdapter:
    """Async ``get_metric_statistics`` over a sync boto3 CloudWatch client.

    The GuardrailMonitor requests percentile stats (e.g. ``p99``) and reads the
    value under that key; CloudWatch returns percentiles under
    ``ExtendedStatistics``. This adapter maps both directions so the monitor's
    real CloudWatch queries work unchanged.
    """

    def __init__(self, cloudwatch_client: Any):
        self._cw = cloudwatch_client

    async def get_metric_statistics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: list[dict[str, str]],
        start_time: float,
        end_time: float,
        period: int,
        statistics: list[str],
    ) -> dict[str, Any]:
        from datetime import datetime, timezone

        standard = [s for s in statistics if s and s[0].isupper()]  # e.g. "Average"
        extended = [s for s in statistics if s and s.startswith("p")]  # e.g. "p99"

        kwargs: dict[str, Any] = {
            "Namespace": namespace,
            "MetricName": metric_name,
            "Dimensions": dimensions,
            "StartTime": datetime.fromtimestamp(start_time, tz=timezone.utc),
            "EndTime": datetime.fromtimestamp(end_time, tz=timezone.utc),
            "Period": period,
        }
        if standard:
            kwargs["Statistics"] = standard
        if extended:
            kwargs["ExtendedStatistics"] = extended

        resp = await asyncio.to_thread(self._cw.get_metric_statistics, **kwargs)

        # Flatten ExtendedStatistics into top-level keys the monitor reads.
        datapoints = []
        for dp in resp.get("Datapoints", []):
            flat = dict(dp)
            for k, v in (dp.get("ExtendedStatistics") or {}).items():
                flat[k] = v
            datapoints.append(flat)
        return {"Datapoints": datapoints}


# ---------------------------------------------------------------------------
# Guardrail check adapter
# ---------------------------------------------------------------------------


class GuardrailCheckAdapter:
    """Adapts the real GuardrailMonitor to the pipeline's ``check`` interface.

    ``ModelPromotionGovernanceAgent`` calls ``guardrail_monitor.check(model_name)``
    expecting a list of violation strings. The real monitor exposes
    ``check_guardrails(model_name) -> GuardrailCheckResult``; this adapter
    translates a violation into the descriptive string list the pipeline uses.
    """

    def __init__(self, monitor: Any):
        self._monitor = monitor

    async def check(self, model_name: str) -> list[str]:
        result = await self._monitor.check_guardrails(model_name)
        if not result.is_violated:
            return []
        reason = result.violation_reason or "guardrail_violation"
        if reason == "latency_regression":
            return [
                f"latency_regression: canary p99 {result.canary_latency_p99:.2f}ms "
                f"vs stable {result.stable_latency_p99:.2f}ms"
            ]
        if reason == "error_rate_breach":
            return [f"error_rate_breach: canary error rate {result.canary_error_rate:.4f}"]
        return [reason]
