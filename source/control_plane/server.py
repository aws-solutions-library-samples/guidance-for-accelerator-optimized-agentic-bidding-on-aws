"""MCP tool registration for the campaign/deals control plane.

THIS FILE IS THE ONLY PLACE THAT IMPORTS THE MCP SDK, and that is the point. Every
operation it exposes lives in operations.py, every record rule in records.py and every
entitlement decision in scope.py, all tested without the SDK. The SDK is imported
inside ``create_server`` so this module can be imported, and its pure helpers tested,
where the SDK is not installed.

TOKENS ARE VERIFIED HERE TOO, NOT ONLY BY AGENTCORE. AgentCore's inbound JWT authorizer
verifies the token before a tool is invoked. This second check is cheap on a
control-plane call, reuses a verifier tested against real signatures, and means a
misconfigured runtime does not silently open a write path to the deals table. It needs
``Authorization`` to reach the container, which on AgentCore means the header must be
on the runtime's request-header allowlist; the deploy sets that. Without it every call
is refused, which is the safe direction to fail.

SCOPES PER TOOL. Campaign writes need ``<prefix>/campaigns:write``; campaign reads
accept ``campaigns:read`` or ``campaigns:write``. Deals likewise. A token for one record
type does not open the other.

RUNS STATELESS. Every operation is one request and one response, so
``stateless_http=True``. AgentCore still supplies an Mcp-Session-Id for continuity.

Environment (set by deployment/deploy_control_plane.sh on the runtime):
  ARTF_CONTROL_TABLE               the deals table name            (required)
  ARTF_CONTROL_TOKEN_ISSUER        the Cognito pool's issuer URL   (required)
  ARTF_CONTROL_DEAL_ORIGIN         Deal.origin for created deals   (required)
  ARTF_CONTROL_SCOPE_PREFIX        resource server id, default artf-control
  ARTF_CONTROL_CLIENT_ENTITLEMENTS client-id=principals;... (see scope.py)
  ARTF_CONTROL_PRICE_CEILING       max campaign CPM incl. uplift, default 50
  ARTF_CONTROL_LOG_LEVEL           default INFO
  AWS_REGION                       the table's and the pool's region
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Mapping, Optional
from urllib.request import urlopen

from . import auth as auth_module
from .operations import (
    ControlPlaneOperations,
    DynamoStore,
    InvalidRequest,
    NotFound,
)
from .scope import Entitlement, NotEntitled, entitlement_from_claims, parse_client_entitlements

logger = logging.getLogger("artf.control_plane")

#: AgentCore expects the MCP streamable-HTTP endpoint here.
BIND_HOST = "0.0.0.0"  # noqa: S104 - AgentCore requires this bind address
BIND_PORT = 8000

DEFAULT_SCOPE_PREFIX = "artf-control"


class Unauthorized(PermissionError):
    """The caller's token could not be verified."""


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required and is not set")
    return value


def scope_names(prefix: str) -> dict[str, str]:
    """The four scopes as Cognito issues them: ``<resource server id>/<scope name>``."""
    return {
        "campaigns:read": f"{prefix}/campaigns:read",
        "campaigns:write": f"{prefix}/campaigns:write",
        "deals:read": f"{prefix}/deals:read",
        "deals:write": f"{prefix}/deals:write",
    }


def scopes_for(prefix: str) -> dict[str, tuple[str, ...]]:
    """Tool name to the scopes that satisfy it (any one suffices)."""
    s = scope_names(prefix)
    campaign_read = (s["campaigns:read"], s["campaigns:write"])
    campaign_write = (s["campaigns:write"],)
    deal_read = (s["deals:read"], s["deals:write"])
    deal_write = (s["deals:write"],)
    return {
        "create_campaign": campaign_write,
        "update_campaign": campaign_write,
        "activate_campaign": campaign_write,
        "pause_campaign": campaign_write,
        "get_campaign": campaign_read,
        "list_campaigns": campaign_read,
        "create_deal": deal_write,
        "update_deal": deal_write,
        "get_deal": deal_read,
        "list_deals": deal_read,
    }


def build_operations() -> tuple[ControlPlaneOperations, Callable[[], Mapping[str, Any]], str]:
    """Wire the operations to DynamoDB and the token verifier to the pool's JWKS.

    Reads the environment once. Every missing REQUIRED variable is fatal at startup,
    because a runtime that starts without them would accept the deploy's health check
    and then refuse or misroute every call.
    """
    import boto3

    table_name = _required("ARTF_CONTROL_TABLE")
    issuer = _required("ARTF_CONTROL_TOKEN_ISSUER").rstrip("/")
    origin = _required("ARTF_CONTROL_DEAL_ORIGIN")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    ceiling = float(os.environ.get("ARTF_CONTROL_PRICE_CEILING") or 50.0)

    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    operations = ControlPlaneOperations(DynamoStore(table), deal_origin=origin, price_ceiling=ceiling)

    pool_id = issuer.rsplit("/", 1)[-1]
    jwks_url = auth_module.cognito_jwks_url(region, pool_id)

    def fetch_jwks() -> Mapping[str, Any]:
        with urlopen(jwks_url, timeout=5) as response:  # noqa: S310 - fixed AWS host
            return json.loads(response.read())

    return operations, auth_module.JwksCache(fetch_jwks).get, issuer


def entitlement_for(
    authorization: Optional[str],
    *,
    jwks_provider: Callable[[], Mapping[str, Any]],
    issuer: str,
    required_scopes: tuple[str, ...] = (),
    client_entitlements: Optional[Mapping[str, Any]] = None,
) -> Entitlement:
    """Verify the caller's token and resolve what it may act for.

    Raises Unauthorized rather than returning an empty entitlement, so a verification
    failure cannot be mistaken for a caller who is simply entitled to nothing.
    """
    result = auth_module.verify(
        auth_module.bearer_token(authorization),
        jwks=jwks_provider(),
        issuer=issuer,
        required_scopes=required_scopes,
        now=time.time(),
    )
    if not result.ok:
        logger.warning(json.dumps({
            "event": "control_plane_auth_refused",
            "outcome": result.outcome.value,
            "detail": result.detail,
        }))
        raise Unauthorized(f"token rejected: {result.outcome.value}")
    return entitlement_from_claims(result.claims, client_entitlements=client_entitlements)


def headers_from_context(ctx: Any) -> Mapping[str, str]:
    """Inbound HTTP headers for the request a tool is serving.

    Read from ``ctx.request_context.request``, which FastMCP populates with the Starlette
    request under streamable-HTTP transport. Verified against mcp 1.29.0 in the embedded
    DSP guidance rather than assumed. An absent request yields no headers, no headers
    means no token, and no token means refusal.
    """
    request = getattr(getattr(ctx, "request_context", None), "request", None)
    headers = getattr(request, "headers", None)
    if headers is None:
        return {}
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


def guarded(
    fn: Callable[..., dict],
    *,
    resolve: Callable[[], Entitlement],
    tool: str,
    **kwargs: Any,
) -> dict:
    """Resolve the caller, run the operation, and map errors onto stable shapes.

    Errors are returned as structured results rather than raised through the transport,
    because an agent needs to read WHY a write was refused; an opaque transport error
    tells it only that something went wrong. The message never echoes the token, the
    entitlement map or another principal's record.
    """
    started = time.monotonic()
    entitlement: Optional[Entitlement] = None
    try:
        entitlement = resolve()
    except Unauthorized as exc:
        return {"error": "unauthorized", "message": str(exc)}
    outcome = "ok"
    try:
        return fn(entitlement, **kwargs)
    except NotEntitled as exc:
        outcome = "forbidden"
        return {"error": "forbidden", "message": str(exc)}
    except NotFound as exc:
        outcome = "not_found"
        return {"error": "not_found", "message": str(exc)}
    except InvalidRequest as exc:
        outcome = "invalid_request"
        return {"error": "invalid_request", "message": str(exc)}
    except Exception:  # noqa: BLE001 - the caller gets a generic error; the log gets the trace
        outcome = "internal_error"
        logger.exception(json.dumps({"event": "control_plane_tool_failed", "tool": tool}))
        return {"error": "internal_error", "message": "the operation failed; see the runtime log"}
    finally:
        logger.info(json.dumps({
            "event": "control_plane_tool", "tool": tool, "outcome": outcome,
            "client_id": getattr(entitlement, "client_id", None),
            "ms": round((time.monotonic() - started) * 1000, 1),
        }))


def create_server(
    operations: ControlPlaneOperations,
    *,
    jwks_provider: Callable[[], Mapping[str, Any]],
    issuer: str,
    scope_prefix: str = DEFAULT_SCOPE_PREFIX,
    header_reader: Callable[[Any], Mapping[str, str]] = headers_from_context,
    client_entitlements: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Register the ten tools on a FastMCP server.

    ``header_reader`` takes the tool's Context and returns its inbound headers. Injected
    so a test can drive authorisation without a live transport, and so a future SDK
    change is one function to update rather than ten tools.
    """
    from mcp.server.fastmcp import Context, FastMCP  # see the module note

    server = FastMCP(host=BIND_HOST, port=BIND_PORT, stateless_http=True)
    scopes = scopes_for(scope_prefix)

    def _run(tool: str, fn: Callable[..., dict], ctx: Any, **kwargs: Any) -> dict:
        def resolve() -> Entitlement:
            return entitlement_for(
                header_reader(ctx).get("authorization"),
                jwks_provider=jwks_provider,
                issuer=issuer,
                required_scopes=scopes[tool],
                client_entitlements=client_entitlements,
            )
        return guarded(fn, resolve=resolve, tool=tool, **kwargs)

    # Every tool takes `ctx` so the inbound headers, and therefore the caller's identity,
    # are available. FastMCP injects it and keeps it out of the published input schema.

    # ------------------------------------------------------------- campaigns

    @server.tool()
    def create_campaign(
        ctx: Context,
        campaign_id: str,
        campaign_name: str,
        adomain: str,
        creative_id: str,
        width: int,
        height: int,
        declared_cpm: float,
        deal_ids: list[str] | None = None,
        open_market: bool = False,
        target_categories: list[str] | None = None,
        media_type: str = "banner",
        duration: int | None = None,
        audience_uplift: dict | None = None,
        note: str | None = None,
    ) -> dict:
        """Create a campaign for the artfhouse demand endpoint. Created PAUSED; call
        activate_campaign to make it bid.

        campaign_id: lowercase letters, digits and hyphens. adomain: the advertiser's
        domain (the caller must be entitled to it). deal_ids: deals this campaign buys
        on; each must exist and be held by no other campaign. open_market: may bid with
        no deal. media_type: banner or video (video needs duration, seconds).
        audience_uplift: {"segment_ids": [...], "cpm": extra CPM for a matched audience}.
        """
        return _run("create_campaign", operations.create_campaign, ctx,
                    campaign_id=campaign_id, campaign_name=campaign_name, adomain=adomain,
                    creative_id=creative_id, width=width, height=height, declared_cpm=declared_cpm,
                    deal_ids=deal_ids, open_market=open_market, target_categories=target_categories,
                    media_type=media_type, duration=duration, audience_uplift=audience_uplift, note=note)

    @server.tool()
    def update_campaign(
        ctx: Context,
        campaign_id: str,
        campaign_name: str | None = None,
        adomain: str | None = None,
        creative_id: str | None = None,
        width: int | None = None,
        height: int | None = None,
        declared_cpm: float | None = None,
        deal_ids: list[str] | None = None,
        open_market: bool | None = None,
        target_categories: list[str] | None = None,
        media_type: str | None = None,
        duration: int | None = None,
        audience_uplift: dict | None = None,
        note: str | None = None,
    ) -> dict:
        """Change a campaign. Only supplied fields change; status moves through
        activate_campaign and pause_campaign."""
        return _run("update_campaign", operations.update_campaign, ctx,
                    campaign_id=campaign_id, campaign_name=campaign_name, adomain=adomain,
                    creative_id=creative_id, width=width, height=height, declared_cpm=declared_cpm,
                    deal_ids=deal_ids, open_market=open_market, target_categories=target_categories,
                    media_type=media_type, duration=duration, audience_uplift=audience_uplift, note=note)

    @server.tool()
    def activate_campaign(ctx: Context, campaign_id: str) -> dict:
        """Make a campaign bid. Refuses one that holds no active deal and is not open
        market, because activating it would change nothing."""
        return _run("activate_campaign", operations.activate_campaign, ctx, campaign_id=campaign_id)

    @server.tool()
    def pause_campaign(ctx: Context, campaign_id: str) -> dict:
        """Stop a campaign bidding. It stays listed and can be activated again. There is
        no delete."""
        return _run("pause_campaign", operations.pause_campaign, ctx, campaign_id=campaign_id)

    @server.tool()
    def get_campaign(ctx: Context, campaign_id: str) -> dict:
        """One campaign's full record, its status, and which of its deals are active."""
        return _run("get_campaign", operations.get_campaign, ctx, campaign_id=campaign_id)

    @server.tool()
    def list_campaigns(ctx: Context, status: str | None = None, advertiser: str | None = None) -> dict:
        """Campaigns the caller is entitled to see, optionally by status (active or
        paused) or advertiser domain."""
        return _run("list_campaigns", operations.list_campaigns, ctx, status=status, advertiser=advertiser)

    # ----------------------------------------------------------------- deals

    @server.tool()
    def create_deal(ctx: Context, deal: dict) -> dict:
        """Create an IAB Tech Lab Deals API v1.0 deal. ``deal`` is the Deal object:
        id (lowercase letters, digits, hyphens), seller (required here: the caller must
        be entitled to it), terms {dealfloor, cur, pricetype, guar, ...}, inventory
        {sitedomains or appbundles, cat, cattax, ...}, adtypes, desc, name, sellerstatus
        (0 active, default). origin is set by the stack and may not be supplied."""
        return _run("create_deal", operations.create_deal, ctx, deal=deal)

    @server.tool()
    def update_deal(ctx: Context, deal_id: str, changes: dict) -> dict:
        """Change a deal's fields. Use sellerstatus to pause (1) or reactivate (0) it;
        4 (archived) is final. origin and id cannot change."""
        return _run("update_deal", operations.update_deal, ctx, deal_id=deal_id, changes=changes)

    @server.tool()
    def get_deal(ctx: Context, deal_id: str) -> dict:
        """One deal's full Deals API record, whether it is active, and which campaign holds it."""
        return _run("get_deal", operations.get_deal, ctx, deal_id=deal_id)

    @server.tool()
    def list_deals(ctx: Context, sellerstatus: int | None = None, seller: str | None = None) -> dict:
        """Deals the caller is entitled to see, optionally by sellerstatus (0 to 4) or
        seller domain."""
        return _run("list_deals", operations.list_deals, ctx, sellerstatus=sellerstatus, seller=seller)

    return server


def main() -> None:
    # Our own records are JSON objects (event, tool, outcome, client_id, ms); the SDK's
    # and boto3's are plain text, so the line format stays plain and the JSON sits in
    # the message field. No token, secret or entitlement map is ever logged.
    logging.basicConfig(
        level=os.environ.get("ARTF_CONTROL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    operations, jwks_provider, issuer = build_operations()
    prefix = os.environ.get("ARTF_CONTROL_SCOPE_PREFIX") or DEFAULT_SCOPE_PREFIX

    client_entitlements = parse_client_entitlements(os.environ.get("ARTF_CONTROL_CLIENT_ENTITLEMENTS"))
    if client_entitlements:
        # Logged at startup so the grants in force are recoverable from the log. The
        # principals are not secret; the client SECRET is never here.
        logger.info(json.dumps({"event": "control_plane_entitlements", "grants": client_entitlements}))
    else:
        # A client_credentials caller carries no custom:advertisers claim, so without a
        # map every tool call from one is denied. Warned, not fatal: a deployment may
        # intend to serve only tokens that carry their own claim.
        logger.warning(json.dumps({
            "event": "control_plane_no_entitlements",
            "message": "ARTF_CONTROL_CLIENT_ENTITLEMENTS is empty; every machine caller will be denied",
        }))

    server = create_server(
        operations,
        jwks_provider=jwks_provider,
        issuer=issuer,
        scope_prefix=prefix,
        client_entitlements=client_entitlements,
    )
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
