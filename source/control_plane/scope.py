"""Which advertisers and sellers a caller may touch. Deny by default.

AgentCore verifies the inbound token before a tool is ever invoked, and auth.py
verifies it again. What this module does is the part neither can know: mapping a
verified caller onto the set of principals it is entitled to act for. A campaign's
principal is its ``adomain`` (the advertiser); a deal's is its ``seller``.

DENY BY DEFAULT. A token with no entitlement resolves to an EMPTY set, and an empty
set authorises nothing; it is not read as "unrestricted". The failure mode of the
opposite default is an agent quietly writing campaigns for advertisers it has no
relationship with, or taking another seller's deals off the market.

A DENIAL IS A STRUCTURED ERROR, NOT AN EMPTY RESULT. Returning "no campaigns" for
something the caller may not see tells them the campaign does not exist, which is
false and sends them to debug the wrong thing. The one exception is the read of a
single record by id: there "not found" and "not yours" are deliberately the same
answer, so a caller cannot enumerate other principals' ids by probing.

Ported from the embedded DSP guidance's scope module; the principal is generalised
from "advertiser" to any domain so one entitlement covers both record types.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: Claim carrying the principals a caller may act for, as a space- or comma-delimited
#: list. Present on a human-backed token; never on a client_credentials one.
DEFAULT_CLAIM = "custom:advertisers"

#: The one value that means "every principal". Spelled out rather than implied by an
#: absent claim, so granting full access is always a deliberate act recorded in config.
WILDCARD = "*"


class NotEntitled(PermissionError):
    """The caller may not act for this principal."""


@dataclass(frozen=True)
class Entitlement:
    """The principals (advertiser or seller domains) a verified caller may act for."""

    principals: frozenset[str]
    unrestricted: bool
    #: For the audit log. Which client the entitlement came from.
    client_id: Optional[str] = None

    def permits(self, principal: str) -> bool:
        if self.unrestricted:
            return True
        return _norm(principal) in self.principals

    def require(self, principal: str, *, kind: str = "principal") -> None:
        """Raise unless the caller may act for this principal."""
        if not principal:
            raise NotEntitled(f"no {kind} was supplied")
        if not self.permits(principal):
            raise NotEntitled(f"caller is not entitled to act for {kind} {principal!r}")


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _split(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [_norm(v) for v in raw if _norm(v)]
    if isinstance(raw, str):
        # Both delimiters, because a claim written by hand and one written by a
        # provisioning script rarely agree on which to use.
        return [_norm(v) for v in raw.replace(",", " ").split() if _norm(v)]
    return []


def parse_client_entitlements(raw: Optional[str]) -> dict[str, list[str]]:
    """Parse a ``client_id=principals`` map for machine-to-machine callers.

    WHY THIS EXISTS. A ``client_credentials`` token has NO USER behind it, so Cognito
    puts no custom attributes on it; ``custom:advertisers`` cannot be present on a
    machine token however the pool is configured. An external agent authenticating
    with a client id and secret therefore arrives with nothing this module could read,
    and deny-by-default would deny every call. So the grant is attached to the CLIENT,
    declared in the runtime's environment where it is visible in ``get-agent-runtime``
    and changeable without a code deploy.

    Format, chosen to survive a shell and a CLI list:

        client-a=acme.example,globex.example;client-b=*

    Semicolons separate clients, commas separate principals, ``*`` means every one.
    An unparseable entry is SKIPPED rather than guessed at: a malformed grant that
    silently became a wildcard would be the worst possible failure here.
    """
    mapping: dict[str, list[str]] = {}
    if not raw:
        return mapping
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        client_id, _, principals = entry.partition("=")
        client_id = client_id.strip()
        if not client_id:
            continue
        values = [_norm(v) for v in principals.split(",") if _norm(v)]
        if values:
            mapping[client_id] = values
    return mapping


def format_client_entitlements(mapping: Mapping[str, Any]) -> str:
    """The inverse of ``parse_client_entitlements``, for tests and tooling."""
    return ";".join(f"{client}={','.join(_split(values))}" for client, values in mapping.items() if _split(values))


def entitlement_from_claims(
    claims: Optional[Mapping[str, Any]],
    *,
    claim_name: str = DEFAULT_CLAIM,
    client_entitlements: Optional[Mapping[str, Any]] = None,
) -> Entitlement:
    """Resolve a verified caller's entitlement.

    TWO SOURCES, in a deliberate order:

      1. the token's own claim, for a caller that has one (a human-backed token, or a
         provider that can attach custom claims to a machine token)
      2. the client-id map, for a ``client_credentials`` caller, which cannot carry one

    The claim wins when present, so a token that states its own entitlement is never
    widened by configuration.
    """
    if not claims:
        return Entitlement(frozenset(), unrestricted=False)

    client_id = claims.get("client_id") or claims.get("sub")
    client_id_str = str(client_id) if client_id else None

    values = _split(claims.get(claim_name))

    if not values and client_entitlements and client_id_str:
        # The machine-to-machine path. Looked up by the client id the token asserts,
        # which Cognito sets itself; a caller cannot choose which grant it matches.
        values = _split(client_entitlements.get(client_id_str))

    if WILDCARD in values:
        return Entitlement(frozenset(), unrestricted=True, client_id=client_id_str)

    return Entitlement(frozenset(values), unrestricted=False, client_id=client_id_str)


def visible(entitlement: Entitlement, items: list[dict], *, key: str) -> list[dict]:
    """Filter records to those the caller may see, by the principal field ``key``.

    Used only for LIST operations, where filtering is the correct behaviour: the caller
    asked "what may I see". A single-record read uses ``require`` instead.
    """
    if entitlement.unrestricted:
        return list(items)
    return [i for i in items if _norm(i.get(key) or "") in entitlement.principals]
