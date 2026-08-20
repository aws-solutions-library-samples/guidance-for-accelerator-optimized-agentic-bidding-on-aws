"""Signal Associator — associates late-arriving downstream signals to originating bids.

Handles impression, click, and conversion signals that arrive after the initial
bid outcome event. Looks up the original bid context by `request_id` and emits
enriched BidOutcomeEvents to Kinesis with the updated signal data.

Uses an in-memory LRU cache with TTL for the prototype.
NOTE: Production deployments should replace this with Redis or DynamoDB for
durability across restarts and horizontal scaling.

Requirements: 1.4
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from shared.feedback_collector import FeedbackCollector
from shared.feedback_models import BidOutcomeEvent

logger = logging.getLogger(__name__)

# Default TTL for cached bid contexts (24 hours)
_DEFAULT_TTL_SECONDS = 86400

# Default maximum number of entries in the LRU cache
_DEFAULT_MAX_ENTRIES = 100_000


class SignalType(str, Enum):
    """Types of downstream signals that can arrive after the initial bid."""

    IMPRESSION = "impression"
    CLICK = "click"
    CONVERSION = "conversion"


class DownstreamSignal(BaseModel):
    """Incoming downstream signal to associate with an originating bid.

    Accepts impression, click, or conversion signals identified by request_id.
    """

    request_id: str
    signal_type: SignalType
    conversion_value: Optional[float] = None
    timestamp: float

    @model_validator(mode="after")
    def _validate_signal(self) -> "DownstreamSignal":
        """Validate that conversion_value is only provided for conversion signals."""
        if self.signal_type != SignalType.CONVERSION and self.conversion_value is not None:
            raise ValueError(
                "conversion_value should only be provided for conversion signals"
            )
        return self


class _CacheEntry:
    """Internal cache entry holding bid context and its expiration time."""

    __slots__ = ("event", "expires_at")

    def __init__(self, event: BidOutcomeEvent, ttl_seconds: float) -> None:
        self.event = event
        self.expires_at = time.monotonic() + ttl_seconds


class SignalAssociator:
    """Associates late-arriving downstream signals to originating bids.

    Maintains an in-memory LRU cache of bid contexts keyed by request_id.
    When a downstream signal arrives, looks up the original bid context and
    emits an enriched BidOutcomeEvent to Kinesis.

    NOTE: This in-memory implementation is suitable for single-instance
    prototypes. Production deployments should use Redis or DynamoDB for:
    - Durability across process restarts
    - Shared state across horizontally-scaled orchestrator instances
    - Larger cache capacity without memory pressure

    Parameters
    ----------
    feedback_collector : FeedbackCollector
        The collector used to emit enriched events to Kinesis.
    ttl_seconds : float
        Time-to-live for cached bid contexts (default: 24 hours).
    max_entries : int
        Maximum number of entries in the LRU cache (default: 100,000).
    """

    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._collector = feedback_collector
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        # OrderedDict for LRU eviction — most recently used at the end
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_bid(self, event: BidOutcomeEvent) -> None:
        """Store a bid context for later signal association.

        Should be called when the initial bid outcome event is emitted,
        so that downstream signals can be associated later.
        """
        self._evict_expired()
        self._cache[event.request_id] = _CacheEntry(event, self._ttl_seconds)
        # Move to end (most recently used)
        self._cache.move_to_end(event.request_id)
        # Enforce max size — evict oldest (LRU)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    async def handle_signal(self, signal: DownstreamSignal) -> bool:
        """Process a downstream signal and emit an enriched event.

        Associates the signal with the originating bid by request_id. If the
        original bid context is found, emits an updated BidOutcomeEvent with
        the enriched signal data. Handles out-of-order signals by filling in
        the signal chain (e.g., conversion implies click implies impression).

        Parameters
        ----------
        signal : DownstreamSignal
            The downstream signal to process.

        Returns
        -------
        bool
            True if the signal was successfully associated and emitted,
            False if the original bid context was not found (expired or
            never registered).
        """
        entry = self._get(signal.request_id)
        if entry is None:
            logger.warning(
                "Signal for request_id=%s dropped: bid context not found "
                "(expired or never registered). signal_type=%s",
                signal.request_id,
                signal.signal_type.value,
            )
            return False

        # Build the enriched event with updated signals
        enriched_event = self._enrich_event(entry.event, signal)

        # Update the cache with the enriched event so subsequent signals
        # build on top of previous enrichments
        entry.event = enriched_event
        self._cache.move_to_end(signal.request_id)

        # Emit the enriched event to Kinesis
        await self._collector.emit(enriched_event)

        logger.info(
            "Signal associated: request_id=%s signal_type=%s",
            signal.request_id,
            signal.signal_type.value,
        )
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache_size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._cache)

    @property
    def ttl_seconds(self) -> float:
        """Configured TTL for cache entries."""
        return self._ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, request_id: str) -> Optional[_CacheEntry]:
        """Look up a cache entry by request_id, respecting TTL and LRU order."""
        entry = self._cache.get(request_id)
        if entry is None:
            return None

        # Check TTL expiry
        if time.monotonic() > entry.expires_at:
            del self._cache[request_id]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(request_id)
        return entry

    def _evict_expired(self) -> None:
        """Remove expired entries from the front of the ordered dict.

        Since entries are ordered by access time, expired entries tend to
        cluster at the front. We scan from the front and stop at the first
        non-expired entry for efficiency.
        """
        now = time.monotonic()
        keys_to_remove = []
        for key, entry in self._cache.items():
            if now > entry.expires_at:
                keys_to_remove.append(key)
            else:
                # Entries at the front are oldest; once we find a valid one,
                # later ones are likely valid too. But since TTLs are uniform,
                # we continue scanning to be safe.
                break
        for key in keys_to_remove:
            del self._cache[key]

    @staticmethod
    def _enrich_event(
        original: BidOutcomeEvent, signal: DownstreamSignal
    ) -> BidOutcomeEvent:
        """Create an enriched BidOutcomeEvent with the downstream signal applied.

        Handles out-of-order signals by filling in the monotonic chain:
        - conversion implies click implies impression implies won
        - click implies impression implies won
        - impression implies won

        The original event's `won` field is preserved; it was set at auction time.
        """
        # Start from the current state of the event
        impression = original.impression
        click = original.click
        conversion = original.conversion
        conversion_value = original.conversion_value

        # Apply the signal and fill the chain
        if signal.signal_type == SignalType.CONVERSION:
            conversion = True
            conversion_value = signal.conversion_value
            # Monotonic chain: conversion → click → impression
            click = True
            impression = True
        elif signal.signal_type == SignalType.CLICK:
            click = True
            # Monotonic chain: click → impression
            impression = True
        elif signal.signal_type == SignalType.IMPRESSION:
            impression = True

        # BidOutcomeEvent is frozen, so we need to create a new instance
        return BidOutcomeEvent(
            request_id=original.request_id,
            timestamp=original.timestamp,
            model_version=original.model_version,
            source=original.source,
            original_price=original.original_price,
            shaded_price=original.shaded_price,
            bid_floor=original.bid_floor,
            won=True if impression else original.won,  # impression implies won
            price_paid=original.price_paid,
            impression=impression,
            click=click,
            conversion=conversion,
            conversion_value=conversion_value,
            user_id_hash=original.user_id_hash,
            site_domain=original.site_domain,
            device_type=original.device_type,
            hour_of_day=original.hour_of_day,
            shade_factor_used=original.shade_factor_used,
            conversion_value_estimate_used=original.conversion_value_estimate_used,
        )
