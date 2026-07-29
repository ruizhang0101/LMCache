# SPDX-License-Identifier: Apache-2.0
"""Routes applied cache-event batches into the coordinator's consumers.

The key directory is the ordering/dedup gate for the fleet's single
cache-event stream; batches it *applies* are fanned out here to the
other consumers (usage ledger, eviction LRU). Any future ingestion path
(e.g. a message-queue consumer) calls the same router, so consumers are
independent of the transport.
"""

# First Party
from lmcache.v1.distributed.tiers import Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType
from lmcache.v1.mp_coordinator.cache_control.eviction_manager import L2EvictionManager
from lmcache.v1.mp_coordinator.cache_control.usage_manager import L2UsageManager


class CacheEventRouter:
    """Fans one applied cache-event batch out to the quota-side consumers.

    Only ``tier=l2`` batches affect the L2 usage ledger and eviction LRU;
    other tiers are ignored (the key directory consumes every tier
    itself). Thread-safe: the consumers lock internally.

    Args:
        usage_manager: Per-``cache_salt`` L2 byte ledger.
        eviction_manager: Per-``cache_salt`` L2 eviction LRU.
    """

    def __init__(
        self,
        usage_manager: L2UsageManager,
        eviction_manager: L2EvictionManager,
    ) -> None:
        self._usage_manager = usage_manager
        self._eviction_manager = eviction_manager

    def route(self, batch: CacheEventBatch) -> None:
        """Apply one batch to the usage ledger and eviction LRU.

        Call only for batches the key directory applied — replays and
        stale incarnations are already dropped there, so consumers see
        each event at most once per delivery attempt.

        Args:
            batch: The applied batch.
        """
        if batch.tier != Tier.L2:
            return
        for entry in batch.entries:
            key = entry.key.to_object_key()
            if batch.event_type == CacheEventType.STORE:
                self._usage_manager.record_stored(key, entry.size_bytes)
                self._eviction_manager.on_store(key)
            elif batch.event_type == CacheEventType.ACCESS:
                self._eviction_manager.on_lookup(key)
            elif batch.event_type == CacheEventType.DELETE:
                self._usage_manager.record_evicted(key)
                self._eviction_manager.on_remove(key)
