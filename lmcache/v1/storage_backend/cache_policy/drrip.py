# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional
import random

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

logger = init_logger(__name__)


class DRRIPCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """
    Dynamic Re-Reference Interval Prediction (DRRIP) cache policy.

    This implementation follows the behavior described in the original DRRIP
    paper, with the following simplifications that fit LMCache's dict-based
    cache storage:

    * All cache entries share a single RRPV domain (no explicit sets).
    * Leader sets are emulated via hashing to dedicate a tiny subset of keys
      for SRRIP/BRRIP policy selection counters.
    """

    # Default values (can be overridden via config)
    DEFAULT_MAX_RRPV = 3
    DEFAULT_SRRIP_INSERT_RRPV = DEFAULT_MAX_RRPV - 1
    DEFAULT_BRRIP_SHORT_INSERT_PROB = 32  # 1 / 32 inserts use short RRPV
    DEFAULT_PSEL_MAX = 1023
    DEFAULT_LEADER_SET_MASK = 0x1F  # Use lower 5 bits to pick leader keys (~1/32 ratio)

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        max_rrpv: int = DEFAULT_MAX_RRPV,
        srrip_insert_rrpv: int = DEFAULT_SRRIP_INSERT_RRPV,
        brrip_short_insert_prob: int = DEFAULT_BRRIP_SHORT_INSERT_PROB,
        psel_max: int = DEFAULT_PSEL_MAX,
        leader_set_mask: int = DEFAULT_LEADER_SET_MASK,
    ) -> None:
        self.MAX_RRPV = max_rrpv
        self.SRRIP_INSERT_RRPV = srrip_insert_rrpv
        self.BRRIP_INSERT_RRPV = max_rrpv
        self.BRRIP_SHORT_INSERT_PROB = brrip_short_insert_prob
        self.PSEL_MAX = psel_max
        self.LEADER_SET_MASK = leader_set_mask

        self.key_to_rrpv: Dict[CacheEngineKey, int] = {}
        self.rrpv_buckets: Dict[int, OrderedDict[CacheEngineKey, None]] = defaultdict(
            OrderedDict
        )
        self.policy_selector = 0  # Saturating counter in [-PSEL_MAX, PSEL_MAX]
        self.rand = rng or random.Random()
        self._last_evict_keys: list[CacheEngineKey] = []

        logger.info(
            f"Initializing DRRIPCachePolicy with max_rrpv={max_rrpv}, "
            f"srrip_insert_rrpv={srrip_insert_rrpv}, "
            f"brrip_short_insert_prob={brrip_short_insert_prob}, "
            f"psel_max={psel_max}, leader_set_mask={leader_set_mask:#x}"
        )

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
    ) -> None:
        # Reset the key's RRPV to 0 (most recently used state).
        self._move_key_to_rrpv(key, 0)

    def update_on_put(
        self,
        key: CacheEngineKey,
    ) -> None:
        leader_policy = self._leader_policy_for(key)
        policy = leader_policy or self._current_policy()
        rrpv = self._initial_rrpv(policy)

        # Update the policy selector using leader feedback.
        if leader_policy == "SRRIP":
            self.policy_selector = min(self.policy_selector + 1, self.PSEL_MAX)
        elif leader_policy == "BRRIP":
            self.policy_selector = max(self.policy_selector - 1, -self.PSEL_MAX)

        self.key_to_rrpv[key] = rrpv
        self._insert_into_bucket(key, rrpv)

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        self._remove_key(key)

    def get_evict_candidates(
        self,
        cache_dict: dict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        evict_keys: list[CacheEngineKey] = []
        while len(evict_keys) < num_candidates and cache_dict:
            bucket = self.rrpv_buckets[self.MAX_RRPV]
            # Try to evict from the MAX_RRPV bucket first.
            for key in list(bucket.keys()):
                if key not in cache_dict:
                    # Key might already be removed by the backend.
                    self._remove_key(key)
                    continue
                cache_entry = cache_dict[key]
                if not cache_entry.can_evict:
                    continue
                bucket.pop(key, None)
                self.key_to_rrpv.pop(key, None)
                evict_keys.append(key)
                if len(evict_keys) == num_candidates:
                    break

            if len(evict_keys) == num_candidates:
                break

            # Check if all keys are already at MAX_RRPV and still pinned
            if len(bucket) == len(cache_dict) - len(evict_keys):
                # All remaining keys are at MAX_RRPV but pinned - best effort return
                break

            # No eligible entry at MAX_RRPV, age all keys to increase RRPVs.
            self._increment_all_rrpv()

        self._last_evict_keys = list(evict_keys)
        return evict_keys

    # Internal helpers -------------------------------------------------

    def _leader_policy_for(self, key: CacheEngineKey) -> Optional[str]:
        selector = hash(key) & self.LEADER_SET_MASK
        if selector == 0:
            return "SRRIP"
        if selector == 1:
            return "BRRIP"
        return None

    def _current_policy(self) -> str:
        return "SRRIP" if self.policy_selector >= 0 else "BRRIP"

    def _initial_rrpv(self, policy: str) -> int:
        if policy == "SRRIP":
            return self.SRRIP_INSERT_RRPV
        # BRRIP mostly inserts with long RRPV, but occasionally short.
        if self.rand.randrange(self.BRRIP_SHORT_INSERT_PROB) == 0:
            return self.SRRIP_INSERT_RRPV
        return self.BRRIP_INSERT_RRPV

    def _move_key_to_rrpv(self, key: CacheEngineKey, new_rrpv: int) -> None:
        curr_rrpv = self.key_to_rrpv.get(key)
        if curr_rrpv is None:
            # Key might have been removed already; nothing to do.
            return

        if curr_rrpv == new_rrpv:
            return

        bucket = self.rrpv_buckets[curr_rrpv]
        bucket.pop(key, None)
        self.key_to_rrpv[key] = new_rrpv
        self._insert_into_bucket(key, new_rrpv)

    def _insert_into_bucket(self, key: CacheEngineKey, rrpv: int) -> None:
        self.rrpv_buckets[rrpv][key] = None

    def _remove_key(self, key: CacheEngineKey) -> None:
        rrpv = self.key_to_rrpv.pop(key, None)
        if rrpv is None:
            return
        self.rrpv_buckets[rrpv].pop(key, None)

    def _increment_all_rrpv(self) -> None:
        # Age keys by increasing their RRPV until some reach MAX_RRPV.
        for rrpv in range(self.MAX_RRPV):
            bucket = self.rrpv_buckets[rrpv]
            if not bucket:
                continue
            for key in list(bucket.keys()):
                bucket.pop(key, None)
                new_rrpv = min(rrpv + 1, self.MAX_RRPV)
                self.key_to_rrpv[key] = new_rrpv
                self._insert_into_bucket(key, new_rrpv)

    def get_debug_stats(self, cache_dict: dict[CacheEngineKey, Any]) -> dict[str, Any]:
        """
        Return fine-grained stats useful for debugging/testing.

        The structure contains:
            bucket: mapping from RRPV value to ordered list of keys.
            evict_keys: keys returned by the last get_evict_candidates call.
            psel: current policy selector value.
        """
        bucket_view = {
            rrpv: list(self.rrpv_buckets[rrpv].keys())
            for rrpv in range(self.MAX_RRPV + 1)
        }
        hash_table_view = {
            key: hash(key) & self.LEADER_SET_MASK for key in cache_dict.keys()
        }
        return {
            "bucket": bucket_view,
            "hash_table": hash_table_view,
            "evict_keys": list(self._last_evict_keys),
            "psel": self.policy_selector,
        }
