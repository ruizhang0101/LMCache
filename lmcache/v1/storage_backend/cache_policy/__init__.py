# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Optional, Type

# First Party
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.drrip import DRRIPCachePolicy
from lmcache.v1.storage_backend.cache_policy.fifo import FIFOCachePolicy
from lmcache.v1.storage_backend.cache_policy.lfu import LFUCachePolicy
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy
from lmcache.v1.storage_backend.cache_policy.mru import MRUCachePolicy

# Cache policy mapping
POLICY_MAPPING: Dict[str, Type[BaseCachePolicy]] = {
    "LRU": LRUCachePolicy,
    "LFU": LFUCachePolicy,
    "FIFO": FIFOCachePolicy,
    "MRU": MRUCachePolicy,
    "DRRIP": DRRIPCachePolicy,
}


def get_cache_policy(policy_name: str, config: Optional[Any] = None) -> BaseCachePolicy:
    """
    Factory function to get the cache policy instance based on the policy name.

    Args:
        policy_name: Name of the cache policy (case-insensitive, e.g., "LRU", "lru").
        config: Optional LMCacheEngineConfig instance for policy configurations.

    Returns:
        Instance of the corresponding cache policy.

    Raises:
        ValueError: If the policy name is not supported.
    """
    if not policy_name:
        raise ValueError("Cache policy name cannot be empty")

    upper_policy_name = policy_name.upper()

    try:
        policy_class = POLICY_MAPPING[upper_policy_name]
        # DRRIP policy accepts config parameters
        if upper_policy_name == "DRRIP" and config is not None:
            return policy_class(
                max_rrpv=config.drrip_max_rrpv,
                brrip_short_insert_prob=config.drrip_brrip_short_insert_prob,
                psel_max=config.drrip_psel_max,
                leader_set_mask=config.drrip_leader_set_mask,
            )
        return policy_class()
    except KeyError:
        raise ValueError(
            f"Unknown cache policy: {upper_policy_name}."
            f" Supported policies are: {list(POLICY_MAPPING.keys())}"
        ) from None
