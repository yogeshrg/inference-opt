"""KV Cache management service."""

import logging
import threading
from collections import OrderedDict
from typing import Any

import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class KVCacheManager:
    """Manager for KV cache with LRU eviction and offloading."""

    def __init__(self) -> None:
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = settings.max_kv_cache_size
        self._offload_enabled = settings.kv_cache_offload
        self._offload_dir = settings.model_cache_dir / "kv_cache_offload"
        self._offload_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._offloaded = 0

    def get(self, key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get KV cache entry."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (MRU)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            self._misses += 1
            return None

    def put(self, key: str, key_tensor: torch.Tensor, value_tensor: torch.Tensor) -> None:
        """Put KV cache entry."""
        with self._lock:
            # Check if we need to evict
            entry_size = key_tensor.element_size() * key_tensor.nelement()
            entry_size += value_tensor.element_size() * value_tensor.nelement()

            while self._get_current_size() + entry_size > self._max_size and self._cache:
                self._evict_lru()

            self._cache[key] = (key_tensor, value_tensor)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        key, (k_tensor, v_tensor) = self._cache.popitem(last=False)
        self._evictions += 1

        if self._offload_enabled:
            self._offload_to_disk(key, k_tensor, v_tensor)

    def _offload_to_disk(self, key: str, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> None:
        """Offload evicted entry to disk."""
        try:
            path = self._offload_dir / f"{key}.pt"
            torch.save({"key": k_tensor, "value": v_tensor}, path)
            self._offloaded += 1
        except Exception as e:
            logger.warning(f"Failed to offload KV cache: {e}")

    def _load_from_disk(self, key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load offloaded entry from disk."""
        try:
            path = self._offload_dir / f"{key}.pt"
            if path.exists():
                data = torch.load(path)
                path.unlink()  # Remove after loading
                return data["key"], data["value"]
        except Exception as e:
            logger.warning(f"Failed to load KV cache from disk: {e}")
        return None

    def _get_current_size(self) -> int:
        """Get current cache size in bytes."""
        total = 0
        for k_tensor, v_tensor in self._cache.values():
            total += k_tensor.element_size() * k_tensor.nelement()
            total += v_tensor.element_size() * v_tensor.nelement()
        return total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "total_entries": len(self._cache),
                "cache_size_mb": self._get_current_size() / (1024 * 1024),
                "max_size_mb": self._max_size / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "offloaded": self._offloaded,
            }

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._offloaded = 0


# Global instance
kv_cache_manager = KVCacheManager()