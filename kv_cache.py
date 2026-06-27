"""
Module 3: KV Cache Coordination Layer
Scientifically correct with proper hit/miss tracking and store-before-retrieve ordering.
"""
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import threading

logger = logging.getLogger("AdaptiveSD.KVCache")

class SyncStrategy(Enum):
    LAZY = "lazy"
    EAGER = "eager"
    ADAPTIVE = "adaptive"

@dataclass
class KVCacheConfig:
    n_layers: int = 28
    n_heads: int = 14
    n_kv_heads: int = 0
    head_dim: int = 64
    max_context: int = 8192
    quantize_draft: bool = True
    sync_strategy: SyncStrategy = SyncStrategy.ADAPTIVE
    lazy_sync_min_accepted: int = 2
    shadow_buffer_tokens: int = 64

    def __post_init__(self):
        if self.n_kv_heads <= 0:
            self.n_kv_heads = self.n_heads

@dataclass
class KVCacheStats:
    total_syncs: int = 0
    total_rollbacks: int = 0
    bytes_saved_by_quantization: int = 0
    bytes_transferred: int = 0
    partial_rollbacks: int = 0
    full_rollbacks: int = 0
    avg_rollback_tokens: float = 0.0
    shadow_hits: int = 0
    shadow_misses: int = 0
    total_sync_latency_ms: float = 0.0
    total_rollback_latency_ms: float = 0.0
    # FIX: Track draft model compression independently
    draft_bytes_full: int = 0
    draft_bytes_compressed: int = 0

class KVCacheCoordinator:
    def __init__(self, config: Optional[KVCacheConfig] = None):
        self.cfg = config or KVCacheConfig()
        self._lock = threading.Lock()
        self._committed_length = 0
        self._draft_end = 0
        self._draft_stack: List[int] = []
        self._shadow_buffer: Dict[int, list] = {}
        self._shadow_valid: Dict[int, bool] = {}
        self.stats = KVCacheStats()
        
        logger.info(
            "KVCacheCoordinator initialised (layers=%d, heads=%d, kv_heads=%d, quantize=%s)",
            self.cfg.n_layers, self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.quantize_draft
        )

    def begin_draft(self, n_tokens: int, starting_pos: int) -> List[int]:
        with self._lock:
            starting_pos = max(0, starting_pos)
            self._committed_length = starting_pos
            positions = list(range(starting_pos, starting_pos + n_tokens))
            self._draft_stack = positions.copy()
            self._draft_end = starting_pos + n_tokens
            return positions

    def accept_tokens(self, n_accepted: int) -> int:
        sync_start = time.perf_counter()
        with self._lock:
            # Always count as a sync (the +1 verified target token is always committed)
            self.stats.total_syncs += 1
            if n_accepted <= 0:
                self.stats.bytes_transferred += self._estimate_kv_bytes(1)
                self.stats.total_sync_latency_ms += (time.perf_counter() - sync_start) * 1000.0
                return self._committed_length
            n_accepted = min(n_accepted, len(self._draft_stack))
            if n_accepted == 0:
                self.stats.total_sync_latency_ms += (time.perf_counter() - sync_start) * 1000.0
                return self._committed_length

            accept_positions = self._draft_stack[:n_accepted]
            self._committed_length += n_accepted
            self._draft_stack = self._draft_stack[n_accepted:]

            for pos in accept_positions:
                self._shadow_valid[pos] = True

            self.stats.bytes_transferred += self._estimate_kv_bytes(n_accepted + 1)
            self._evict_old_shadow(self._committed_length)

        self.stats.total_sync_latency_ms += (time.perf_counter() - sync_start) * 1000.0
        return self._committed_length

    def reject_tokens(self, n_rejected: int) -> Tuple[int, bool]:
        rollback_start = time.perf_counter()
        with self._lock:
            if n_rejected <= 0:
                return self._draft_end, False
            total_before = len(self._draft_stack)
            if total_before == 0:
                return self._draft_end, False

            n_rejected = min(n_rejected, total_before)
            rejected_positions = self._draft_stack[-n_rejected:]
            self._draft_stack = self._draft_stack[:-n_rejected]
            self._draft_end -= n_rejected

            for pos in rejected_positions:
                self._shadow_valid[pos] = False
                self._shadow_buffer.pop(pos, None)

            self.stats.total_rollbacks += 1
            if n_rejected < total_before:
                self.stats.partial_rollbacks += 1
            else:
                self.stats.full_rollbacks += 1

            n = self.stats.total_rollbacks
            self.stats.avg_rollback_tokens = (
                (self.stats.avg_rollback_tokens * (n - 1) + n_rejected) / n
            )

        self.stats.total_rollback_latency_ms += (time.perf_counter() - rollback_start) * 1000.0
        return self._draft_end, True

    def store_draft_kv(self, pos: int, keys: np.ndarray, values: np.ndarray) -> int:
        """Store draft KV for a position. Call BEFORE retrieve_draft_kv."""
        full_bytes = keys.nbytes + values.nbytes
        
        if self.cfg.quantize_draft:
            keys_q, scale_k = self._quantize_int8(keys)
            vals_q, scale_v = self._quantize_int8(values)
            entry = [keys_q, vals_q, scale_k, scale_v]
            compressed_bytes = keys_q.nbytes + vals_q.nbytes + scale_k.nbytes + scale_v.nbytes
            saved = max(0, full_bytes - compressed_bytes)
            
            # FIX: Track draft compression accurately (separate from target model transfer)
            self.stats.draft_bytes_full += full_bytes
            self.stats.draft_bytes_compressed += compressed_bytes
        else:
            entry = [keys.copy(), values.copy(), None, None]
            saved = 0
            self.stats.draft_bytes_full += full_bytes
            self.stats.draft_bytes_compressed += full_bytes

        with self._lock:
            self._shadow_buffer[pos] = entry
            self._shadow_valid[pos] = True
            self.stats.bytes_saved_by_quantization += saved
        return saved

    def retrieve_draft_kv(self, pos: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached KV for a position. Call AFTER store_draft_kv."""
        with self._lock:
            if pos not in self._shadow_buffer:
                self.stats.shadow_misses += 1
                return None
            if not self._shadow_valid.get(pos, False):
                self.stats.shadow_misses += 1
                return None
            self.stats.shadow_hits += 1
            entry = self._shadow_buffer[pos]

        keys_q, vals_q, scale_k, scale_v = entry
        if scale_k is not None:
            keys = self._dequantize_int8(keys_q, scale_k)
            vals = self._dequantize_int8(vals_q, scale_v)
        else:
            keys = keys_q.copy()
            vals = vals_q.copy()
        return keys, vals

    def can_reuse_kv(self, pos: int) -> bool:
        with self._lock:
            return pos in self._shadow_buffer and self._shadow_valid.get(pos, False)

    @property
    def committed_length(self) -> int:
        return self._committed_length

    @property
    def draft_length(self) -> int:
        return len(self._draft_stack)

    @property
    def memory_pressure(self) -> float:
        total = self._committed_length + len(self._draft_stack)
        return total / self.cfg.max_context if self.cfg.max_context > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        total = self.stats.shadow_hits + self.stats.shadow_misses
        return self.stats.shadow_hits / total if total > 0 else 0.0

    @property
    def compression_ratio(self) -> float:
        # FIX: Calculate ratio based purely on draft model sizes (FP32 vs INT8)
        if self.stats.draft_bytes_full <= 0:
            return 1.0
        return self.stats.draft_bytes_full / max(1, self.stats.draft_bytes_compressed)

    def get_stats(self) -> dict:
        s = self.stats
        total_syncs = max(s.total_syncs, 1)
        total_rollbacks = max(s.total_rollbacks, 1)
        return {
            "committed_length": self._committed_length,
            "draft_length": self.draft_length,
            "total_syncs": s.total_syncs,
            "total_rollbacks": s.total_rollbacks,
            "partial_rollbacks": s.partial_rollbacks,
            "full_rollbacks": s.full_rollbacks,
            "avg_rollback_tokens": round(s.avg_rollback_tokens, 2),
            "shadow_hits": s.shadow_hits,
            "shadow_misses": s.shadow_misses,
            "hit_rate": round(self.hit_rate, 4),
            "compression_ratio": round(self.compression_ratio, 2),
            "memory_pressure": round(self.memory_pressure, 4),
            "sync_latency_avg_ms": round(s.total_sync_latency_ms / total_syncs, 3),
            "rollback_latency_avg_ms": round(s.total_rollback_latency_ms / total_rollbacks, 3),
            "bytes_saved_by_quantization_KB": round(s.bytes_saved_by_quantization / 1024, 1),
            "bytes_transferred_KB": round(s.bytes_transferred / 1024, 1),
            "shadow_buffer_entries": len(self._shadow_buffer),
        }

    def _estimate_kv_bytes(self, n_tokens: int) -> int:
        return 2 * self.cfg.n_layers * self.cfg.n_kv_heads * self.cfg.head_dim * 4 * n_tokens

    @staticmethod
    def _quantize_int8(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_val = float(np.max(np.abs(arr))) + 1e-8
        scale = max_val / 127.0
        q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        return q, np.array([scale], dtype=np.float32)

    @staticmethod
    def _dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return q.astype(np.float32) * float(scale[0])

    def _evict_old_shadow(self, current_pos: int):
        evict_before = max(0, current_pos - self.cfg.shadow_buffer_tokens)
        to_evict = [p for p in self._shadow_buffer if p < evict_before]
        for p in to_evict:
            del self._shadow_buffer[p]
            self._shadow_valid.pop(p, None)