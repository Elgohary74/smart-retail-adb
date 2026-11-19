"""Configurable FIFO/LRU buffer pool with dirty-page write-back."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from .page_file import PageBackend, PAGE_SIZE


@dataclass
class BufferStats:
    hits: int = 0
    misses: int = 0
    disk_reads: int = 0
    disk_writes: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class BufferPool:
    """Caches pages. read_page returns immutable bytes; writes are cached dirty."""
    def __init__(self, backend: PageBackend, capacity: int = 16,
                 policy: str = "LRU"):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        policy = policy.upper()
        if policy not in {"FIFO", "LRU"}:
            raise ValueError("policy must be 'FIFO' or 'LRU'")
        self.backend = backend
        self.capacity = capacity
        self.policy = policy
        # OrderedDict keeps insertion order for FIFO and is moved on access for LRU.
        self._frames: OrderedDict[int, bytes] = OrderedDict()
        self._dirty: set[int] = set()
        self.stats = BufferStats()

    @property
    def page_count(self) -> int:
        return self.backend.page_count

    def allocate_page(self) -> int:
        return self.backend.allocate_page()

    def _evict_if_needed(self) -> None:
        if len(self._frames) < self.capacity:
            return
        page_id, data = self._frames.popitem(last=False)
        if page_id in self._dirty:
            self.backend.write_page(page_id, data)
            self.stats.disk_writes += 1
            self._dirty.remove(page_id)
        self.stats.evictions += 1

    def _touch(self, page_id: int) -> None:
        if self.policy == "LRU":
            self._frames.move_to_end(page_id)

    def read_page(self, page_id: int) -> bytes:
        if page_id in self._frames:
            self.stats.hits += 1
            self._touch(page_id)
            return self._frames[page_id]
        self.stats.misses += 1
        data = self.backend.read_page(page_id)
        self.stats.disk_reads += 1
        self._evict_if_needed()
        self._frames[page_id] = data
        return data

    def write_page(self, page_id: int, data: bytes) -> None:
        if len(data) != PAGE_SIZE:
            raise ValueError(f"Page must be exactly {PAGE_SIZE} bytes; got {len(data)}")
        if page_id in self._frames:
            self._touch(page_id)
        else:
            self._evict_if_needed()
        self._frames[page_id] = bytes(data)
        self._dirty.add(page_id)

    def flush(self) -> None:
        for page_id in list(self._dirty):
            self.backend.write_page(page_id, self._frames[page_id])
            self.stats.disk_writes += 1
            self._dirty.remove(page_id)

    def close(self) -> None:
        self.flush()
        self.backend.close()

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy,
            "capacity": self.capacity,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": round(self.stats.hit_rate, 4),
            "disk_reads": self.stats.disk_reads,
            "disk_writes": self.stats.disk_writes,
            "evictions": self.stats.evictions,
        }
