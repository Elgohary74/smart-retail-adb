"""Configurable FIFO/LRU buffer pool shared across page streams."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .page_file import PageBackend

DEFAULT_STREAM = "default"
FrameKey = tuple[str, int]


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
    """Caches pages keyed by (stream_id, page_id) with dirty write-back and pinning."""

    def __init__(
        self,
        backend: PageBackend | None = None,
        capacity: int = 16,
        policy: str = "LRU",
    ):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        normalized_policy = policy.upper()
        if normalized_policy not in {"FIFO", "LRU"}:
            raise ValueError("policy must be 'FIFO' or 'LRU'")
        self.capacity = capacity
        self.policy = normalized_policy
        self._backends: dict[str, PageBackend] = {}
        self._frames: OrderedDict[FrameKey, bytes] = OrderedDict()
        self._dirty: set[FrameKey] = set()
        self._pins: dict[FrameKey, int] = {}
        self.stats = BufferStats()
        if backend is not None:
            self.register_backend(DEFAULT_STREAM, backend)

    def register_backend(self, stream_id: str, backend: PageBackend) -> None:
        if stream_id in self._backends:
            raise ValueError(f"Stream {stream_id!r} is already registered")
        self._backends[stream_id] = backend

    def unregister_backend(self, stream_id: str, close_backend: bool = False) -> None:
        backend = self._get_backend(stream_id)
        if any(key[0] == stream_id and count > 0 for key, count in self._pins.items()):
            raise RuntimeError(f"Cannot unregister stream {stream_id!r} with pinned frames")
        self.flush(stream_id=stream_id)
        for key in [k for k in self._frames if k[0] == stream_id]:
            del self._frames[key]
            self._pins.pop(key, None)
        del self._backends[stream_id]
        if close_backend:
            backend.close()

    def _get_backend(self, stream_id: str) -> PageBackend:
        if stream_id not in self._backends:
            raise KeyError(f"Unknown storage stream: {stream_id!r}")
        return self._backends[stream_id]

    @property
    def page_count(self) -> int:
        return self.stream_page_count(DEFAULT_STREAM)

    def stream_page_count(self, stream_id: str = DEFAULT_STREAM) -> int:
        return self._get_backend(stream_id).page_count

    def stream_page_size(self, stream_id: str = DEFAULT_STREAM) -> int:
        return self._get_backend(stream_id).page_size

    def allocate_page(self, stream_id: str = DEFAULT_STREAM) -> int:
        return self._get_backend(stream_id).allocate_page()

    def _touch(self, key: FrameKey) -> None:
        if self.policy == "LRU":
            self._frames.move_to_end(key)

    def _evict_if_needed(self) -> None:
        if len(self._frames) < self.capacity:
            return
        victim = next((key for key in self._frames if self._pins.get(key, 0) == 0), None)
        if victim is None:
            raise RuntimeError("Buffer pool is full and all frames are pinned")
        data = self._frames.pop(victim)
        self._pins.pop(victim, None)
        if victim in self._dirty:
            self._get_backend(victim[0]).write_page(victim[1], data)
            self.stats.disk_writes += 1
            self._dirty.remove(victim)
        self.stats.evictions += 1

    def _validate_page_id(self, backend: PageBackend, page_id: int) -> None:
        if not isinstance(page_id, int) or isinstance(page_id, bool) or page_id < 0:
            raise ValueError("page_id must be a non-negative integer")
        if page_id >= backend.page_count:
            raise IndexError(f"Page {page_id} does not exist; page_count={backend.page_count}")

    def read_page(self, page_id: int, stream_id: str = DEFAULT_STREAM) -> bytes:
        key = (stream_id, page_id)
        if key in self._frames:
            self.stats.hits += 1
            self._touch(key)
            return self._frames[key]
        backend = self._get_backend(stream_id)
        self.stats.misses += 1
        data = backend.read_page(page_id)
        self.stats.disk_reads += 1
        self._evict_if_needed()
        self._frames[key] = data
        return data

    def write_page(self, page_id: int, data: bytes, stream_id: str = DEFAULT_STREAM) -> None:
        backend = self._get_backend(stream_id)
        self._validate_page_id(backend, page_id)
        if len(data) != backend.page_size:
            raise ValueError(f"Page must be exactly {backend.page_size} bytes; got {len(data)}")
        key = (stream_id, page_id)
        if key in self._frames:
            self._touch(key)
        else:
            self._evict_if_needed()
        self._frames[key] = bytes(data)
        self._dirty.add(key)

    def pin_page(self, page_id: int, stream_id: str = DEFAULT_STREAM) -> bytes:
        data = self.read_page(page_id, stream_id=stream_id)
        key = (stream_id, page_id)
        self._pins[key] = self._pins.get(key, 0) + 1
        return data

    def unpin_page(self, page_id: int, stream_id: str = DEFAULT_STREAM) -> None:
        key = (stream_id, page_id)
        count = self._pins.get(key, 0)
        if count <= 0:
            raise ValueError(f"Page {key} is not pinned")
        if count == 1:
            del self._pins[key]
        else:
            self._pins[key] = count - 1

    def flush(self, stream_id: str | None = None) -> None:
        keys = [
            key for key in list(self._dirty)
            if stream_id is None or key[0] == stream_id
        ]
        for key in keys:
            self._get_backend(key[0]).write_page(key[1], self._frames[key])
            self.stats.disk_writes += 1
            self._dirty.remove(key)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            for backend in self._backends.values():
                backend.close()

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy,
            "capacity": self.capacity,
            "cached_frames": len(self._frames),
            "dirty_frames": len(self._dirty),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": round(self.stats.hit_rate, 4),
            "disk_reads": self.stats.disk_reads,
            "disk_writes": self.stats.disk_writes,
            "evictions": self.stats.evictions,
        }
