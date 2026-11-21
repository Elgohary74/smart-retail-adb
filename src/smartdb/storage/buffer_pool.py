"""Shared configurable FIFO/LRU buffer pool with pinning and LSN hooks."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from ..core.config import DEFAULT_BUFFER_FRAMES, DEFAULT_REPLACEMENT_POLICY
from .backends import PageBackend, StorageManager


@dataclass
class BufferFrame:
    file_id: int
    page_id: int
    data: bytearray
    pin_count: int = 0
    dirty: bool = False
    page_lsn: int = 0
    rec_lsn: int = 0


@dataclass
class BufferStats:
    hits: int = 0
    misses: int = 0
    disk_reads: int = 0
    disk_writes: int = 0
    evictions: int = 0
    dirty_evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class BufferPool:
    """Caches pages keyed by (file_id, page_id) with pin/unpin and dirty write-back."""

    def __init__(
        self,
        backend: PageBackend | None = None,
        capacity: int = DEFAULT_BUFFER_FRAMES,
        policy: str = DEFAULT_REPLACEMENT_POLICY,
        storage_manager: StorageManager | None = None,
        wal_flushed_lsn_fn: Callable[[], int] | None = None,
    ):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        policy = policy.upper()
        if policy not in {"FIFO", "LRU"}:
            raise ValueError("policy must be 'FIFO' or 'LRU'")
        self.capacity = capacity
        self.policy = policy
        self.storage_manager = storage_manager
        self._wal_flushed_lsn_fn = wal_flushed_lsn_fn
        self._backends: dict[int, PageBackend] = {}
        self._next_file_id = 0
        if backend is not None:
            self.register_backend(backend, file_id=0)
        self._frames: OrderedDict[tuple[int, int], BufferFrame] = OrderedDict()
        self.stats = BufferStats()

    def register_backend(self, backend: PageBackend, file_id: int | None = None) -> int:
        if file_id is None:
            while self._next_file_id in self._backends:
                self._next_file_id += 1
            file_id = self._next_file_id
        self._backends[file_id] = backend
        self._next_file_id = max(self._next_file_id, file_id + 1)
        return file_id

    def _get_backend(self, file_id: int) -> PageBackend:
        if file_id in self._backends:
            return self._backends[file_id]
        if self.storage_manager is not None:
            backend = self.storage_manager.get_backend(file_id)
            self._backends[file_id] = backend
            return backend
        raise KeyError(f"Unknown file_id: {file_id}")

    def _touch(self, key: tuple[int, int]) -> None:
        if self.policy == "LRU":
            self._frames.move_to_end(key)

    def _flush_frame(self, frame: BufferFrame) -> None:
        if not frame.dirty:
            return
        if self._wal_flushed_lsn_fn is not None and frame.page_lsn > self._wal_flushed_lsn_fn():
            raise RuntimeError(
                f"Cannot flush page {frame.page_id} with page_lsn={frame.page_lsn} "
                f"before WAL reaches that LSN"
            )
        backend = self._get_backend(frame.file_id)
        backend.write_page(frame.page_id, bytes(frame.data))
        self.stats.disk_writes += 1
        frame.dirty = False

    def _evict_if_needed(self) -> None:
        if len(self._frames) < self.capacity:
            return
        for key, frame in self._frames.items():
            if frame.pin_count == 0:
                if frame.dirty:
                    self._flush_frame(frame)
                    self.stats.dirty_evictions += 1
                del self._frames[key]
                self.stats.evictions += 1
                return
        raise RuntimeError("All buffer frames are pinned; cannot evict")

    def fetch_page(self, page_id: int, file_id: int = 0) -> BufferFrame:
        key = (file_id, page_id)
        if key in self._frames:
            self.stats.hits += 1
            self._touch(key)
            frame = self._frames[key]
            frame.pin_count += 1
            return frame
        self.stats.misses += 1
        backend = self._get_backend(file_id)
        raw = backend.read_page(page_id)
        self.stats.disk_reads += 1
        self._evict_if_needed()
        frame = BufferFrame(file_id=file_id, page_id=page_id, data=bytearray(raw), pin_count=1)
        self._frames[key] = frame
        return frame

    def new_page(self, file_id: int = 0) -> tuple[int, BufferFrame]:
        backend = self._get_backend(file_id)
        self._evict_if_needed()
        page_id = backend.allocate_page()
        key = (file_id, page_id)
        frame = BufferFrame(
            file_id=file_id,
            page_id=page_id,
            data=bytearray(backend.usable_page_size),
            pin_count=1,
            dirty=True,
        )
        self._frames[key] = frame
        return page_id, frame

    def unpin_page(
        self,
        page_id: int,
        is_dirty: bool = False,
        lsn: int | None = None,
        file_id: int = 0,
    ) -> None:
        key = (file_id, page_id)
        frame = self._frames.get(key)
        if frame is None:
            raise KeyError(f"Page {key} is not in buffer pool")
        if frame.pin_count <= 0:
            raise ValueError(f"Page {key} is already unpinned")
        frame.pin_count -= 1
        if is_dirty:
            frame.dirty = True
            if lsn is not None:
                frame.page_lsn = max(frame.page_lsn, lsn)
                if frame.rec_lsn == 0:
                    frame.rec_lsn = lsn

    @contextmanager
    def pin_page(self, page_id: int, file_id: int = 0, dirty: bool = False) -> Iterator[BufferFrame]:
        frame = self.fetch_page(page_id, file_id=file_id)
        try:
            yield frame
        finally:
            self.unpin_page(page_id, is_dirty=dirty, file_id=file_id)

    def flush_page(self, page_id: int, file_id: int = 0) -> None:
        frame = self._frames.get((file_id, page_id))
        if frame is not None and frame.dirty:
            self._flush_frame(frame)

    def flush_file(self, file_id: int) -> None:
        for (fid, _), frame in list(self._frames.items()):
            if fid == file_id and frame.dirty:
                self._flush_frame(frame)

    def flush_all(self) -> None:
        for frame in list(self._frames.values()):
            if frame.dirty:
                self._flush_frame(frame)

    def flush(self) -> None:
        self.flush_all()

    def unregister_file(self, file_id: int, drop_dirty: bool = True) -> None:
        if not drop_dirty:
            self.flush_file(file_id)
        for key in [k for k in self._frames if k[0] == file_id]:
            del self._frames[key]
        self._backends.pop(file_id, None)

    def close(self) -> None:
        self.flush_all()
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        self._frames.clear()

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
            "dirty_evictions": self.stats.dirty_evictions,
        }
