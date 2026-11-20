"""
Module: src/storage/buffer_pool.py
Layer: Layer 1 — Buffer Pool Manager & Frame Pinning
Lock Domain: BufferPoolLock (``self._lock = threading.RLock()``; acquired before DiskManagerLock)

Purpose:
    Manages a fixed-capacity pool of 8,192-byte in-memory page frames backed by
    a ``DiskManager``, enforcing pin-count eviction protection, dirty page
    write-back, pluggable replacement policies (``LRUReplacer`` /
    ``FIFOReplacer``), and hit/miss/eviction telemetry.

Key Invariants:
    1. Lock acquisition strictly follows ``BufferPoolLock -> DiskManagerLock``;
       no ``DiskManager`` call ever re-enters ``BufferPoolManager``.
    2. A frame with ``pin_count > 0`` is never selected for eviction; when all
       frames are pinned on a cache miss or ``new_page()``,
       ``BufferPoolFullError`` is raised before any disk read or allocation.
    3. Failed disk I/O during miss handling or page allocation never leaks a
       frame or drops dirty/resident metadata.
    4. Every frame's ``data`` buffer stays fixed at ``PAGE_SIZE`` (8,192 bytes)
       so zero-copy ``memoryview`` slices returned to callers remain valid.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Any

from src.common.config import DEFAULT_BUFFER_POOL_SIZE, PAGE_SIZE
from src.common.exceptions import BufferPoolFullError, StorageError
from src.storage.disk_manager import DiskManager
from src.storage.replacer import LRUReplacer, ReplacementPolicy

_ZERO_PAGE: bytes = bytes(PAGE_SIZE)


def _validate_page_id_type(page_id: int) -> None:
    """Validate that ``page_id`` is a non-bool integer."""
    if isinstance(page_id, bool) or not isinstance(page_id, int):
        raise StorageError(f"Page ID must be an int, got {type(page_id).__name__}")


@dataclass
class FrameHeader:
    """In-memory descriptor and 8,192-byte page buffer for one buffer pool frame."""

    frame_id: int
    page_id: int | None = None
    pin_count: int = 0
    is_dirty: bool = False
    data: bytearray = field(default_factory=lambda: bytearray(PAGE_SIZE))

    def reset(self) -> None:
        """Clear page metadata and zero-fill the 8,192-byte frame buffer in place."""
        self.page_id = None
        self.pin_count = 0
        self.is_dirty = False
        # In-place slice overwrite preserves buffer size if any memoryview still exists.
        self.data[:] = _ZERO_PAGE


class BufferPoolManager:
    """Thread-safe fixed-size page cache with pin protection and dirty write-back.

    Frame Lifecycle State Machine:
    ```text
                       new_page() / fetch_page() [MISS]
      +------------+ --------------------------------------> +-------------------+
      | FREE FRAME |                                         |  PINNED IN POOL   |
      +------------+ <-------------------------------------- +-------------------+
                        delete_page() [pin_count == 0]        |   ^           |
                                                              |   |           |
                                              fetch_page()    |   |           | unpin_page()
                                              [HIT: pin++]    |   |           | [pin-- > 0]
                                                              +---+           |
                                                                              | unpin_page()
                                                                              | [pin == 0]
                                                                              v
      +------------+  victim() selected & is_dirty == True   +-------------------+
      | DISK WRITE | <-------------------------------------- | UNPINNED IN       |
      | (pwrite)   | --------------------------------------> | REPLACER          |
      +------------+  reassigned to new page_id (pin = 1)    | (LRU or FIFO)     |
                                                             +-------------------+
    ```
    """

    def __init__(
        self,
        disk_manager: DiskManager,
        pool_size: int = DEFAULT_BUFFER_POOL_SIZE,
        replacer: ReplacementPolicy | None = None,
    ) -> None:
        """Initialize a buffer pool backed by ``disk_manager``.

        Args:
            disk_manager: Physical 8,192-byte page file manager.
            pool_size: Number of resident 8,192-byte frames (must be ``>= 1``).
            replacer: Optional eviction policy; defaults to ``LRUReplacer(pool_size)``.

        Raises:
            TypeError: If ``pool_size`` is not an int or ``replacer`` is not a
                ``ReplacementPolicy`` instance.
            ValueError: If ``pool_size < 1`` or ``replacer`` has fewer frames
                than ``pool_size``.

        Complexity:
            ``O(pool_size * PAGE_SIZE)`` memory allocation for frame buffers.
        """
        if isinstance(pool_size, bool) or not isinstance(pool_size, int):
            raise TypeError(f"pool_size must be an int, got {type(pool_size).__name__}")
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")

        if replacer is None:
            resolved_replacer: ReplacementPolicy = LRUReplacer(pool_size)
        else:
            if not isinstance(replacer, ReplacementPolicy):
                raise TypeError(
                    f"replacer must be a ReplacementPolicy, got {type(replacer).__name__}"
                )
            if hasattr(replacer, "num_frames") and replacer.num_frames < pool_size:
                raise ValueError(
                    f"replacer num_frames ({replacer.num_frames}) < pool_size ({pool_size})"
                )
            resolved_replacer = replacer

        self._disk_manager: DiskManager = disk_manager
        self.pool_size: int = pool_size
        self._replacer: ReplacementPolicy = resolved_replacer
        self._lock: threading.RLock = threading.RLock()

        self._frames: list[FrameHeader] = [
            FrameHeader(frame_id=i) for i in range(pool_size)
        ]
        self._free_list: deque[int] = deque(range(pool_size))
        self._page_table: dict[int, int] = {}

        self.hit_count: int = 0
        self.miss_count: int = 0
        self.eviction_write_count: int = 0

    @property
    def disk_manager(self) -> DiskManager:
        """Return the underlying ``DiskManager`` instance."""
        return self._disk_manager

    def _flush_frame_if_dirty(self, frame: FrameHeader) -> bool:
        """Write ``frame`` to disk if dirty and clear its dirty flag.

        Returns:
            ``True`` if a disk write occurred, else ``False``.
        """
        if frame.page_id is None or not frame.is_dirty:
            return False
        # M4 ARIES WAL hook: enforce flushed_lsn >= PageHeader.get_page_lsn(frame.data) here.
        self._disk_manager.write_page(frame.page_id, frame.data)
        frame.is_dirty = False
        return True

    def _install_page(
        self,
        frame_id: int,
        page_id: int,
        payload: bytes | bytearray,
        *,
        is_dirty: bool,
    ) -> memoryview:
        """Bind ``page_id`` and ``payload`` into ``frame_id`` with ``pin_count == 1``."""
        frame = self._frames[frame_id]
        if frame.page_id is not None:
            self._page_table.pop(frame.page_id, None)
        frame.data[:] = payload
        frame.page_id = page_id
        frame.pin_count = 1
        frame.is_dirty = is_dirty
        self._page_table[page_id] = frame_id
        # Defensive pin in case a custom replacer still tracks the selected frame.
        self._replacer.pin(frame_id)
        return memoryview(frame.data)

    def _obtain_frame_for_replacement(self) -> tuple[int, bool]:
        """Return an available ``(frame_id, from_free_list)`` under ``self._lock``.

        Flushes dirty victim contents to disk before returning a victim frame,
        keeping the victim's ``page_table`` mapping intact until the caller
        completes its subsequent disk read or page allocation.
        """
        if self._free_list:
            return self._free_list.popleft(), True

        victim_id = self._replacer.victim()
        if victim_id is None:
            raise BufferPoolFullError(
                f"Buffer pool full: all {self.pool_size} frames are pinned"
            )

        frame = self._frames[victim_id]
        if frame.pin_count != 0:
            raise BufferPoolFullError(
                f"Replacer selected pinned frame {victim_id} (pin_count={frame.pin_count})"
            )

        try:
            flushed = self._flush_frame_if_dirty(frame)
        except Exception:
            # Re-register in replacer so a failed write-back never strands the frame.
            self._replacer.unpin(victim_id)
            raise
        if flushed:
            self.eviction_write_count += 1
        return victim_id, False

    def _restore_unused_frame(self, frame_id: int, from_free_list: bool) -> None:
        """Restore a frame after a failed disk read or page allocation."""
        frame = self._frames[frame_id]
        if from_free_list or frame.page_id is None:
            self._free_list.appendleft(frame_id)
        else:
            # The previous page's buffer and page_table entry are still intact.
            self._replacer.unpin(frame_id)

    def fetch_page(self, page_id: int) -> memoryview:
        """Pin and return a writable 8,192-byte ``memoryview`` for ``page_id``.

        Args:
            page_id: Zero-based physical page ID to fetch from pool or disk.

        Returns:
            Writable ``memoryview`` backed directly by the resident frame buffer.

        Raises:
            BufferPoolFullError: On a cache miss when every frame in the pool is
                pinned (raised before reading from disk).
            StorageError: If ``page_id`` is invalid, out of range, or disk I/O fails.
            PageCorruptedError: If disk checksum verification fails on load.

        Complexity:
            $O(1)$ on cache hit; $O(1)$ plus 1–2 ``PAGE_SIZE`` disk I/Os on miss.
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            _validate_page_id_type(page_id)
            frame_id = self._page_table.get(page_id)
            if frame_id is not None:
                self.hit_count += 1
                frame = self._frames[frame_id]
                frame.pin_count += 1
                self._replacer.pin(frame_id)
                return memoryview(frame.data)

            self.miss_count += 1
            frame_id, from_free_list = self._obtain_frame_for_replacement()

            try:
                raw_page = self._disk_manager.read_page(page_id)
                if len(raw_page) != PAGE_SIZE:
                    raise StorageError(
                        f"Expected {PAGE_SIZE}-byte page from DiskManager, got {len(raw_page)}"
                    )
            except Exception:
                self._restore_unused_frame(frame_id, from_free_list)
                raise

            return self._install_page(frame_id, page_id, raw_page, is_dirty=False)

    def unpin_page(self, page_id: int, is_dirty: bool = False) -> bool:
        """Decrement ``page_id``'s pin count and mark it dirty if modified.

        Args:
            page_id: Physical page ID to unpin.
            is_dirty: ``True`` if the caller mutated the page buffer while pinned.

        Returns:
            ``True`` if ``page_id`` was resident with ``pin_count > 0`` and was
            unpinned; ``False`` if absent or already at ``pin_count == 0``.

        Complexity:
            $O(1)$ time and space; thread-safe under ``self._lock``.
        """
        with self._lock:
            frame_id = self._page_table.get(page_id)
            if frame_id is None:
                return False
            frame = self._frames[frame_id]
            if frame.pin_count <= 0:
                return False

            frame.pin_count -= 1
            if is_dirty:
                frame.is_dirty = True
            if frame.pin_count == 0:
                self._replacer.unpin(frame_id)
            return True

    def new_page(self) -> tuple[int, memoryview]:
        """Allocate a new zeroed page on disk and pin it in the buffer pool.

        Secures a buffer frame (evicting and flushing a dirty victim if needed)
        before allocating on disk so ``BufferPoolFullError`` never leaks an
        orphan page in the FSM.

        Returns:
            Tuple ``(page_id, memoryview)`` with ``pin_count == 1`` and
            ``is_dirty == True``.

        Raises:
            BufferPoolFullError: If no free or unpinned frame is available.
            StorageError: If dirty victim write-back or FSM page allocation fails.

        Complexity:
            $O(1)$ pool operations plus FSM allocation (and optional dirty write-back).
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            frame_id, from_free_list = self._obtain_frame_for_replacement()

            try:
                page_id = self._disk_manager.allocate_page()
            except Exception:
                self._restore_unused_frame(frame_id, from_free_list)
                raise

            return page_id, self._install_page(
                frame_id, page_id, _ZERO_PAGE, is_dirty=True
            )

    def flush_page(self, page_id: int, sync: bool = False) -> bool:
        """Write a resident dirty page back to disk and clear its dirty flag.

        Args:
            page_id: Physical page ID to flush.
            sync: When ``True``, also call ``disk_manager.sync()`` after flushing.

        Returns:
            ``True`` if ``page_id`` is resident in the pool, else ``False``.

        Raises:
            StorageError: If writing to disk or syncing fails.

        Complexity:
            ``O(PAGE_SIZE)`` when dirty; $O(1)$ when clean or non-resident.
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            frame_id = self._page_table.get(page_id)
            if frame_id is None:
                return False
            self._flush_frame_if_dirty(self._frames[frame_id])
            if sync:
                self._disk_manager.sync()
            return True

    def flush_all_pages(self) -> None:
        """Flush all dirty resident pages to disk and issue ``disk_manager.sync()``.

        Raises:
            StorageError: If any page write or ``fsync`` fails.

        Complexity:
            ``O(pool_size * PAGE_SIZE)`` worst case plus one ``os.fsync``.
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            for frame_id in list(self._page_table.values()):
                self._flush_frame_if_dirty(self._frames[frame_id])
            self._disk_manager.sync()

    def delete_page(self, page_id: int) -> bool:
        """Evict an unpinned page from the pool and deallocate it in the FSM.

        Args:
            page_id: Physical data page ID (``>= 1``) to delete.

        Returns:
            ``False`` if ``page_id`` is currently pinned (``pin_count > 0``);
            ``True`` if deleted or not resident.

        Raises:
            StorageError: If ``page_id`` is not an int, is ``<= 0`` (e.g.,
                reserved FSM Page 0), or disk deallocation fails.

        Complexity:
            ``O(PAGE_SIZE)`` when deallocating on disk; $O(1)$ when rejected
            due to active pin. Thread-safe under ``self._lock``.
        """
        with self._lock:
            _validate_page_id_type(page_id)
            if page_id <= 0:
                if page_id == 0:
                    self._disk_manager.deallocate_page(0)
                raise StorageError(f"Invalid page_id {page_id} for deletion")

            frame_id = self._page_table.get(page_id)
            if frame_id is not None:
                frame = self._frames[frame_id]
                if frame.pin_count > 0:
                    return False
                if self._disk_manager.is_allocated(page_id):
                    self._disk_manager.deallocate_page(page_id)
                self._page_table.pop(page_id, None)
                remove_fn = getattr(self._replacer, "remove", self._replacer.pin)
                remove_fn(frame_id)
                frame.reset()
                self._free_list.append(frame_id)
                return True

            if self._disk_manager.is_allocated(page_id):
                self._disk_manager.deallocate_page(page_id)
            return True

    def is_resident(self, page_id: int) -> bool:
        """Return whether ``page_id`` is currently loaded in a buffer frame.

        Args:
            page_id: Physical page ID to look up.

        Returns:
            ``True`` if ``page_id`` resides in ``page_table``, else ``False``.

        Complexity:
            $O(1)$ under ``self._lock``.
        """
        with self._lock:
            return page_id in self._page_table

    def get_stats(self) -> dict[str, Any]:
        """Return buffer pool cache hit/miss, eviction, and disk I/O telemetry.

        Returns:
            Dictionary with keys ``hit_count``, ``miss_count``, ``hit_ratio``,
            ``eviction_write_count``, ``disk_reads``, ``disk_writes``,
            ``pinned_frames``, and ``pool_size``.

        Complexity:
            ``O(pool_size)`` to count pinned frames under ``self._lock``.
        """
        with self._lock:
            total_fetches = self.hit_count + self.miss_count
            hit_ratio = (
                self.hit_count / total_fetches if total_fetches > 0 else 0.0
            )
            pinned_frames = sum(1 for frame in self._frames if frame.pin_count > 0)
            return {
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_ratio": hit_ratio,
                "eviction_write_count": self.eviction_write_count,
                "disk_reads": self._disk_manager.disk_reads,
                "disk_writes": self._disk_manager.disk_writes,
                "pinned_frames": pinned_frames,
                "pool_size": self.pool_size,
            }


__all__ = ["BufferPoolManager", "FrameHeader"]
