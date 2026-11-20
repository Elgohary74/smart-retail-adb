"""
Module: src/storage/replacer.py
Layer: Layer 1 — Buffer Pool Replacement Policy Interface & Implementations
Lock Domain: BufferPoolLock (caller ``BufferPoolManager`` synchronizes access)

Purpose:
    Defines the abstract ``ReplacementPolicy`` interface and concrete $O(1)$
    (amortized) eviction policies: ``LRUReplacer`` (Least Recently Used) and
    ``FIFOReplacer`` (First-In, First-Out).

Key Invariants:
    1. Only frames with ``pin_count == 0`` may be chosen by ``victim()``; both
       policies reject out-of-range frame IDs outside ``[0, num_frames)``.
    2. ``LRUReplacer.unpin(frame_id)`` moves or inserts ``frame_id`` at the
       most-recently-used tail, whereas ``victim()`` pops from the LRU head.
    3. ``FIFOReplacer`` preserves each resident frame's original arrival order
       across temporary ``pin()``/``unpin()`` cycles on cache hits while still
       excluding currently pinned frames from ``victim()`` and ``size()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict


def _validate_num_frames(num_frames: int) -> int:
    """Validate that a replacer capacity is a positive non-bool integer."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError(f"num_frames must be an int, got {type(num_frames).__name__}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    return num_frames


def _validate_frame_id(frame_id: int, num_frames: int) -> None:
    """Validate that ``frame_id`` is a non-bool integer in ``[0, num_frames)``."""
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise TypeError(f"frame_id must be an int, got {type(frame_id).__name__}")
    if frame_id < 0 or frame_id >= num_frames:
        raise ValueError(f"frame_id {frame_id} out of range [0, {num_frames})")


class ReplacementPolicy(ABC):
    """Abstract buffer pool frame eviction policy."""

    @abstractmethod
    def victim(self) -> int | None:
        """Select and remove the next eviction candidate frame ID.

        Returns:
            Frame index to evict, or ``None`` if no unpinned frames exist.
        """

    @abstractmethod
    def pin(self, frame_id: int) -> None:
        """Mark ``frame_id`` as pinned so it cannot be chosen as a victim.

        Args:
            frame_id: Frame index currently pinned by a caller.
        """

    @abstractmethod
    def unpin(self, frame_id: int) -> None:
        """Record ``frame_id`` as eligible for eviction once its pin count hits zero.

        Args:
            frame_id: Frame index whose pin count reached zero.
        """

    @abstractmethod
    def size(self) -> int:
        """Return the number of frames currently eligible for eviction.

        Returns:
            Count of unpinned frames tracked by this policy.
        """

    def remove(self, frame_id: int) -> None:
        """Remove ``frame_id`` permanently when its page is deleted from the pool.

        Defaults to ``pin(frame_id)`` so subclasses that retain arrival state
        across temporary ``pin()``/``unpin()`` cycles (such as ``FIFOReplacer``)
        can override it to clear stale history without breaking custom policies.

        Args:
            frame_id: Frame index returned to the buffer pool free list.
        """
        self.pin(frame_id)


class LRUReplacer(ReplacementPolicy):
    """Least-Recently-Used frame replacement policy backed by ``OrderedDict``."""

    def __init__(self, num_frames: int) -> None:
        """Initialize an empty LRU replacer managing ``num_frames`` frame slots.

        Args:
            num_frames: Total number of frames in the owning buffer pool (``>= 1``).

        Raises:
            TypeError: If ``num_frames`` is not an integer.
            ValueError: If ``num_frames < 1``.

        Complexity:
            $O(1)$ initialization.
        """
        self._num_frames: int = _validate_num_frames(num_frames)
        self._od: OrderedDict[int, None] = OrderedDict()

    @property
    def num_frames(self) -> int:
        """Return the configured number of frame slots."""
        return self._num_frames

    def victim(self) -> int | None:
        """Pop and return the least-recently-unpinned frame ID, or ``None`` if empty.

        Returns:
            LRU frame ID, or ``None`` when all frames are pinned or free.

        Complexity:
            $O(1)$ time and space; caller must hold ``BufferPoolLock``.
        """
        if not self._od:
            return None
        frame_id, _ = self._od.popitem(last=False)
        return frame_id

    def pin(self, frame_id: int) -> None:
        """Remove ``frame_id`` from the evictable set if currently tracked.

        Args:
            frame_id: Frame index in ``[0, num_frames)`` being pinned.

        Raises:
            TypeError: If ``frame_id`` is not an integer.
            ValueError: If ``frame_id`` lies outside ``[0, num_frames)``.

        Complexity:
            $O(1)$ time and space; caller must hold ``BufferPoolLock``.
        """
        _validate_frame_id(frame_id, self._num_frames)
        self._od.pop(frame_id, None)

    def unpin(self, frame_id: int) -> None:
        """Move or insert ``frame_id`` at the most-recently-used end.

        Args:
            frame_id: Frame index in ``[0, num_frames)`` whose pin count hit zero.

        Raises:
            TypeError: If ``frame_id`` is not an integer.
            ValueError: If ``frame_id`` lies outside ``[0, num_frames)``.

        Complexity:
            $O(1)$ time and space; caller must hold ``BufferPoolLock``.
        """
        _validate_frame_id(frame_id, self._num_frames)
        if frame_id in self._od:
            self._od.move_to_end(frame_id, last=True)
        else:
            self._od[frame_id] = None

    def size(self) -> int:
        """Return the count of unpinned frames currently eligible for eviction.

        Returns:
            Number of unpinned frames in the LRU queue.

        Complexity:
            $O(1)$ time and space.
        """
        return len(self._od)


class FIFOReplacer(ReplacementPolicy):
    """First-In, First-Out frame replacement policy preserving original arrival order."""

    def __init__(self, num_frames: int) -> None:
        """Initialize an empty FIFO replacer managing ``num_frames`` frame slots.

        Args:
            num_frames: Total number of frames in the owning buffer pool (``>= 1``).

        Raises:
            TypeError: If ``num_frames`` is not an integer.
            ValueError: If ``num_frames < 1``.

        Complexity:
            $O(1)$ initialization.
        """
        self._num_frames: int = _validate_num_frames(num_frames)
        self._od: OrderedDict[int, None] = OrderedDict()
        # Keep _pinned <= _od.keys() so temporary pin/unpin hits preserve original FIFO order.
        self._pinned: set[int] = set()

    @property
    def num_frames(self) -> int:
        """Return the configured number of frame slots."""
        return self._num_frames

    def victim(self) -> int | None:
        """Pop and return the oldest-inserted unpinned frame ID, or ``None``.

        Returns:
            Oldest unpinned frame ID in FIFO order, or ``None`` if none exist.

        Complexity:
            $O(1)$ when the oldest frame is unpinned; $O(P + 1)$ worst-case
            where $P$ is the number of currently pinned resident frames. Caller
            must hold ``BufferPoolLock``.
        """
        if len(self._od) == len(self._pinned):
            return None
        for frame_id in self._od:
            if frame_id not in self._pinned:
                del self._od[frame_id]
                return frame_id
        return None

    def pin(self, frame_id: int) -> None:
        """Mark ``frame_id`` pinned while preserving its FIFO position if resident.

        Args:
            frame_id: Frame index in ``[0, num_frames)`` being pinned.

        Raises:
            TypeError: If ``frame_id`` is not an integer.
            ValueError: If ``frame_id`` lies outside ``[0, num_frames)``.

        Complexity:
            $O(1)$ time and space; caller must hold ``BufferPoolLock``.
        """
        _validate_frame_id(frame_id, self._num_frames)
        if frame_id in self._od:
            self._pinned.add(frame_id)

    def unpin(self, frame_id: int) -> None:
        """Make ``frame_id`` eligible for eviction without refreshing its FIFO position.

        If ``frame_id`` is already recorded in the FIFO queue from an earlier
        insertion and was temporarily pinned on a cache hit, it simply becomes
        evictable again in its original slot. Only newly loaded frames are
        appended to the FIFO tail.

        Args:
            frame_id: Frame index in ``[0, num_frames)`` whose pin count hit zero.

        Raises:
            TypeError: If ``frame_id`` is not an integer.
            ValueError: If ``frame_id`` lies outside ``[0, num_frames)``.

        Complexity:
            $O(1)$ time and space; caller must hold ``BufferPoolLock``.
        """
        _validate_frame_id(frame_id, self._num_frames)
        if frame_id in self._od:
            self._pinned.discard(frame_id)
        else:
            self._od[frame_id] = None

    def remove(self, frame_id: int) -> None:
        """Remove ``frame_id`` from FIFO history when its page is deleted or freed.

        Args:
            frame_id: Frame index in ``[0, num_frames)`` returned to the free list.

        Raises:
            TypeError: If ``frame_id`` is not an integer.
            ValueError: If ``frame_id`` lies outside ``[0, num_frames)``.

        Complexity:
            $O(1)$ time and space; caller must hold ``BufferPoolLock``.
        """
        _validate_frame_id(frame_id, self._num_frames)
        self._pinned.discard(frame_id)
        self._od.pop(frame_id, None)

    def size(self) -> int:
        """Return the count of unpinned frames currently eligible for eviction.

        Returns:
            Number of unpinned frames tracked by this FIFO policy.

        Complexity:
            $O(1)$ time and space.
        """
        return len(self._od) - len(self._pinned)


__all__ = ["FIFOReplacer", "LRUReplacer", "ReplacementPolicy"]
