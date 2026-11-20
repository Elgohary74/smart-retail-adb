"""
Module: src/storage/replacer.py
Layer: Layer 1 — Buffer Pool Replacement Policy Interface
Lock Domain: BufferPoolLock (caller ``BufferPoolManager`` synchronizes access)

Purpose:
    Defines the abstract ``ReplacementPolicy`` interface implemented by
    concrete eviction policies (``LRUReplacer``, ``FIFOReplacer``).

Key Invariants:
    1. Only frames with ``pin_count == 0`` may reside in the replacer.
    2. ``victim()`` returns ``None`` when all frames in the pool are pinned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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


__all__ = ["ReplacementPolicy"]
