"""
Module: src/index/base_index.py
Layer: Layer 2 — Unified Index Interface
Lock Domain: IndexLock (implemented by concrete subclasses)

Purpose:
    Defines the ``BaseIndex`` abstract base class shared identically by
    ``BPlusTreeIndex`` and ``LSMTreeIndex`` so ``SimpleExecutor`` and benchmarks
    can swap access methods transparently.

Key Invariants:
    1. ``get(key)`` returns matching ``RID`` values in deterministic ascending
       order, or ``[]`` when absent.
    2. ``range_scan(low_key, high_key, inclusive)`` returns ``(key, RID)`` pairs
       in ascending composite key order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.common.types import RID


class BaseIndex(ABC):
    """Abstract secondary and primary index contract for M2 access methods."""

    @abstractmethod
    def insert(self, key: Any, rid: RID) -> None:
        """Insert a ``(key, rid)`` entry into the index.

        Args:
            key: Logical key value matching the index ``DataType``.
            rid: Physical ``RID`` locating the tuple in heap storage.

        Raises:
            DuplicateKeyError: If this index enforces uniqueness and ``key`` exists.
        """

    @abstractmethod
    def get(self, key: Any) -> list[RID]:
        """Look up all ``RID`` entries matching ``key``.

        Args:
            key: Logical key value to locate.

        Returns:
            Sorted list of matching ``RID`` instances (empty if not found).
        """

    @abstractmethod
    def range_scan(
        self, low_key: Any, high_key: Any, inclusive: bool = True
    ) -> list[tuple[Any, RID]]:
        """Return all indexed ``(key, RID)`` entries between ``low_key`` and ``high_key``.

        Args:
            low_key: Lower bound key.
            high_key: Upper bound key.
            inclusive: Include endpoints when ``True``; exclude both bounds when ``False``.

        Returns:
            Sorted list of ``(decoded_key, RID)`` tuples within the requested range.
        """

    @abstractmethod
    def delete(self, key: Any, rid: RID | None = None) -> bool:
        """Remove entries matching ``key`` (and optionally a specific ``rid``).

        Args:
            key: Logical key value to delete.
            rid: Optional specific ``RID`` to remove for non-unique indexes; when
                ``None``, remove all entries matching ``key``.

        Returns:
            ``True`` if at least one matching entry was deleted, ``False`` otherwise.
        """

    @abstractmethod
    def flush(self) -> None:
        """Persist all dirty index buffers or in-memory tables durably to disk."""

    @abstractmethod
    def close(self) -> None:
        """Flush buffered state and close underlying file descriptors cleanly."""

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return index telemetry counters (height, entry count, I/O, or Bloom stats)."""


__all__ = ["BaseIndex"]
