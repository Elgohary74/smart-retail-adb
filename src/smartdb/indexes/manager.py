"""Index selection, lifecycle management, and unified lookup/range API."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import (
    AUTO_DEFAULT_READS,
    AUTO_DEFAULT_WRITES,
    AUTO_WRITE_RATIO,
    LSM_COMPACTION_THRESHOLD,
    LSM_FLUSH_THRESHOLD,
    RID,
    DataType,
)
from ..storage import BufferPool, StorageManager
from .base import IndexProtocol
from .bplus import BPlusTree
from .lsm import LSMTree


@dataclass(frozen=True)
class WorkloadProfile:
    """Observed or estimated operation counts used for auto index selection."""

    reads: int = AUTO_DEFAULT_READS
    writes: int = AUTO_DEFAULT_WRITES
    range_reads: int = 0


def choose_index_type(reads: int, writes: int, range_reads: int = 0) -> str:
    """Documented workload rule: LSM for write-heavy point workloads, B+ Tree otherwise."""
    if reads < 0 or writes < 0 or range_reads < 0:
        raise ValueError("reads, writes, and range_reads cannot be negative")
    if writes == 0:
        return "BPLUS"
    if reads == 0 and range_reads == 0:
        return "LSM"
    return "LSM" if writes > reads * AUTO_WRITE_RATIO and range_reads == 0 else "BPLUS"


class IndexManager:
    """Manages page-backed B+ Tree and SSTable LSM Tree indexes."""

    def __init__(
        self,
        directory: str | Path,
        storage_manager: StorageManager,
        pool: BufferPool,
        lsm_flush_threshold: int = LSM_FLUSH_THRESHOLD,
        lsm_compaction_threshold: int = LSM_COMPACTION_THRESHOLD,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.storage_manager = storage_manager
        self.pool = pool
        self.lsm_flush_threshold = lsm_flush_threshold
        self.lsm_compaction_threshold = lsm_compaction_threshold
        self.indexes: dict[tuple[str, str], tuple[str, IndexProtocol]] = {}

    def drop(self, table: str, field: str) -> None:
        """Destroy and remove any active or on-disk index state for (table, field)."""
        existing = self.indexes.pop((table, field), None)
        if existing is not None:
            existing[1].destroy()
        self.storage_manager.drop_file(f"index:{table}.{field}:bplus")
        shutil.rmtree(self.directory / table / field, ignore_errors=True)

    @staticmethod
    def _resolve_kind(index_type: str, workload: WorkloadProfile) -> str:
        if index_type.lower() == "auto":
            return choose_index_type(workload.reads, workload.writes, workload.range_reads)
        kind = index_type.upper()
        if kind in {"B+TREE", "BPLUS", "BPTREE"}:
            return "BPLUS"
        if kind == "LSM":
            return "LSM"
        raise ValueError("index_type must be 'auto', 'BPLUS', or 'LSM'")

    def create(
        self,
        table: str,
        field: str,
        index_type: str = "auto",
        key_dtype: DataType = DataType.ANY,
        workload: WorkloadProfile | None = None,
        replace: bool = True,
    ) -> tuple[str, IndexProtocol]:
        """Create or open a B+ Tree or LSM index according to workload or explicit type."""
        kind = self._resolve_kind(index_type, workload or WorkloadProfile())
        if replace:
            self.drop(table, field)
        name = f"{table}.{field}"
        if kind == "BPLUS":
            index: IndexProtocol = BPlusTree(
                storage_manager=self.storage_manager, pool=self.pool, name=name, key_dtype=key_dtype
            )
        else:
            index = LSMTree(
                self.directory / table / field,
                flush_threshold=self.lsm_flush_threshold,
                compaction_threshold=self.lsm_compaction_threshold,
                storage_manager=self.storage_manager,
                pool=self.pool,
                name_prefix=name,
                key_dtype=key_dtype,
            )
        self.indexes[(table, field)] = (kind, index)
        return kind, index

    def add(self, table: str, field: str, key: Any, rid: RID) -> None:
        """Insert (key, RID) into the index on (table, field) if present."""
        item = self.indexes.get((table, field))
        if item is not None:
            item[1].insert(key, rid)

    def remove(self, table: str, field: str, key: Any, rid: RID | None = None) -> None:
        """Delete matching RID (or all RIDs) for key from the index if present."""
        item = self.indexes.get((table, field))
        if item is not None:
            item[1].delete(key, rid)

    def search(self, table: str, field: str, key: Any) -> list[RID] | None:
        """Lookup RIDs matching key; returns None if no index exists, [] on type mismatch."""
        item = self.indexes.get((table, field))
        if item is None:
            return None
        try:
            return item[1].search(key)
        except TypeError:
            return []

    def range_search(
        self,
        table: str,
        field: str,
        low: Any | None = None,
        high: Any | None = None,
        include_low: bool = True,
        include_high: bool = True,
    ) -> list[tuple[Any, RID]] | None:
        """Lookup ordered (key, RID) pairs in bounds; returns None if unindexed, [] on type mismatch."""
        item = self.indexes.get((table, field))
        if item is None:
            return None
        try:
            return item[1].range_search(low=low, high=high, include_low=include_low, include_high=include_high)
        except TypeError:
            return []

    def kind(self, table: str, field: str) -> str | None:
        """Return 'BPLUS' or 'LSM' for an active index, or None if unindexed."""
        item = self.indexes.get((table, field))
        return item[0] if item else None

    def disk_bytes(self, table: str, field: str) -> int:
        """Return physical bytes used by the index on (table, field)."""
        item = self.indexes.get((table, field))
        return item[1].disk_bytes() if item is not None else 0

    def flush_all(self) -> None:
        """Flush all active indexes to backend storage."""
        for _, index in self.indexes.values():
            index.flush()

    def close(self) -> None:
        """Flush and close all active indexes."""
        for _, index in self.indexes.values():
            index.close()
