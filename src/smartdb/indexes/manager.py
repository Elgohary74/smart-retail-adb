"""Index selection, lifecycle management, and unified lookup/range API."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..core import RID, DataType
from ..storage import BufferPool, StorageManager
from .bplus import BPlusTree
from .lsm import LSMTree


def choose_index_type(reads: int, writes: int, range_reads: int = 0) -> str:
    """Documented workload rule: LSM for write-heavy point workloads, B+ Tree otherwise."""
    if reads < 0 or writes < 0 or range_reads < 0:
        raise ValueError("reads, writes, and range_reads cannot be negative")
    if writes == 0:
        return "BPLUS"
    if reads == 0 and range_reads == 0:
        return "LSM"
    return "LSM" if writes > reads * 2 and range_reads == 0 else "BPLUS"


class IndexManager:
    """Manages page-backed B+ Tree and SSTable LSM Tree indexes."""

    def __init__(self, directory: str | Path, storage_manager: StorageManager, pool: BufferPool):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.storage_manager = storage_manager
        self.pool = pool
        self.indexes: dict[tuple[str, str], tuple[str, BPlusTree | LSMTree]] = {}

    def drop(self, table: str, field: str) -> None:
        key = (table, field)
        existing = self.indexes.pop(key, None)
        if existing is not None:
            kind, idx = existing
            if kind == "BPLUS":
                assert isinstance(idx, BPlusTree)
                self.pool.unregister_file(idx._file_id, drop_dirty=True)
                self.storage_manager.drop_file(f"index:{idx.name}:bplus")
            else:
                assert isinstance(idx, LSMTree)
                if not idx._wal_file.closed:
                    idx._wal_file.close()
                for run in idx.runs:
                    self.pool.unregister_file(run.file_id, drop_dirty=True)
                    self.storage_manager.drop_file(run.logical_name)
        # Clean any leftover files on disk for both index types
        safe = f"{table}.{field}".replace(":", "_")
        self.storage_manager.drop_file(f"index:{table}.{field}:bplus")
        (self.directory / table / field).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.directory / table / field, ignore_errors=True)
        bplus_path = self.storage_manager.root / "indexes" / f"{safe}.bplus.pages"
        bplus_path.unlink(missing_ok=True)
        bplus_obj = bplus_path.with_suffix(bplus_path.suffix + ".objects")
        if bplus_obj.exists():
            shutil.rmtree(bplus_obj, ignore_errors=True)

    def create(
        self,
        table: str,
        field: str,
        index_type: str = "auto",
        key_dtype: DataType = DataType.ANY,
        reads: int = 100,
        writes: int = 10,
        range_reads: int = 0,
        replace: bool = True,
    ) -> tuple[str, BPlusTree | LSMTree]:
        if index_type.lower() == "auto":
            kind = choose_index_type(reads, writes, range_reads)
        else:
            kind = index_type.upper()
            if kind in {"B+TREE", "BPLUS", "BPTREE"}:
                kind = "BPLUS"
            elif kind != "LSM":
                raise ValueError("index_type must be 'auto', 'BPLUS', or 'LSM'")

        if replace:
            self.drop(table, field)

        name = f"{table}.{field}"
        if kind == "BPLUS":
            index: BPlusTree | LSMTree = BPlusTree(
                storage_manager=self.storage_manager,
                pool=self.pool,
                name=name,
                key_dtype=key_dtype,
            )
        else:
            index = LSMTree(
                self.directory / table / field,
                storage_manager=self.storage_manager,
                pool=self.pool,
                name_prefix=name,
                key_dtype=key_dtype,
            )
        self.indexes[(table, field)] = (kind, index)
        return kind, index

    def add(self, table: str, field: str, key: Any, rid: RID) -> None:
        item = self.indexes.get((table, field))
        if item is not None:
            item[1].insert(key, rid)

    def remove(self, table: str, field: str, key: Any, rid: RID | None = None) -> None:
        item = self.indexes.get((table, field))
        if item is not None:
            item[1].delete(key, rid)

    def search(self, table: str, field: str, key: Any) -> list[RID] | None:
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
        item = self.indexes.get((table, field))
        if item is None:
            return None
        return item[1].range_search(low=low, high=high, include_low=include_low, include_high=include_high)

    def kind(self, table: str, field: str) -> str | None:
        item = self.indexes.get((table, field))
        return item[0] if item else None

    def disk_bytes(self, table: str, field: str) -> int:
        item = self.indexes.get((table, field))
        return item[1].disk_bytes() if item is not None else 0

    def flush_all(self) -> None:
        for _, index in self.indexes.values():
            index.flush()

    def close(self) -> None:
        for _, index in self.indexes.values():
            index.close()
