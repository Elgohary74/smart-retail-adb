"""Index selection and common interface for B+ Tree and LSM Tree indexes."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .bplus_tree import BPlusTree
from .lsm_tree import LSMTree


def choose_index_type(reads: int, writes: int) -> str:
    """Heuristic: choose LSM for write-heavy workloads; B+ Tree otherwise."""
    if reads < 0 or writes < 0:
        raise ValueError("reads and writes cannot be negative")
    return "LSM" if writes > max(1, reads) * 2 else "BPLUS"


class IndexManager:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.indexes: dict[tuple[str, str], tuple[str, Any]] = {}

    def create(self, table: str, field: str, index_type: str = "auto",
               reads: int = 100, writes: int = 10):
        key = (table, field)
        if index_type.lower() == "auto":
            kind = choose_index_type(reads, writes)
        else:
            kind = index_type.upper()
        if kind in {"B+TREE", "BPLUS", "BPTREE"}:
            index = BPlusTree()
            kind = "BPLUS"
        elif kind == "LSM":
            index = LSMTree(self.directory / table / field)
        else:
            raise ValueError("index_type must be 'auto', 'BPLUS', or 'LSM'")
        self.indexes[key] = (kind, index)
        return kind, index

    def add(self, table: str, field: str, value: Any, locator: tuple[int, int]) -> None:
        item = self.indexes.get((table, field))
        if item is not None:
            item[1].insert(value, locator)

    def search(self, table: str, field: str, value: Any) -> list[tuple[int, int]] | None:
        item = self.indexes.get((table, field))
        if item is None:
            return None
        return item[1].search(value)

    def kind(self, table: str, field: str) -> str | None:
        item = self.indexes.get((table, field))
        return item[0] if item else None

    def close(self) -> None:
        for _, index in self.indexes.values():
            close = getattr(index, "close", None)
            if close:
                close()
