"""Disk-backed SSTable LSM Tree with Bloom filters, fence pointers, WAL, and atomic compaction."""
from __future__ import annotations

import json
from bisect import bisect_left
from operator import itemgetter
from pathlib import Path
from typing import Any

from ...core import (
    LSM_COMPACTION_THRESHOLD,
    LSM_FLUSH_THRESHOLD,
    RID,
    DataType,
    atomic_write_json,
    decode_key,
    encode_key,
)
from ...storage import BufferPool, StorageManager
from ..base import key_in_range
from .bloom import BloomFilter
from .sstable import OpList, SSTableRun, read_sstable_meta, read_sstable_page, write_sstable_run
from .wal import LSMWal

__all__ = ["BloomFilter", "LSMTree"]
_KEY_0 = itemgetter(0)


def _resolve_ops(ops: OpList) -> list[RID]:
    live: dict[RID, bool] = {}
    for rid, _, is_deleted in sorted(ops, key=lambda o: o[1]):
        if rid is None and is_deleted:
            live.clear()
        elif rid is not None:
            live[rid] = not is_deleted
    return [rid for rid, ok in live.items() if ok]


def _start_page_for_key(run: SSTableRun, key_bytes: bytes | None) -> int:
    if key_bytes is None or not run.fence:
        return 0
    idx = bisect_left(run.fence_keys, key_bytes)
    if idx < len(run.fence) and run.fence_keys[idx] == key_bytes:
        return run.fence[idx][1]
    return run.fence[max(0, idx - 1)][1]


class LSMTree:
    """Write-oriented LSM index storing SSTables in StorageManager/BufferPool pages."""

    def __init__(
        self,
        directory: str | Path,
        flush_threshold: int = LSM_FLUSH_THRESHOLD,
        compaction_threshold: int = LSM_COMPACTION_THRESHOLD,
        storage_manager: StorageManager | None = None,
        pool: BufferPool | None = None,
        name_prefix: str = "lsm",
        key_dtype: DataType = DataType.ANY,
    ):
        if flush_threshold < 1 or compaction_threshold < 2:
            raise ValueError("flush_threshold must be >= 1 and compaction_threshold >= 2")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_threshold, self.compaction_threshold = flush_threshold, compaction_threshold
        self.name_prefix = name_prefix.replace(":", "_")
        self.key_dtype = key_dtype

        self._owns_storage = storage_manager is None or pool is None
        if self._owns_storage:
            storage_manager = StorageManager(self.directory)
            pool = BufferPool(capacity=32, storage_manager=storage_manager)
        if storage_manager is None or pool is None:
            raise RuntimeError("LSMTree requires both storage_manager and pool")
        self.storage_manager, self.pool = storage_manager, pool

        self.manifest_path = self.directory / f"{self.name_prefix}_manifest.json"
        self.wal_path = self.directory / f"{self.name_prefix}_memtable.wal"
        self.runs: list[SSTableRun] = []
        self._next_run_id = 0
        self._next_seq = 1
        self.flushes = 0
        self.compactions = 0

        self._load_manifest()
        self._wal = LSMWal(self.wal_path, self.storage_manager)
        self.memtable, self.memtable_entries, max_seq = self._wal.replay()
        self._next_seq = max(self._next_seq, max_seq + 1)

    @property
    def wal(self) -> LSMWal:
        """Return the memtable write-ahead log handle."""
        return self._wal

    @property
    def run_count(self) -> int:
        """Number of immutable SSTable runs currently active."""
        return len(self.runs)

    def disk_bytes(self) -> int:
        """Total physical bytes occupied by SSTable runs and the active WAL."""
        total = self.wal_path.stat().st_size if self.wal_path.exists() else 0
        for run in self.runs:
            backend = self.storage_manager.get_backend(run.file_id)
            total += backend.page_count * backend.physical_page_size
        return total

    def _run_rel_path(self, run_id: int) -> str:
        prefix = "" if self._owns_storage else f"indexes/{self.name_prefix}/"
        return f"{prefix}{self.name_prefix}_run_{run_id:08d}.sst"

    def _save_manifest(self) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "next_run_id": self._next_run_id,
                "next_seq": self._next_seq,
                "runs": [{"run_id": r.run_id, "logical_name": r.logical_name, "meta_pid": r.data_pages} for r in self.runs],
            },
        )

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._next_run_id = data.get("next_run_id", 0)
        self._next_seq = data.get("next_seq", 1)
        for item in data.get("runs", []):
            rid, lname, meta_pid = item["run_id"], item["logical_name"], item["meta_pid"]
            fid, _ = self.storage_manager.open_file(lname, self._run_rel_path(rid))
            self.runs.append(read_sstable_meta(self.pool, rid, lname, fid, meta_pid))

    def _record_op(self, key: Any, rid: RID | None, is_deleted: bool) -> None:
        kb = encode_key(self.key_dtype, key)
        seq = self._next_seq
        self._next_seq += 1
        self.memtable.setdefault(kb, []).append((rid, seq, is_deleted))
        self.memtable_entries += 1
        self._wal.append(kb, rid, seq, is_deleted)
        if self.memtable_entries >= self.flush_threshold:
            self.flush()

    def insert(self, key: Any, rid: RID) -> None:
        """Append a (key, RID) insert into the memtable and WAL."""
        self._record_op(key, rid, False)

    def delete(self, key: Any, rid: RID | None = None) -> None:
        """Append a RID-level or key-level tombstone into the memtable and WAL."""
        self._record_op(key, rid, True)

    def _create_run(self, sorted_items: list[tuple[bytes, OpList]]) -> SSTableRun:
        run_id = self._next_run_id
        self._next_run_id += 1
        lname = f"index:{self.name_prefix}:run_{run_id:08d}"
        return write_sstable_run(
            self.storage_manager, self.pool, run_id, lname, self._run_rel_path(run_id), sorted_items
        )

    def _clear_memtable_and_commit(self) -> None:
        self.memtable.clear()
        self.memtable_entries = 0
        self._save_manifest()
        self._wal.truncate()

    def flush(self) -> None:
        """Flush the active memtable to a new sorted SSTable run and truncate WAL."""
        if not self.memtable:
            return
        run = self._create_run(sorted(self.memtable.items(), key=lambda item: item[0]))
        self.runs.append(run)
        self._clear_memtable_and_commit()
        self.flushes += 1
        if len(self.runs) >= self.compaction_threshold:
            self.compact()

    def _merge_all_ops(self) -> list[tuple[bytes, OpList]]:
        merged: dict[bytes, OpList] = {}
        for run in self.runs:
            for pid in range(run.data_pages):
                for kb, ops in read_sstable_page(self.pool, run.file_id, pid):
                    merged.setdefault(kb, []).extend(ops)
        for kb, ops in self.memtable.items():
            merged.setdefault(kb, []).extend(ops)
        final_items: list[tuple[bytes, OpList]] = []
        seq = self._next_seq
        for kb in sorted(merged):
            if live := _resolve_ops(merged[kb]):
                final_items.append((kb, [(r, seq, False) for r in live]))
        if final_items:
            self._next_seq += 1
        return final_items

    def compact(self) -> None:
        """Merge all SSTable runs and memtable entries into a single compacted run."""
        if len(self.runs) < 2 and not self.memtable:
            return
        final_items = self._merge_all_ops()
        old_runs = list(self.runs)
        self.runs = [self._create_run(final_items)] if final_items else []
        self._clear_memtable_and_commit()
        for old in old_runs:
            self.pool.unregister_file(old.file_id, drop_dirty=True)
            self.storage_manager.drop_file(old.logical_name)
        self.compactions += 1

    def search(self, key: Any) -> list[RID]:
        """Return all live RIDs indexed under key across memtable and SSTable runs."""
        kb = encode_key(self.key_dtype, key)
        ops: OpList = list(self.memtable.get(kb, []))
        for run in reversed(self.runs):
            if kb < run.min_key or kb > run.max_key or not run.bloom.might_contain(kb):
                continue
            for pid in range(_start_page_for_key(run, kb), run.data_pages):
                entries = read_sstable_page(self.pool, run.file_id, pid)
                pos = bisect_left(entries, kb, key=_KEY_0)
                while pos < len(entries) and entries[pos][0] == kb:
                    ops.extend(entries[pos][1])
                    pos += 1
                if entries and entries[-1][0] > kb:
                    break
        return _resolve_ops(ops)

    def _collect_run_range(
        self, run: SSTableRun, low_b: bytes | None, high_b: bytes | None,
        include_low: bool, include_high: bool, collected: dict[bytes, OpList],
    ) -> None:
        if (low_b is not None and run.max_key < low_b) or (high_b is not None and run.min_key > high_b):
            return
        for pid in range(_start_page_for_key(run, low_b), run.data_pages):
            for kb, ops in read_sstable_page(self.pool, run.file_id, pid):
                if high_b is not None and (kb > high_b or (not include_high and kb == high_b)):
                    return
                if key_in_range(kb, low_b, high_b, include_low, include_high):
                    collected.setdefault(kb, []).extend(ops)

    def range_search(
        self, low: Any | None = None, high: Any | None = None,
        include_low: bool = True, include_high: bool = True,
    ) -> list[tuple[Any, RID]]:
        """Return ordered (key, RID) pairs within optional lower/upper bounds."""
        low_b = encode_key(self.key_dtype, low) if low is not None else None
        high_b = encode_key(self.key_dtype, high) if high is not None else None
        collected: dict[bytes, OpList] = {}
        for kb, ops in self.memtable.items():
            if key_in_range(kb, low_b, high_b, include_low, include_high):
                collected.setdefault(kb, []).extend(ops)
        for run in self.runs:
            self._collect_run_range(run, low_b, high_b, include_low, include_high, collected)
        return [
            (decode_key(self.key_dtype, kb), rid)
            for kb in sorted(collected)
            for rid in _resolve_ops(collected[kb])
        ]

    def close(self) -> None:
        """Flush pending memtable entries to disk and close owned handles."""
        if not self._wal.closed:
            self.flush()
            self._wal.close()
        if self._owns_storage:
            self.pool.close()
            self.storage_manager.close_all()

    def destroy(self) -> None:
        """Remove WAL, manifest, memtable state, and all SSTable run files."""
        self._wal.destroy()
        self.memtable.clear()
        self.memtable_entries = 0
        for run in self.runs:
            self.pool.unregister_file(run.file_id, drop_dirty=True)
            self.storage_manager.drop_file(run.logical_name)
        self.runs.clear()
        self.manifest_path.unlink(missing_ok=True)
        self.manifest_path.with_suffix(".tmp").unlink(missing_ok=True)

    def __enter__(self) -> "LSMTree":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
