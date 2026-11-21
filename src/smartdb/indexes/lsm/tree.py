"""Disk-backed SSTable LSM Tree with Bloom filters, fence pointers, WAL, and atomic compaction."""
from __future__ import annotations

import json
import os
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

from ...core import (
    LSM_COMPACTION_THRESHOLD,
    LSM_FLUSH_THRESHOLD,
    RID,
    DataType,
    decode_key,
    encode_key,
)
from ...storage import BufferPool, StorageManager
from .bloom import BloomFilter
from .wal import LSMWal
from .sstable import OpList, SSTableRun, read_sstable_meta, read_sstable_page, write_sstable_run

__all__ = ["BloomFilter", "LSMTree"]


def _resolve_ops(ops: OpList) -> list[RID]:
    live: dict[RID, bool] = {}
    for rid, _, is_deleted in sorted(ops, key=lambda o: o[1]):
        if rid is None and is_deleted:
            live.clear()
        elif rid is not None:
            live[rid] = not is_deleted
    return [rid for rid, ok in live.items() if ok]


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
        assert storage_manager is not None and pool is not None
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
        self._wal_file = self._wal  # expose handle for crash-simulation tests

    @property
    def run_count(self) -> int:
        return len(self.runs)

    def disk_bytes(self) -> int:
        total = self.wal_path.stat().st_size if self.wal_path.exists() else 0
        for run in self.runs:
            backend = self.storage_manager.get_backend(run.file_id)
            total += backend.page_count * backend.physical_page_size
        return total

    def _run_rel_path(self, run_id: int) -> str:
        prefix = "" if self._owns_storage else f"indexes/{self.name_prefix}/"
        return f"{prefix}{self.name_prefix}_run_{run_id:08d}.sst"

    def _save_manifest(self) -> None:
        data = {
            "next_run_id": self._next_run_id,
            "next_seq": self._next_seq,
            "runs": [{"run_id": r.run_id, "logical_name": r.logical_name, "meta_pid": r.data_pages} for r in self.runs],
        }
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

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
        self._record_op(key, rid, False)

    def delete(self, key: Any, rid: RID | None = None) -> None:
        self._record_op(key, rid, True)

    def flush(self) -> None:
        if not self.memtable:
            return
        run_id = self._next_run_id
        self._next_run_id += 1
        lname = f"index:{self.name_prefix}:run_{run_id:08d}"
        run = write_sstable_run(
            self.storage_manager, self.pool, run_id, lname, self._run_rel_path(run_id),
            sorted(self.memtable.items(), key=lambda item: item[0]),
        )
        self.runs.append(run)
        self.memtable.clear()
        self.memtable_entries = 0
        self._save_manifest()
        self._wal.truncate()
        self.flushes += 1
        if len(self.runs) >= self.compaction_threshold:
            self.compact()

    def compact(self) -> None:
        if len(self.runs) < 2 and not self.memtable:
            return
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

        old_runs = list(self.runs)
        new_runs: list[SSTableRun] = []
        if final_items:
            run_id = self._next_run_id
            self._next_run_id += 1
            lname = f"index:{self.name_prefix}:run_{run_id:08d}"
            new_runs.append(
                write_sstable_run(self.storage_manager, self.pool, run_id, lname, self._run_rel_path(run_id), final_items)
            )
        self.runs = new_runs
        self.memtable.clear()
        self.memtable_entries = 0
        self._save_manifest()
        self._wal.truncate()
        for old in old_runs:
            self.pool.unregister_file(old.file_id, drop_dirty=True)
            self.storage_manager.drop_file(old.logical_name)
        self.compactions += 1

    def search(self, key: Any) -> list[RID]:
        kb = encode_key(self.key_dtype, key)
        ops: OpList = list(self.memtable.get(kb, []))
        for run in reversed(self.runs):
            if kb < run.min_key or kb > run.max_key or not run.bloom.might_contain(kb):
                continue
            idx = max(0, bisect_right([fk for fk, _ in run.fence], kb) - 1)
            for pid in range(run.fence[idx][1], run.data_pages):
                entries = read_sstable_page(self.pool, run.file_id, pid)
                pos = bisect_left([e[0] for e in entries], kb)
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
        start_pid = 0
        if low_b is not None and run.fence:
            start_pid = run.fence[max(0, bisect_right([fk for fk, _ in run.fence], low_b) - 1)][1]
        for pid in range(start_pid, run.data_pages):
            for kb, ops in read_sstable_page(self.pool, run.file_id, pid):
                if high_b is not None and (kb > high_b or (not include_high and kb == high_b)):
                    return
                if low_b is None or (kb > low_b or (include_low and kb == low_b)):
                    collected.setdefault(kb, []).extend(ops)

    def range_search(
        self, low: Any | None = None, high: Any | None = None,
        include_low: bool = True, include_high: bool = True,
    ) -> list[tuple[Any, RID]]:
        low_b = encode_key(self.key_dtype, low) if low is not None else None
        high_b = encode_key(self.key_dtype, high) if high is not None else None
        collected: dict[bytes, OpList] = {}
        for kb, ops in self.memtable.items():
            ok_low = low_b is None or (kb > low_b or (include_low and kb == low_b))
            ok_high = high_b is None or (kb < high_b or (include_high and kb == high_b))
            if ok_low and ok_high:
                collected.setdefault(kb, []).extend(ops)
        for run in self.runs:
            self._collect_run_range(run, low_b, high_b, include_low, include_high, collected)
        return [
            (decode_key(self.key_dtype, kb), rid)
            for kb in sorted(collected)
            for rid in _resolve_ops(collected[kb])
        ]

    def close(self) -> None:
        if not self._wal.closed:
            self.flush()
            self._wal.close()
        if self._owns_storage:
            self.pool.close()
            self.storage_manager.close_all()
