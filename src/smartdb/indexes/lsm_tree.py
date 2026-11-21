"""Disk-backed SSTable LSM Tree with Bloom filters, fence pointers, WAL, and atomic compaction."""
from __future__ import annotations

import hashlib
import json
import os
import struct
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import RID, DataType, decode_key, encode_key
from ..storage.buffer_pool import BufferPool
from ..storage.manager import StorageManager
from ..storage.slotted_page import HEADER_SIZE, PageHeader, PageType, read_header, write_header

_OP_STRUCT = "<IIQB"  # page_id, slot_id (0xFFFFFFFF for key-tombstone), seq, is_deleted
_OP_SIZE = struct.calcsize(_OP_STRUCT)
_WAL_OP_STRUCT = "<BQiiH"  # op_type, seq, pid, sid, key_len
_KEY_TOMBSTONE_PID = 0xFFFFFFFF


class BloomFilter:
    """Compact bitset Bloom filter using blake2b double hashing."""

    def __init__(self, num_bits: int = 1024, num_hashes: int = 7, bits: bytearray | None = None):
        self.num_bits = max(64, num_bits)
        self.num_hashes = max(1, num_hashes)
        byte_len = (self.num_bits + 7) // 8
        self.bits = bits if bits is not None else bytearray(byte_len)

    def _hashes(self, key_bytes: bytes) -> tuple[int, int]:
        digest = hashlib.blake2b(key_bytes, digest_size=16).digest()
        h1, h2 = struct.unpack("<QQ", digest)
        return h1, h2 or 1

    def add(self, key_bytes: bytes) -> None:
        h1, h2 = self._hashes(key_bytes)
        for i in range(self.num_hashes):
            pos = (h1 + i * h2) % self.num_bits
            self.bits[pos // 8] |= 1 << (pos % 8)

    def might_contain(self, key_bytes: bytes) -> bool:
        h1, h2 = self._hashes(key_bytes)
        for i in range(self.num_hashes):
            pos = (h1 + i * h2) % self.num_bits
            if not (self.bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def to_bytes(self) -> bytes:
        return struct.pack("<II", self.num_bits, self.num_hashes) + bytes(self.bits)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "BloomFilter":
        num_bits, num_hashes = struct.unpack_from("<II", raw, 0)
        return cls(num_bits=num_bits, num_hashes=num_hashes, bits=bytearray(raw[8:]))


@dataclass
class _SSTableRun:
    run_id: int
    logical_name: str
    file_id: int
    data_pages: int
    min_key: bytes
    max_key: bytes
    bloom: BloomFilter
    fence: list[tuple[bytes, int]]  # [(first_key_bytes, data_page_id)]


def _resolve_ops(ops: list[tuple[RID | None, int, bool]]) -> list[RID]:
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
        flush_threshold: int = 128,
        compaction_threshold: int = 4,
        storage_manager: StorageManager | None = None,
        pool: BufferPool | None = None,
        name_prefix: str = "lsm",
        key_dtype: DataType = DataType.ANY,
    ):
        if flush_threshold < 1:
            raise ValueError("flush_threshold must be >= 1")
        if compaction_threshold < 2:
            raise ValueError("compaction_threshold must be >= 2")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_threshold = flush_threshold
        self.compaction_threshold = compaction_threshold
        self.name_prefix = name_prefix.replace(":", "_")
        self.key_dtype = key_dtype

        self._owns_storage = storage_manager is None or pool is None
        if self._owns_storage:
            storage_manager = StorageManager(self.directory)
            pool = BufferPool(capacity=32, storage_manager=storage_manager)
        assert storage_manager is not None and pool is not None
        self.storage_manager = storage_manager
        self.pool = pool

        self.manifest_path = self.directory / f"{self.name_prefix}_manifest.json"
        self.wal_path = self.directory / f"{self.name_prefix}_memtable.wal"

        self.memtable: dict[bytes, list[tuple[RID | None, int, bool]]] = {}
        self.memtable_entries = 0
        self.runs: list[_SSTableRun] = []
        self._next_run_id = 0
        self._next_seq = 1
        self.flushes = 0
        self.compactions = 0

        self._load_manifest()
        self._replay_wal()
        self._wal_file = self.wal_path.open("ab")

    @property
    def run_count(self) -> int:
        return len(self.runs)

    def disk_bytes(self) -> int:
        total = self.wal_path.stat().st_size if self.wal_path.exists() else 0
        for run in self.runs:
            backend = self.storage_manager.get_backend(run.file_id)
            total += backend.page_count * backend.physical_page_size
        return total

    # --- Manifest & WAL ---

    def _save_manifest(self) -> None:
        data = {
            "next_run_id": self._next_run_id,
            "next_seq": self._next_seq,
            "runs": [
                {
                    "run_id": r.run_id,
                    "logical_name": r.logical_name,
                    "meta_pid": r.data_pages,
                }
                for r in self.runs
            ],
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
            run_id = item["run_id"]
            lname = item["logical_name"]
            meta_pid = item["meta_pid"]
            fid, _ = self.storage_manager.open_file(lname, self._run_rel_path(run_id))
            run = self._read_run_meta(run_id, lname, fid, meta_pid)
            self.runs.append(run)

    def _run_rel_path(self, run_id: int) -> str:
        if self._owns_storage:
            return f"{self.name_prefix}_run_{run_id:08d}.sst"
        return f"indexes/{self.name_prefix}/run_{run_id:08d}.sst"

    def _append_wal(self, key_bytes: bytes, rid: RID | None, seq: int, is_deleted: bool) -> None:
        pid = -1 if rid is None else rid[0]
        sid = 0 if rid is None else rid[1]
        raw = struct.pack(_WAL_OP_STRUCT, 1 if is_deleted else 0, seq, pid, sid, len(key_bytes)) + key_bytes
        enc = self.storage_manager.encrypt_record(raw)
        self._wal_file.write(struct.pack("<I", len(enc)) + enc)
        self._wal_file.flush()

    def _replay_wal(self) -> None:
        if not self.wal_path.exists():
            return
        with self.wal_path.open("rb") as f:
            while True:
                hdr = f.read(4)
                if len(hdr) < 4:
                    break
                length = struct.unpack("<I", hdr)[0]
                enc = f.read(length)
                if len(enc) < length:
                    break  # Torn trailing record from crash
                raw = self.storage_manager.decrypt_record(enc)
                op_flag, seq, pid, sid, klen = struct.unpack_from(_WAL_OP_STRUCT, raw, 0)
                key_bytes = raw[struct.calcsize(_WAL_OP_STRUCT):struct.calcsize(_WAL_OP_STRUCT) + klen]
                rid: RID | None = None if pid == -1 else (pid, sid)
                self.memtable.setdefault(key_bytes, []).append((rid, seq, op_flag == 1))
                self.memtable_entries += 1
                self._next_seq = max(self._next_seq, seq + 1)

    def _truncate_wal(self) -> None:
        self._wal_file.flush()
        os.fsync(self._wal_file.fileno())
        self._wal_file.close()
        self.wal_path.write_bytes(b"")
        self._wal_file = self.wal_path.open("ab")

    # --- SSTable serialization ---

    def _write_run(
        self, run_id: int, sorted_items: list[tuple[bytes, list[tuple[RID | None, int, bool]]]]
    ) -> _SSTableRun:
        lname = f"index:{self.name_prefix}:run_{run_id:08d}"
        fid, backend = self.storage_manager.open_file(lname, self._run_rel_path(run_id))
        usable = backend.usable_page_size - HEADER_SIZE

        bloom = BloomFilter(num_bits=max(256, len(sorted_items) * 10), num_hashes=7)
        fence: list[tuple[bytes, int]] = []
        cur_entries: list[tuple[bytes, list[tuple[RID | None, int, bool]]]] = []
        cur_size = 0

        def flush_data_page() -> None:
            nonlocal cur_entries, cur_size
            if not cur_entries:
                return
            pid, frame = self.pool.new_page(file_id=fid)
            fence.append((cur_entries[0][0], pid))
            buf = frame.data
            pos = HEADER_SIZE
            for kb, ops in cur_entries:
                struct.pack_into("<HI", buf, pos, len(kb), len(ops))
                pos += 6
                buf[pos:pos + len(kb)] = kb
                pos += len(kb)
                for rid, seq, is_del in ops:
                    r_pid = _KEY_TOMBSTONE_PID if rid is None else rid[0]
                    r_sid = 0 if rid is None else rid[1]
                    struct.pack_into(_OP_STRUCT, buf, pos, r_pid, r_sid, seq, 1 if is_del else 0)
                    pos += _OP_SIZE
            write_header(buf, PageHeader(int(PageType.LSM_DATA), 0, len(cur_entries), pos, len(buf)))
            self.pool.unpin_page(pid, is_dirty=True, file_id=fid)
            cur_entries = []
            cur_size = 0

        for kb, ops in sorted_items:
            bloom.add(kb)
            entry_bytes = 6 + len(kb) + len(ops) * _OP_SIZE
            if entry_bytes <= usable:
                if cur_entries and cur_size + entry_bytes > usable:
                    flush_data_page()
                cur_entries.append((kb, ops))
                cur_size += entry_bytes
            else:
                # Split huge op list for one key across multiple pages
                flush_data_page()
                max_ops = max(1, (usable - 6 - len(kb)) // _OP_SIZE)
                for i in range(0, len(ops), max_ops):
                    cur_entries.append((kb, ops[i:i + max_ops]))
                    flush_data_page()
        flush_data_page()

        data_pages = backend.page_count
        meta_blob = self._encode_meta(sorted_items[0][0], sorted_items[-1][0], bloom, fence)
        # Write meta blob across chained LSM_META pages starting at data_pages
        meta_chunks = [meta_blob[i:i + usable - 4] for i in range(0, len(meta_blob), usable - 4)]
        meta_pids: list[int] = []
        for idx, chunk in enumerate(meta_chunks):
            mpid, mframe = self.pool.new_page(file_id=fid)
            meta_pids.append(mpid)
            struct.pack_into("<I", mframe.data, HEADER_SIZE, len(chunk))
            mframe.data[HEADER_SIZE + 4:HEADER_SIZE + 4 + len(chunk)] = chunk
            write_header(
                mframe.data,
                PageHeader(int(PageType.LSM_META), 0, 1, HEADER_SIZE + 4 + len(chunk), len(mframe.data)),
            )
            self.pool.unpin_page(mpid, is_dirty=True, file_id=fid)
        for idx in range(len(meta_pids) - 1):
            with self.pool.pin_page(meta_pids[idx], file_id=fid, dirty=True) as mframe:
                hdr = read_header(mframe.data)
                hdr.next_page = meta_pids[idx + 1]
                write_header(mframe.data, hdr)

        self.pool.flush_file(fid)
        return _SSTableRun(
            run_id=run_id,
            logical_name=lname,
            file_id=fid,
            data_pages=data_pages,
            min_key=sorted_items[0][0],
            max_key=sorted_items[-1][0],
            bloom=bloom,
            fence=fence,
        )

    @staticmethod
    def _encode_meta(min_k: bytes, max_k: bytes, bloom: BloomFilter, fence: list[tuple[bytes, int]]) -> bytes:
        bloom_raw = bloom.to_bytes()
        parts = [
            struct.pack("<III", len(min_k), len(max_k), len(bloom_raw)),
            min_k,
            max_k,
            bloom_raw,
            struct.pack("<I", len(fence)),
        ]
        for fk, pid in fence:
            parts.append(struct.pack("<HI", len(fk), pid) + fk)
        return b"".join(parts)

    def _read_run_meta(self, run_id: int, lname: str, fid: int, meta_pid: int) -> _SSTableRun:
        chunks: list[bytes] = []
        pid = meta_pid
        while pid != -1:
            with self.pool.pin_page(pid, file_id=fid) as frame:
                hdr = read_header(frame.data)
                clen = struct.unpack_from("<I", frame.data, HEADER_SIZE)[0]
                chunks.append(bytes(frame.data[HEADER_SIZE + 4:HEADER_SIZE + 4 + clen]))
                pid = hdr.next_page
        blob = b"".join(chunks)
        min_len, max_len, bloom_len = struct.unpack_from("<III", blob, 0)
        pos = 12
        min_k = blob[pos:pos + min_len]
        pos += min_len
        max_k = blob[pos:pos + max_len]
        pos += max_len
        bloom = BloomFilter.from_bytes(blob[pos:pos + bloom_len])
        pos += bloom_len
        fcount = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        fence: list[tuple[bytes, int]] = []
        for _ in range(fcount):
            fklen, dpid = struct.unpack_from("<HI", blob, pos)
            pos += 6
            fence.append((blob[pos:pos + fklen], dpid))
            pos += fklen
        return _SSTableRun(run_id, lname, fid, meta_pid, min_k, max_k, bloom, fence)

    def _read_data_page(self, fid: int, pid: int) -> list[tuple[bytes, list[tuple[RID | None, int, bool]]]]:
        with self.pool.pin_page(pid, file_id=fid) as frame:
            hdr = read_header(frame.data)
            pos = HEADER_SIZE
            entries: list[tuple[bytes, list[tuple[RID | None, int, bool]]]] = []
            for _ in range(hdr.item_count):
                klen, op_count = struct.unpack_from("<HI", frame.data, pos)
                pos += 6
                kb = bytes(frame.data[pos:pos + klen])
                pos += klen
                ops: list[tuple[RID | None, int, bool]] = []
                for _ in range(op_count):
                    r_pid, r_sid, seq, is_del = struct.unpack_from(_OP_STRUCT, frame.data, pos)
                    pos += _OP_SIZE
                    rid: RID | None = None if r_pid == _KEY_TOMBSTONE_PID else (r_pid, r_sid)
                    ops.append((rid, seq, is_del == 1))
                entries.append((kb, ops))
            return entries

    # --- Public operations ---

    def insert(self, key: Any, rid: RID) -> None:
        kb = encode_key(self.key_dtype, key)
        seq = self._next_seq
        self._next_seq += 1
        self.memtable.setdefault(kb, []).append((rid, seq, False))
        self.memtable_entries += 1
        self._append_wal(kb, rid, seq, False)
        if self.memtable_entries >= self.flush_threshold:
            self.flush()

    def delete(self, key: Any, rid: RID | None = None) -> None:
        kb = encode_key(self.key_dtype, key)
        seq = self._next_seq
        self._next_seq += 1
        self.memtable.setdefault(kb, []).append((rid, seq, True))
        self.memtable_entries += 1
        self._append_wal(kb, rid, seq, True)
        if self.memtable_entries >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self.memtable:
            return
        ordered = sorted(self.memtable.items(), key=lambda item: item[0])
        run_id = self._next_run_id
        self._next_run_id += 1
        run = self._write_run(run_id, ordered)
        self.runs.append(run)
        self.memtable.clear()
        self.memtable_entries = 0
        self._save_manifest()
        self._truncate_wal()
        self.flushes += 1
        if len(self.runs) >= self.compaction_threshold:
            self.compact()

    def compact(self) -> None:
        if len(self.runs) < 2 and not self.memtable:
            return
        merged: dict[bytes, list[tuple[RID | None, int, bool]]] = {}
        for run in self.runs:
            for pid in range(run.data_pages):
                for kb, ops in self._read_data_page(run.file_id, pid):
                    merged.setdefault(kb, []).extend(ops)
        for kb, ops in self.memtable.items():
            merged.setdefault(kb, []).extend(ops)

        final_items: list[tuple[bytes, list[tuple[RID | None, int, bool]]]] = []
        for kb in sorted(merged):
            live_rids = _resolve_ops(merged[kb])
            if live_rids:
                seq = self._next_seq
                final_items.append((kb, [(r, seq, False) for r in live_rids]))
        if final_items:
            self._next_seq += 1

        old_runs = list(self.runs)
        new_runs: list[_SSTableRun] = []
        if final_items:
            run_id = self._next_run_id
            self._next_run_id += 1
            new_runs.append(self._write_run(run_id, final_items))

        self.runs = new_runs
        self.memtable.clear()
        self.memtable_entries = 0
        self._save_manifest()
        self._truncate_wal()

        for old in old_runs:
            self.pool.unregister_file(old.file_id, drop_dirty=True)
            self.storage_manager.drop_file(old.logical_name)
        self.compactions += 1

    def search(self, key: Any) -> list[RID]:
        kb = encode_key(self.key_dtype, key)
        ops: list[tuple[RID | None, int, bool]] = list(self.memtable.get(kb, []))
        for run in reversed(self.runs):
            if kb < run.min_key or kb > run.max_key or not run.bloom.might_contain(kb):
                continue
            fkeys = [fk for fk, _ in run.fence]
            idx = max(0, bisect_right(fkeys, kb) - 1)
            for pid in range(run.fence[idx][1], run.data_pages):
                entries = self._read_data_page(run.file_id, pid)
                ekeys = [e[0] for e in entries]
                pos = bisect_left(ekeys, kb)
                while pos < len(entries) and entries[pos][0] == kb:
                    ops.extend(entries[pos][1])
                    pos += 1
                if entries and entries[-1][0] > kb:
                    break
        return _resolve_ops(ops)

    def range_search(
        self,
        low: Any | None = None,
        high: Any | None = None,
        include_low: bool = True,
        include_high: bool = True,
    ) -> list[tuple[Any, RID]]:
        low_b = encode_key(self.key_dtype, low) if low is not None else None
        high_b = encode_key(self.key_dtype, high) if high is not None else None

        def in_bounds(kb: bytes) -> bool:
            if low_b is not None and (kb < low_b or (not include_low and kb == low_b)):
                return False
            if high_b is not None and (kb > high_b or (not include_high and kb == high_b)):
                return False
            return True

        collected: dict[bytes, list[tuple[RID | None, int, bool]]] = {}
        for kb, ops in self.memtable.items():
            if in_bounds(kb):
                collected.setdefault(kb, []).extend(ops)

        for run in self.runs:
            if low_b is not None and run.max_key < low_b:
                continue
            if high_b is not None and run.min_key > high_b:
                continue
            start_pid = 0
            if low_b is not None and run.fence:
                fkeys = [fk for fk, _ in run.fence]
                idx = max(0, bisect_right(fkeys, low_b) - 1)
                start_pid = run.fence[idx][1]
            for pid in range(start_pid, run.data_pages):
                entries = self._read_data_page(run.file_id, pid)
                stop = False
                for kb, ops in entries:
                    if high_b is not None and (kb > high_b or (not include_high and kb == high_b)):
                        stop = True
                        break
                    if in_bounds(kb):
                        collected.setdefault(kb, []).extend(ops)
                if stop:
                    break

        out: list[tuple[Any, RID]] = []
        for kb in sorted(collected):
            decoded = decode_key(self.key_dtype, kb)
            for rid in _resolve_ops(collected[kb]):
                out.append((decoded, rid))
        return out

    def close(self) -> None:
        if not self._wal_file.closed:
            self.flush()
            self._wal_file.close()
        if self._owns_storage:
            self.pool.close()
            self.storage_manager.close_all()
