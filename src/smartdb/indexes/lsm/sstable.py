"""Immutable SSTable run serialization over StorageManager and BufferPool."""
from __future__ import annotations

import struct
from dataclasses import dataclass

from ...core import RID
from ...storage import BufferPool, StorageManager
from ...storage.pages import HEADER_SIZE, PageHeader, PageType, read_header, write_header
from .bloom import BloomFilter

OP_STRUCT = "<IIQB"
OP_SIZE = struct.calcsize(OP_STRUCT)
KEY_TOMBSTONE_PID = 0xFFFFFFFF

OpList = list[tuple[RID | None, int, bool]]


@dataclass
class SSTableRun:
    run_id: int
    logical_name: str
    file_id: int
    data_pages: int
    min_key: bytes
    max_key: bytes
    bloom: BloomFilter
    fence: list[tuple[bytes, int]]


def _pack_data_page(buf: bytearray, entries: list[tuple[bytes, OpList]]) -> None:
    pos = HEADER_SIZE
    for kb, ops in entries:
        struct.pack_into("<HI", buf, pos, len(kb), len(ops))
        pos += 6
        buf[pos:pos + len(kb)] = kb
        pos += len(kb)
        for rid, seq, is_del in ops:
            r_pid = KEY_TOMBSTONE_PID if rid is None else rid[0]
            r_sid = 0 if rid is None else rid[1]
            struct.pack_into(OP_STRUCT, buf, pos, r_pid, r_sid, seq, 1 if is_del else 0)
            pos += OP_SIZE
    write_header(buf, PageHeader(int(PageType.LSM_DATA), 0, len(entries), pos, len(buf)))


def _flush_page(pool: BufferPool, fid: int, entries: list[tuple[bytes, OpList]],
                fence: list[tuple[bytes, int]]) -> None:
    if not entries:
        return
    pid, frame = pool.new_page(file_id=fid)
    fence.append((entries[0][0], pid))
    _pack_data_page(frame.data, entries)
    pool.unpin_page(pid, is_dirty=True, file_id=fid)


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


def _write_meta_pages(pool: BufferPool, fid: int, usable: int, meta_blob: bytes) -> None:
    chunks = [meta_blob[i:i + usable - 4] for i in range(0, len(meta_blob), usable - 4)]
    meta_pids: list[int] = []
    for chunk in chunks:
        mpid, mframe = pool.new_page(file_id=fid)
        meta_pids.append(mpid)
        struct.pack_into("<I", mframe.data, HEADER_SIZE, len(chunk))
        mframe.data[HEADER_SIZE + 4:HEADER_SIZE + 4 + len(chunk)] = chunk
        write_header(mframe.data, PageHeader(int(PageType.LSM_META), 0, 1, HEADER_SIZE + 4 + len(chunk), len(mframe.data)))
        pool.unpin_page(mpid, is_dirty=True, file_id=fid)
    for idx in range(len(meta_pids) - 1):
        with pool.pin_page(meta_pids[idx], file_id=fid, dirty=True) as mframe:
            hdr = read_header(mframe.data)
            hdr.next_page = meta_pids[idx + 1]
            write_header(mframe.data, hdr)


def write_sstable_run(
    storage_manager: StorageManager,
    pool: BufferPool,
    run_id: int,
    logical_name: str,
    rel_path: str,
    sorted_items: list[tuple[bytes, OpList]],
) -> SSTableRun:
    fid, backend = storage_manager.open_file(logical_name, rel_path)
    usable = backend.usable_page_size - HEADER_SIZE
    bloom = BloomFilter(num_bits=max(256, len(sorted_items) * 10), num_hashes=7)
    fence: list[tuple[bytes, int]] = []
    cur_entries: list[tuple[bytes, OpList]] = []
    cur_size = 0

    for kb, ops in sorted_items:
        bloom.add(kb)
        entry_bytes = 6 + len(kb) + len(ops) * OP_SIZE
        if entry_bytes <= usable:
            if cur_entries and cur_size + entry_bytes > usable:
                _flush_page(pool, fid, cur_entries, fence)
                cur_entries, cur_size = [], 0
            cur_entries.append((kb, ops))
            cur_size += entry_bytes
        else:
            _flush_page(pool, fid, cur_entries, fence)
            cur_entries, cur_size = [], 0
            max_ops = max(1, (usable - 6 - len(kb)) // OP_SIZE)
            for i in range(0, len(ops), max_ops):
                _flush_page(pool, fid, [(kb, ops[i:i + max_ops])], fence)
    _flush_page(pool, fid, cur_entries, fence)

    data_pages = backend.page_count
    meta_blob = _encode_meta(sorted_items[0][0], sorted_items[-1][0], bloom, fence)
    _write_meta_pages(pool, fid, usable, meta_blob)
    pool.flush_file(fid)
    return SSTableRun(run_id, logical_name, fid, data_pages, sorted_items[0][0], sorted_items[-1][0], bloom, fence)


def read_sstable_meta(pool: BufferPool, run_id: int, lname: str, fid: int, meta_pid: int) -> SSTableRun:
    chunks: list[bytes] = []
    pid = meta_pid
    while pid != -1:
        with pool.pin_page(pid, file_id=fid) as frame:
            hdr = read_header(frame.data)
            clen = struct.unpack_from("<I", frame.data, HEADER_SIZE)[0]
            chunks.append(bytes(frame.data[HEADER_SIZE + 4:HEADER_SIZE + 4 + clen]))
            pid = hdr.next_page
    blob = b"".join(chunks)
    min_len, max_len, bloom_len = struct.unpack_from("<III", blob, 0)
    pos = 12
    min_k, max_k = blob[pos:pos + min_len], blob[pos + min_len:pos + min_len + max_len]
    pos += min_len + max_len
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
    return SSTableRun(run_id, lname, fid, meta_pid, min_k, max_k, bloom, fence)


def read_sstable_page(pool: BufferPool, fid: int, pid: int) -> list[tuple[bytes, OpList]]:
    with pool.pin_page(pid, file_id=fid) as frame:
        hdr = read_header(frame.data)
        pos = HEADER_SIZE
        entries: list[tuple[bytes, OpList]] = []
        for _ in range(hdr.item_count):
            klen, op_count = struct.unpack_from("<HI", frame.data, pos)
            pos += 6
            kb = bytes(frame.data[pos:pos + klen])
            pos += klen
            ops: OpList = []
            for _ in range(op_count):
                r_pid, r_sid, seq, is_del = struct.unpack_from(OP_STRUCT, frame.data, pos)
                pos += OP_SIZE
                rid: RID | None = None if r_pid == KEY_TOMBSTONE_PID else (r_pid, r_sid)
                ops.append((rid, seq, is_del == 1))
            entries.append((kb, ops))
        return entries
