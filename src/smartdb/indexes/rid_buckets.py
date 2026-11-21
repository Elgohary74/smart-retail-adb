"""Overflow RID-bucket page chains for duplicate keys in B+ Tree leaves."""
from __future__ import annotations

import struct

from ..schema import RID
from ..storage.buffer_pool import BufferPool
from ..storage.slotted_page import HEADER_SIZE, PageHeader, PageType, read_header, write_header

_RID_STRUCT = "<II"
_RID_SIZE = 8


def _new_bucket_page(pool: BufferPool, file_id: int, rid: RID) -> int:
    pid, frame = pool.new_page(file_id=file_id)
    write_header(frame.data, PageHeader(int(PageType.OVERFLOW), 0, 1, HEADER_SIZE + _RID_SIZE, len(frame.data)))
    struct.pack_into(_RID_STRUCT, frame.data, HEADER_SIZE, rid[0], rid[1])
    pool.unpin_page(pid, is_dirty=True, file_id=file_id)
    return pid


def bucket_append(pool: BufferPool, file_id: int, usable_bytes: int, first_bucket_pid: int, rid: RID) -> int:
    max_rids = usable_bytes // _RID_SIZE
    if first_bucket_pid == -1:
        return _new_bucket_page(pool, file_id, rid)
    cur = first_bucket_pid
    while True:
        with pool.pin_page(cur, file_id=file_id, dirty=True) as frame:
            hdr = read_header(frame.data)
            if hdr.next_page == -1 and hdr.item_count < max_rids:
                struct.pack_into(_RID_STRUCT, frame.data, HEADER_SIZE + hdr.item_count * _RID_SIZE, rid[0], rid[1])
                hdr.item_count += 1
                write_header(frame.data, hdr)
                return first_bucket_pid
            if hdr.next_page == -1:
                hdr.next_page = _new_bucket_page(pool, file_id, rid)
                write_header(frame.data, hdr)
                return first_bucket_pid
            cur = hdr.next_page


def bucket_read(pool: BufferPool, file_id: int, first_bucket_pid: int) -> list[RID]:
    out: list[RID] = []
    cur = first_bucket_pid
    while cur != -1:
        with pool.pin_page(cur, file_id=file_id) as frame:
            hdr = read_header(frame.data)
            for i in range(hdr.item_count):
                out.append(struct.unpack_from(_RID_STRUCT, frame.data, HEADER_SIZE + i * _RID_SIZE))
            cur = hdr.next_page
    return out


def bucket_remove(pool: BufferPool, file_id: int, first_bucket_pid: int, target: RID | None) -> tuple[int, int]:
    if first_bucket_pid == -1:
        return -1, 0
    removed = 0
    cur = first_bucket_pid
    while cur != -1:
        with pool.pin_page(cur, file_id=file_id, dirty=True) as frame:
            hdr = read_header(frame.data)
            nxt = hdr.next_page
            if target is None:
                removed += hdr.item_count
                hdr.page_type, hdr.next_page = int(PageType.FREE), -1
                write_header(frame.data, hdr)
            else:
                kept = [struct.unpack_from(_RID_STRUCT, frame.data, HEADER_SIZE + i * _RID_SIZE) for i in range(hdr.item_count)]
                filtered = [r for r in kept if r != target]
                if len(filtered) != len(kept):
                    removed += len(kept) - len(filtered)
                    hdr.item_count = len(filtered)
                    for i, r in enumerate(filtered):
                        struct.pack_into(_RID_STRUCT, frame.data, HEADER_SIZE + i * _RID_SIZE, r[0], r[1])
                    write_header(frame.data, hdr)
            cur = nxt
    return (-1 if target is None else first_bucket_pid), removed
