"""Overflow RID-bucket page chains for duplicate keys in B+ Tree leaves."""
from __future__ import annotations

import struct

from ...core import RID
from ...storage import BufferPool
from ...storage.pages import HEADER_SIZE, PageHeader, PageType, read_header, write_header
from .pages import RID_SIZE, RID_STRUCT


def _new_bucket_page(pool: BufferPool, file_id: int, rid: RID) -> int:
    pid, frame = pool.new_page(file_id=file_id)
    try:
        write_header(frame.data, PageHeader(int(PageType.OVERFLOW), 0, 1, HEADER_SIZE + RID_SIZE, len(frame.data)))
        struct.pack_into(RID_STRUCT, frame.data, HEADER_SIZE, rid[0], rid[1])
        return pid
    finally:
        pool.unpin_page(pid, is_dirty=True, file_id=file_id)


def bucket_append(pool: BufferPool, file_id: int, usable_bytes: int, first_bucket_pid: int, rid: RID) -> int:
    """Append a RID to the overflow bucket chain, returning the head page ID."""
    max_rids = usable_bytes // RID_SIZE
    if first_bucket_pid == -1:
        return _new_bucket_page(pool, file_id, rid)
    cur = first_bucket_pid
    while True:
        with pool.pin_page(cur, file_id=file_id, dirty=True) as frame:
            hdr = read_header(frame.data)
            if hdr.next_page == -1 and hdr.item_count < max_rids:
                struct.pack_into(RID_STRUCT, frame.data, HEADER_SIZE + hdr.item_count * RID_SIZE, rid[0], rid[1])
                hdr.item_count += 1
                write_header(frame.data, hdr)
                return first_bucket_pid
            if hdr.next_page == -1:
                tail_pid = cur
                break
            cur = hdr.next_page
    # Allocate outside the tail pin so at most one buffer frame is pinned at once.
    new_pid = _new_bucket_page(pool, file_id, rid)
    with pool.pin_page(tail_pid, file_id=file_id, dirty=True) as frame:
        hdr = read_header(frame.data)
        hdr.next_page = new_pid
        write_header(frame.data, hdr)
    return first_bucket_pid


def bucket_read(pool: BufferPool, file_id: int, first_bucket_pid: int) -> list[RID]:
    """Read all RIDs stored across an overflow bucket chain."""
    out: list[RID] = []
    cur = first_bucket_pid
    while cur != -1:
        with pool.pin_page(cur, file_id=file_id) as frame:
            hdr = read_header(frame.data)
            for i in range(hdr.item_count):
                out.append(struct.unpack_from(RID_STRUCT, frame.data, HEADER_SIZE + i * RID_SIZE))
            cur = hdr.next_page
    return out


def _clear_bucket_page(buf: bytearray) -> tuple[int, int]:
    hdr = read_header(buf)
    removed, nxt = hdr.item_count, hdr.next_page
    hdr.page_type, hdr.item_count, hdr.next_page = int(PageType.FREE), 0, -1
    write_header(buf, hdr)
    return removed, nxt


def _filter_bucket_page(buf: bytearray, target: RID) -> tuple[int, int, int]:
    hdr = read_header(buf)
    kept = [struct.unpack_from(RID_STRUCT, buf, HEADER_SIZE + i * RID_SIZE) for i in range(hdr.item_count)]
    filtered = [r for r in kept if r != target]
    removed = len(kept) - len(filtered)
    nxt = hdr.next_page
    if not filtered:
        hdr.page_type, hdr.item_count, hdr.next_page = int(PageType.FREE), 0, -1
        write_header(buf, hdr)
    elif removed > 0:
        hdr.item_count = len(filtered)
        for i, r in enumerate(filtered):
            struct.pack_into(RID_STRUCT, buf, HEADER_SIZE + i * RID_SIZE, r[0], r[1])
        write_header(buf, hdr)
    return removed, len(filtered), nxt


def _relink_prev(pool: BufferPool, file_id: int, prev_pid: int, next_pid: int) -> None:
    if prev_pid == -1:
        return
    with pool.pin_page(prev_pid, file_id=file_id, dirty=True) as frame:
        hdr = read_header(frame.data)
        hdr.next_page = next_pid
        write_header(frame.data, hdr)


def bucket_remove(pool: BufferPool, file_id: int, first_bucket_pid: int, target: RID | None) -> tuple[int, int]:
    """Remove all or matching RIDs from a bucket chain; returns (new_head_pid, removed)."""
    if first_bucket_pid == -1:
        return -1, 0
    if target is None:
        removed, cur = 0, first_bucket_pid
        while cur != -1:
            with pool.pin_page(cur, file_id=file_id, dirty=True) as frame:
                delta, cur = _clear_bucket_page(frame.data)
                removed += delta
        return -1, removed

    head, prev_pid, cur, removed = first_bucket_pid, -1, first_bucket_pid, 0
    while cur != -1:
        with pool.pin_page(cur, file_id=file_id, dirty=True) as frame:
            delta, remaining, nxt = _filter_bucket_page(frame.data, target)
        removed += delta
        if remaining == 0:
            if prev_pid == -1:
                head = nxt
            else:
                _relink_prev(pool, file_id, prev_pid, nxt)
        else:
            prev_pid = cur
        cur = nxt
    return head, removed
