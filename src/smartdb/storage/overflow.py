"""Linked overflow page chains for records or RID buckets exceeding one page."""
from __future__ import annotations

import struct
from .buffer_pool import BufferPool
from .slotted_page import HEADER_SIZE, PageHeader, PageType, read_header, write_header

_CHUNK_LEN_STRUCT = "<H"


def write_overflow_chain(pool: BufferPool, file_id: int, data: bytes) -> int:
    """Write arbitrary bytes across linked OVERFLOW pages and return first page_id."""
    if not data:
        raise ValueError("Overflow payload must be non-empty")
    usable = pool._get_backend(file_id).usable_page_size - HEADER_SIZE - 2
    chunks = [data[i:i + usable] for i in range(0, len(data), usable)]
    first_pid = -1
    prev_pid = -1
    for chunk in chunks:
        pid, frame = pool.new_page(file_id=file_id)
        if first_pid == -1:
            first_pid = pid
        if prev_pid != -1:
            with pool.pin_page(prev_pid, file_id=file_id, dirty=True) as prev_frame:
                hdr = read_header(prev_frame.data)
                hdr.next_page = pid
                write_header(prev_frame.data, hdr)
        write_header(
            frame.data,
            PageHeader(
                page_type=int(PageType.OVERFLOW),
                flags=0,
                item_count=1,
                free_start=HEADER_SIZE + 2 + len(chunk),
                free_end=len(frame.data),
                next_page=-1,
                prev_page=prev_pid,
            ),
        )
        struct.pack_into(_CHUNK_LEN_STRUCT, frame.data, HEADER_SIZE, len(chunk))
        frame.data[HEADER_SIZE + 2:HEADER_SIZE + 2 + len(chunk)] = chunk
        pool.unpin_page(pid, is_dirty=True, file_id=file_id)
        prev_pid = pid
    return first_pid


def read_overflow_chain(pool: BufferPool, file_id: int, first_page_id: int, total_len: int) -> bytes:
    """Read a linked OVERFLOW chain back into a contiguous bytes object."""
    parts: list[bytes] = []
    pid = first_page_id
    while pid != -1:
        with pool.pin_page(pid, file_id=file_id) as frame:
            hdr = read_header(frame.data)
            chunk_len = struct.unpack_from(_CHUNK_LEN_STRUCT, frame.data, HEADER_SIZE)[0]
            parts.append(bytes(frame.data[HEADER_SIZE + 2:HEADER_SIZE + 2 + chunk_len]))
            pid = hdr.next_page
    raw = b"".join(parts)
    if len(raw) != total_len:
        raise IOError(f"Corrupt overflow chain: expected {total_len} bytes, got {len(raw)}")
    return raw


def free_overflow_chain(pool: BufferPool, file_id: int, first_page_id: int) -> None:
    """Mark all pages in an overflow chain as FREE."""
    pid = first_page_id
    while pid != -1:
        with pool.pin_page(pid, file_id=file_id, dirty=True) as frame:
            hdr = read_header(frame.data)
            next_pid = hdr.next_page
            hdr.page_type = int(PageType.FREE)
            hdr.next_page = -1
            write_header(frame.data, hdr)
            pid = next_pid
