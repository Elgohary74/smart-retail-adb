"""Linked overflow page chains and stub codecs for oversized records/values."""
from __future__ import annotations

import struct
from ..buffer_pool import BufferPool
from .header import HEADER_SIZE, PageHeader, PageType, read_header, write_header

_CHUNK_LEN_STRUCT = "<H"
OVF_MAGIC = b"\xffOVF"
OVF_STUB_STRUCT = "<4sIi"
OVF_STUB_SIZE = struct.calcsize(OVF_STUB_STRUCT)


def is_overflow_stub(payload: bytes) -> bool:
    return len(payload) == OVF_STUB_SIZE and payload[:4] == OVF_MAGIC


def pack_overflow_stub(total_len: int, first_page_id: int) -> bytes:
    return struct.pack(OVF_STUB_STRUCT, OVF_MAGIC, total_len, first_page_id)


def unpack_overflow_stub(payload: bytes) -> tuple[int, int]:
    _, total_len, first_page_id = struct.unpack(OVF_STUB_STRUCT, payload)
    return total_len, first_page_id


def _write_chunk_page(pool: BufferPool, file_id: int, prev_pid: int, chunk: bytes) -> int:
    pid, frame = pool.new_page(file_id=file_id)
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
    if prev_pid != -1:
        with pool.pin_page(prev_pid, file_id=file_id, dirty=True) as prev_frame:
            hdr = read_header(prev_frame.data)
            hdr.next_page = pid
            write_header(prev_frame.data, hdr)
    return pid


def write_overflow_chain(pool: BufferPool, file_id: int, data: bytes) -> int:
    """Write arbitrary bytes across linked OVERFLOW pages and return first page_id."""
    if not data:
        raise ValueError("Overflow payload must be non-empty")
    usable = pool.usable_page_size(file_id) - HEADER_SIZE - 2
    first_pid, prev_pid = -1, -1
    for i in range(0, len(data), usable):
        prev_pid = _write_chunk_page(pool, file_id, prev_pid, data[i:i + usable])
        if first_pid == -1:
            first_pid = prev_pid
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
