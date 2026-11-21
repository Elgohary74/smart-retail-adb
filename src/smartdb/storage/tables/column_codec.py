"""Bitmap and fixed/variable column page read/write helpers for ColumnTable."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

from ...core import ColumnSpec, DataType, deserialize_value, serialize_value
from ..backends import PageBackend
from ..buffer_pool import BufferPool
from ..pages import (
    HEADER_SIZE,
    PageHeader,
    PageType,
    SlottedPage,
    pack_overflow_stub,
    read_header,
    read_overflow_chain,
    unpack_overflow_stub,
    write_header,
    write_overflow_chain,
)


@dataclass
class ColumnStream:
    """State for a single column's page stream inside ColumnTable."""

    col: ColumnSpec
    file_id: int
    backend: PageBackend
    ovf_fid: int
    max_var_inline: int
    fixed_spec: tuple[int, int, int] | None = None  # (elem_size, slots_per_page, bmap_bytes)
    var_starts: list[int] = field(default_factory=list)


def fixed_elem_size(col: ColumnSpec, usable_page_size: int) -> int | None:
    if col.dtype in (DataType.INT, DataType.FLOAT):
        return 8
    if col.dtype == DataType.BOOL:
        return 1
    if col.dtype == DataType.VECTOR and col.dim is not None and 4 * col.dim <= usable_page_size - HEADER_SIZE - 64:
        return 4 * col.dim
    return None


def fixed_slots_per_page(usable_page_size: int, elem_size: int) -> int:
    avail = usable_page_size - HEADER_SIZE
    s = (avail * 8) // (8 * elem_size + 1)
    while ((s + 7) // 8) + s * elem_size > avail:
        s -= 1
    return max(1, s)


def is_deleted_bit(pool: BufferPool, del_fid: int, del_backend: PageBackend, bits_per_page: int, rowid: int) -> bool:
    pid, bit = rowid // bits_per_page, rowid % bits_per_page
    if pid >= del_backend.page_count:
        return True
    with pool.pin_page(pid, file_id=del_fid) as frame:
        return bool(frame.data[HEADER_SIZE + (bit // 8)] & (1 << (bit % 8)))


def set_deleted_bit(
    pool: BufferPool, del_fid: int, del_backend: PageBackend, bits_per_page: int, rowid: int, deleted: bool
) -> None:
    pid, bit = rowid // bits_per_page, rowid % bits_per_page
    while del_backend.page_count <= pid:
        npid, frame = pool.new_page(file_id=del_fid)
        write_header(frame.data, PageHeader(int(PageType.BITMAP), 0, 0, HEADER_SIZE, len(frame.data)))
        pool.unpin_page(npid, is_dirty=True, file_id=del_fid)
    with pool.pin_page(pid, file_id=del_fid, dirty=True) as frame:
        byte_idx = HEADER_SIZE + (bit // 8)
        if deleted:
            frame.data[byte_idx] |= 1 << (bit % 8)
        else:
            frame.data[byte_idx] &= ~(1 << (bit % 8))


def read_live_rowids(pool: BufferPool, del_fid: int, bits_per_page: int, start: int, end: int) -> list[int]:
    live: list[int] = []
    cur = start
    while cur < end:
        pid = cur // bits_per_page
        page_end = min(end, (pid + 1) * bits_per_page)
        with pool.pin_page(pid, file_id=del_fid) as frame:
            live.extend(_live_in_bitmap_page(frame.data, bits_per_page, cur, page_end))
        cur = page_end
    return live


def _live_in_bitmap_page(buf: bytearray, bits_per_page: int, start: int, end: int) -> list[int]:
    out: list[int] = []
    for rowid in range(start, end):
        bit = rowid % bits_per_page
        if not (buf[HEADER_SIZE + (bit // 8)] & (1 << (bit % 8))):
            out.append(rowid)
    return out


def append_fixed(pool: BufferPool, stream: ColumnStream, rowid: int, value: Any) -> None:
    if stream.fixed_spec is None:
        raise ValueError(f"Column {stream.col.name!r} is not fixed-width")
    elem_size, spp, bmap_len = stream.fixed_spec
    pid, slot = rowid // spp, rowid % spp
    while stream.backend.page_count <= pid:
        npid, frame = pool.new_page(file_id=stream.file_id)
        write_header(frame.data, PageHeader(int(PageType.COL_FIXED), 0, 0, HEADER_SIZE + bmap_len, len(frame.data)))
        pool.unpin_page(npid, is_dirty=True, file_id=stream.file_id)
    with pool.pin_page(pid, file_id=stream.file_id, dirty=True) as frame:
        if value is None:
            frame.data[HEADER_SIZE + (slot // 8)] |= 1 << (slot % 8)
        else:
            frame.data[HEADER_SIZE + (slot // 8)] &= ~(1 << (slot % 8))
            off = HEADER_SIZE + bmap_len + slot * elem_size
            frame.data[off:off + elem_size] = serialize_value(stream.col.dtype, value, stream.col.dim)
        hdr = read_header(frame.data)
        hdr.item_count = max(hdr.item_count, slot + 1)
        write_header(frame.data, hdr)


def append_var(pool: BufferPool, stream: ColumnStream, rowid: int, value: Any) -> None:
    payload = _encode_var_payload(pool, stream, value)
    if stream.backend.page_count > 0:
        with pool.pin_page(stream.backend.page_count - 1, file_id=stream.file_id, dirty=True) as frame:
            if SlottedPage.insert_payload(frame.data, payload) is not None:
                return
    npid, frame = pool.new_page(file_id=stream.file_id)
    try:
        SlottedPage.init(frame.data, PageType.COL_VAR)
        hdr = read_header(frame.data)
        hdr.prev_page = rowid
        write_header(frame.data, hdr)
        if SlottedPage.insert_payload(frame.data, payload) is None:
            raise OverflowError("Column value stub too large")
        stream.var_starts.append(rowid)
    finally:
        pool.unpin_page(npid, is_dirty=True, file_id=stream.file_id)


def _encode_var_payload(pool: BufferPool, stream: ColumnStream, value: Any) -> bytes:
    if value is None:
        return b"\x00"
    raw = b"\x01" + serialize_value(stream.col.dtype, value, stream.col.dim)
    if len(raw) <= stream.max_var_inline:
        return raw
    first_ovf = write_overflow_chain(pool, stream.ovf_fid, raw)
    return b"\x02" + pack_overflow_stub(len(raw), first_ovf)


def decode_var_payload(pool: BufferPool, ovf_fid: int, col: ColumnSpec, payload: bytes) -> Any:
    if payload[0] == 0:
        return None
    if payload[0] == 2:
        total_len, first_ovf = unpack_overflow_stub(payload[1:])
        payload = read_overflow_chain(pool, ovf_fid, first_ovf, total_len)
    return deserialize_value(col.dtype, payload, 1, col.dim)[0]


def _read_fixed_cell(buf: bytearray, col: ColumnSpec, fixed_spec: tuple[int, int, int], rowid: int) -> Any:
    elem_size, spp, bmap_len = fixed_spec
    slot = rowid % spp
    if buf[HEADER_SIZE + (slot // 8)] & (1 << (slot % 8)):
        return None
    return deserialize_value(col.dtype, buf, HEADER_SIZE + bmap_len + slot * elem_size, col.dim)[0]


def read_fixed_slice(pool: BufferPool, stream: ColumnStream, rowids: list[int]) -> list[Any]:
    if stream.fixed_spec is None:
        raise ValueError(f"Column {stream.col.name!r} is not fixed-width")
    spp = stream.fixed_spec[1]
    values: list[Any] = []
    idx = 0
    while idx < len(rowids):
        pid = rowids[idx] // spp
        end = idx
        while end < len(rowids) and rowids[end] // spp == pid:
            end += 1
        with pool.pin_page(pid, file_id=stream.file_id) as frame:
            values.extend(_read_fixed_cell(frame.data, stream.col, stream.fixed_spec, rid) for rid in rowids[idx:end])
        idx = end
    return values


def read_var_slice(pool: BufferPool, stream: ColumnStream, rowids: list[int]) -> list[Any]:
    values: list[Any] = []
    idx = 0
    while idx < len(rowids):
        pid = bisect.bisect_right(stream.var_starts, rowids[idx]) - 1
        first_rid = stream.var_starts[pid]
        with pool.pin_page(pid, file_id=stream.file_id) as frame:
            page_end = first_rid + read_header(frame.data).item_count
            while idx < len(rowids) and rowids[idx] < page_end:
                payload = SlottedPage.get_payload(frame.data, rowids[idx] - first_rid)
                if payload is None:
                    raise IOError(f"Corrupt COL_VAR page {pid}: missing slot {rowids[idx] - first_rid}")
                values.append(decode_var_payload(pool, stream.ovf_fid, stream.col, payload))
                idx += 1
    return values
