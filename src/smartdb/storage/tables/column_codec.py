"""Bitmap and fixed/variable column page read/write helpers for ColumnTable."""
from __future__ import annotations

import bisect
import struct
from typing import Any

from ...core import ColumnSpec, DataType, deserialize_value, serialize_value
from ..buffer_pool import BufferPool
from ..pages import HEADER_SIZE, PageHeader, PageType, SlottedPage, read_header, read_overflow_chain, write_header, write_overflow_chain

_OVF_MAGIC = b"\xffOVF"
_OVF_STUB_STRUCT = "<4sIi"


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


def is_deleted_bit(pool: BufferPool, del_fid: int, del_backend: Any, bits_per_page: int, rowid: int) -> bool:
    pid, bit = rowid // bits_per_page, rowid % bits_per_page
    if pid >= del_backend.page_count:
        return True
    with pool.pin_page(pid, file_id=del_fid) as frame:
        return bool(frame.data[HEADER_SIZE + (bit // 8)] & (1 << (bit % 8)))


def set_deleted_bit(pool: BufferPool, del_fid: int, del_backend: Any, bits_per_page: int, rowid: int, deleted: bool) -> None:
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
        with pool.pin_page(pid, file_id=del_fid) as frame:
            page_end = min(end, (pid + 1) * bits_per_page)
            while cur < page_end:
                bit = cur % bits_per_page
                if not (frame.data[HEADER_SIZE + (bit // 8)] & (1 << (bit % 8))):
                    live.append(cur)
                cur += 1
    return live


def append_fixed(
    pool: BufferPool, fid: int, backend: Any, col: ColumnSpec,
    fixed_spec: tuple[int, int, int], rowid: int, value: Any,
) -> None:
    elem_size, spp, bmap_len = fixed_spec
    pid, slot = rowid // spp, rowid % spp
    while backend.page_count <= pid:
        npid, frame = pool.new_page(file_id=fid)
        write_header(frame.data, PageHeader(int(PageType.COL_FIXED), 0, 0, HEADER_SIZE + bmap_len, len(frame.data)))
        pool.unpin_page(npid, is_dirty=True, file_id=fid)
    with pool.pin_page(pid, file_id=fid, dirty=True) as frame:
        if value is None:
            frame.data[HEADER_SIZE + (slot // 8)] |= 1 << (slot % 8)
        else:
            frame.data[HEADER_SIZE + (slot // 8)] &= ~(1 << (slot % 8))
            off = HEADER_SIZE + bmap_len + slot * elem_size
            frame.data[off:off + elem_size] = serialize_value(col.dtype, value, col.dim)
        hdr = read_header(frame.data)
        hdr.item_count = max(hdr.item_count, slot + 1)
        write_header(frame.data, hdr)


def append_var(
    pool: BufferPool, fid: int, backend: Any, ovf_fid: int,
    max_inline: int, starts: list[int], col: ColumnSpec, rowid: int, value: Any,
) -> None:
    if value is None:
        payload = b"\x00"
    else:
        raw = b"\x01" + serialize_value(col.dtype, value, col.dim)
        if len(raw) <= max_inline:
            payload = raw
        else:
            first_ovf = write_overflow_chain(pool, ovf_fid, raw)
            payload = b"\x02" + struct.pack(_OVF_STUB_STRUCT, _OVF_MAGIC, len(raw), first_ovf)
    if backend.page_count > 0:
        with pool.pin_page(backend.page_count - 1, file_id=fid, dirty=True) as frame:
            if SlottedPage.insert_payload(frame.data, payload) is not None:
                return
    npid, frame = pool.new_page(file_id=fid)
    try:
        SlottedPage.init(frame.data, PageType.COL_VAR)
        hdr = read_header(frame.data)
        hdr.prev_page = rowid
        write_header(frame.data, hdr)
        if SlottedPage.insert_payload(frame.data, payload) is None:
            raise OverflowError("Column value stub too large")
        starts.append(rowid)
    finally:
        pool.unpin_page(npid, is_dirty=True, file_id=fid)


def decode_var_payload(pool: BufferPool, ovf_fid: int, col: ColumnSpec, payload: bytes) -> Any:
    if payload[0] == 0:
        return None
    if payload[0] == 2:
        _, total_len, first_ovf = struct.unpack(_OVF_STUB_STRUCT, payload[1:])
        payload = read_overflow_chain(pool, ovf_fid, first_ovf, total_len)
    val, _ = deserialize_value(col.dtype, payload, 1, col.dim)
    return val


def read_fixed_slice(
    pool: BufferPool, fid: int, col: ColumnSpec, fixed_spec: tuple[int, int, int], rowids: list[int]
) -> list[Any]:
    elem_size, spp, bmap_len = fixed_spec
    values: list[Any] = []
    idx = 0
    while idx < len(rowids):
        pid = rowids[idx] // spp
        with pool.pin_page(pid, file_id=fid) as frame:
            while idx < len(rowids) and rowids[idx] // spp == pid:
                slot = rowids[idx] % spp
                if frame.data[HEADER_SIZE + (slot // 8)] & (1 << (slot % 8)):
                    values.append(None)
                else:
                    off = HEADER_SIZE + bmap_len + slot * elem_size
                    values.append(deserialize_value(col.dtype, frame.data, off, col.dim)[0])
                idx += 1
    return values


def read_var_slice(
    pool: BufferPool, fid: int, ovf_fid: int, col: ColumnSpec, starts: list[int], rowids: list[int]
) -> list[Any]:
    values: list[Any] = []
    idx = 0
    while idx < len(rowids):
        pid = bisect.bisect_right(starts, rowids[idx]) - 1
        first_rid = starts[pid]
        with pool.pin_page(pid, file_id=fid) as frame:
            page_end = first_rid + read_header(frame.data).item_count
            while idx < len(rowids) and rowids[idx] < page_end:
                payload = SlottedPage.get_payload(frame.data, rowids[idx] - first_rid)
                assert payload is not None
                values.append(decode_var_payload(pool, ovf_fid, col, payload))
                idx += 1
    return values
