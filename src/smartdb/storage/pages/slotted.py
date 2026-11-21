"""Common 24-byte page header and binary slotted-page layout."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

HEADER_STRUCT = "<BBHHHqii"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 24 bytes
_SLOT_STRUCT = "<HH"
_SLOT_SIZE = struct.calcsize(_SLOT_STRUCT)  # 4 bytes
_TOMBSTONE_OFFSET = 0xFFFF


class PageType(IntEnum):
    FREE = 0
    ROW_SLOTTED = 1
    OVERFLOW = 2
    COL_FIXED = 3
    COL_VAR = 4
    BITMAP = 5
    BPLUS_META = 6
    BPLUS_INTERNAL = 7
    BPLUS_LEAF = 8
    LSM_DATA = 9
    LSM_META = 10
    COL_META = 11


@dataclass
class PageHeader:
    page_type: int
    flags: int
    item_count: int
    free_start: int
    free_end: int
    lsn: int = 0
    next_page: int = -1
    prev_page: int = -1


def read_header(buf: bytes | bytearray | memoryview) -> PageHeader:
    return PageHeader(*struct.unpack_from(HEADER_STRUCT, buf, 0))


def write_header(buf: bytearray | memoryview, hdr: PageHeader) -> None:
    struct.pack_into(
        HEADER_STRUCT,
        buf,
        0,
        hdr.page_type,
        hdr.flags,
        hdr.item_count,
        hdr.free_start,
        hdr.free_end,
        hdr.lsn,
        hdr.next_page,
        hdr.prev_page,
    )


class SlottedPage:
    """In-page slot directory growing forward and variable payloads growing backward."""

    @staticmethod
    def init(buf: bytearray, page_type: PageType = PageType.ROW_SLOTTED) -> None:
        buf[:] = b"\x00" * len(buf)
        write_header(
            buf,
            PageHeader(
                page_type=int(page_type),
                flags=0,
                item_count=0,
                free_start=HEADER_SIZE,
                free_end=len(buf),
            ),
        )

    @staticmethod
    def _slot_offset(slot_id: int) -> int:
        return HEADER_SIZE + slot_id * _SLOT_SIZE

    @classmethod
    def insert_payload(cls, buf: bytearray, payload: bytes) -> int | None:
        hdr = read_header(buf)
        needed = len(payload) + _SLOT_SIZE
        if hdr.free_end - hdr.free_start < needed:
            cls.compact(buf)
            hdr = read_header(buf)
            if hdr.free_end - hdr.free_start < needed:
                return None
        slot_id = hdr.item_count
        new_end = hdr.free_end - len(payload)
        buf[new_end:hdr.free_end] = payload
        struct.pack_into(_SLOT_STRUCT, buf, cls._slot_offset(slot_id), new_end, len(payload))
        hdr.item_count += 1
        hdr.free_start += _SLOT_SIZE
        hdr.free_end = new_end
        write_header(buf, hdr)
        return slot_id

    @classmethod
    def get_payload(cls, buf: bytes | bytearray | memoryview, slot_id: int) -> bytes | None:
        hdr = read_header(buf)
        if slot_id < 0 or slot_id >= hdr.item_count:
            return None
        off, length = struct.unpack_from(_SLOT_STRUCT, buf, cls._slot_offset(slot_id))
        if off == _TOMBSTONE_OFFSET or length == 0:
            return None
        return bytes(buf[off:off + length])

    @classmethod
    def delete_slot(cls, buf: bytearray, slot_id: int) -> bool:
        hdr = read_header(buf)
        if slot_id < 0 or slot_id >= hdr.item_count:
            return False
        off, length = struct.unpack_from(_SLOT_STRUCT, buf, cls._slot_offset(slot_id))
        if off == _TOMBSTONE_OFFSET or length == 0:
            return False
        struct.pack_into(_SLOT_STRUCT, buf, cls._slot_offset(slot_id), _TOMBSTONE_OFFSET, 0)
        return True

    @classmethod
    def update_payload(cls, buf: bytearray, slot_id: int, payload: bytes) -> bool:
        hdr = read_header(buf)
        if slot_id < 0 or slot_id >= hdr.item_count:
            return False
        off, old_len = struct.unpack_from(_SLOT_STRUCT, buf, cls._slot_offset(slot_id))
        if off == _TOMBSTONE_OFFSET or old_len == 0:
            return False
        if len(payload) <= old_len:
            buf[off:off + len(payload)] = payload
            struct.pack_into(_SLOT_STRUCT, buf, cls._slot_offset(slot_id), off, len(payload))
            return True
        if hdr.free_end - hdr.free_start < len(payload):
            # Temporarily tombstone this slot so compact() frees its old bytes.
            struct.pack_into(_SLOT_STRUCT, buf, cls._slot_offset(slot_id), _TOMBSTONE_OFFSET, 0)
            cls.compact(buf)
            hdr = read_header(buf)
            if hdr.free_end - hdr.free_start < len(payload):
                # Cannot restore old bytes easily after compact; caller must relocate.
                return False
        new_end = hdr.free_end - len(payload)
        buf[new_end:hdr.free_end] = payload
        struct.pack_into(_SLOT_STRUCT, buf, cls._slot_offset(slot_id), new_end, len(payload))
        hdr.free_end = new_end
        write_header(buf, hdr)
        return True

    @classmethod
    def compact(cls, buf: bytearray) -> None:
        hdr = read_header(buf)
        live: list[tuple[int, bytes]] = []
        for slot_id in range(hdr.item_count):
            off, length = struct.unpack_from(_SLOT_STRUCT, buf, cls._slot_offset(slot_id))
            if off != _TOMBSTONE_OFFSET and length > 0:
                live.append((slot_id, bytes(buf[off:off + length])))
        cursor = len(buf)
        for slot_id, payload in live:
            cursor -= len(payload)
            buf[cursor:cursor + len(payload)] = payload
            struct.pack_into(_SLOT_STRUCT, buf, cls._slot_offset(slot_id), cursor, len(payload))
        hdr.free_end = cursor
        write_header(buf, hdr)

    @classmethod
    def live_slots(cls, buf: bytes | bytearray | memoryview) -> Iterator[tuple[int, bytes]]:
        hdr = read_header(buf)
        for slot_id in range(hdr.item_count):
            off, length = struct.unpack_from(_SLOT_STRUCT, buf, cls._slot_offset(slot_id))
            if off != _TOMBSTONE_OFFSET and length > 0:
                yield slot_id, bytes(buf[off:off + length])
