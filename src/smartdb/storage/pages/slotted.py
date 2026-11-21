"""Binary slotted-page layout with in-page compaction."""
from __future__ import annotations

import struct
from typing import Iterator

from .header import HEADER_SIZE, PageHeader, PageType, read_header, write_header

_SLOT_STRUCT = "<HH"
_SLOT_SIZE = struct.calcsize(_SLOT_STRUCT)  # 4 bytes
_TOMBSTONE_OFFSET = 0xFFFF


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
