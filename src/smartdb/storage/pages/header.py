"""Common 24-byte page header envelope and page type tags shared across all page formats."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

HEADER_STRUCT = "<BBHHHqii"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 24 bytes


class PageType(IntEnum):
    """Physical page kind tags stored in byte 0 of every PageHeader."""

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
    """24-byte header shared across all page formats."""

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
