"""
Module: src/storage/page_header.py
Layer: Layer 1 — Universal Binary Page Header & Integrity Checksum
Lock Domain: Stateless (caller holds buffer pool frame pin or DiskManagerLock)

Purpose:
    Defines ``PageType``, ``PageFlags``, and the universal 32-byte
    ``PageHeader`` shared across every 8,192-byte relational row page,
    columnar chunk page, FSM/metadata page, and B+ Tree index page, together
    with 64-bit BLAKE2b page checksums and Milestone 4 ARIES ``page_lsn``
    accessors.

Key Invariants:
    1. Every persistent page begins at byte offset 0 with the exact 32-byte
       little-endian ``<IBBHQIIQ`` header.
    2. Checksums hash ``buf[0:24] + buf[32:8192]`` with 64-bit BLAKE2b,
       excluding bytes 24..31 so checksum stamping is idempotent.
    3. ``page_lsn`` occupies bytes 8..15 so Milestone 4 ARIES recovery and WAL
       flush enforcement operate without altering Milestone 1-2 page layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import hashlib
import struct

from src.common.config import HEADER_SIZE, INVALID_PAGE_ID, PAGE_SIZE

_HEADER_STRUCT = struct.Struct("<IBBHQIIQ")
_U64_STRUCT = struct.Struct("<Q")

_CHECKSUM_START: int = 24
_PAGE_LSN_OFFSET: int = 8


class PageType(IntEnum):
    """Canonical physical page type tags stored at header byte offset 4."""

    FREE_OR_FSM = 0
    SLOTTED_ROW = 1
    COLUMN_CHUNK = 2
    BPLUS_INTERNAL = 3
    BPLUS_LEAF = 4
    OBJECT_DIR = 5


class PageFlags(IntFlag):
    """Bitmask flags stored at header byte offset 5."""

    NONE = 0
    IS_ROOT = 1 << 0
    IS_LEAF = 1 << 1
    COMPRESSED = 1 << 2
    ENCRYPTED = 1 << 3


@dataclass
class PageHeader:
    """Universal 32-byte binary page header for 8,192-byte storage pages.

    Binary Layout (32 bytes, little-endian ``<IBBHQIIQ``):
    ```text
    Byte Offset:  0       4   5   6       8              16             20             24             32
                 +-------+---+---+-------+--------------+--------------+--------------+--------------+
                 |page_id|typ|flg| slots |   page_lsn   |free_space_ptr| next_page_id |   checksum   |
                 |  u32  | u8| u8|  u16  |     u64      |     u32      |     u32      |     u64      |
                 +-------+---+---+-------+--------------+--------------+--------------+--------------+
    ```
    """

    page_id: int = 0
    page_type: PageType | int = PageType.FREE_OR_FSM
    flags: PageFlags | int = PageFlags.NONE
    slot_count: int = 0
    page_lsn: int = 0
    free_space_ptr: int = 0
    next_page_id: int = INVALID_PAGE_ID
    checksum: int = 0

    def pack(self) -> bytes:
        """Serialize this header into a 32-byte little-endian byte string.

        Returns:
            Exact 32-byte representation packed with ``<IBBHQIIQ``.

        Raises:
            struct.error: If any integer field falls outside its unsigned range.

        Complexity:
            $O(1)$ time and space.
        """
        return _HEADER_STRUCT.pack(
            self.page_id,
            int(self.page_type),
            int(self.flags),
            self.slot_count,
            self.page_lsn,
            self.free_space_ptr,
            self.next_page_id,
            self.checksum,
        )

    def pack_into(self, buf: bytearray | memoryview, offset: int = 0) -> None:
        """Write this 32-byte header directly into ``buf`` at ``offset``.

        Args:
            buf: Writable ``bytearray`` or ``memoryview`` with at least
                ``offset + 32`` bytes.
            offset: Byte offset at which to place the header (defaults to ``0``).

        Raises:
            ValueError: If ``offset`` is negative or ``buf`` lacks 32 bytes
                starting at ``offset``.
            struct.error: If any header field overflows its binary width.

        Complexity:
            $O(1)$ in-place write; thread safety is governed by caller pin.
        """
        if offset < 0 or offset + HEADER_SIZE > len(buf):
            raise ValueError(
                f"Buffer length {len(buf)} cannot fit {HEADER_SIZE}-byte header at offset {offset}"
            )
        _HEADER_STRUCT.pack_into(
            buf,
            offset,
            self.page_id,
            int(self.page_type),
            int(self.flags),
            self.slot_count,
            self.page_lsn,
            self.free_space_ptr,
            self.next_page_id,
            self.checksum,
        )

    @classmethod
    def unpack(
        cls, data: bytes | bytearray | memoryview, offset: int = 0
    ) -> PageHeader:
        """Deserialize a 32-byte header from ``data`` starting at ``offset``.

        Args:
            data: Buffer containing at least ``offset + 32`` bytes.
            offset: Byte offset at which the header begins (defaults to ``0``).

        Returns:
            Decoded ``PageHeader`` instance with ``PageType`` and ``PageFlags``
            enumerations restored.

        Raises:
            ValueError: If ``offset`` is out of bounds, fewer than 32 bytes
                remain, or ``page_type`` is not a valid ``PageType`` member.

        Complexity:
            $O(1)$ time and space.
        """
        if offset < 0 or offset + HEADER_SIZE > len(data):
            raise ValueError(
                f"Buffer length {len(data)} too small for {HEADER_SIZE}-byte header at offset {offset}"
            )
        (
            page_id,
            raw_type,
            raw_flags,
            slot_count,
            page_lsn,
            free_space_ptr,
            next_page_id,
            checksum,
        ) = _HEADER_STRUCT.unpack_from(data, offset)
        return cls(
            page_id=page_id,
            page_type=PageType(raw_type),
            flags=PageFlags(raw_flags),
            slot_count=slot_count,
            page_lsn=page_lsn,
            free_space_ptr=free_space_ptr,
            next_page_id=next_page_id,
            checksum=checksum,
        )

    @staticmethod
    def compute_checksum(buf: bytes | bytearray | memoryview) -> int:
        """Compute the 64-bit BLAKE2b integrity digest for an 8,192-byte page.

        Args:
            buf: Full 8,192-byte page buffer.

        Returns:
            Unsigned 64-bit integer checksum decoded in little-endian order.

        Raises:
            ValueError: If ``len(buf) != PAGE_SIZE``.

        Complexity:
            ``O(PAGE_SIZE)`` over 8,184 hashed bytes; zero-copy slicing via
            ``memoryview``.
        """
        mv = memoryview(buf)
        if len(mv) != PAGE_SIZE:
            raise ValueError(f"Page buffer must be {PAGE_SIZE} bytes, got {len(mv)}")
        # Skip bytes 24..31 (the checksum slot itself) so stamp/verify remain consistent.
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(mv[:_CHECKSUM_START])
        hasher.update(mv[HEADER_SIZE:])
        return _U64_STRUCT.unpack(hasher.digest())[0]

    @staticmethod
    def stamp_checksum(buf: bytearray | memoryview) -> int:
        """Compute and write the 64-bit BLAKE2b checksum into bytes ``24..31``.

        Args:
            buf: Writable 8,192-byte page buffer.

        Returns:
            The stamped unsigned 64-bit checksum value.

        Raises:
            ValueError: If ``len(buf) != PAGE_SIZE``.

        Complexity:
            ``O(PAGE_SIZE)`` hash computation plus ``O(1)`` in-place write.
        """
        checksum = PageHeader.compute_checksum(buf)
        _U64_STRUCT.pack_into(buf, _CHECKSUM_START, checksum)
        return checksum

    @staticmethod
    def verify_checksum(buf: bytes | bytearray | memoryview) -> bool:
        """Verify that bytes ``24..31`` match the computed 64-bit BLAKE2b digest.

        Args:
            buf: Full 8,192-byte page buffer to validate.

        Returns:
            ``True`` when the stored checksum matches the computed digest,
            otherwise ``False``.

        Raises:
            ValueError: If ``len(buf) != PAGE_SIZE``.

        Complexity:
            ``O(PAGE_SIZE)`` time, ``O(1)`` extra space.
        """
        expected = PageHeader.compute_checksum(buf)
        stored = _U64_STRUCT.unpack_from(buf, _CHECKSUM_START)[0]
        return stored == expected

    @staticmethod
    def get_page_lsn(buf: bytes | bytearray | memoryview) -> int:
        """Read the 64-bit ``page_lsn`` field from header bytes ``8..15``.

        Args:
            buf: Page or header buffer of at least 32 bytes.

        Returns:
            Unsigned 64-bit Log Sequence Number stored on the page.

        Raises:
            ValueError: If ``len(buf) < HEADER_SIZE``.

        Complexity:
            $O(1)$ time and space.
        """
        if len(buf) < HEADER_SIZE:
            raise ValueError(
                f"Buffer length {len(buf)} smaller than {HEADER_SIZE}-byte header"
            )
        return _U64_STRUCT.unpack_from(buf, _PAGE_LSN_OFFSET)[0]

    @staticmethod
    def set_page_lsn(buf: bytearray | memoryview, lsn: int) -> None:
        """Update the 64-bit ``page_lsn`` field at header bytes ``8..15``.

        Args:
            buf: Writable page or header buffer of at least 32 bytes.
            lsn: New unsigned 64-bit Log Sequence Number.

        Raises:
            ValueError: If ``len(buf) < HEADER_SIZE`` or ``lsn`` is outside
                ``[0, 2**64 - 1]``.

        Complexity:
            $O(1)$ in-place write; caller must re-stamp checksum prior to disk
            write.
        """
        if len(buf) < HEADER_SIZE:
            raise ValueError(
                f"Buffer length {len(buf)} smaller than {HEADER_SIZE}-byte header"
            )
        if lsn < 0 or lsn > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"page_lsn must be uint64, got {lsn}")
        _U64_STRUCT.pack_into(buf, _PAGE_LSN_OFFSET, lsn)


__all__ = ["PageFlags", "PageHeader", "PageType"]
