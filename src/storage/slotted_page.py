"""
Module: src/storage/slotted_page.py
Layer: Layer 1 — Row-Oriented Slotted Page Layout
Lock Domain: Caller must hold an active BufferPoolManager pin on the frame

Purpose:
    Implements an 8,192-byte row-oriented slotted page (``PageType.SLOTTED_ROW``)
    with a forward-growing 4-byte slot directory, backward-growing tuple heap,
    16-byte Milestone 4 MVCC tuple headers (``xmin``, ``xmax``), and
    RID-preserving in-page compaction.

Key Invariants:
    1. Slot IDs are immutable once assigned: ``delete_tuple()`` zeroes a slot's
       ``(offset, length)`` entry to ``(0, 0)`` without shifting subsequent
       slots, and ``insert_tuple()`` always appends at ``slot_count`` so
       external B+ Tree / LSM indexes never suffer dangling or aliased ``RID``s.
    2. Every stored tuple begins with a 16-byte little-endian ``<QQ`` MVCC
       header (``xmin: uint64 = 1``, ``xmax: uint64 = 0`` by default) followed
       immediately by the user payload bytes.
    3. ``compact()`` defragments live tuples toward offset ``8192`` and updates
       each live slot's ``tuple_offset`` in place without changing its
       ``slot_id``.
"""

from __future__ import annotations

from collections.abc import Iterator
import struct

from src.common.config import (
    HEADER_SIZE,
    INVALID_PAGE_ID,
    MVCC_HEADER_SIZE,
    PAGE_SIZE,
    SLOT_ENTRY_SIZE,
)
from src.common.exceptions import RecordTooLargeError
from src.storage.page_header import PageFlags, PageHeader, PageType

_SLOT_STRUCT = struct.Struct("<HH")
_MVCC_STRUCT = struct.Struct("<QQ")
_U16_STRUCT = struct.Struct("<H")
_U32_STRUCT = struct.Struct("<I")
_U64_STRUCT = struct.Struct("<Q")

_PAGE_ID_OFFSET: int = 0
_SLOT_COUNT_OFFSET: int = 6
_FREE_SPACE_PTR_OFFSET: int = 16
_XMAX_OFFSET_IN_TUPLE: int = 8

_MAX_UINT32: int = 0xFFFFFFFF
_MAX_UINT64: int = 0xFFFFFFFFFFFFFFFF
_ZERO_PAGE: bytes = bytes(PAGE_SIZE)


def _validate_writable_page_buffer(buf: bytearray | memoryview) -> None:
    """Validate that ``buf`` is a writable 8,192-byte ``bytearray`` or ``memoryview``."""
    if not isinstance(buf, (bytearray, memoryview)):
        raise TypeError(
            f"SlottedPage requires a bytearray or memoryview, got {type(buf).__name__}"
        )
    if isinstance(buf, memoryview) and buf.readonly:
        raise ValueError("SlottedPage requires a writable memoryview")
    if len(buf) != PAGE_SIZE:
        raise ValueError(f"Page buffer must be {PAGE_SIZE} bytes, got {len(buf)}")


def _validate_uint64(value: int, name: str) -> None:
    """Validate that ``value`` fits inside an unsigned 64-bit integer."""
    if value < 0 or value > _MAX_UINT64:
        raise ValueError(f"{name} must be uint64 in [0, {_MAX_UINT64}], got {value}")


class SlottedPage:
    """8,192-byte row-oriented page with RID-preserving slot directory and MVCC headers.

    Binary Layout (8,192 bytes):
    ```text
    0                  32                             32+4N              free_space_ptr        8192
    +------------------+------------------------------+------------------+--------------------+
    | PageHeader (32B) | Slot Dir (4B * slot_count)   |   Free Space     | Packed Tuples      |
    | <IBBHQIIQ>       | Slot i: <HH (offset, length) |                  | (grow downward)    |
    +------------------+------------------------------+------------------+--------------------+

    Slot Entry (4 bytes, at offset 32 + slot_id * 4):
    0             2             4
    +-------------+-------------+
    |tuple_offset |tuple_length |   Deleted slot sentinel: (0, 0)
    |    u16      |     u16     |
    +-------------+-------------+

    Stored Tuple Layout (tuple_length = 16 + len(user_payload)):
    0                  8                  16                                 tuple_length
    +------------------+------------------+----------------------------------+
    |   xmin (u64)     |   xmax (u64)     |       User Payload Bytes         |
    +------------------+------------------+----------------------------------+
    ```
    """

    def __init__(self, buf: bytearray | memoryview) -> None:
        """Wrap an existing initialized 8,192-byte ``SLOTTED_ROW`` page buffer.

        Args:
            buf: Writable ``bytearray`` or ``memoryview`` (e.g., from
                ``BufferPoolManager.fetch_page``) of exact length ``8192``.

        Raises:
            TypeError: If ``buf`` is neither a ``bytearray`` nor a ``memoryview``.
            ValueError: If ``buf`` is read-only, not ``8192`` bytes long, has a
                ``page_type`` other than ``PageType.SLOTTED_ROW``, or has
                corrupt slot/free-space bounds.

        Complexity:
            $O(1)$ time and space; operates zero-copy on ``buf``.
        """
        _validate_writable_page_buffer(buf)
        header = PageHeader.unpack(buf, 0)
        if header.page_type != PageType.SLOTTED_ROW:
            raise ValueError(
                f"Expected PageType.SLOTTED_ROW, got {header.page_type!r}"
            )
        slot_dir_end = HEADER_SIZE + header.slot_count * SLOT_ENTRY_SIZE
        if header.free_space_ptr < slot_dir_end or header.free_space_ptr > PAGE_SIZE:
            raise ValueError(
                f"Corrupt free_space_ptr {header.free_space_ptr} for slot_count {header.slot_count}"
            )
        self._buf: bytearray | memoryview = buf

    @classmethod
    def init_page(cls, buf: bytearray | memoryview, page_id: int) -> SlottedPage:
        """Initialize ``buf`` as a fresh, empty ``SLOTTED_ROW`` page.

        Args:
            buf: Writable ``8192``-byte ``bytearray`` or ``memoryview``.
            page_id: Zero-based physical page identifier stored in the header.

        Returns:
            A ``SlottedPage`` wrapper bound directly to ``buf``.

        Raises:
            TypeError: If ``buf`` is not a ``bytearray`` or ``memoryview``.
            ValueError: If ``buf`` is read-only, not ``8192`` bytes, or
                ``page_id`` is outside ``uint32`` bounds.

        Complexity:
            ``O(PAGE_SIZE)`` to zero-initialize the page buffer.
        """
        _validate_writable_page_buffer(buf)
        if page_id < 0 or page_id > _MAX_UINT32:
            raise ValueError(
                f"page_id must be uint32 in [0, {_MAX_UINT32}], got {page_id}"
            )

        buf[:] = _ZERO_PAGE
        PageHeader(
            page_id=page_id,
            page_type=PageType.SLOTTED_ROW,
            flags=PageFlags.NONE,
            slot_count=0,
            page_lsn=0,
            free_space_ptr=PAGE_SIZE,
            next_page_id=INVALID_PAGE_ID,
            checksum=0,
        ).pack_into(buf, 0)
        return cls(buf)

    @property
    def buffer(self) -> bytearray | memoryview:
        """Return the underlying 8,192-byte page buffer."""
        return self._buf

    @property
    def page_id(self) -> int:
        """Return the 32-bit physical page ID from header bytes ``0..3``."""
        return _U32_STRUCT.unpack_from(self._buf, _PAGE_ID_OFFSET)[0]

    @property
    def slot_count(self) -> int:
        """Return total allocated slot entries (live + tombstoned) from bytes ``6..7``."""
        return _U16_STRUCT.unpack_from(self._buf, _SLOT_COUNT_OFFSET)[0]

    @property
    def free_space_ptr(self) -> int:
        """Return byte offset where packed tuple storage begins from bytes ``16..19``."""
        return _U32_STRUCT.unpack_from(self._buf, _FREE_SPACE_PTR_OFFSET)[0]

    @property
    def header(self) -> PageHeader:
        """Unpack and return a snapshot of the 32-byte ``PageHeader``."""
        return PageHeader.unpack(self._buf, 0)

    def _set_slot_count(self, count: int) -> None:
        _U16_STRUCT.pack_into(self._buf, _SLOT_COUNT_OFFSET, count)

    def _set_free_space_ptr(self, ptr: int) -> None:
        _U32_STRUCT.pack_into(self._buf, _FREE_SPACE_PTR_OFFSET, ptr)

    def _validate_slot_id(self, slot_id: int) -> int:
        """Ensure ``slot_id`` lies in ``[0, slot_count)`` and return its byte offset."""
        count = self.slot_count
        if slot_id < 0 or slot_id >= count:
            raise IndexError(f"slot_id {slot_id} out of range [0, {count})")
        return HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE

    def _read_slot(self, slot_id: int) -> tuple[int, int]:
        """Return ``(tuple_offset, tuple_length)`` for a validated ``slot_id``."""
        slot_pos = self._validate_slot_id(slot_id)
        return _SLOT_STRUCT.unpack_from(self._buf, slot_pos)

    def _write_slot(self, slot_id: int, offset: int, length: int) -> None:
        """Write ``(tuple_offset, tuple_length)`` into ``slot_id``'s 4-byte entry."""
        slot_pos = HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE
        _SLOT_STRUCT.pack_into(self._buf, slot_pos, offset, length)

    def contiguous_free_space(self) -> int:
        """Return unallocated contiguous bytes between the slot directory and tuples.

        Returns:
            Exact byte count ``free_space_ptr - (32 + 4 * slot_count)``.

        Complexity:
            $O(1)$ time and space.
        """
        return self.free_space_ptr - (HEADER_SIZE + self.slot_count * SLOT_ENTRY_SIZE)

    def total_free_space(self) -> int:
        """Return total reclaimable free space after accounting for fragmented holes.

        Deleted or shrunk tuples leave dead holes inside ``[free_space_ptr, 8192)``,
        while deleted slot entries still occupy their 4-byte directory slots to
        preserve ``RID`` immutability.

        Returns:
            Contiguous bytes that would be free if ``compact()`` were executed.

        Complexity:
            ``O(slot_count)`` time, $O(1)$ space.
        """
        count = self.slot_count
        live_tuple_bytes = 0
        for slot_id in range(count):
            _, length = _SLOT_STRUCT.unpack_from(
                self._buf, HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE
            )
            if length > 0:
                live_tuple_bytes += length
        slot_dir_bytes = count * SLOT_ENTRY_SIZE
        return PAGE_SIZE - HEADER_SIZE - slot_dir_bytes - live_tuple_bytes

    def insert_tuple(
        self, payload: bytes, xmin: int = 1, xmax: int = 0
    ) -> int | None:
        """Insert a tuple prefixed with a 16-byte MVCC header into the page.

        Args:
            payload: Serialized user column bytes (may be empty ``b""``).
            xmin: Creating transaction ID written to MVCC bytes ``0..7``
                (defaults to ``1`` for M1-M2 committed bootstrap writes).
            xmax: Deleting transaction ID written to MVCC bytes ``8..15``
                (defaults to ``0`` for active tuples).

        Returns:
            Newly assigned immutable ``slot_id``, or ``None`` if total free space
            is insufficient even after automatic compaction.

        Raises:
            RecordTooLargeError: If ``len(payload) + 16 + 4 > PAGE_SIZE - HEADER_SIZE``.
            ValueError: If ``xmin`` or ``xmax`` falls outside ``uint64`` bounds.

        Complexity:
            $O(1)$ on fast path; ``O(slot_count + PAGE_SIZE)`` if
            fragmentation triggers automatic ``compact()``.
        """
        _validate_uint64(xmin, "xmin")
        _validate_uint64(xmax, "xmax")

        tuple_len = MVCC_HEADER_SIZE + len(payload)
        required = tuple_len + SLOT_ENTRY_SIZE
        if required > PAGE_SIZE - HEADER_SIZE:
            raise RecordTooLargeError(
                f"Tuple of {len(payload)} payload bytes ({required} bytes with MVCC + slot) "
                f"exceeds max page capacity {PAGE_SIZE - HEADER_SIZE}"
            )

        if self.contiguous_free_space() < required:
            if self.total_free_space() < required:
                return None
            self.compact()

        # Always append at slot_count rather than reusing deleted slots so old RIDs never alias new rows.
        slot_id = self.slot_count
        new_offset = self.free_space_ptr - tuple_len
        _MVCC_STRUCT.pack_into(self._buf, new_offset, xmin, xmax)
        if payload:
            self._buf[new_offset + MVCC_HEADER_SIZE : new_offset + tuple_len] = payload
        self._write_slot(slot_id, new_offset, tuple_len)
        self._set_slot_count(slot_id + 1)
        self._set_free_space_ptr(new_offset)
        return slot_id

    def get_tuple(self, slot_id: int, include_mvcc: bool = False) -> bytes | None:
        """Fetch a tuple's bytes by ``slot_id``, or ``None`` if deleted.

        Args:
            slot_id: Zero-based slot index in ``[0, slot_count)``.
            include_mvcc: When ``True``, include the leading 16-byte
                ``(xmin, xmax)`` MVCC header; otherwise strip it.

        Returns:
            Tuple payload bytes (or full 16B + payload when ``include_mvcc=True``),
            or ``None`` if the slot has been deleted.

        Raises:
            IndexError: If ``slot_id`` is outside ``[0, slot_count)``.

        Complexity:
            ``O(tuple_length)`` slice copy.
        """
        offset, length = self._read_slot(slot_id)
        if length == 0:
            return None
        start = offset if include_mvcc else offset + MVCC_HEADER_SIZE
        return bytes(self._buf[start : offset + length])

    def update_tuple(
        self,
        slot_id: int,
        new_payload: bytes,
        xmin: int | None = None,
        xmax: int | None = None,
    ) -> bool:
        """Update a live tuple in place, via intra-page move/compaction, or signal overflow.

        Args:
            slot_id: Zero-based live slot index in ``[0, slot_count)``.
            new_payload: Replacement serialized user payload bytes.
            xmin: Optional replacement ``xmin``; preserves existing ``xmin`` when ``None``.
            xmax: Optional replacement ``xmax``; preserves existing ``xmax`` when ``None``.

        Returns:
            ``True`` if the update succeeded on this page; ``False`` if the enlarged
            tuple does not fit even after compaction (signaling ``RowTableHeap`` to
            relocate the tuple to another page).

        Raises:
            IndexError: If ``slot_id`` is outside ``[0, slot_count)``.
            ValueError: If ``slot_id`` is already deleted, or ``xmin`` / ``xmax``
                fall outside ``uint64`` bounds.
            RecordTooLargeError: If ``len(new_payload) + 16 + 4 > PAGE_SIZE - HEADER_SIZE``.

        Complexity:
            ``O(len(new_payload))`` when updating in place or moving into
            contiguous free space; ``O(slot_count + PAGE_SIZE)`` if
            compaction is needed to reclaim fragmented space.
        """
        old_offset, old_len = self._read_slot(slot_id)
        if old_len == 0:
            raise ValueError(f"Cannot update deleted slot {slot_id}")

        if xmin is not None:
            _validate_uint64(xmin, "xmin")
        if xmax is not None:
            _validate_uint64(xmax, "xmax")

        new_len = MVCC_HEADER_SIZE + len(new_payload)
        if new_len + SLOT_ENTRY_SIZE > PAGE_SIZE - HEADER_SIZE:
            raise RecordTooLargeError(
                f"Updated tuple of {len(new_payload)} payload bytes exceeds max page capacity"
            )

        old_xmin, old_xmax = _MVCC_STRUCT.unpack_from(self._buf, old_offset)
        eff_xmin = old_xmin if xmin is None else xmin
        eff_xmax = old_xmax if xmax is None else xmax

        if new_len <= old_len:
            _MVCC_STRUCT.pack_into(self._buf, old_offset, eff_xmin, eff_xmax)
            if new_payload:
                self._buf[old_offset + MVCC_HEADER_SIZE : old_offset + new_len] = (
                    new_payload
                )
            self._write_slot(slot_id, old_offset, new_len)
            return True

        # Existing slot already occupies its 4B directory entry, so only new_len tuple bytes are needed.
        if self.contiguous_free_space() < new_len:
            # Exclude the old tuple's bytes when checking post-compaction fit so an update
            # does not needlessly fail because compact() preserved the stale version.
            if self.total_free_space() + old_len < new_len:
                return False
            self._write_slot(slot_id, 0, 0)
            self.compact()

        new_offset = self.free_space_ptr - new_len
        _MVCC_STRUCT.pack_into(self._buf, new_offset, eff_xmin, eff_xmax)
        if new_payload:
            self._buf[new_offset + MVCC_HEADER_SIZE : new_offset + new_len] = (
                new_payload
            )
        self._write_slot(slot_id, new_offset, new_len)
        self._set_free_space_ptr(new_offset)
        return True

    def delete_tuple(self, slot_id: int) -> bool:
        """Tombstone ``slot_id`` as ``(0, 0)`` without shifting slot indices.

        Args:
            slot_id: Zero-based slot index in ``[0, slot_count)``.

        Returns:
            ``True`` if the slot was live and is now marked deleted; ``False`` if
            it was already deleted.

        Raises:
            IndexError: If ``slot_id`` is outside ``[0, slot_count)``.

        Complexity:
            $O(1)$ time and space.
        """
        _, length = self._read_slot(slot_id)
        if length == 0:
            return False
        self._write_slot(slot_id, 0, 0)
        return True

    def compact(self) -> None:
        """Defragment all live tuples contiguously toward byte ``8192``.

        Updates each live slot's ``tuple_offset`` while keeping every ``slot_id``
        (including tombstoned ``(0, 0)`` slots) at its original directory index so
        external ``RID(page_id, slot_id)`` pointers remain immutable.

        Complexity:
            ``O(slot_count + PAGE_SIZE)`` time and up to
            ``O(PAGE_SIZE)`` temporary memory for safe non-overlapping copy.
        """
        count = self.slot_count
        live_entries: list[tuple[int, bytes]] = []
        for slot_id in range(count):
            offset, length = _SLOT_STRUCT.unpack_from(
                self._buf, HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE
            )
            if length > 0:
                live_entries.append((slot_id, bytes(self._buf[offset : offset + length])))

        cursor = PAGE_SIZE
        for slot_id, raw_tuple in live_entries:
            tup_len = len(raw_tuple)
            cursor -= tup_len
            self._buf[cursor : cursor + tup_len] = raw_tuple
            self._write_slot(slot_id, cursor, tup_len)

        slot_dir_end = HEADER_SIZE + count * SLOT_ENTRY_SIZE
        if cursor > slot_dir_end:
            self._buf[slot_dir_end:cursor] = bytes(cursor - slot_dir_end)
        self._set_free_space_ptr(cursor)

    def iter_live_slots(self, include_mvcc: bool = False) -> Iterator[tuple[int, bytes]]:
        """Yield ``(slot_id, payload)`` for every non-deleted slot in order.

        Args:
            include_mvcc: When ``True``, include the 16-byte MVCC header prefix;
                otherwise yield user payload bytes only (default ``False``).

        Yields:
            Tuples of ``(slot_id, bytes)`` for each live slot.

        Complexity:
            ``O(slot_count + live_bytes)`` across full iteration.
        """
        count = self.slot_count
        for slot_id in range(count):
            payload = self.get_tuple(slot_id, include_mvcc=include_mvcc)
            if payload is not None:
                yield slot_id, payload

    def get_tuple_mvcc_header(self, slot_id: int) -> tuple[int, int]:
        """Read the 16-byte ``(xmin, xmax)`` MVCC header of a live tuple.

        Args:
            slot_id: Zero-based live slot index in ``[0, slot_count)``.

        Returns:
            Tuple ``(xmin, xmax)`` of unsigned 64-bit transaction IDs.

        Raises:
            IndexError: If ``slot_id`` is outside ``[0, slot_count)``.
            ValueError: If ``slot_id`` is deleted.

        Complexity:
            $O(1)$ time and space.
        """
        offset, length = self._read_slot(slot_id)
        if length == 0:
            raise ValueError(f"Slot {slot_id} is deleted")
        return _MVCC_STRUCT.unpack_from(self._buf, offset)

    def set_tuple_xmax(self, slot_id: int, xmax: int) -> None:
        """Update the 64-bit ``xmax`` field in a live tuple's MVCC header in place.

        Args:
            slot_id: Zero-based live slot index in ``[0, slot_count)``.
            xmax: Unsigned 64-bit deleting transaction ID.

        Raises:
            IndexError: If ``slot_id`` is outside ``[0, slot_count)``.
            ValueError: If ``slot_id`` is deleted or ``xmax`` falls outside ``uint64``.

        Complexity:
            $O(1)$ in-place write; caller must mark the buffer pool frame dirty.
        """
        _validate_uint64(xmax, "xmax")
        offset, length = self._read_slot(slot_id)
        if length == 0:
            raise ValueError(f"Slot {slot_id} is deleted")
        _U64_STRUCT.pack_into(self._buf, offset + _XMAX_OFFSET_IN_TUPLE, xmax)


__all__ = ["SlottedPage"]
