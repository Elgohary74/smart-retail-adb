"""
Module: src/storage/table_heap.py
Layer: Layer 1 — Multi-Page Row & Column Table Heaps
Lock Domain: TableHeapLock (``self._lock = threading.RLock()``; acquired before BufferPoolLock)

Purpose:
    Implements multi-page table storage abstractions over ``BufferPoolManager``:
    ``RowTableHeap`` (backed by 8,192-byte ``SlottedPage`` heaps for OLTP
    tables like ``customers``, ``products``, ``inventory``, and ``orders``) and
    ``ColumnTableHeap`` (backed by aligned per-column ``ColumnPage`` files for
    OLAP analytics on ``order_items`` and Milestone 3/5 vectorized scans).

Key Invariants:
    1. Lock ordering strictly follows ``TableHeapLock -> BufferPoolLock ->
       DiskManagerLock``; every pinned frame is released inside ``try/finally``
       before returning or relocating across pages.
    2. ``RowTableHeap.update()`` relocates overflowing tuples to another page
       while tombstoning the old slot as ``(0, 0)``, preserving ``slot_id``
       immutability for unaffected records and returning the new ``RID``.
    3. ``ColumnTableHeap`` keeps every column's ``.col`` file aligned in
       lockstep so logical row ordinal ``r`` maps to identical
       ``(page_id, slot_id)`` coordinates across all columns (up to
       ``chunk_rows = 1020`` rows per page for 8-byte scalar schemas).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import threading
from typing import Any

import numpy as np

from src.common.config import (
    CHUNK_ROWS,
    HEADER_SIZE,
    MVCC_HEADER_SIZE,
    PAGE_SIZE,
    SLOT_ENTRY_SIZE,
)
from src.common.exceptions import KeyNotFoundError, RecordTooLargeError, StorageError
from src.common.types import RID, ColumnDef, DataType, LayoutType, Schema
from src.storage.buffer_pool import BufferPoolManager
from src.storage.column_page import ColumnPage
from src.storage.page_header import PageHeader
from src.storage.slotted_page import SlottedPage

_MAX_SLOTTED_PAYLOAD: int = PAGE_SIZE - HEADER_SIZE - MVCC_HEADER_SIZE - SLOT_ENTRY_SIZE
_MAX_COLUMN_VARCHAR_BYTES: int = PAGE_SIZE - HEADER_SIZE - SLOT_ENTRY_SIZE

_INT_BOUNDS: dict[DataType, tuple[int, int]] = {
    DataType.INT32: (-(1 << 31), (1 << 31) - 1),
    DataType.INT64: (-(1 << 63), (1 << 63) - 1),
    DataType.UINT64: (0, (1 << 64) - 1),
}


def _validate_column_value(col: ColumnDef, value: Any) -> int | None:
    """Validate one column value against ``col`` and return UTF-8 byte length for VARCHAR."""
    if value is None:
        raise ValueError(f"Column {col.name!r} cannot be None in binary column storage")

    if col.dtype == DataType.VARCHAR:
        if isinstance(value, str):
            byte_len = len(value.encode("utf-8"))
        elif isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            raw.decode("utf-8")
            byte_len = len(raw)
        else:
            raise TypeError(
                f"VARCHAR column {col.name!r} expects str or UTF-8 bytes, got {type(value).__name__}"
            )
        if byte_len > _MAX_COLUMN_VARCHAR_BYTES:
            raise RecordTooLargeError(
                f"VARCHAR column {col.name!r} value of {byte_len} bytes exceeds max page "
                f"capacity {_MAX_COLUMN_VARCHAR_BYTES}"
            )
        if byte_len > col.max_len:
            raise ValueError(
                f"Column {col.name!r} length {byte_len} exceeds max_len={col.max_len}"
            )
        return byte_len

    if col.dtype == DataType.FLOAT64:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"FLOAT64 column {col.name!r} expects float or int, got {type(value).__name__}"
            )
        return None

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{col.dtype.name} column {col.name!r} expects int, got {type(value).__name__}"
        )
    iv = int(value)
    lo, hi = _INT_BOUNDS[col.dtype]
    if iv < lo or iv > hi:
        raise ValueError(
            f"{col.dtype.name} column {col.name!r} value {iv} out of range [{lo}, {hi}]"
        )
    return None


class RowTableHeap:
    """Multi-page row-oriented table heap managing 8,192-byte ``SlottedPage`` files.

    Page & RID Coordination:
    ```text
    Disk File: <table_name>.db
    +--------------------+--------------------+--------------------+-----+
    | Page 0: FSM Bitmap | Page 1: SlottedRow | Page 2: SlottedRow | ... |
    | (allocated_pages   | slots 0..k-1       | slots 0..m-1       |     |
    |  excludes Page 0)  | (cached last page) | <- new_page()      |     |
    +--------------------+--------------------+--------------------+-----+
    Insert: tries cached _last_page_id first -> allocates bpm.new_page() on full.
    Update: updates in place / compacts page -> relocates + tombstones old slot on overflow.
    ```
    """

    def __init__(self, schema: Schema, bpm: BufferPoolManager) -> None:
        """Initialize a row table heap bound to ``schema`` and ``bpm``.

        Discovers any existing allocated data pages (``page_id >= 1``) from
        ``bpm.disk_manager.allocated_pages()`` and caches the highest page ID as
        ``_last_page_id`` for $O(1)$ append placement.

        Args:
            schema: Relational ``Schema`` governing row encoding and decoding.
            bpm: ``BufferPoolManager`` backed by this table's ``*.db`` file.

        Raises:
            TypeError: If ``schema`` is not a ``Schema`` or ``bpm`` is not a
                ``BufferPoolManager``.
            StorageError: If reading FSM allocation state fails.

        Complexity:
            ``O(FSM_BYTES)`` on initialization to scan existing allocated pages.
        """
        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema, got {type(schema).__name__}")
        if not isinstance(bpm, BufferPoolManager):
            raise TypeError(
                f"bpm must be a BufferPoolManager, got {type(bpm).__name__}"
            )

        self.schema: Schema = schema
        self._bpm: BufferPoolManager = bpm
        self._lock: threading.RLock = threading.RLock()
        self._closed: bool = False

        allocated = self._bpm.disk_manager.allocated_pages()
        self._last_page_id: int | None = allocated[-1] if allocated else None

    @property
    def bpm(self) -> BufferPoolManager:
        """Return the underlying ``BufferPoolManager`` for this table."""
        return self._bpm

    def _ensure_open(self) -> None:
        """Raise ``StorageError`` if this table heap has been closed."""
        if self._closed:
            raise StorageError(f"RowTableHeap for {self.schema.table_name!r} is closed")

    def _insert_payload(self, payload: bytes) -> RID:
        """Insert serialized row payload under ``self._lock`` and return its ``RID``."""
        if self._last_page_id is not None:
            target_pid = self._last_page_id
            buf = self._bpm.fetch_page(target_pid)
            dirty = False
            try:
                sp = SlottedPage(buf)
                slot_id = sp.insert_tuple(payload)
                if slot_id is not None:
                    dirty = True
                    return RID(page_id=target_pid, slot_id=slot_id)
            finally:
                self._bpm.unpin_page(target_pid, is_dirty=dirty)

        new_pid, buf = self._bpm.new_page()
        try:
            sp = SlottedPage.init_page(buf, page_id=new_pid)
            slot_id = sp.insert_tuple(payload)
            if slot_id is None:
                raise RecordTooLargeError(
                    f"Encoded row ({len(payload)} bytes) exceeds empty SlottedPage capacity"
                )
            self._last_page_id = new_pid
            return RID(page_id=new_pid, slot_id=slot_id)
        finally:
            self._bpm.unpin_page(new_pid, is_dirty=True)

    def insert(self, row: Mapping[str, Any] | Sequence[Any]) -> RID:
        """Encode and insert ``row`` into the heap, returning its assigned ``RID``.

        Args:
            row: Column-name mapping or ordered value sequence matching ``schema``.

        Returns:
            Newly assigned ``RID(page_id, slot_id)``.

        Raises:
            KeyError: If a required column is missing from a mapping input.
            TypeError: If any column value has an incompatible Python type.
            ValueError: If positional length mismatches, a value is ``None``, or
                a ``VARCHAR`` exceeds ``col.max_len``.
            RecordTooLargeError: If the encoded tuple plus 16B MVCC header and
                4B slot entry exceeds single-page capacity (8,160 bytes).
            BufferPoolFullError: If all buffer frames are pinned on miss/alloc.
            StorageError: If the heap is closed or disk I/O fails.

        Complexity:
            $O(1)$ buffer pool pin/unpin on fast path; ``O(PAGE_SIZE)`` when
            compacting or initializing a new page. Thread-safe under ``self._lock``.
        """
        payload = self.schema.encode_row(row)
        # Validate single-page capacity before locking/allocating so oversized rows never leak pages.
        if len(payload) > _MAX_SLOTTED_PAYLOAD:
            raise RecordTooLargeError(
                f"Encoded row payload ({len(payload)} bytes) exceeds max SlottedPage "
                f"payload {_MAX_SLOTTED_PAYLOAD}"
            )
        with self._lock:
            self._ensure_open()
            return self._insert_payload(payload)

    def get(self, rid: RID) -> dict[str, Any] | None:
        """Fetch and decode the row at ``rid``, or ``None`` if deleted or absent.

        Args:
            rid: Target ``RID(page_id, slot_id)``.

        Returns:
            Decoded column dictionary, or ``None`` if ``rid`` points to an
            unallocated page, an out-of-range slot, or a deleted ``(0, 0)`` slot.

        Raises:
            TypeError: If ``rid`` is not a ``RID``.
            StorageError: If the heap is closed or disk read fails.
            PageCorruptedError: If page checksum verification fails.
            BufferPoolFullError: If all frames in the pool are pinned on miss.

        Complexity:
            $O(1)$ buffer pool lookup plus tuple decoding. Thread-safe under
            ``self._lock``.
        """
        if not isinstance(rid, RID):
            raise TypeError(f"rid must be a RID, got {type(rid).__name__}")
        with self._lock:
            self._ensure_open()
            if (
                rid.page_id <= 0
                or rid.slot_id < 0
                or not self._bpm.disk_manager.is_allocated(rid.page_id)
            ):
                return None

            buf = self._bpm.fetch_page(rid.page_id)
            try:
                sp = SlottedPage(buf)
                if rid.slot_id >= sp.slot_count:
                    return None
                payload = sp.get_tuple(rid.slot_id, include_mvcc=False)
                if payload is None:
                    return None
                return self.schema.decode_row(payload)
            finally:
                self._bpm.unpin_page(rid.page_id, is_dirty=False)

    def update(self, rid: RID, new_row: Mapping[str, Any] | Sequence[Any]) -> RID:
        """Update the tuple at ``rid``, relocating to a new page if it overflows.

        Args:
            rid: Existing live ``RID(page_id, slot_id)`` to update.
            new_row: Replacement row mapping or positional sequence.

        Returns:
            Same ``rid`` if updated in place or moved within the same page, or
            a **new ``RID``** if page overflow forced relocation.

        Raises:
            TypeError: If ``rid`` is not a ``RID`` or column types are invalid.
            KeyError: If a column is missing from ``new_row``.
            ValueError: If ``new_row`` violates column constraints.
            RecordTooLargeError: If ``new_row`` exceeds single-page capacity.
            KeyNotFoundError: If ``rid`` references an unallocated page,
                out-of-range slot, or already-deleted slot.
            BufferPoolFullError: If no evictable buffer frame is available.
            StorageError: If the heap is closed or disk I/O fails.

        Complexity:
            $O(1)$ on in-place update; up to ``O(PAGE_SIZE)`` on intra-page
            compaction or cross-page relocation. Thread-safe under ``self._lock``.
        """
        if not isinstance(rid, RID):
            raise TypeError(f"rid must be a RID, got {type(rid).__name__}")

        payload = self.schema.encode_row(new_row)
        if len(payload) > _MAX_SLOTTED_PAYLOAD:
            raise RecordTooLargeError(
                f"Updated row payload ({len(payload)} bytes) exceeds max SlottedPage "
                f"payload {_MAX_SLOTTED_PAYLOAD}"
            )

        with self._lock:
            self._ensure_open()
            if (
                rid.page_id <= 0
                or rid.slot_id < 0
                or not self._bpm.disk_manager.is_allocated(rid.page_id)
            ):
                raise KeyNotFoundError(
                    f"Cannot update RID {rid}: page is not allocated"
                )

            buf = self._bpm.fetch_page(rid.page_id)
            dirty = False
            overflowed = False
            try:
                sp = SlottedPage(buf)
                if rid.slot_id >= sp.slot_count or sp.get_tuple(rid.slot_id) is None:
                    raise KeyNotFoundError(
                        f"Cannot update RID {rid}: slot is missing or deleted"
                    )
                if sp.update_tuple(rid.slot_id, payload):
                    dirty = True
                    return rid
                overflowed = True
            finally:
                self._bpm.unpin_page(rid.page_id, is_dirty=dirty)

            # Secure the relocated tuple before tombstoning the old slot so a
            # BufferPoolFullError during allocation never drops the existing row.
            assert overflowed
            new_rid = self._insert_payload(payload)
            try:
                old_buf = self._bpm.fetch_page(rid.page_id)
                old_dirty = False
                try:
                    old_sp = SlottedPage(old_buf)
                    old_dirty = old_sp.delete_tuple(rid.slot_id)
                finally:
                    self._bpm.unpin_page(rid.page_id, is_dirty=old_dirty)
            except Exception:
                # Compensating cleanup avoids leaving duplicate live rows if tombstoning fails.
                try:
                    self.delete(new_rid)
                except Exception:
                    pass
                raise
            return new_rid

    def delete(self, rid: RID) -> bool:
        """Tombstone the tuple at ``rid`` without shifting sibling slot IDs.

        Args:
            rid: Target ``RID(page_id, slot_id)``.

        Returns:
            ``True`` if the slot existed and was newly deleted, else ``False``.

        Raises:
            TypeError: If ``rid`` is not a ``RID``.
            StorageError: If the heap is closed or disk I/O fails.
            BufferPoolFullError: If all buffer frames are pinned on miss.

        Complexity:
            $O(1)$ buffer pool pin/unpin. Thread-safe under ``self._lock``.
        """
        if not isinstance(rid, RID):
            raise TypeError(f"rid must be a RID, got {type(rid).__name__}")
        with self._lock:
            self._ensure_open()
            if (
                rid.page_id <= 0
                or rid.slot_id < 0
                or not self._bpm.disk_manager.is_allocated(rid.page_id)
            ):
                return False

            buf = self._bpm.fetch_page(rid.page_id)
            deleted = False
            try:
                sp = SlottedPage(buf)
                if rid.slot_id < sp.slot_count:
                    deleted = sp.delete_tuple(rid.slot_id)
                return deleted
            finally:
                self._bpm.unpin_page(rid.page_id, is_dirty=deleted)

    def full_scan(self) -> list[tuple[RID, dict[str, Any]]]:
        """Scan every allocated data page and return all live ``(RID, row)`` pairs.

        Returns a concrete ``list`` (rather than a single-use iterator) so
        callers such as query verification and index equivalence suites can
        inspect ``len(rows)`` and iterate repeatedly.

        Returns:
            List of ``(RID, decoded_row_dict)`` ordered by ascending
            ``(page_id, slot_id)``.

        Raises:
            StorageError: If the heap is closed or disk I/O fails.
            PageCorruptedError: If any page fails checksum validation.
            BufferPoolFullError: If all buffer frames are pinned on miss.

        Complexity:
            $O(P + N)$ where $P$ is allocated data pages and $N$ is live tuples.
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            rows: list[tuple[RID, dict[str, Any]]] = []
            for page_id in self._bpm.disk_manager.allocated_pages():
                buf = self._bpm.fetch_page(page_id)
                try:
                    sp = SlottedPage(buf)
                    for slot_id, payload in sp.iter_live_slots(include_mvcc=False):
                        rows.append(
                            (
                                RID(page_id=page_id, slot_id=slot_id),
                                self.schema.decode_row(payload),
                            )
                        )
                finally:
                    self._bpm.unpin_page(page_id, is_dirty=False)
            return rows

    def page_count(self) -> int:
        """Return the number of allocated data pages (excluding reserved Page 0).

        Returns:
            Count of allocated data pages in ``*.db``.

        Raises:
            StorageError: If the heap is closed.

        Complexity:
            ``O(FSM_BYTES)`` bitmap scan under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            return len(self._bpm.disk_manager.allocated_pages())

    def flush(self) -> None:
        """Flush all dirty pages in the buffer pool and ``fsync`` the disk file.

        Raises:
            StorageError: If the heap is closed or flushing fails.
        """
        with self._lock:
            self._ensure_open()
            self._bpm.flush_all_pages()

    def close(self) -> None:
        """Flush dirty pages and idempotently close the underlying ``DiskManager``.

        Raises:
            StorageError: If flushing or closing the disk file fails.
        """
        with self._lock:
            if self._closed:
                return
            try:
                self._bpm.flush_all_pages()
            finally:
                self._closed = True
                self._bpm.disk_manager.close()

    def get_stats(self) -> dict[str, Any]:
        """Return heap page count and buffer pool telemetry."""
        with self._lock:
            self._ensure_open()
            return {
                "table_name": self.schema.table_name,
                "layout": LayoutType.ROW.value,
                "page_count": len(self._bpm.disk_manager.allocated_pages()),
                "buffer_pool": self._bpm.get_stats(),
            }


class ColumnTableHeap:
    """Multi-file columnar table heap storing one aligned ``.col`` file per column.

    Cross-Column Chunk Alignment:
    ```text
    Columns: c_0 (<tbl>.c_0.col), c_1 (<tbl>.c_1.col), ..., c_{k-1} (<tbl>.c_{k-1}.col)
    +-----------------------------------------------------------------------------+
    | Logical Chunk Page p (page_id >= 1, up to chunk_rows = 1020 rows/page):     |
    |   c_0.col[p]: [v_0[0], v_0[1], ..., v_0[m-1]]  (slot_count == m)            |
    |   c_1.col[p]: [v_1[0], v_1[1], ..., v_1[m-1]]  (slot_count == m)            |
    |   ...                                                                       |
    |   RID(p, s) selects slot s on physical page p across EVERY column file.     |
    +-----------------------------------------------------------------------------+
    ```
    """

    def __init__(
        self,
        schema: Schema,
        column_bpms: Mapping[str, BufferPoolManager],
        chunk_rows: int = CHUNK_ROWS,
    ) -> None:
        """Initialize a columnar table heap across per-column buffer pools.

        Args:
            schema: Relational ``Schema`` defining ordered columns.
            column_bpms: Mapping from each column name in ``schema`` to its
                dedicated ``BufferPoolManager`` (backing ``<table>.<col>.col``).
            chunk_rows: Maximum row count per aligned column chunk page
                (defaults to ``CHUNK_ROWS = 1020``).

        Raises:
            TypeError: If ``schema``, ``column_bpms``, or ``chunk_rows`` has an
                invalid type.
            ValueError: If ``chunk_rows <= 0`` or ``column_bpms`` keys do not
                match ``schema.columns``.
            StorageError: If existing column files have mismatched allocated
                page sets.

        Complexity:
            ``O(K * FSM_BYTES)`` for ``K`` columns to verify page alignment.
        """
        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema, got {type(schema).__name__}")
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int):
            raise TypeError(f"chunk_rows must be an int, got {type(chunk_rows).__name__}")
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
        if not isinstance(column_bpms, Mapping):
            raise TypeError(
                f"column_bpms must be a mapping, got {type(column_bpms).__name__}"
            )

        schema_col_names = {col.name for col in schema.columns}
        provided_names = set(column_bpms.keys())
        missing = schema_col_names - provided_names
        if missing:
            raise ValueError(
                f"Missing BufferPoolManager(s) for columns {sorted(missing)} "
                f"in schema {schema.table_name!r}"
            )
        extra = provided_names - schema_col_names
        if extra:
            raise ValueError(
                f"Unexpected BufferPoolManager(s) for unknown columns {sorted(extra)} "
                f"in schema {schema.table_name!r}"
            )

        ordered_bpms: dict[str, BufferPoolManager] = {}
        for col in schema.columns:
            bpm = column_bpms[col.name]
            if not isinstance(bpm, BufferPoolManager):
                raise TypeError(
                    f"column_bpms[{col.name!r}] must be a BufferPoolManager, "
                    f"got {type(bpm).__name__}"
                )
            ordered_bpms[col.name] = bpm

        # Cap effective chunk_rows by the tightest fixed-width column physical page limit
        # so all columns always advance to a new page at the exact same row boundary.
        payload_bytes = PAGE_SIZE - HEADER_SIZE
        fixed_caps = [
            payload_bytes // col.dtype.fixed_size
            for col in schema.columns
            if col.dtype != DataType.VARCHAR
        ]
        max_aligned_rows = (
            min(fixed_caps)
            if fixed_caps
            else payload_bytes // SLOT_ENTRY_SIZE
        )

        self.schema: Schema = schema
        self._column_bpms: dict[str, BufferPoolManager] = ordered_bpms
        self._chunk_rows: int = min(chunk_rows, max_aligned_rows)
        self._has_varchar: bool = any(
            col.dtype == DataType.VARCHAR for col in schema.columns
        )
        self._lock: threading.RLock = threading.RLock()
        self._closed: bool = False

        first_name = schema.columns[0].name
        expected_pages = ordered_bpms[first_name].disk_manager.allocated_pages()
        for col in schema.columns[1:]:
            col_pages = ordered_bpms[col.name].disk_manager.allocated_pages()
            if col_pages != expected_pages:
                raise StorageError(
                    f"Column files for {schema.table_name!r} are misaligned: "
                    f"{first_name!r} has pages {expected_pages}, "
                    f"{col.name!r} has {col_pages}"
                )
        self._last_page_id: int | None = expected_pages[-1] if expected_pages else None

    @property
    def chunk_rows(self) -> int:
        """Return the aligned row capacity per column chunk page."""
        return self._chunk_rows

    @property
    def column_bpms(self) -> dict[str, BufferPoolManager]:
        """Return a shallow mapping copy from column name to ``BufferPoolManager``."""
        return dict(self._column_bpms)

    def _ensure_open(self) -> None:
        """Raise ``StorageError`` if this column table heap has been closed."""
        if self._closed:
            raise StorageError(
                f"ColumnTableHeap for {self.schema.table_name!r} is closed"
            )

    def _extract_row_values(
        self, row: Mapping[str, Any] | Sequence[Any]
    ) -> tuple[list[Any], list[int | None]]:
        """Extract ordered column values from ``row`` and validate schema constraints."""
        if isinstance(row, Mapping):
            ordered = [row[col.name] for col in self.schema.columns]
        else:
            if len(row) != len(self.schema.columns):
                raise ValueError(
                    f"Expected {len(self.schema.columns)} values for "
                    f"{self.schema.table_name!r}, got {len(row)}"
                )
            ordered = list(row)

        byte_lens: list[int | None] = []
        for col, val in zip(self.schema.columns, ordered):
            byte_lens.append(_validate_column_value(col, val))
        return ordered, byte_lens

    def _page_available_rows_single(
        self, page_id: int, byte_lens: list[int | None]
    ) -> tuple[bool, int]:
        """Check whether all columns can accept one row on ``page_id``."""
        first_col = self.schema.columns[0]
        first_bpm = self._column_bpms[first_col.name]
        buf = first_bpm.fetch_page(page_id)
        try:
            cp = ColumnPage(buf, first_col.dtype, max_rows=self._chunk_rows)
            slot_count = cp.slot_count
            if slot_count >= cp.capacity:
                return False, slot_count
            if first_col.dtype == DataType.VARCHAR:
                needed = SLOT_ENTRY_SIZE + (byte_lens[0] or 0)
                if cp.contiguous_free_space() < needed:
                    return False, slot_count
        finally:
            first_bpm.unpin_page(page_id, is_dirty=False)

        if self._has_varchar:
            for idx, col in enumerate(self.schema.columns[1:], start=1):
                if col.dtype != DataType.VARCHAR:
                    continue
                bpm = self._column_bpms[col.name]
                cbuf = bpm.fetch_page(page_id)
                try:
                    ccp = ColumnPage(cbuf, col.dtype, max_rows=self._chunk_rows)
                    needed = SLOT_ENTRY_SIZE + (byte_lens[idx] or 0)
                    if ccp.slot_count >= ccp.capacity or ccp.contiguous_free_space() < needed:
                        return False, slot_count
                finally:
                    bpm.unpin_page(page_id, is_dirty=False)

        return True, slot_count

    def _delete_page_all_columns(self, page_id: int) -> None:
        """Best-effort unpin and FSM deletion of ``page_id`` across every column."""
        for col in self.schema.columns:
            bpm = self._column_bpms[col.name]
            try:
                bpm.unpin_page(page_id, is_dirty=False)
            except Exception:
                pass
            try:
                bpm.delete_page(page_id)
            except Exception:
                pass

    def _rollback_page_writes(
        self,
        page_id: int,
        was_new_page: bool,
        mutated_cols: list[tuple[BufferPoolManager, int, int]],
    ) -> None:
        """Restore column pages after a partial multi-column write failure."""
        if was_new_page:
            self._delete_page_all_columns(page_id)
            return

        for bpm, old_slots, old_free in mutated_cols:
            try:
                buf = bpm.fetch_page(page_id)
                restored = False
                try:
                    header = PageHeader.unpack(buf, 0)
                    header.slot_count = old_slots
                    header.free_space_ptr = old_free
                    header.pack_into(buf, 0)
                    restored = True
                finally:
                    bpm.unpin_page(page_id, is_dirty=restored)
            except Exception:
                pass

    def _allocate_aligned_page(self) -> int:
        """Allocate and initialize an empty aligned ``ColumnPage`` across all columns.

        Rolls back any previously allocated column pages if any column fails
        (e.g., ``BufferPoolFullError``, page ID mismatch, or init error) so
        column files never diverge or leak FSM pages.
        """
        created: list[tuple[BufferPoolManager, int]] = []
        target_pid: int | None = None
        try:
            for col in self.schema.columns:
                bpm = self._column_bpms[col.name]
                pid, buf = bpm.new_page()
                created.append((bpm, pid))
                try:
                    if target_pid is None:
                        target_pid = pid
                    elif pid != target_pid:
                        raise StorageError(
                            f"Allocated page ID mismatch in {self.schema.table_name!r}: "
                            f"expected {target_pid}, got {pid} on column {col.name!r}"
                        )
                    ColumnPage.init_page(
                        buf, page_id=pid, dtype=col.dtype, max_rows=self._chunk_rows
                    )
                finally:
                    bpm.unpin_page(pid, is_dirty=True)
        except Exception:
            for col_bpm, col_pid in created:
                try:
                    col_bpm.unpin_page(col_pid, is_dirty=False)
                except Exception:
                    pass
                try:
                    col_bpm.delete_page(col_pid)
                except Exception:
                    pass
            raise

        assert target_pid is not None
        return target_pid

    def _compute_batch_fit(
        self,
        page_id: int,
        col_values: list[list[Any]],
        varchar_lens: dict[int, list[int]],
        offset: int,
        total: int,
    ) -> tuple[int, int]:
        """Return ``(start_slot, fit_count)`` across all columns for ``page_id``."""
        first_col = self.schema.columns[0]
        first_bpm = self._column_bpms[first_col.name]
        buf = first_bpm.fetch_page(page_id)
        try:
            cp = ColumnPage(buf, first_col.dtype, max_rows=self._chunk_rows)
            start_slot = cp.slot_count
            fit_count = min(cp.capacity - start_slot, total - offset)
            if fit_count <= 0:
                return start_slot, 0
            if first_col.dtype == DataType.VARCHAR:
                fit_count = self._max_varchar_fit(
                    cp.contiguous_free_space(), varchar_lens[0], offset, fit_count
                )
        finally:
            first_bpm.unpin_page(page_id, is_dirty=False)

        if fit_count > 0 and self._has_varchar:
            for idx, col in enumerate(self.schema.columns[1:], start=1):
                if col.dtype != DataType.VARCHAR:
                    continue
                bpm = self._column_bpms[col.name]
                cbuf = bpm.fetch_page(page_id)
                try:
                    ccp = ColumnPage(cbuf, col.dtype, max_rows=self._chunk_rows)
                    fit_count = min(fit_count, ccp.capacity - ccp.slot_count)
                    fit_count = self._max_varchar_fit(
                        ccp.contiguous_free_space(),
                        varchar_lens[idx],
                        offset,
                        fit_count,
                    )
                finally:
                    bpm.unpin_page(page_id, is_dirty=False)
                if fit_count == 0:
                    break
        return start_slot, fit_count

    @staticmethod
    def _max_varchar_fit(
        free_bytes: int, byte_lens: list[int], offset: int, limit: int
    ) -> int:
        """Compute how many consecutive VARCHAR entries from ``offset`` fit in ``free_bytes``."""
        used = 0
        taken = 0
        for idx in range(offset, offset + limit):
            need = SLOT_ENTRY_SIZE + byte_lens[idx]
            if used + need > free_bytes:
                break
            used += need
            taken += 1
        return taken

    def insert(self, row: Mapping[str, Any] | Sequence[Any]) -> RID:
        """Append a single row across all aligned column files and return its ``RID``.

        Args:
            row: Column-name mapping or ordered sequence matching ``schema``.

        Returns:
            ``RID(page_id, slot_id)`` shared across every column's ``.col`` file.

        Raises:
            KeyError: If a column is missing from a mapping input.
            TypeError: If any column value has an incompatible Python type.
            ValueError: If positional length mismatches, a value is ``None``, or
                a ``VARCHAR`` exceeds ``col.max_len``.
            RecordTooLargeError: If a ``VARCHAR`` exceeds single-page capacity.
            BufferPoolFullError: If all buffer frames are pinned on miss/alloc.
            StorageError: If the heap is closed or disk I/O fails.

        Complexity:
            ``O(K)`` buffer pool pin/unpin operations for ``K`` columns.
            Thread-safe under ``self._lock``.
        """
        values, byte_lens = self._extract_row_values(row)
        with self._lock:
            self._ensure_open()
            prev_last_page_id = self._last_page_id
            target_pid: int | None = None
            was_new_page = False
            if self._last_page_id is not None:
                fits, _ = self._page_available_rows_single(self._last_page_id, byte_lens)
                if fits:
                    target_pid = self._last_page_id

            if target_pid is None:
                target_pid = self._allocate_aligned_page()
                was_new_page = True
                self._last_page_id = target_pid

            mutated_cols: list[tuple[BufferPoolManager, int, int]] = []
            assigned_slot: int | None = None
            try:
                for idx, col in enumerate(self.schema.columns):
                    bpm = self._column_bpms[col.name]
                    buf = bpm.fetch_page(target_pid)
                    mutated = False
                    try:
                        cp = ColumnPage(buf, col.dtype, max_rows=self._chunk_rows)
                        old_slots = cp.slot_count
                        old_free = cp.free_space_ptr
                        slot_id = cp.append_value(values[idx])
                        if slot_id is None:
                            raise StorageError(
                                f"Column {col.name!r} overflowed unexpectedly on page {target_pid}"
                            )
                        mutated = True
                        mutated_cols.append((bpm, old_slots, old_free))
                        if assigned_slot is None:
                            assigned_slot = slot_id
                        elif slot_id != assigned_slot:
                            raise StorageError(
                                f"Slot alignment mismatch on page {target_pid} for column "
                                f"{col.name!r}: expected {assigned_slot}, got {slot_id}"
                            )
                    finally:
                        bpm.unpin_page(target_pid, is_dirty=mutated)
            except Exception:
                self._rollback_page_writes(target_pid, was_new_page, mutated_cols)
                self._last_page_id = prev_last_page_id
                raise

            assert assigned_slot is not None
            return RID(page_id=target_pid, slot_id=assigned_slot)

    def insert_batch(
        self, rows: Sequence[Mapping[str, Any] | Sequence[Any]]
    ) -> list[RID]:
        """Append ``rows`` across aligned column chunk pages using ``ColumnPage.append_batch``.

        Args:
            rows: Sequence of row mappings or ordered sequences matching ``schema``.

        Returns:
            List of ``RID(page_id, slot_id)`` in exact input order.

        Raises:
            KeyError: If a column is missing from a mapping row.
            TypeError: If any column value has an incompatible Python type.
            ValueError: If positional row length mismatches, a value is ``None``,
                or a ``VARCHAR`` exceeds ``col.max_len``.
            RecordTooLargeError: If any ``VARCHAR`` exceeds single-page capacity.
            BufferPoolFullError: If all buffer frames are pinned on miss/alloc.
            StorageError: If the heap is closed or column files diverge.

        Complexity:
            ``O(N * K)`` CPU with only ``O((N / chunk_rows) * K)`` buffer pool
            pin/unpin transitions. Thread-safe under ``self._lock``.
        """
        total = len(rows)
        if total == 0:
            return []

        num_cols = len(self.schema.columns)
        col_names = [col.name for col in self.schema.columns]
        col_values: list[list[Any]] = [[] for _ in range(num_cols)]
        varchar_lens: dict[int, list[int]] = {
            idx: []
            for idx, col in enumerate(self.schema.columns)
            if col.dtype == DataType.VARCHAR
        }

        for row in rows:
            if isinstance(row, Mapping):
                ordered = [row[name] for name in col_names]
            else:
                if len(row) != num_cols:
                    raise ValueError(
                        f"Expected {num_cols} values for {self.schema.table_name!r}, "
                        f"got {len(row)}"
                    )
                ordered = list(row)

            for idx, (col, val) in enumerate(zip(self.schema.columns, ordered)):
                vlen = _validate_column_value(col, val)
                col_values[idx].append(val)
                if vlen is not None:
                    varchar_lens[idx].append(vlen)

        with self._lock:
            self._ensure_open()
            initial_last_page_id = self._last_page_id
            undo_log: list[
                tuple[int, bool, list[tuple[BufferPoolManager, int, int]]]
            ] = []
            rids: list[RID] = []
            offset = 0
            try:
                while offset < total:
                    was_new_page = False
                    if self._last_page_id is None:
                        target_pid = self._allocate_aligned_page()
                        was_new_page = True
                        self._last_page_id = target_pid
                        undo_log.append((target_pid, True, []))
                    else:
                        target_pid = self._last_page_id

                    start_slot, take = self._compute_batch_fit(
                        target_pid, col_values, varchar_lens, offset, total
                    )
                    if take == 0:
                        if was_new_page:
                            raise RecordTooLargeError(
                                f"Row at batch index {offset} cannot fit in an empty ColumnPage"
                            )
                        target_pid = self._allocate_aligned_page()
                        was_new_page = True
                        self._last_page_id = target_pid
                        undo_log.append((target_pid, True, []))
                        start_slot, take = self._compute_batch_fit(
                            target_pid, col_values, varchar_lens, offset, total
                        )
                        if take == 0:
                            raise RecordTooLargeError(
                                f"Row at batch index {offset} cannot fit in an empty ColumnPage"
                            )

                    mutated_cols: list[tuple[BufferPoolManager, int, int]] = []
                    if not was_new_page:
                        undo_log.append((target_pid, False, mutated_cols))

                    end = offset + take
                    for idx, col in enumerate(self.schema.columns):
                        bpm = self._column_bpms[col.name]
                        buf = bpm.fetch_page(target_pid)
                        mutated = False
                        try:
                            cp = ColumnPage(buf, col.dtype, max_rows=self._chunk_rows)
                            old_slots = cp.slot_count
                            old_free = cp.free_space_ptr
                            mutated = True
                            mutated_cols.append((bpm, old_slots, old_free))
                            written = cp.append_batch(col_values[idx][offset:end])
                            if written == 0:
                                mutated = False
                                mutated_cols.pop()
                            if written != take:
                                raise StorageError(
                                    f"Column {col.name!r} wrote {written} rows instead of "
                                    f"{take} on page {target_pid}"
                                )
                        finally:
                            bpm.unpin_page(target_pid, is_dirty=mutated)

                    rids.extend(
                        RID(page_id=target_pid, slot_id=slot)
                        for slot in range(start_slot, start_slot + take)
                    )
                    offset = end

                return rids
            except Exception:
                for page_id, was_new, mutated_cols in reversed(undo_log):
                    self._rollback_page_writes(page_id, was_new, mutated_cols)
                self._last_page_id = initial_last_page_id
                raise

    def get(self, rid: RID) -> dict[str, Any] | None:
        """Reconstruct a single row across all column files at ``rid``.

        Args:
            rid: Target ``RID(page_id, slot_id)``.

        Returns:
            Decoded column dictionary, or ``None`` if ``rid`` points to an
            unallocated page or out-of-range slot.

        Raises:
            TypeError: If ``rid`` is not a ``RID``.
            StorageError: If the heap is closed or disk read fails.
            PageCorruptedError: If checksum validation fails.
            BufferPoolFullError: If all buffer frames are pinned on miss.

        Complexity:
            ``O(K)`` buffer pool pin/unpin operations for ``K`` columns.
            Thread-safe under ``self._lock``.
        """
        if not isinstance(rid, RID):
            raise TypeError(f"rid must be a RID, got {type(rid).__name__}")
        with self._lock:
            self._ensure_open()
            first_col = self.schema.columns[0]
            first_dm = self._column_bpms[first_col.name].disk_manager
            if (
                rid.page_id <= 0
                or rid.slot_id < 0
                or not first_dm.is_allocated(rid.page_id)
            ):
                return None

            row: dict[str, Any] = {}
            for col in self.schema.columns:
                bpm = self._column_bpms[col.name]
                buf = bpm.fetch_page(rid.page_id)
                try:
                    cp = ColumnPage(buf, col.dtype, max_rows=self._chunk_rows)
                    if rid.slot_id >= cp.slot_count:
                        return None
                    row[col.name] = cp.get_value(rid.slot_id)
                finally:
                    bpm.unpin_page(rid.page_id, is_dirty=False)
            return row

    def iter_column_pages(self, column_name: str) -> Iterator[tuple[int, ColumnPage]]:
        """Yield ``(page_id, ColumnPage)`` chunks for ``column_name`` while pinned.

        Guarantees ``unpin_page(page_id, is_dirty=False)`` inside ``try/finally``
        even if the caller breaks early or raises during iteration.

        Args:
            column_name: Target column present in ``schema``.

        Yields:
            Tuples ``(page_id, ColumnPage)`` backed by a currently pinned frame.

        Raises:
            KeyError: If ``column_name`` is not in ``schema``.
            StorageError: If the heap is closed or disk read fails.
        """
        col = self.schema.get_column(column_name)
        bpm = self._column_bpms[col.name]
        with self._lock:
            self._ensure_open()
            page_ids = bpm.disk_manager.allocated_pages()

        def _page_generator() -> Iterator[tuple[int, ColumnPage]]:
            for page_id in page_ids:
                buf = bpm.fetch_page(page_id)
                try:
                    yield page_id, ColumnPage(buf, col.dtype, max_rows=self._chunk_rows)
                finally:
                    bpm.unpin_page(page_id, is_dirty=False)

        return _page_generator()

    def iter_batches(
        self, columns: Sequence[str] | None = None
    ) -> Iterator[dict[str, np.ndarray[Any, Any]]]:
        """Yield Milestone 3 zero-copy ``{col: np.ndarray}`` batches per page chunk.

        Fixed-width columns yield true zero-copy ``ColumnPage.as_numpy_array()``
        views while their underlying buffer frames stay pinned across the
        ``yield`` boundary; ``try/finally`` guarantees every pinned column page
        is unpinned when advancing or exiting early.

        Args:
            columns: Optional sequence of column names to project; defaults to
                all columns in ``schema``.

        Yields:
            Dictionaries mapping each requested column name to a 1D
            ``np.ndarray`` of up to ``chunk_rows`` elements.

        Raises:
            KeyError: If any column in ``columns`` is not in ``schema``.
            ValueError: If ``columns`` is explicitly an empty sequence.
            StorageError: If the heap is closed or disk read fails.
        """
        if columns is None:
            target_cols = list(self.schema.columns)
        else:
            if isinstance(columns, str):
                raise TypeError("columns must be a sequence of column names, not str")
            if len(columns) == 0:
                raise ValueError("columns must not be empty")
            target_cols = [self.schema.get_column(name) for name in columns]

        first_bpm = self._column_bpms[self.schema.columns[0].name]
        with self._lock:
            self._ensure_open()
            page_ids = first_bpm.disk_manager.allocated_pages()

        def _batch_generator() -> Iterator[dict[str, np.ndarray[Any, Any]]]:
            for page_id in page_ids:
                pinned: list[tuple[BufferPoolManager, int]] = []
                try:
                    batch: dict[str, np.ndarray[Any, Any]] = {}
                    for col in target_cols:
                        bpm = self._column_bpms[col.name]
                        buf = bpm.fetch_page(page_id)
                        pinned.append((bpm, page_id))
                        cp = ColumnPage(buf, col.dtype, max_rows=self._chunk_rows)
                        if col.dtype == DataType.VARCHAR:
                            batch[col.name] = np.array(
                                [cp.get_value(i) for i in range(cp.slot_count)],
                                dtype=object,
                            )
                        else:
                            batch[col.name] = cp.as_numpy_array()
                    yield batch
                finally:
                    for bpm, pid in reversed(pinned):
                        bpm.unpin_page(pid, is_dirty=False)

        return _batch_generator()

    def full_scan(self) -> list[tuple[RID, dict[str, Any]]]:
        """Scan all aligned column chunks and return ``(RID, row_dict)`` pairs as a ``list``.

        Returns:
            List of ``(RID, decoded_row_dict)`` in ascending ``(page_id, slot_id)``
            order.

        Raises:
            StorageError: If the heap is closed or column pages diverge.
            PageCorruptedError: If checksum validation fails.
            BufferPoolFullError: If all buffer frames are pinned on miss.

        Complexity:
            ``O(N * K)`` row reconstruction using ``np.ndarray.tolist()`` per
            column chunk page. Thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            first_name = self.schema.columns[0].name
            page_ids = self._column_bpms[first_name].disk_manager.allocated_pages()
            if not page_ids:
                return []

            col_names = [col.name for col in self.schema.columns]
            rows: list[tuple[RID, dict[str, Any]]] = []
            for page_id in page_ids:
                col_lists: list[list[Any]] = []
                row_count: int | None = None
                for col in self.schema.columns:
                    bpm = self._column_bpms[col.name]
                    buf = bpm.fetch_page(page_id)
                    try:
                        cp = ColumnPage(buf, col.dtype, max_rows=self._chunk_rows)
                        count = cp.slot_count
                        if row_count is None:
                            row_count = count
                        elif count != row_count:
                            raise StorageError(
                                f"Column page {page_id} row-count mismatch in "
                                f"{self.schema.table_name!r}: expected {row_count}, "
                                f"got {count} on {col.name!r}"
                            )
                        if col.dtype == DataType.VARCHAR:
                            col_lists.append([cp.get_value(i) for i in range(count)])
                        else:
                            # .tolist() yields native Python int/float values in C speed.
                            col_lists.append(cp.as_numpy_array().tolist())
                    finally:
                        bpm.unpin_page(page_id, is_dirty=False)

                assert row_count is not None
                for slot_id in range(row_count):
                    row_dict = {
                        col_name: col_lists[col_idx][slot_id]
                        for col_idx, col_name in enumerate(col_names)
                    }
                    rows.append((RID(page_id=page_id, slot_id=slot_id), row_dict))

            return rows

    def page_count(self) -> int:
        """Return the number of aligned data chunk pages (excluding Page 0).

        Returns:
            Allocated data page count per column file.

        Raises:
            StorageError: If the heap is closed.
        """
        with self._lock:
            self._ensure_open()
            first_col = self.schema.columns[0].name
            return len(self._column_bpms[first_col].disk_manager.allocated_pages())

    def flush(self) -> None:
        """Flush all dirty pages and ``fsync`` every column file.

        Raises:
            StorageError: If the heap is closed or flushing fails.
        """
        with self._lock:
            self._ensure_open()
            for col in self.schema.columns:
                self._column_bpms[col.name].flush_all_pages()

    def close(self) -> None:
        """Flush all column buffer pools and close all column ``DiskManager`` instances.

        Raises:
            StorageError: If flushing or closing any column file fails (still
                closes remaining column descriptors before re-raising).
        """
        with self._lock:
            if self._closed:
                return
            first_exc: Exception | None = None
            try:
                for col in self.schema.columns:
                    bpm = self._column_bpms[col.name]
                    try:
                        bpm.flush_all_pages()
                    except Exception as exc:
                        if first_exc is None:
                            first_exc = exc
                    finally:
                        try:
                            bpm.disk_manager.close()
                        except Exception as exc:
                            if first_exc is None:
                                first_exc = exc
            finally:
                self._closed = True
            if first_exc is not None:
                raise first_exc

    def get_stats(self) -> dict[str, Any]:
        """Return columnar chunk page count and per-column buffer pool telemetry."""
        with self._lock:
            self._ensure_open()
            return {
                "table_name": self.schema.table_name,
                "layout": LayoutType.COLUMN.value,
                "page_count": self.page_count(),
                "chunk_rows": self._chunk_rows,
                "columns": {
                    col.name: self._column_bpms[col.name].get_stats()
                    for col in self.schema.columns
                },
            }


__all__ = ["ColumnTableHeap", "RowTableHeap"]
