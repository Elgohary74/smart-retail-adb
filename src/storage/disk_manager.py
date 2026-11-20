"""
Module: src/storage/disk_manager.py
Layer: Layer 1 — Physical Page I/O & Free Space Map (FSM) Manager
Lock Domain: DiskManagerLock (``self._lock = threading.RLock()``; leaf lock)

Purpose:
    Manages a single fixed-page-size binary database file (``*.db``,
    ``*.col``, or ``*.bpt``), providing thread-safe 8,192-byte positioned page
    I/O, 64-bit BLAKE2b checksum enforcement on every read/write, persistent
    Page 0 Free Space Map (FSM) allocation/reuse, and OS durability syncing.

Key Invariants:
    1. Page 0 is permanently reserved for the FSM bitmap (and optional header
       metadata in bytes ``32..fsm_offset - 1``, such as B+ Tree root/height
       metadata when ``fsm_offset = 64``); bit 0 is always ``1``.
    2. ``write_page(0, data)`` overlays the authoritative in-memory FSM bitmap
       onto ``data[fsm_offset:8192]``, while ``_persist_fsm()`` preserves
       bytes ``0..fsm_offset - 1`` from disk so metadata and FSM never clobber
       each other.
    3. Uninitialized all-zero pages bypass checksum failure on ``read_page()``,
       whereas any non-zero page with a mismatched checksum raises
       ``PageCorruptedError``.
    4. ``DiskManagerLock`` is the lowest lock in the acquisition hierarchy and
       never acquires higher-level locks.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading

from src.common.config import (
    DEFAULT_FSM_OFFSET,
    HEADER_SIZE,
    INVALID_PAGE_ID,
    PAGE_SIZE,
)
from src.common.exceptions import PageCorruptedError, StorageError
from src.storage.page_header import PageFlags, PageHeader, PageType

_HAS_PREAD: bool = hasattr(os, "pread")
_HAS_PWRITE: bool = hasattr(os, "pwrite")
_ZERO_PAGE: bytes = bytes(PAGE_SIZE)


class DiskManager:
    """Thread-safe 8,192-byte page file manager with Page 0 FSM allocation.

    Physical File & Page 0 Layout:
    ```text
    Page 0 (8,192 Bytes — FSM & Optional Metadata):
    0              32                     fsm_offset (32 or 64)                          8192
    +--------------+----------------------+-----------------------------------------------+
    | PageHeader   | Reserved Metadata    | Free Space Map (FSM) Bitmap                   |
    | (32 Bytes)   | (e.g. 32 B for B+)   | (8,160 B = 65,280 bits; 8,128 B = 65,024 bits)|
    +--------------+----------------------+-----------------------------------------------+
    Bit i (1 = allocated, 0 = free); bit 0 is always 1.

    Pages 1..N-1 (8,192 Bytes Each — Data / Index Pages):
    0              32                                                                   8192
    +--------------+--------------------------------------------------------------------+
    | PageHeader   | Page Payload (SlottedPage, ColumnPage, or B+ Tree Node)            |
    | (32 Bytes)   | (BLAKE2b-64 checksum over [0:24] + [32:8192] stored at [24:32])    |
    +--------------+--------------------------------------------------------------------+
    ```
    """

    def __init__(
        self, file_path: str | Path, fsm_offset: int = DEFAULT_FSM_OFFSET
    ) -> None:
        """Open or initialize an 8,192-byte paged database file.

        Args:
            file_path: Path to the underlying binary storage file.
            fsm_offset: Byte offset on Page 0 where the FSM bitmap starts
                (``32`` for table/column heaps, ``64`` for B+ Tree indexes).

        Raises:
            ValueError: If ``fsm_offset`` is outside ``[32, 8191]``.
            StorageError: If an existing file has a size that is not a positive
                multiple of ``PAGE_SIZE``, or if opening/initializing fails.
            PageCorruptedError: If an existing non-zero Page 0 fails checksum
                verification.

        Complexity:
            $O(1)$ disk I/O (reads or writes Page 0 once).
        """
        if fsm_offset < HEADER_SIZE or fsm_offset >= PAGE_SIZE:
            raise ValueError(
                f"fsm_offset must be in [{HEADER_SIZE}, {PAGE_SIZE - 1}], got {fsm_offset}"
            )

        self.file_path: Path = Path(file_path)
        self.fsm_offset: int = fsm_offset
        self._fsm_bytes: int = PAGE_SIZE - fsm_offset
        self._max_pages: int = self._fsm_bytes * 8

        self._lock: threading.RLock = threading.RLock()
        self.disk_reads: int = 0
        self.disk_writes: int = 0

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        open_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        try:
            self._fd: int | None = os.open(os.fspath(self.file_path), open_flags, 0o644)
        except OSError as exc:
            raise StorageError(f"Failed to open page file {self.file_path}: {exc}") from exc

        try:
            self._init_or_reload_file()
        except Exception:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise

    def _write_initial_page0(self) -> None:
        """Initialize Page 0 header, mark Page 0 allocated, and sync to disk."""
        self._fsm_bitmap = bytearray(self._fsm_bytes)
        # Page 0 itself is permanently reserved for the FSM and metadata.
        self._fsm_bitmap[0] = 0x01

        page0 = bytearray(PAGE_SIZE)
        PageHeader(
            page_id=0,
            page_type=PageType.FREE_OR_FSM,
            flags=PageFlags.NONE,
            slot_count=0,
            page_lsn=0,
            free_space_ptr=self.fsm_offset,
            next_page_id=INVALID_PAGE_ID,
            checksum=0,
        ).pack_into(page0, 0)
        page0[self.fsm_offset :] = self._fsm_bitmap
        PageHeader.stamp_checksum(page0)
        self._pwrite_exact(0, page0)
        os.fsync(self._ensure_open())

    def _init_or_reload_file(self) -> None:
        """Initialize a fresh Page 0 FSM or checksum-verify and reload an existing file."""
        fd = self._ensure_open()
        file_size = os.fstat(fd).st_size
        if file_size == 0:
            self._num_pages = 1
            self._write_initial_page0()
            return

        if file_size < PAGE_SIZE or file_size % PAGE_SIZE != 0:
            raise StorageError(
                f"Invalid page file size {file_size} for {self.file_path}; "
                f"expected non-zero multiple of {PAGE_SIZE}"
            )

        self._num_pages = file_size // PAGE_SIZE
        page0 = self._pread_exact(0, PAGE_SIZE)
        if not PageHeader.verify_checksum(page0):
            if page0 == _ZERO_PAGE:
                # Only an all-zero Page 0 is uninitialized; never overwrite torn bytes.
                self._write_initial_page0()
                return
            raise PageCorruptedError(
                f"Page 0 FSM checksum mismatch in {self.file_path}"
            )
        self._fsm_bitmap = bytearray(page0[self.fsm_offset :])
        self._fsm_bitmap[0] |= 0x01

    def _validate_page_id(self, page_id: int, *, allow_zero: bool) -> None:
        """Validate that ``page_id`` is a non-bool integer within file page bounds."""
        if not isinstance(page_id, int) or isinstance(page_id, bool):
            raise StorageError(
                f"Page ID must be an int, got {type(page_id).__name__}"
            )
        if not allow_zero and page_id == 0:
            raise StorageError("Cannot deallocate reserved FSM Page 0")
        min_pid = 0 if allow_zero else 1
        if page_id < min_pid or page_id >= self._num_pages:
            raise StorageError(
                f"Page ID {page_id} out of range [{min_pid}, {self._num_pages - 1}]"
            )

    def _ensure_open(self) -> int:
        """Return the active file descriptor or raise ``StorageError`` if closed."""
        if self._fd is None:
            raise StorageError(f"DiskManager for {self.file_path} is closed")
        return self._fd

    def _pread_exact(self, offset: int, size: int) -> bytearray:
        """Read exactly ``size`` bytes at ``offset`` under ``self._lock``."""
        fd = self._ensure_open()
        buf: bytearray | None = None
        while True:
            pos = offset if buf is None else offset + len(buf)
            remaining = size if buf is None else size - len(buf)
            try:
                if _HAS_PREAD:
                    chunk = os.pread(fd, remaining, pos)
                else:
                    os.lseek(fd, pos, os.SEEK_SET)
                    chunk = os.read(fd, remaining)
            except OSError as exc:
                raise StorageError(
                    f"Disk read failed at offset {pos} in {self.file_path}: {exc}"
                ) from exc
            if not chunk:
                raise StorageError(
                    f"Unexpected EOF reading {size} bytes at offset {offset} in {self.file_path}"
                )
            if buf is None and len(chunk) == size:
                return bytearray(chunk)
            if buf is None:
                buf = bytearray(chunk)
            else:
                buf.extend(chunk)
            if len(buf) >= size:
                return buf

    def _pwrite_exact(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        """Write all bytes of ``data`` at ``offset`` under ``self._lock``."""
        fd = self._ensure_open()
        mv = memoryview(data)
        total = len(mv)
        written = 0
        while written < total:
            pos = offset + written
            try:
                if _HAS_PWRITE:
                    n = os.pwrite(fd, mv[written:], pos)
                else:
                    os.lseek(fd, pos, os.SEEK_SET)
                    n = os.write(fd, mv[written:])
            except OSError as exc:
                raise StorageError(
                    f"Disk write failed at offset {pos} in {self.file_path}: {exc}"
                ) from exc
            if n <= 0:
                raise StorageError(
                    f"Short write ({n} bytes) at offset {pos} in {self.file_path}"
                )
            written += n

    def _persist_fsm(self) -> None:
        """Flush ``self._fsm_bitmap`` to Page 0 without overwriting ``0..fsm_offset - 1``.

        Reading existing Page 0 bytes ``0..fsm_offset - 1`` preserves both
        ``PageHeader.page_lsn`` and B+ Tree root/schema metadata stored in
        bytes ``32..55`` when ``fsm_offset == 64``.
        """
        page0 = self._pread_exact(0, PAGE_SIZE)
        page0[self.fsm_offset :] = self._fsm_bitmap
        page0[self.fsm_offset] |= 0x01
        PageHeader.stamp_checksum(page0)
        self._pwrite_exact(0, page0)

    def _set_bit(self, page_id: int) -> None:
        self._fsm_bitmap[page_id >> 3] |= 1 << (page_id & 7)

    def _clear_bit(self, page_id: int) -> None:
        self._fsm_bitmap[page_id >> 3] &= ~(1 << (page_id & 7))

    def _get_bit(self, page_id: int) -> bool:
        return bool(self._fsm_bitmap[page_id >> 3] & (1 << (page_id & 7)))

    def allocate_page(self) -> int:
        """Allocate and return the lowest available ``page_id >= 1`` from the FSM.

        Extends the backing file with zero-filled 8,192-byte pages when reusing
        no freed slot, then persists the updated FSM bitmap on Page 0.

        Returns:
            Newly allocated physical ``page_id`` (``>= 1``).

        Raises:
            StorageError: If the manager is closed, the single-page FSM bitmap
                capacity is exhausted, or disk extension/flush fails.

        Complexity:
            ``O(FSM_BYTES)`` worst-case byte scan with ``O(1)`` skipping over
            ``0xFF`` full bytes; thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            candidate: int | None = None
            for byte_idx, byte_val in enumerate(self._fsm_bitmap):
                if byte_val == 0xFF:
                    continue
                for bit in range(8):
                    pid = (byte_idx << 3) + bit
                    if pid == 0:
                        continue
                    if not (byte_val & (1 << bit)):
                        candidate = pid
                        break
                if candidate is not None:
                    break

            if candidate is None or candidate >= self._max_pages:
                raise StorageError(
                    f"FSM capacity exhausted (max {self._max_pages} pages) in {self.file_path}"
                )

            prev_num_pages = self._num_pages
            self._set_bit(candidate)
            try:
                if candidate >= self._num_pages:
                    for ext_pid in range(self._num_pages, candidate + 1):
                        self._pwrite_exact(ext_pid * PAGE_SIZE, _ZERO_PAGE)
                    self._num_pages = candidate + 1
                self._persist_fsm()
            except Exception:
                self._clear_bit(candidate)
                self._num_pages = prev_num_pages
                raise
            return candidate

    def deallocate_page(self, page_id: int) -> None:
        """Free ``page_id`` in the Page 0 FSM and zero out its disk page.

        Args:
            page_id: Physical data page ID (must satisfy ``1 <= page_id < num_pages()``).

        Raises:
            StorageError: If ``page_id == 0`` (reserved FSM page), ``page_id``
                is out of range, the manager is closed, or disk I/O fails.

        Complexity:
            ``O(PAGE_SIZE)`` disk zero-fill plus Page 0 FSM flush; thread-safe
            under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            self._validate_page_id(page_id, allow_zero=False)
            self._clear_bit(page_id)
            self._pwrite_exact(page_id * PAGE_SIZE, _ZERO_PAGE)
            self._persist_fsm()

    def read_page(self, page_id: int) -> bytearray:
        """Read and checksum-verify an 8,192-byte page from disk.

        Uninitialized all-zero pages (such as newly extended or recently
        deallocated pages prior to their first write) bypass checksum failure
        and return a mutable zero-filled ``bytearray(8192)``.

        Args:
            page_id: Zero-based physical page index in ``[0, num_pages() - 1]``.

        Returns:
            Mutable ``bytearray`` of length ``PAGE_SIZE`` (8,192 bytes).

        Raises:
            StorageError: If ``page_id`` is out of bounds, the manager is
                closed, or disk read fails.
            PageCorruptedError: If a non-zero page fails 64-bit BLAKE2b
                checksum verification.

        Complexity:
            ``O(PAGE_SIZE)`` I/O and checksum verification; thread-safe under
            ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            self._validate_page_id(page_id, allow_zero=True)
            buf = self._pread_exact(page_id * PAGE_SIZE, PAGE_SIZE)
            self.disk_reads += 1
            # Newly allocated or deallocated pages are zero-filled before first use.
            if buf != _ZERO_PAGE and not PageHeader.verify_checksum(buf):
                raise PageCorruptedError(
                    f"Checksum mismatch on page {page_id} of {self.file_path}"
                )
            return buf

    def write_page(self, page_id: int, data: bytes | bytearray | memoryview) -> None:
        """Stamp the BLAKE2b checksum onto ``data`` and write page ``page_id`` to disk.

        When writing Page 0, the authoritative in-memory FSM bitmap is copied
        into ``bytes[fsm_offset:8192]`` prior to checksum computation so a
        stale buffer-pool Page 0 frame cannot revert prior allocations.

        Args:
            page_id: Target physical page index in ``[0, num_pages() - 1]``.
            data: Exactly 8,192 bytes of page content.

        Raises:
            StorageError: If ``len(data) != PAGE_SIZE``, ``page_id`` is out of
                bounds, the manager is closed, or disk write fails.

        Complexity:
            ``O(PAGE_SIZE)`` checksum and positioned write; thread-safe under
            ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            if len(data) != PAGE_SIZE:
                raise StorageError(
                    f"Expected {PAGE_SIZE}-byte page buffer, got {len(data)}"
                )
            self._validate_page_id(page_id, allow_zero=True)
            buf = bytearray(data)
            if page_id == 0:
                # Never allow an older Page 0 frame copy to overwrite live FSM allocations.
                buf[self.fsm_offset :] = self._fsm_bitmap
                buf[self.fsm_offset] |= 0x01
            PageHeader.stamp_checksum(buf)
            self._pwrite_exact(page_id * PAGE_SIZE, buf)
            self.disk_writes += 1

    def is_allocated(self, page_id: int) -> bool:
        """Return whether ``page_id`` is currently marked allocated in the FSM.

        Args:
            page_id: Physical page index to inspect.

        Returns:
            ``True`` if the FSM bit for ``page_id`` is ``1``, else ``False``.

        Raises:
            StorageError: If the manager is closed.

        Complexity:
            $O(1)$ bit lookup; thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            if not isinstance(page_id, int) or isinstance(page_id, bool):
                raise StorageError(
                    f"Page ID must be an int, got {type(page_id).__name__}"
                )
            if page_id < 0 or page_id >= self._max_pages:
                return False
            return self._get_bit(page_id)

    def allocated_pages(self) -> list[int]:
        """Return sorted list of all currently allocated data page IDs (``>= 1``).

        Returns:
            Ascending list of allocated page IDs excluding reserved Page 0.

        Raises:
            StorageError: If the manager is closed.

        Complexity:
            ``O(FSM_BYTES)`` bitmap scan; thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            pages: list[int] = []
            for byte_idx, byte_val in enumerate(self._fsm_bitmap):
                if byte_val == 0:
                    continue
                base = byte_idx << 3
                for bit in range(8):
                    if byte_val & (1 << bit):
                        pid = base + bit
                        if pid > 0:
                            pages.append(pid)
            return pages

    def num_pages(self) -> int:
        """Return total number of 8,192-byte pages currently backed by the file.

        Returns:
            Total page count including Page 0 (always ``>= 1`` while open).

        Raises:
            StorageError: If the manager is closed.

        Complexity:
            $O(1)$ under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            return self._num_pages

    def sync(self) -> None:
        """Force dirty OS kernel file buffers to physical storage via ``os.fsync``.

        Raises:
            StorageError: If the manager is closed or ``os.fsync`` fails.

        Complexity:
            OS kernel flush latency; thread-safe under ``self._lock``.
        """
        with self._lock:
            fd = self._ensure_open()
            try:
                os.fsync(fd)
            except OSError as exc:
                raise StorageError(f"fsync failed for {self.file_path}: {exc}") from exc

    def close(self) -> None:
        """Sync and close the underlying OS file descriptor idempotently.

        Raises:
            StorageError: If syncing fails prior to closing the descriptor.

        Complexity:
            $O(1)$ plus OS fsync; safe to call multiple times under ``self._lock``.
        """
        with self._lock:
            if self._fd is None:
                return
            fd = self._fd
            try:
                os.fsync(fd)
            except OSError as exc:
                raise StorageError(f"fsync failed while closing {self.file_path}: {exc}") from exc
            finally:
                self._fd = None
                os.close(fd)

    def __enter__(self) -> DiskManager:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


__all__ = ["DiskManager"]
