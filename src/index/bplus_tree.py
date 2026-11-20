"""
Module: src/index/bplus_tree.py
Layer: Layer 2 — Disk-Backed B+ Tree Access Method
Lock Domain: IndexLock (``self._lock = threading.RLock()``; acquired before BufferPoolLock)

Purpose:
    Implements a persistent, buffer-pool-backed B+ Tree index (``BPlusTreeIndex``)
    over fixed 8,192-byte pages managed by ``DiskManager(fsm_offset=64)`` and
    ``BufferPoolManager``, supporting unique primary keys and duplicate secondary
    keys via fixed-width order-preserving composite keys.

Key Invariants:
    1. Lock ordering strictly follows ``IndexLock -> BufferPoolLock -> DiskManagerLock``.
    2. Top-down traversal and sibling leaf scans unpin each parent/predecessor
       before fetching the next node, and node splits unpin both halves before
       updating ancestors, keeping at most 2 frames pinned simultaneously.
    3. Every node entry is keyed by ``encode_key(key) + struct.pack(">II", rid.page_id, rid.slot_id)``
       (``effective_key_size = key_size + 8``), ensuring strict byte-order
       uniqueness even on duplicate secondary keys.
    4. Internal nodes route strictly with ``bisect_right`` semantics (``k == sep_i``
       descends into ``child_{i+1}``) and preserve ``max(child_i) < sep_i <= min(child_{i+1})``.
    5. Page 0 stores B+ Tree metadata (``<IBHHQ?6x``) at bytes ``32..55`` while
       ``DiskManager`` maintains the FSM bitmap in bytes ``64..8191``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import threading
from typing import Any

from src.common.config import (
    BPLUS_FSM_OFFSET,
    DEFAULT_BUFFER_POOL_SIZE,
    HEADER_SIZE,
    INVALID_PAGE_ID,
    PAGE_SIZE,
)
from src.common.exceptions import DuplicateKeyError, PageCorruptedError, StorageError
from src.common.types import (
    RID,
    DataType,
    decode_key,
    encode_key,
    make_composite_key,
)
from src.index.base_index import BaseIndex
from src.storage.buffer_pool import BufferPoolManager
from src.storage.disk_manager import DiskManager
from src.storage.page_header import PageFlags, PageHeader, PageType

_META_STRUCT = struct.Struct("<IBHHQ?6x")
_U16_STRUCT = struct.Struct("<H")
_U32_STRUCT = struct.Struct("<I")
_RID_LE_STRUCT = struct.Struct("<II")
_RID_BE_STRUCT = struct.Struct(">II")

_PAGE_TYPE_OFFSET: int = 4
_FLAGS_OFFSET: int = 5
_SLOT_COUNT_OFFSET: int = 6
_FREE_SPACE_PTR_OFFSET: int = 16
_NEXT_PAGE_ID_OFFSET: int = 20

_META_OFFSET: int = HEADER_SIZE
_META_END: int = _META_OFFSET + _META_STRUCT.size  # 56

_INTERNAL_CHILDREN_OFFSET: int = HEADER_SIZE  # 32
_INTERNAL_ENTRIES_OFFSET: int = HEADER_SIZE + 4  # 36
_LEAF_ENTRIES_OFFSET: int = HEADER_SIZE  # 32

_ZERO_RID_SUFFIX: bytes = b"\x00" * 8
_MAX_RID_SUFFIX: bytes = b"\xff" * 8
_ZERO_PAGE: bytes = bytes(PAGE_SIZE)


@dataclass(frozen=True)
class _LeafAudit:
    """Snapshot of one leaf node collected during structural invariant checking."""

    page_id: int
    next_page_id: int
    slot_count: int
    first_comp: bytes
    last_comp: bytes


class BPlusTreeIndex(BaseIndex):
    """Buffer-pool-backed B+ Tree index persisted inside an 8,192-byte ``*.bpt`` file.

    Binary Page Layouts (8,192 bytes each):
    ```text
    [A] Metadata + FSM Page (page_id = 0, fsm_offset = 64):
    0              32              56        64                                         8192
    +--------------+---------------+---------+------------------------------------------+
    | PageHeader   | B+ Metadata   | Padding | Free Space Map (FSM) Bitmap              |
    | (32 Bytes)   | <IBHHQ?6x     | (8 B)   | (8,128 Bytes = 65,024 page bits)         |
    +--------------+---------------+---------+------------------------------------------+
      Bytes 32..55: root_page_id (u32), key_dtype (u8), key_size (u16), height (u16),
                    entry_count (u64), unique (bool), 6B zero padding.

    [B] Internal Node (page_type = 3 = PageType.BPLUS_INTERNAL):
    0              32        36                          36 + k*(eks+4)                 8192
    +--------------+---------+---------------------------+------------------------------+
    | PageHeader   | child_0 | [sep_key_0 | child_1] ... |      Unused Space            |
    | slots = k    | (4B <I) | (eks B + 4B <I) * k       |  M_int = (8192-36)//(eks+4) |
    +--------------+---------+---------------------------+------------------------------+

    [C] Leaf Node (page_type = 4 = PageType.BPLUS_LEAF):
    0              32                                    32 + k*(eks+8)                 8192
    +--------------+-------------------------------------+------------------------------+
    | PageHeader   | [entry_key_i | rid_page | rid_slot] |      Unused Space            |
    | slots = k    | (eks B + 4B <I + 4B <I) * k         |  M_leaf = (8192-32)//(eks+8)|
    +--------------+-------------------------------------+------------------------------+
      eks = effective_key_size = key_size + 8 (encode_key(key) || >II(rid_page, rid_slot))
    ```
    """

    def __init__(
        self,
        file_path: str | Path,
        key_dtype: DataType,
        key_size: int = 8,
        unique: bool = False,
        buffer_pool_size: int = DEFAULT_BUFFER_POOL_SIZE,
    ) -> None:
        """Open or initialize a buffer-pool-backed B+ Tree index file.

        Args:
            file_path: Path to the ``*.bpt`` index file.
            key_dtype: Logical ``DataType`` of the indexed key column.
            key_size: Fixed serialized byte width of the base key (``4`` for
                ``INT32``, ``8`` for 64-bit numerics, ``max_len`` for ``VARCHAR``).
            unique: When ``True``, reject duplicate logical keys with
                ``DuplicateKeyError``.
            buffer_pool_size: Number of 8,192-byte buffer frames allocated to
                this index (must be ``>= 1``; ``>= 3`` recommended for splits).

        Raises:
            TypeError: If ``key_dtype`` is not a ``DataType`` or ``buffer_pool_size``
                is not an integer.
            ValueError: If ``key_size`` mismatches ``key_dtype`` or leaves fewer
                than 2 slots per node, or if ``buffer_pool_size < 1``.
            StorageError: If an existing ``.bpt`` file has incompatible metadata
                (``key_dtype``, ``key_size``, or ``unique``) or invalid Page 0 state.
            PageCorruptedError: If Page 0 fails 64-bit BLAKE2b checksum verification.

        Complexity:
            $O(1)$ disk reads to load or initialize Page 0 metadata.
        """
        if not isinstance(key_dtype, DataType):
            raise TypeError(f"key_dtype must be a DataType, got {type(key_dtype).__name__}")
        if isinstance(key_size, bool) or not isinstance(key_size, int) or key_size <= 0:
            raise ValueError(f"key_size must be a positive int, got {key_size!r}")
        if key_dtype != DataType.VARCHAR and key_size != key_dtype.fixed_size:
            raise ValueError(
                f"{key_dtype.name} requires key_size={key_dtype.fixed_size}, got {key_size}"
            )
        if not isinstance(unique, bool):
            raise TypeError(f"unique must be a bool, got {type(unique).__name__}")

        effective_key_size = key_size + 8
        int_stride = effective_key_size + 4
        leaf_stride = effective_key_size + 8
        m_int = (PAGE_SIZE - _INTERNAL_ENTRIES_OFFSET) // int_stride
        m_leaf = (PAGE_SIZE - _LEAF_ENTRIES_OFFSET) // leaf_stride
        if m_int < 2 or m_leaf < 2:
            raise ValueError(
                f"key_size={key_size} leaves too little fanout (m_int={m_int}, m_leaf={m_leaf})"
            )

        self.file_path: Path = Path(file_path)
        self._key_dtype: DataType = key_dtype
        self._key_size: int = key_size
        self._effective_key_size: int = effective_key_size
        self._int_stride: int = int_stride
        self._leaf_stride: int = leaf_stride
        self._m_int: int = m_int
        self._m_leaf: int = m_leaf
        self._unique: bool = unique

        self._lock: threading.RLock = threading.RLock()
        self._closed: bool = False

        pre_exists = self.file_path.exists()
        pre_size = self.file_path.stat().st_size if pre_exists else 0

        self._disk_manager: DiskManager = DiskManager(
            self.file_path, fsm_offset=BPLUS_FSM_OFFSET
        )
        try:
            self._bpm: BufferPoolManager = BufferPoolManager(
                self._disk_manager, pool_size=buffer_pool_size
            )
            self._root_page_id: int = INVALID_PAGE_ID
            self._height: int = 0
            self._entry_count: int = 0
            self._load_or_init_metadata(is_new_file=(not pre_exists or pre_size == 0))
        except Exception:
            self._disk_manager.close()
            self._closed = True
            raise

    def _load_or_init_metadata(self, *, is_new_file: bool) -> None:
        """Initialize Page 0 metadata on new files or validate and reload on reopen."""
        buf = self._bpm.fetch_page(0)
        dirty = False
        try:
            if is_new_file:
                _META_STRUCT.pack_into(
                    buf,
                    _META_OFFSET,
                    INVALID_PAGE_ID,
                    int(self._key_dtype),
                    self._key_size,
                    0,
                    0,
                    self._unique,
                )
                dirty = True
                return

            header = PageHeader.unpack(buf, 0)
            if (
                header.page_id != 0
                or header.page_type != PageType.FREE_OR_FSM
                or header.free_space_ptr != BPLUS_FSM_OFFSET
            ):
                raise StorageError(
                    f"File {self.file_path} is not a valid B+ Tree Page 0 "
                    f"(page_type={header.page_type}, fsm_offset={header.free_space_ptr})"
                )

            if bytes(buf[_META_OFFSET:_META_END]) == b"\x00" * _META_STRUCT.size:
                if self._disk_manager.num_pages() != 1:
                    raise StorageError(
                        f"Uninitialized B+ Tree metadata on multi-page file {self.file_path}"
                    )
                _META_STRUCT.pack_into(
                    buf,
                    _META_OFFSET,
                    INVALID_PAGE_ID,
                    int(self._key_dtype),
                    self._key_size,
                    0,
                    0,
                    self._unique,
                )
                dirty = True
                return

            root_pid, stored_dtype, stored_key_size, height, entry_count, stored_unique = (
                _META_STRUCT.unpack_from(buf, _META_OFFSET)
            )
            if stored_dtype != int(self._key_dtype):
                raise StorageError(
                    f"B+ Tree key_dtype mismatch in {self.file_path}: "
                    f"expected {self._key_dtype.name} ({int(self._key_dtype)}), "
                    f"found {stored_dtype}"
                )
            if stored_key_size != self._key_size:
                raise StorageError(
                    f"B+ Tree key_size mismatch in {self.file_path}: "
                    f"expected {self._key_size}, found {stored_key_size}"
                )
            if bool(stored_unique) != self._unique:
                raise StorageError(
                    f"B+ Tree unique flag mismatch in {self.file_path}: "
                    f"expected {self._unique}, found {bool(stored_unique)}"
                )
            if root_pid == INVALID_PAGE_ID:
                if height != 0 or entry_count != 0:
                    raise StorageError(
                        f"Inconsistent empty root metadata in {self.file_path}: "
                        f"height={height}, entry_count={entry_count}"
                    )
            else:
                if root_pid == 0 or root_pid >= self._disk_manager.num_pages() or height < 1:
                    raise StorageError(
                        f"Corrupt B+ Tree root metadata in {self.file_path}: "
                        f"root_page_id={root_pid}, height={height}"
                    )

            self._root_page_id = root_pid
            self._height = height
            self._entry_count = entry_count
        finally:
            buf.release()
            self._bpm.unpin_page(0, is_dirty=dirty)

        if is_new_file or dirty:
            self._bpm.flush_page(0, sync=True)

    def _ensure_open(self) -> None:
        """Raise ``StorageError`` if this index has already been closed."""
        if self._closed:
            raise StorageError(f"BPlusTreeIndex {self.file_path} is closed")

    def _persist_metadata(self, *, flush_to_disk: bool = False) -> None:
        """Update Page 0 metadata bytes ``32..55`` through the buffer pool."""
        buf = self._bpm.fetch_page(0)
        dirty = False
        try:
            _META_STRUCT.pack_into(
                buf,
                _META_OFFSET,
                self._root_page_id,
                int(self._key_dtype),
                self._key_size,
                self._height,
                self._entry_count,
                self._unique,
            )
            dirty = True
        finally:
            buf.release()
            self._bpm.unpin_page(0, is_dirty=dirty)

        if flush_to_disk:
            self._bpm.flush_page(0)

    def _internal_bisect_right(
        self, buf: memoryview, count: int, comp_key: bytes
    ) -> int:
        """Return child index ``i`` where ``sep[i-1] <= comp_key < sep[i]``."""
        lo = 0
        hi = count
        eks = self._effective_key_size
        stride = self._int_stride
        while lo < hi:
            mid = (lo + hi) >> 1
            off = _INTERNAL_ENTRIES_OFFSET + mid * stride
            if comp_key < bytes(buf[off : off + eks]):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _leaf_bisect_left(
        self, buf: memoryview, count: int, comp_key: bytes
    ) -> int:
        """Return smallest index ``i`` in leaf where ``entry_key[i] >= comp_key``."""
        lo = 0
        hi = count
        eks = self._effective_key_size
        stride = self._leaf_stride
        while lo < hi:
            mid = (lo + hi) >> 1
            off = _LEAF_ENTRIES_OFFSET + mid * stride
            if bytes(buf[off : off + eks]) < comp_key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _leaf_bisect_right(
        self, buf: memoryview, count: int, comp_key: bytes
    ) -> int:
        """Return smallest index ``i`` in leaf where ``entry_key[i] > comp_key``."""
        lo = 0
        hi = count
        eks = self._effective_key_size
        stride = self._leaf_stride
        while lo < hi:
            mid = (lo + hi) >> 1
            off = _LEAF_ENTRIES_OFFSET + mid * stride
            if comp_key < bytes(buf[off : off + eks]):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _find_leaf(
        self,
        search_key: bytes,
        *,
        track_path: list[tuple[int, int]] | None = None,
        track_left_sep_key: bytes | None = None,
    ) -> tuple[int, memoryview, bool]:
        """Descend from root to target leaf, returning ``(leaf_pid, buf, saw_left_sep)``.

        Unpins each parent immediately before fetching its child so at most one
        frame stays pinned during descent. Caller owns the returned pin on
        ``leaf_pid`` and must release ``buf`` and unpin ``leaf_pid``.
        """
        pinned_pid: int | None = self._root_page_id
        curr_buf: memoryview | None = self._bpm.fetch_page(pinned_pid)
        saw_left_sep = False
        try:
            while True:
                page_type = curr_buf[_PAGE_TYPE_OFFSET]
                if page_type == PageType.BPLUS_LEAF:
                    leaf_pid = pinned_pid
                    leaf_buf = curr_buf
                    pinned_pid = None
                    curr_buf = None
                    return leaf_pid, leaf_buf, saw_left_sep
                if page_type != PageType.BPLUS_INTERNAL:
                    raise PageCorruptedError(
                        f"Unexpected page_type {page_type} on internal node {pinned_pid}"
                    )
                slot_count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                child_idx = self._internal_bisect_right(curr_buf, slot_count, search_key)
                if track_path is not None:
                    track_path.append((pinned_pid, child_idx))
                if track_left_sep_key is not None and child_idx > 0:
                    sep_off = _INTERNAL_ENTRIES_OFFSET + (child_idx - 1) * self._int_stride
                    if bytes(curr_buf[sep_off : sep_off + self._key_size]) == track_left_sep_key:
                        # A left separator sharing base key means an earlier delete/reinsert may reside left.
                        saw_left_sep = True

                next_pid = _U32_STRUCT.unpack_from(
                    curr_buf, _INTERNAL_CHILDREN_OFFSET + child_idx * self._int_stride
                )[0]
                # Unpin parent before fetching child to enforce strict <=2 pin discipline.
                curr_buf.release()
                curr_buf = None
                self._bpm.unpin_page(pinned_pid, is_dirty=False)
                pinned_pid = None

                next_buf = self._bpm.fetch_page(next_pid)
                pinned_pid = next_pid
                curr_buf = next_buf
        finally:
            if curr_buf is not None:
                curr_buf.release()
            if pinned_pid is not None:
                self._bpm.unpin_page(pinned_pid, is_dirty=False)

    def _contains_key_bytes(self, key_bytes: bytes) -> bool:
        """Return ``True`` if any entry in the tree has base key ``key_bytes``."""
        if self._root_page_id == INVALID_PAGE_ID:
            return False
        low_comp = key_bytes + _ZERO_RID_SUFFIX
        curr_pid, curr_buf, _ = self._find_leaf(low_comp)
        try:
            while True:
                count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                idx = self._leaf_bisect_left(curr_buf, count, low_comp)
                if idx < count:
                    off = _LEAF_ENTRIES_OFFSET + idx * self._leaf_stride
                    return bytes(curr_buf[off : off + self._key_size]) == key_bytes
                next_pid = _U32_STRUCT.unpack_from(curr_buf, _NEXT_PAGE_ID_OFFSET)[0]
                if next_pid == INVALID_PAGE_ID:
                    return False
                curr_buf.release()
                self._bpm.unpin_page(curr_pid, is_dirty=False)
                curr_pid = next_pid
                curr_buf = self._bpm.fetch_page(curr_pid)
        finally:
            if curr_buf is not None:
                curr_buf.release()
            if curr_pid != INVALID_PAGE_ID:
                self._bpm.unpin_page(curr_pid, is_dirty=False)

    def _peek_next_leaf_first_base_key(self, next_pid: int) -> bytes | None:
        """Fetch ``next_pid`` briefly to read its first base key, keeping caller's pin."""
        if next_pid == INVALID_PAGE_ID:
            return None
        next_buf = self._bpm.fetch_page(next_pid)
        try:
            count = _U16_STRUCT.unpack_from(next_buf, _SLOT_COUNT_OFFSET)[0]
            if count == 0:
                return None
            return bytes(next_buf[_LEAF_ENTRIES_OFFSET : _LEAF_ENTRIES_OFFSET + self._key_size])
        finally:
            next_buf.release()
            self._bpm.unpin_page(next_pid, is_dirty=False)

    def _split_leaf_and_insert(
        self,
        left_pid: int,
        left_buf: memoryview,
        ins_idx: int,
        new_record: bytes,
        *,
        clear_root_flag: bool,
    ) -> tuple[bytes, int]:
        """Split a full leaf (``M_leaf`` entries) plus ``new_record`` into two leaves.

        Allocates ``right_pid``, moves the upper half ``(M_leaf + 1) // 2``
        entries into ``right_pid``, links ``left -> right -> old_next``, and
        returns ``(right_first_comp_key, right_pid)`` without releasing the caller's
        pin on ``left_pid`` (caller unpins both or lets helper unpin ``right_pid``).
        """
        n = self._m_leaf
        stride = self._leaf_stride
        eks = self._effective_key_size
        left_count = (n + 2) // 2
        right_count = (n + 1) - left_count

        old_next = _U32_STRUCT.unpack_from(left_buf, _NEXT_PAGE_ID_OFFSET)[0]
        right_pid, right_buf = self._bpm.new_page()
        right_dirty = False
        try:
            right_buf[:] = _ZERO_PAGE
            if ins_idx < left_count:
                # Last `right_count` existing entries move to right; new record fits in left.
                src_start = _LEAF_ENTRIES_OFFSET + (left_count - 1) * stride
                src_end = _LEAF_ENTRIES_OFFSET + n * stride
                right_buf[_LEAF_ENTRIES_OFFSET : _LEAF_ENTRIES_OFFSET + right_count * stride] = (
                    left_buf[src_start:src_end]
                )
                if ins_idx < left_count - 1:
                    shift_src_start = _LEAF_ENTRIES_OFFSET + ins_idx * stride
                    shift_src_end = _LEAF_ENTRIES_OFFSET + (left_count - 1) * stride
                    shift_dst_start = _LEAF_ENTRIES_OFFSET + (ins_idx + 1) * stride
                    shift_dst_end = _LEAF_ENTRIES_OFFSET + left_count * stride
                    left_buf[shift_dst_start:shift_dst_end] = left_buf[
                        shift_src_start:shift_src_end
                    ]
                ins_off = _LEAF_ENTRIES_OFFSET + ins_idx * stride
                left_buf[ins_off : ins_off + stride] = new_record
            else:
                r_idx = ins_idx - left_count
                if r_idx > 0:
                    pre_start = _LEAF_ENTRIES_OFFSET + left_count * stride
                    pre_end = _LEAF_ENTRIES_OFFSET + ins_idx * stride
                    right_buf[_LEAF_ENTRIES_OFFSET : _LEAF_ENTRIES_OFFSET + r_idx * stride] = (
                        left_buf[pre_start:pre_end]
                    )
                r_ins_off = _LEAF_ENTRIES_OFFSET + r_idx * stride
                right_buf[r_ins_off : r_ins_off + stride] = new_record
                if ins_idx < n:
                    post_start = _LEAF_ENTRIES_OFFSET + ins_idx * stride
                    post_end = _LEAF_ENTRIES_OFFSET + n * stride
                    right_buf[
                        r_ins_off + stride : _LEAF_ENTRIES_OFFSET + right_count * stride
                    ] = left_buf[post_start:post_end]

            # Zero trailing stale bytes on left leaf so page contents stay clean.
            left_end = _LEAF_ENTRIES_OFFSET + left_count * stride
            old_end = _LEAF_ENTRIES_OFFSET + n * stride
            if left_end < old_end:
                left_buf[left_end:old_end] = bytes(old_end - left_end)

            PageHeader(
                page_id=right_pid,
                page_type=PageType.BPLUS_LEAF,
                flags=PageFlags.IS_LEAF,
                slot_count=right_count,
                page_lsn=0,
                free_space_ptr=_LEAF_ENTRIES_OFFSET + right_count * stride,
                next_page_id=old_next,
                checksum=0,
            ).pack_into(right_buf, 0)

            _U16_STRUCT.pack_into(left_buf, _SLOT_COUNT_OFFSET, left_count)
            _U32_STRUCT.pack_into(left_buf, _FREE_SPACE_PTR_OFFSET, left_end)
            _U32_STRUCT.pack_into(left_buf, _NEXT_PAGE_ID_OFFSET, right_pid)
            if clear_root_flag:
                left_buf[_FLAGS_OFFSET] &= ~int(PageFlags.IS_ROOT)

            promoted_key = bytes(right_buf[_LEAF_ENTRIES_OFFSET : _LEAF_ENTRIES_OFFSET + eks])
            right_dirty = True
            return promoted_key, right_pid
        finally:
            right_buf.release()
            self._bpm.unpin_page(right_pid, is_dirty=right_dirty)

    def _split_internal_and_insert(
        self,
        left_pid: int,
        left_buf: memoryview,
        ins_idx: int,
        new_sep: bytes,
        new_child: int,
        *,
        clear_root_flag: bool,
    ) -> tuple[bytes, int]:
        """Split a full internal node (``M_int`` keys) plus ``(new_sep, new_child)``.

        Moves the upper ``M_int - M_int // 2`` separator entries into a newly
        allocated right sibling node and promotes the middle separator upward.
        """
        n = self._m_int
        eks = self._effective_key_size
        stride = self._int_stride
        mid = n // 2
        right_keys = n - mid

        new_entry = new_sep + _U32_STRUCT.pack(new_child)
        right_pid, right_buf = self._bpm.new_page()
        right_dirty = False
        try:
            right_buf[:] = _ZERO_PAGE
            if ins_idx == mid:
                promoted_key = new_sep
                _U32_STRUCT.pack_into(right_buf, _INTERNAL_CHILDREN_OFFSET, new_child)
                src_start = _INTERNAL_ENTRIES_OFFSET + mid * stride
                src_end = _INTERNAL_ENTRIES_OFFSET + n * stride
                right_buf[
                    _INTERNAL_ENTRIES_OFFSET : _INTERNAL_ENTRIES_OFFSET + right_keys * stride
                ] = left_buf[src_start:src_end]
            elif ins_idx < mid:
                prom_off = _INTERNAL_ENTRIES_OFFSET + (mid - 1) * stride
                promoted_key = bytes(left_buf[prom_off : prom_off + eks])
                # Offset 32 + mid*stride holds C[mid], followed contiguously by K[mid..n-1] & C[mid+1..n].
                src_start = _INTERNAL_CHILDREN_OFFSET + mid * stride
                src_end = _INTERNAL_ENTRIES_OFFSET + n * stride
                right_buf[
                    _INTERNAL_CHILDREN_OFFSET : _INTERNAL_ENTRIES_OFFSET + right_keys * stride
                ] = left_buf[src_start:src_end]
                if ins_idx < mid - 1:
                    s_start = _INTERNAL_ENTRIES_OFFSET + ins_idx * stride
                    s_end = _INTERNAL_ENTRIES_OFFSET + (mid - 1) * stride
                    d_start = _INTERNAL_ENTRIES_OFFSET + (ins_idx + 1) * stride
                    d_end = _INTERNAL_ENTRIES_OFFSET + mid * stride
                    left_buf[d_start:d_end] = left_buf[s_start:s_end]
                ins_off = _INTERNAL_ENTRIES_OFFSET + ins_idx * stride
                left_buf[ins_off : ins_off + stride] = new_entry
            else:
                prom_off = _INTERNAL_ENTRIES_OFFSET + mid * stride
                promoted_key = bytes(left_buf[prom_off : prom_off + eks])
                r_idx = ins_idx - (mid + 1)
                pre_src_start = _INTERNAL_CHILDREN_OFFSET + (mid + 1) * stride
                pre_src_end = _INTERNAL_ENTRIES_OFFSET + ins_idx * stride
                right_buf[
                    _INTERNAL_CHILDREN_OFFSET : _INTERNAL_ENTRIES_OFFSET + r_idx * stride
                ] = left_buf[pre_src_start:pre_src_end]
                r_ins_off = _INTERNAL_ENTRIES_OFFSET + r_idx * stride
                right_buf[r_ins_off : r_ins_off + stride] = new_entry
                if ins_idx < n:
                    post_src_start = _INTERNAL_ENTRIES_OFFSET + ins_idx * stride
                    post_src_end = _INTERNAL_ENTRIES_OFFSET + n * stride
                    right_buf[
                        r_ins_off + stride : _INTERNAL_ENTRIES_OFFSET + right_keys * stride
                    ] = left_buf[post_src_start:post_src_end]

            left_end = _INTERNAL_ENTRIES_OFFSET + mid * stride
            old_end = _INTERNAL_ENTRIES_OFFSET + n * stride
            if left_end < old_end:
                left_buf[left_end:old_end] = bytes(old_end - left_end)

            PageHeader(
                page_id=right_pid,
                page_type=PageType.BPLUS_INTERNAL,
                flags=PageFlags.NONE,
                slot_count=right_keys,
                page_lsn=0,
                free_space_ptr=_INTERNAL_ENTRIES_OFFSET + right_keys * stride,
                next_page_id=INVALID_PAGE_ID,
                checksum=0,
            ).pack_into(right_buf, 0)

            _U16_STRUCT.pack_into(left_buf, _SLOT_COUNT_OFFSET, mid)
            _U32_STRUCT.pack_into(left_buf, _FREE_SPACE_PTR_OFFSET, left_end)
            if clear_root_flag:
                left_buf[_FLAGS_OFFSET] &= ~int(PageFlags.IS_ROOT)

            right_dirty = True
            return promoted_key, right_pid
        finally:
            right_buf.release()
            self._bpm.unpin_page(right_pid, is_dirty=right_dirty)

    def _propagate_split_upward(
        self,
        path: list[tuple[int, int]],
        promoted_key: bytes,
        promoted_child: int,
    ) -> None:
        """Insert ``(promoted_key, promoted_child)`` along ancestor ``path``, splitting as needed."""
        stride = self._int_stride
        eks = self._effective_key_size

        for parent_pid, child_idx in reversed(path):
            parent_buf = self._bpm.fetch_page(parent_pid)
            parent_dirty = False
            try:
                pn = _U16_STRUCT.unpack_from(parent_buf, _SLOT_COUNT_OFFSET)[0]
                if pn < self._m_int:
                    if child_idx < pn:
                        src_start = _INTERNAL_ENTRIES_OFFSET + child_idx * stride
                        src_end = _INTERNAL_ENTRIES_OFFSET + pn * stride
                        dst_start = _INTERNAL_ENTRIES_OFFSET + (child_idx + 1) * stride
                        dst_end = _INTERNAL_ENTRIES_OFFSET + (pn + 1) * stride
                        parent_buf[dst_start:dst_end] = parent_buf[src_start:src_end]
                    ins_off = _INTERNAL_ENTRIES_OFFSET + child_idx * stride
                    parent_buf[ins_off : ins_off + eks] = promoted_key
                    _U32_STRUCT.pack_into(parent_buf, ins_off + eks, promoted_child)
                    _U16_STRUCT.pack_into(parent_buf, _SLOT_COUNT_OFFSET, pn + 1)
                    _U32_STRUCT.pack_into(
                        parent_buf,
                        _FREE_SPACE_PTR_OFFSET,
                        _INTERNAL_ENTRIES_OFFSET + (pn + 1) * stride,
                    )
                    parent_dirty = True
                    return

                is_root = parent_pid == self._root_page_id
                promoted_key, promoted_child = self._split_internal_and_insert(
                    parent_pid,
                    parent_buf,
                    child_idx,
                    promoted_key,
                    promoted_child,
                    clear_root_flag=is_root,
                )
                parent_dirty = True
            finally:
                parent_buf.release()
                self._bpm.unpin_page(parent_pid, is_dirty=parent_dirty)

        # Previous root split; allocate a new root internal node above both halves.
        new_root_pid, new_root_buf = self._bpm.new_page()
        root_dirty = False
        try:
            new_root_buf[:] = _ZERO_PAGE
            PageHeader(
                page_id=new_root_pid,
                page_type=PageType.BPLUS_INTERNAL,
                flags=PageFlags.IS_ROOT,
                slot_count=1,
                page_lsn=0,
                free_space_ptr=_INTERNAL_ENTRIES_OFFSET + stride,
                next_page_id=INVALID_PAGE_ID,
                checksum=0,
            ).pack_into(new_root_buf, 0)
            _U32_STRUCT.pack_into(
                new_root_buf, _INTERNAL_CHILDREN_OFFSET, self._root_page_id
            )
            new_root_buf[_INTERNAL_ENTRIES_OFFSET : _INTERNAL_ENTRIES_OFFSET + eks] = (
                promoted_key
            )
            _U32_STRUCT.pack_into(
                new_root_buf, _INTERNAL_ENTRIES_OFFSET + eks, promoted_child
            )
            root_dirty = True
        finally:
            new_root_buf.release()
            self._bpm.unpin_page(new_root_pid, is_dirty=root_dirty)

        self._root_page_id = new_root_pid
        self._height += 1

    def _insert_into_empty_tree(self, comp_key: bytes, rid: RID) -> None:
        """Allocate the first leaf root page and persist initial metadata."""
        leaf_pid, leaf_buf = self._bpm.new_page()
        leaf_dirty = False
        try:
            leaf_buf[:] = _ZERO_PAGE
            PageHeader(
                page_id=leaf_pid,
                page_type=PageType.BPLUS_LEAF,
                flags=PageFlags.IS_ROOT | PageFlags.IS_LEAF,
                slot_count=1,
                page_lsn=0,
                free_space_ptr=_LEAF_ENTRIES_OFFSET + self._leaf_stride,
                next_page_id=INVALID_PAGE_ID,
                checksum=0,
            ).pack_into(leaf_buf, 0)
            leaf_buf[
                _LEAF_ENTRIES_OFFSET : _LEAF_ENTRIES_OFFSET + self._effective_key_size
            ] = comp_key
            _RID_LE_STRUCT.pack_into(
                leaf_buf,
                _LEAF_ENTRIES_OFFSET + self._effective_key_size,
                rid.page_id,
                rid.slot_id,
            )
            leaf_dirty = True
        finally:
            leaf_buf.release()
            self._bpm.unpin_page(leaf_pid, is_dirty=leaf_dirty)

        self._root_page_id = leaf_pid
        self._height = 1
        self._entry_count = 1
        self._persist_metadata(flush_to_disk=True)

    def _has_local_unique_duplicate(
        self,
        leaf_buf: memoryview,
        count: int,
        ins_idx: int,
        key_bytes: bytes,
    ) -> bool:
        """Return ``True`` if ``key_bytes`` already exists at ``ins_idx``, ``ins_idx-1``, or next leaf."""
        stride = self._leaf_stride
        ksize = self._key_size
        if ins_idx < count:
            off = _LEAF_ENTRIES_OFFSET + ins_idx * stride
            if bytes(leaf_buf[off : off + ksize]) == key_bytes:
                return True
        if ins_idx > 0:
            prev_off = _LEAF_ENTRIES_OFFSET + (ins_idx - 1) * stride
            if bytes(leaf_buf[prev_off : prev_off + ksize]) == key_bytes:
                return True
        if ins_idx == count:
            next_pid = _U32_STRUCT.unpack_from(leaf_buf, _NEXT_PAGE_ID_OFFSET)[0]
            if (
                next_pid != INVALID_PAGE_ID
                and self._peek_next_leaf_first_base_key(next_pid) == key_bytes
            ):
                return True
        return False

    def insert(self, key: Any, rid: RID) -> None:
        """Insert a ``(key, rid)`` entry into the B+ Tree.

        Args:
            key: Logical key matching the configured ``DataType``.
            rid: Physical ``RID(page_id, slot_id)`` of the indexed tuple.

        Raises:
            TypeError: If ``rid`` is not a ``RID`` or ``key`` mismatches ``key_dtype``.
            ValueError: If ``key`` overflows ``key_size`` or is ``NaN``.
            DuplicateKeyError: If ``unique=True`` and ``key`` already exists.
            StorageError: If the index is closed or disk/pool operations fail.

        Complexity:
            $O(\\log_B N)$ page accesses; holds at most 2 pinned frames simultaneously.
            Thread-safe under ``self._lock``.
        """
        if not isinstance(rid, RID):
            raise TypeError(f"rid must be a RID, got {type(rid).__name__}")
        key_bytes = encode_key(key, self._key_dtype, self._key_size)
        comp_key = make_composite_key(key_bytes, rid)

        with self._lock:
            self._ensure_open()
            if self._root_page_id == INVALID_PAGE_ID:
                self._insert_into_empty_tree(comp_key, rid)
                return

            path: list[tuple[int, int]] = []
            track_sep = key_bytes if self._unique else None
            leaf_pid, leaf_buf, saw_left_sep = self._find_leaf(
                comp_key, track_path=path, track_left_sep_key=track_sep
            )
            leaf_dirty = False
            try:
                count = _U16_STRUCT.unpack_from(leaf_buf, _SLOT_COUNT_OFFSET)[0]
                ins_idx = self._leaf_bisect_left(leaf_buf, count, comp_key)
                stride = self._leaf_stride
                eks = self._effective_key_size

                if self._unique:
                    if self._has_local_unique_duplicate(leaf_buf, count, ins_idx, key_bytes):
                        raise DuplicateKeyError(
                            f"Duplicate key {key!r} violates unique B+ Tree constraint"
                        )
                    if ins_idx == 0 and saw_left_sep:
                        leaf_buf.release()
                        self._bpm.unpin_page(leaf_pid, is_dirty=False)
                        leaf_pid = INVALID_PAGE_ID
                        if self._contains_key_bytes(key_bytes):
                            raise DuplicateKeyError(
                                f"Duplicate key {key!r} violates unique B+ Tree constraint"
                            )
                        leaf_pid = self._reacquire_leaf_id(comp_key)
                        leaf_buf = self._bpm.fetch_page(leaf_pid)
                        count = _U16_STRUCT.unpack_from(leaf_buf, _SLOT_COUNT_OFFSET)[0]
                        ins_idx = self._leaf_bisect_left(leaf_buf, count, comp_key)
                else:
                    if ins_idx < count:
                        off = _LEAF_ENTRIES_OFFSET + ins_idx * stride
                        if bytes(leaf_buf[off : off + eks]) == comp_key:
                            return

                new_record = comp_key + _RID_LE_STRUCT.pack(rid.page_id, rid.slot_id)
                if count < self._m_leaf:
                    if ins_idx < count:
                        src_start = _LEAF_ENTRIES_OFFSET + ins_idx * stride
                        src_end = _LEAF_ENTRIES_OFFSET + count * stride
                        dst_start = _LEAF_ENTRIES_OFFSET + (ins_idx + 1) * stride
                        dst_end = _LEAF_ENTRIES_OFFSET + (count + 1) * stride
                        leaf_buf[dst_start:dst_end] = leaf_buf[src_start:src_end]
                    ins_off = _LEAF_ENTRIES_OFFSET + ins_idx * stride
                    leaf_buf[ins_off : ins_off + stride] = new_record
                    _U16_STRUCT.pack_into(leaf_buf, _SLOT_COUNT_OFFSET, count + 1)
                    _U32_STRUCT.pack_into(
                        leaf_buf,
                        _FREE_SPACE_PTR_OFFSET,
                        _LEAF_ENTRIES_OFFSET + (count + 1) * stride,
                    )
                    leaf_dirty = True
                    self._entry_count += 1
                    root_changed = False
                else:
                    prev_root = self._root_page_id
                    prev_height = self._height
                    promoted_key, promoted_child = self._split_leaf_and_insert(
                        leaf_pid,
                        leaf_buf,
                        ins_idx,
                        new_record,
                        clear_root_flag=(len(path) == 0),
                    )
                    leaf_dirty = True
                    self._entry_count += 1
                    leaf_buf.release()
                    self._bpm.unpin_page(leaf_pid, is_dirty=True)
                    leaf_pid = INVALID_PAGE_ID
                    self._propagate_split_upward(path, promoted_key, promoted_child)
                    root_changed = (
                        self._root_page_id != prev_root or self._height != prev_height
                    )
            finally:
                if leaf_pid != INVALID_PAGE_ID:
                    leaf_buf.release()
                    self._bpm.unpin_page(leaf_pid, is_dirty=leaf_dirty)

            self._persist_metadata(flush_to_disk=root_changed)

    def _reacquire_leaf_id(self, comp_key: bytes) -> int:
        """Descend to find the leaf page ID for ``comp_key`` and immediately unpin it."""
        leaf_pid, leaf_buf, _ = self._find_leaf(comp_key)
        leaf_buf.release()
        self._bpm.unpin_page(leaf_pid, is_dirty=False)
        return leaf_pid

    def get(self, key: Any) -> list[RID]:
        """Look up all ``RID`` values associated with ``key``.

        Args:
            key: Logical key matching the configured ``DataType``.

        Returns:
            Sorted list of matching ``RID`` instances, or ``[]`` if absent.

        Raises:
            TypeError: If ``key`` has an incompatible Python type.
            ValueError: If ``key`` overflows ``key_size`` or is ``NaN``.
            StorageError: If the index is closed or a page cannot be read.
            PageCorruptedError: If an accessed page fails checksum validation.

        Complexity:
            $O(\\log_B N + k)$ where $k$ is the number of matching duplicates;
            holds at most 1 pinned frame at any instant. Thread-safe under
            ``self._lock``.
        """
        key_bytes = encode_key(key, self._key_dtype, self._key_size)
        low_comp = key_bytes + _ZERO_RID_SUFFIX

        with self._lock:
            self._ensure_open()
            if self._root_page_id == INVALID_PAGE_ID:
                return []

            results: list[RID] = []
            curr_pid, curr_buf, _ = self._find_leaf(low_comp)
            try:
                count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                idx = self._leaf_bisect_left(curr_buf, count, low_comp)
                stride = self._leaf_stride
                eks = self._effective_key_size
                ksize = self._key_size

                while True:
                    while idx < count:
                        off = _LEAF_ENTRIES_OFFSET + idx * stride
                        entry_base = bytes(curr_buf[off : off + ksize])
                        if entry_base > key_bytes:
                            return results
                        if entry_base == key_bytes:
                            rid_pid, rid_slot = _RID_LE_STRUCT.unpack_from(
                                curr_buf, off + eks
                            )
                            results.append(RID(page_id=rid_pid, slot_id=rid_slot))
                            if self._unique:
                                return results
                        idx += 1

                    next_pid = _U32_STRUCT.unpack_from(curr_buf, _NEXT_PAGE_ID_OFFSET)[0]
                    if next_pid == INVALID_PAGE_ID:
                        return results
                    # Unpin current leaf before fetching right sibling to keep <= 1 pinned frame.
                    curr_buf.release()
                    curr_buf = None
                    self._bpm.unpin_page(curr_pid, is_dirty=False)
                    curr_pid = INVALID_PAGE_ID

                    next_buf = self._bpm.fetch_page(next_pid)
                    curr_pid = next_pid
                    curr_buf = next_buf
                    count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                    idx = 0
            finally:
                if curr_buf is not None:
                    curr_buf.release()
                if curr_pid != INVALID_PAGE_ID:
                    self._bpm.unpin_page(curr_pid, is_dirty=False)

    def range_scan(
        self, low_key: Any, high_key: Any, inclusive: bool = True
    ) -> list[tuple[Any, RID]]:
        """Scan all ``(key, RID)`` pairs with ``low_key <= key <= high_key`` (or strict).

        Args:
            low_key: Lower range bound matching ``key_dtype``.
            high_key: Upper range bound matching ``key_dtype``.
            inclusive: Include both ``low_key`` and ``high_key`` when ``True``;
                exclude both endpoints when ``False``.

        Returns:
            List of ``(decoded_key, RID)`` tuples in ascending composite key order.

        Raises:
            TypeError: If ``low_key`` or ``high_key`` has an incompatible type.
            ValueError: If either bound overflows ``key_size`` or is ``NaN``.
            StorageError: If the index is closed or disk/pool operations fail.
            PageCorruptedError: If an accessed page fails checksum validation.

        Complexity:
            $O(\\log_B N + k)$ page visits for $k$ returned entries; holds at
            most 1 pinned frame at any instant. Thread-safe under ``self._lock``.
        """
        low_bytes = encode_key(low_key, self._key_dtype, self._key_size)
        high_bytes = encode_key(high_key, self._key_dtype, self._key_size)
        if low_bytes > high_bytes or (not inclusive and low_bytes == high_bytes):
            return []

        with self._lock:
            self._ensure_open()
            if self._root_page_id == INVALID_PAGE_ID:
                return []

            search_comp = (
                low_bytes + _ZERO_RID_SUFFIX
                if inclusive
                else low_bytes + _MAX_RID_SUFFIX
            )

            results: list[tuple[Any, RID]] = []
            prev_base: bytes | None = None
            prev_decoded: Any = None

            curr_pid, curr_buf, _ = self._find_leaf(search_comp)
            try:
                count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                idx = (
                    self._leaf_bisect_left(curr_buf, count, search_comp)
                    if inclusive
                    else self._leaf_bisect_right(curr_buf, count, search_comp)
                )
                stride = self._leaf_stride
                eks = self._effective_key_size
                ksize = self._key_size

                while True:
                    while idx < count:
                        off = _LEAF_ENTRIES_OFFSET + idx * stride
                        entry_base = bytes(curr_buf[off : off + ksize])
                        if inclusive:
                            if entry_base > high_bytes:
                                return results
                            in_lower = entry_base >= low_bytes
                        else:
                            if entry_base >= high_bytes:
                                return results
                            in_lower = entry_base > low_bytes

                        if in_lower:
                            if entry_base != prev_base:
                                prev_base = entry_base
                                prev_decoded = decode_key(entry_base, self._key_dtype)
                            rid_pid, rid_slot = _RID_LE_STRUCT.unpack_from(
                                curr_buf, off + eks
                            )
                            results.append((prev_decoded, RID(page_id=rid_pid, slot_id=rid_slot)))
                        idx += 1

                    next_pid = _U32_STRUCT.unpack_from(curr_buf, _NEXT_PAGE_ID_OFFSET)[0]
                    if next_pid == INVALID_PAGE_ID:
                        return results
                    curr_buf.release()
                    curr_buf = None
                    self._bpm.unpin_page(curr_pid, is_dirty=False)
                    curr_pid = INVALID_PAGE_ID

                    next_buf = self._bpm.fetch_page(next_pid)
                    curr_pid = next_pid
                    curr_buf = next_buf
                    count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                    idx = 0
            finally:
                if curr_buf is not None:
                    curr_buf.release()
                if curr_pid != INVALID_PAGE_ID:
                    self._bpm.unpin_page(curr_pid, is_dirty=False)

    def _find_prev_leaf(self, path: list[tuple[int, int]]) -> int | None:
        """Return the immediate left sibling leaf of the node at ``path``, or ``None``."""
        anc_pid: int | None = None
        anc_cidx: int = 0
        for pid, cidx in reversed(path):
            if cidx > 0:
                anc_pid = pid
                anc_cidx = cidx
                break
        if anc_pid is None:
            return None

        anc_buf = self._bpm.fetch_page(anc_pid)
        try:
            walk_pid = _U32_STRUCT.unpack_from(
                anc_buf, _INTERNAL_CHILDREN_OFFSET + (anc_cidx - 1) * self._int_stride
            )[0]
        finally:
            anc_buf.release()
            self._bpm.unpin_page(anc_pid, is_dirty=False)

        while True:
            walk_buf = self._bpm.fetch_page(walk_pid)
            try:
                page_type = walk_buf[_PAGE_TYPE_OFFSET]
                if page_type == PageType.BPLUS_LEAF:
                    return walk_pid
                if page_type != PageType.BPLUS_INTERNAL:
                    raise PageCorruptedError(
                        f"Unexpected page_type {page_type} while locating prev leaf on {walk_pid}"
                    )
                wn = _U16_STRUCT.unpack_from(walk_buf, _SLOT_COUNT_OFFSET)[0]
                next_walk_pid = _U32_STRUCT.unpack_from(
                    walk_buf, _INTERNAL_CHILDREN_OFFSET + wn * self._int_stride
                )[0]
            finally:
                walk_buf.release()
                self._bpm.unpin_page(walk_pid, is_dirty=False)
            walk_pid = next_walk_pid

    def _rebalance_single_child_internal(
        self,
        parent_pid: int,
        only_child: int,
        grandparent_pid: int,
        gp_idx: int,
    ) -> bool:
        """Merge or borrow from an adjacent sibling when ``parent_pid`` drops to 0 keys.

        Returns ``True`` if ``parent_pid`` was merged into its sibling and deleted
        (so the caller must remove ``parent_pid`` from ``grandparent_pid``), or
        ``False`` if a key/child was borrowed and the tree is now balanced.
        """
        stride = self._int_stride
        eks = self._effective_key_size

        gp_buf = self._bpm.fetch_page(grandparent_pid)
        try:
            if gp_idx > 0:
                sib_is_left = True
                p_sep_idx = gp_idx - 1
                sib_pid = _U32_STRUCT.unpack_from(
                    gp_buf, _INTERNAL_CHILDREN_OFFSET + (gp_idx - 1) * stride
                )[0]
            else:
                sib_is_left = False
                p_sep_idx = 0
                sib_pid = _U32_STRUCT.unpack_from(
                    gp_buf, _INTERNAL_CHILDREN_OFFSET + stride
                )[0]
            p_sep_off = _INTERNAL_ENTRIES_OFFSET + p_sep_idx * stride
            p_sep = bytes(gp_buf[p_sep_off : p_sep_off + eks])
        finally:
            gp_buf.release()
            self._bpm.unpin_page(grandparent_pid, is_dirty=False)

        sib_buf = self._bpm.fetch_page(sib_pid)
        sib_dirty = False
        borrowed_key: bytes | None = None
        borrowed_child: int = INVALID_PAGE_ID
        try:
            sn = _U16_STRUCT.unpack_from(sib_buf, _SLOT_COUNT_OFFSET)[0]
            if sn < self._m_int:
                if sib_is_left:
                    ins_off = _INTERNAL_ENTRIES_OFFSET + sn * stride
                    sib_buf[ins_off : ins_off + eks] = p_sep
                    _U32_STRUCT.pack_into(sib_buf, ins_off + eks, only_child)
                else:
                    old_c0 = _U32_STRUCT.unpack_from(sib_buf, _INTERNAL_CHILDREN_OFFSET)[0]
                    src_start = _INTERNAL_ENTRIES_OFFSET
                    src_end = _INTERNAL_ENTRIES_OFFSET + sn * stride
                    dst_start = _INTERNAL_ENTRIES_OFFSET + stride
                    dst_end = _INTERNAL_ENTRIES_OFFSET + (sn + 1) * stride
                    sib_buf[dst_start:dst_end] = sib_buf[src_start:src_end]
                    _U32_STRUCT.pack_into(sib_buf, _INTERNAL_CHILDREN_OFFSET, only_child)
                    sib_buf[_INTERNAL_ENTRIES_OFFSET : _INTERNAL_ENTRIES_OFFSET + eks] = (
                        p_sep
                    )
                    _U32_STRUCT.pack_into(sib_buf, _INTERNAL_ENTRIES_OFFSET + eks, old_c0)
                _U16_STRUCT.pack_into(sib_buf, _SLOT_COUNT_OFFSET, sn + 1)
                _U32_STRUCT.pack_into(
                    sib_buf,
                    _FREE_SPACE_PTR_OFFSET,
                    _INTERNAL_ENTRIES_OFFSET + (sn + 1) * stride,
                )
                sib_dirty = True
            else:
                if sib_is_left:
                    b_off = _INTERNAL_ENTRIES_OFFSET + (sn - 1) * stride
                    borrowed_key = bytes(sib_buf[b_off : b_off + eks])
                    borrowed_child = _U32_STRUCT.unpack_from(sib_buf, b_off + eks)[0]
                    sib_buf[b_off : b_off + stride] = bytes(stride)
                else:
                    borrowed_child = _U32_STRUCT.unpack_from(
                        sib_buf, _INTERNAL_CHILDREN_OFFSET
                    )[0]
                    borrowed_key = bytes(
                        sib_buf[_INTERNAL_ENTRIES_OFFSET : _INTERNAL_ENTRIES_OFFSET + eks]
                    )
                    new_sib_c0 = _U32_STRUCT.unpack_from(
                        sib_buf, _INTERNAL_ENTRIES_OFFSET + eks
                    )[0]
                    _U32_STRUCT.pack_into(sib_buf, _INTERNAL_CHILDREN_OFFSET, new_sib_c0)
                    src_start = _INTERNAL_ENTRIES_OFFSET + stride
                    src_end = _INTERNAL_ENTRIES_OFFSET + sn * stride
                    dst_start = _INTERNAL_ENTRIES_OFFSET
                    dst_end = _INTERNAL_ENTRIES_OFFSET + (sn - 1) * stride
                    sib_buf[dst_start:dst_end] = sib_buf[src_start:src_end]
                    sib_buf[dst_end:src_end] = bytes(stride)
                _U16_STRUCT.pack_into(sib_buf, _SLOT_COUNT_OFFSET, sn - 1)
                _U32_STRUCT.pack_into(
                    sib_buf,
                    _FREE_SPACE_PTR_OFFSET,
                    _INTERNAL_ENTRIES_OFFSET + (sn - 1) * stride,
                )
                sib_dirty = True
        finally:
            sib_buf.release()
            self._bpm.unpin_page(sib_pid, is_dirty=sib_dirty)

        if borrowed_key is None:
            self._bpm.delete_page(parent_pid)
            return True

        parent_buf = self._bpm.fetch_page(parent_pid)
        try:
            c0, c1 = (
                (borrowed_child, only_child) if sib_is_left else (only_child, borrowed_child)
            )
            _U32_STRUCT.pack_into(parent_buf, _INTERNAL_CHILDREN_OFFSET, c0)
            parent_buf[_INTERNAL_ENTRIES_OFFSET : _INTERNAL_ENTRIES_OFFSET + eks] = p_sep
            _U32_STRUCT.pack_into(parent_buf, _INTERNAL_ENTRIES_OFFSET + eks, c1)
            _U16_STRUCT.pack_into(parent_buf, _SLOT_COUNT_OFFSET, 1)
            _U32_STRUCT.pack_into(
                parent_buf, _FREE_SPACE_PTR_OFFSET, _INTERNAL_ENTRIES_OFFSET + stride
            )
            parent_buf[
                _INTERNAL_ENTRIES_OFFSET + stride : _INTERNAL_ENTRIES_OFFSET + 2 * stride
            ] = bytes(stride)
        finally:
            parent_buf.release()
            self._bpm.unpin_page(parent_pid, is_dirty=True)

        gp_buf = self._bpm.fetch_page(grandparent_pid)
        try:
            gp_buf[p_sep_off : p_sep_off + eks] = borrowed_key
        finally:
            gp_buf.release()
            self._bpm.unpin_page(grandparent_pid, is_dirty=True)
        return False

    def _remove_child_at_path(self, path: list[tuple[int, int]]) -> None:
        """Remove the child at ``path[-1]`` and rebalance/collapse empty internal parents."""
        stride = self._int_stride
        eks = self._effective_key_size
        level = len(path) - 1

        while level >= 0:
            parent_pid, child_idx = path[level]
            parent_buf = self._bpm.fetch_page(parent_pid)
            parent_dirty = False
            only_child: int | None = None
            try:
                pn = _U16_STRUCT.unpack_from(parent_buf, _SLOT_COUNT_OFFSET)[0]
                if pn >= 2:
                    if child_idx == 0:
                        new_c0 = _U32_STRUCT.unpack_from(
                            parent_buf, _INTERNAL_ENTRIES_OFFSET + eks
                        )[0]
                        _U32_STRUCT.pack_into(parent_buf, _INTERNAL_CHILDREN_OFFSET, new_c0)
                        src_start = _INTERNAL_ENTRIES_OFFSET + stride
                        src_end = _INTERNAL_ENTRIES_OFFSET + pn * stride
                        dst_start = _INTERNAL_ENTRIES_OFFSET
                        dst_end = _INTERNAL_ENTRIES_OFFSET + (pn - 1) * stride
                        if src_start < src_end:
                            parent_buf[dst_start:dst_end] = parent_buf[src_start:src_end]
                    elif child_idx < pn:
                        src_start = _INTERNAL_ENTRIES_OFFSET + child_idx * stride
                        src_end = _INTERNAL_ENTRIES_OFFSET + pn * stride
                        dst_start = _INTERNAL_ENTRIES_OFFSET + (child_idx - 1) * stride
                        dst_end = _INTERNAL_ENTRIES_OFFSET + (pn - 1) * stride
                        parent_buf[dst_start:dst_end] = parent_buf[src_start:src_end]
                    new_end = _INTERNAL_ENTRIES_OFFSET + (pn - 1) * stride
                    old_end = _INTERNAL_ENTRIES_OFFSET + pn * stride
                    parent_buf[new_end:old_end] = bytes(old_end - new_end)
                    _U16_STRUCT.pack_into(parent_buf, _SLOT_COUNT_OFFSET, pn - 1)
                    _U32_STRUCT.pack_into(parent_buf, _FREE_SPACE_PTR_OFFSET, new_end)
                    parent_dirty = True
                    return

                # pn == 1: removing child_idx leaves parent with one surviving child.
                only_child = _U32_STRUCT.unpack_from(
                    parent_buf, _INTERNAL_CHILDREN_OFFSET + (1 - child_idx) * stride
                )[0]
            finally:
                parent_buf.release()
                self._bpm.unpin_page(parent_pid, is_dirty=parent_dirty)

            if level == 0:
                self._bpm.delete_page(parent_pid)
                child_buf = self._bpm.fetch_page(only_child)
                try:
                    child_buf[_FLAGS_OFFSET] |= int(PageFlags.IS_ROOT)
                finally:
                    child_buf.release()
                    self._bpm.unpin_page(only_child, is_dirty=True)
                self._root_page_id = only_child
                self._height -= 1
                return

            grandparent_pid, gp_idx = path[level - 1]
            if not self._rebalance_single_child_internal(
                parent_pid, only_child, grandparent_pid, gp_idx
            ):
                return
            level -= 1

    def _prune_empty_leaf(
        self, leaf_pid: int, next_pid: int, path: list[tuple[int, int]]
    ) -> None:
        """Unlink and deallocate an emptied leaf page and collapse ancestors as needed."""
        if not path:
            self._bpm.delete_page(leaf_pid)
            self._root_page_id = INVALID_PAGE_ID
            self._height = 0
            return

        prev_leaf_pid = self._find_prev_leaf(path)
        if prev_leaf_pid is not None:
            prev_buf = self._bpm.fetch_page(prev_leaf_pid)
            try:
                _U32_STRUCT.pack_into(prev_buf, _NEXT_PAGE_ID_OFFSET, next_pid)
            finally:
                prev_buf.release()
                self._bpm.unpin_page(prev_leaf_pid, is_dirty=True)

        self._bpm.delete_page(leaf_pid)
        self._remove_child_at_path(path)

    def _delete_single_rid(self, key_bytes: bytes, rid: RID) -> bool:
        """Delete a single ``(key_bytes, rid)`` entry; returns ``True`` if found."""
        comp_key = make_composite_key(key_bytes, rid)
        path: list[tuple[int, int]] = []
        leaf_pid, leaf_buf, _ = self._find_leaf(comp_key, track_path=path)
        leaf_dirty = False
        deleted = False
        prune_next: int | None = None
        try:
            count = _U16_STRUCT.unpack_from(leaf_buf, _SLOT_COUNT_OFFSET)[0]
            idx = self._leaf_bisect_left(leaf_buf, count, comp_key)
            stride = self._leaf_stride
            off = _LEAF_ENTRIES_OFFSET + idx * stride
            if (
                idx >= count
                or bytes(leaf_buf[off : off + self._effective_key_size]) != comp_key
            ):
                return False

            if count == 1:
                prune_next = _U32_STRUCT.unpack_from(leaf_buf, _NEXT_PAGE_ID_OFFSET)[0]
            else:
                if idx < count - 1:
                    src_start = _LEAF_ENTRIES_OFFSET + (idx + 1) * stride
                    src_end = _LEAF_ENTRIES_OFFSET + count * stride
                    dst_start = _LEAF_ENTRIES_OFFSET + idx * stride
                    dst_end = _LEAF_ENTRIES_OFFSET + (count - 1) * stride
                    leaf_buf[dst_start:dst_end] = leaf_buf[src_start:src_end]
                new_end = _LEAF_ENTRIES_OFFSET + (count - 1) * stride
                old_end = _LEAF_ENTRIES_OFFSET + count * stride
                leaf_buf[new_end:old_end] = bytes(old_end - new_end)
                _U16_STRUCT.pack_into(leaf_buf, _SLOT_COUNT_OFFSET, count - 1)
                _U32_STRUCT.pack_into(leaf_buf, _FREE_SPACE_PTR_OFFSET, new_end)
                leaf_dirty = True
            deleted = True
        finally:
            leaf_buf.release()
            self._bpm.unpin_page(leaf_pid, is_dirty=leaf_dirty)

        if not deleted:
            return False
        if prune_next is not None:
            self._prune_empty_leaf(leaf_pid, prune_next, path)
        self._entry_count -= 1
        return True

    def _delete_all_for_key(self, key_bytes: bytes) -> bool:
        """Delete every entry whose base key equals ``key_bytes`` across sibling leaves."""
        low_comp = key_bytes + _ZERO_RID_SUFFIX
        high_comp = key_bytes + _MAX_RID_SUFFIX
        stride = self._leaf_stride
        eks = self._effective_key_size
        deleted_total = 0

        curr_pid, curr_buf, _ = self._find_leaf(low_comp)
        curr_dirty = False
        try:
            count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
            del_start = self._leaf_bisect_left(curr_buf, count, low_comp)
            while True:
                if del_start < count:
                    del_end = self._leaf_bisect_right(curr_buf, count, high_comp)
                    if del_end <= del_start:
                        break
                    removed_here = del_end - del_start
                    next_pid = _U32_STRUCT.unpack_from(
                        curr_buf, _NEXT_PAGE_ID_OFFSET
                    )[0]

                    if removed_here < count:
                        if del_end < count:
                            src_start = _LEAF_ENTRIES_OFFSET + del_end * stride
                            src_end = _LEAF_ENTRIES_OFFSET + count * stride
                            dst_start = _LEAF_ENTRIES_OFFSET + del_start * stride
                            dst_end = (
                                _LEAF_ENTRIES_OFFSET + (count - removed_here) * stride
                            )
                            curr_buf[dst_start:dst_end] = curr_buf[src_start:src_end]
                        new_end = _LEAF_ENTRIES_OFFSET + (count - removed_here) * stride
                        old_end = _LEAF_ENTRIES_OFFSET + count * stride
                        curr_buf[new_end:old_end] = bytes(old_end - new_end)
                        _U16_STRUCT.pack_into(
                            curr_buf, _SLOT_COUNT_OFFSET, count - removed_here
                        )
                        _U32_STRUCT.pack_into(curr_buf, _FREE_SPACE_PTR_OFFSET, new_end)
                        curr_dirty = True
                        deleted_total += removed_here
                        if del_end < count:
                            # Encountered a strictly larger key on this leaf; no more matches exist.
                            break
                    else:
                        # Whole leaf holds only key_bytes; record anchor key before pruning.
                        anchor_comp = bytes(
                            curr_buf[_LEAF_ENTRIES_OFFSET : _LEAF_ENTRIES_OFFSET + eks]
                        )
                        curr_buf.release()
                        curr_buf = None
                        self._bpm.unpin_page(curr_pid, is_dirty=False)
                        curr_pid = INVALID_PAGE_ID

                        prune_path: list[tuple[int, int]] = []
                        found_pid, found_buf, _ = self._find_leaf(
                            anchor_comp, track_path=prune_path
                        )
                        found_buf.release()
                        self._bpm.unpin_page(found_pid, is_dirty=False)
                        self._prune_empty_leaf(found_pid, next_pid, prune_path)
                        deleted_total += removed_here

                        if next_pid == INVALID_PAGE_ID:
                            break
                        curr_buf = self._bpm.fetch_page(next_pid)
                        curr_pid = next_pid
                        curr_dirty = False
                        count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                        del_start = 0
                        continue
                else:
                    next_pid = _U32_STRUCT.unpack_from(
                        curr_buf, _NEXT_PAGE_ID_OFFSET
                    )[0]

                if next_pid == INVALID_PAGE_ID:
                    break
                curr_buf.release()
                curr_buf = None
                self._bpm.unpin_page(curr_pid, is_dirty=curr_dirty)
                curr_pid = INVALID_PAGE_ID
                curr_dirty = False

                curr_buf = self._bpm.fetch_page(next_pid)
                curr_pid = next_pid
                count = _U16_STRUCT.unpack_from(curr_buf, _SLOT_COUNT_OFFSET)[0]
                del_start = 0
        finally:
            if curr_buf is not None:
                curr_buf.release()
            if curr_pid != INVALID_PAGE_ID:
                self._bpm.unpin_page(curr_pid, is_dirty=curr_dirty)

        if deleted_total == 0:
            return False
        self._entry_count -= deleted_total
        return True

    def delete(self, key: Any, rid: RID | None = None) -> bool:
        """Remove indexed entry or entries matching ``key`` (and optional ``rid``).

        Args:
            key: Logical key matching the configured ``DataType``.
            rid: When provided, delete only ``(key, rid)``; when ``None``, delete
                all entries matching ``key`` across one or more sibling leaves.

        Returns:
            ``True`` if at least one matching entry was deleted, else ``False``.

        Raises:
            TypeError: If ``key`` or ``rid`` has an invalid Python type.
            ValueError: If ``key`` overflows ``key_size`` or is ``NaN``.
            StorageError: If the index is closed or disk/pool operations fail.
            PageCorruptedError: If an accessed page fails checksum validation.

        Complexity:
            $O(\\log_B N + d)$ for $d$ deleted entries; holds at most 1 pinned
            frame at any instant. Thread-safe under ``self._lock``.
        """
        if rid is not None and not isinstance(rid, RID):
            raise TypeError(f"rid must be a RID or None, got {type(rid).__name__}")
        key_bytes = encode_key(key, self._key_dtype, self._key_size)

        with self._lock:
            self._ensure_open()
            if self._root_page_id == INVALID_PAGE_ID:
                return False

            prev_root = self._root_page_id
            prev_height = self._height

            deleted = (
                self._delete_single_rid(key_bytes, rid)
                if rid is not None
                else self._delete_all_for_key(key_bytes)
            )
            if not deleted:
                return False

            root_changed = (
                self._root_page_id != prev_root or self._height != prev_height
            )
            self._persist_metadata(flush_to_disk=root_changed)
            return True

    def validate_invariants(self) -> bool:
        """Verify structural, ordering, depth, bound, and leaf-chain B+ Tree invariants.

        Checks:
            1. Page 0 metadata matches in-memory root, schema, height, and entry count.
            2. Every leaf sits at identical depth ``height``.
            3. Keys inside every internal and leaf node are strictly sorted.
            4. Internal separators bound subtrees (``max(child_i) < sep_i <= min(child_{i+1})``).
            5. Left-to-right leaf ``next_page_id`` linkage matches tree order and
               total entries equal ``entry_count``.

        Returns:
            ``True`` when all invariants hold; ``False`` if any structural violation
            is detected.

        Raises:
            StorageError: If the index is closed.

        Complexity:
            $O(N)$ across all nodes; holds at most 1 pinned frame at a time.
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()

            meta_buf = self._bpm.fetch_page(0)
            try:
                root_pid, dtype_val, ksize, height, entry_count, uniq = (
                    _META_STRUCT.unpack_from(meta_buf, _META_OFFSET)
                )
            finally:
                meta_buf.release()
                self._bpm.unpin_page(0, is_dirty=False)

            if (
                root_pid != self._root_page_id
                or dtype_val != int(self._key_dtype)
                or ksize != self._key_size
                or height != self._height
                or entry_count != self._entry_count
                or bool(uniq) != self._unique
            ):
                return False

            if self._root_page_id == INVALID_PAGE_ID:
                return self._height == 0 and self._entry_count == 0

            if self._height < 1:
                return False

            visited: set[int] = set()
            leaf_audits: list[_LeafAudit] = []
            bounds = self._validate_subtree(
                self._root_page_id,
                depth=1,
                lower_bound=None,
                upper_bound=None,
                visited=visited,
                leaf_audits=leaf_audits,
            )
            if bounds is None or not leaf_audits:
                return False

            total_entries = 0
            for idx, audit in enumerate(leaf_audits):
                total_entries += audit.slot_count
                if idx + 1 < len(leaf_audits):
                    nxt = leaf_audits[idx + 1]
                    if audit.next_page_id != nxt.page_id:
                        return False
                    if not (audit.last_comp < nxt.first_comp):
                        return False
                    if self._unique and not (
                        audit.last_comp[: self._key_size]
                        < nxt.first_comp[: self._key_size]
                    ):
                        return False
                else:
                    if audit.next_page_id != INVALID_PAGE_ID:
                        return False

            return total_entries == self._entry_count

    def _validate_subtree(
        self,
        page_id: int,
        *,
        depth: int,
        lower_bound: bytes | None,
        upper_bound: bytes | None,
        visited: set[int],
        leaf_audits: list[_LeafAudit],
    ) -> tuple[bytes, bytes] | None:
        """Validate one subtree without holding pins across recursive calls."""
        if (
            page_id == 0
            or page_id >= self._disk_manager.num_pages()
            or not self._disk_manager.is_allocated(page_id)
            or page_id in visited
        ):
            return None
        visited.add(page_id)

        buf = self._bpm.fetch_page(page_id)
        eks = self._effective_key_size
        ksize = self._key_size
        try:
            header = PageHeader.unpack(buf, 0)
            if header.page_id != page_id:
                return None

            expected_root = depth == 1
            if bool(header.flags & PageFlags.IS_ROOT) != expected_root:
                return None

            if header.page_type == PageType.BPLUS_LEAF:
                if depth != self._height or not (header.flags & PageFlags.IS_LEAF):
                    return None
                n = header.slot_count
                if n < 1 or n > self._m_leaf:
                    return None
                stride = self._leaf_stride
                if header.free_space_ptr != _LEAF_ENTRIES_OFFSET + n * stride:
                    return None

                prev_comp: bytes | None = None
                first_comp: bytes | None = None
                for i in range(n):
                    off = _LEAF_ENTRIES_OFFSET + i * stride
                    comp = bytes(buf[off : off + eks])
                    comp_pid, comp_slot = _RID_BE_STRUCT.unpack_from(buf, off + ksize)
                    val_pid, val_slot = _RID_LE_STRUCT.unpack_from(buf, off + eks)
                    if comp_pid != val_pid or comp_slot != val_slot:
                        return None
                    if prev_comp is not None:
                        if not (prev_comp < comp):
                            return None
                        if self._unique and not (prev_comp[:ksize] < comp[:ksize]):
                            return None
                    else:
                        first_comp = comp
                    prev_comp = comp

                assert first_comp is not None and prev_comp is not None
                if lower_bound is not None and first_comp < lower_bound:
                    return None
                if upper_bound is not None and prev_comp >= upper_bound:
                    return None

                leaf_audits.append(
                    _LeafAudit(
                        page_id=page_id,
                        next_page_id=header.next_page_id,
                        slot_count=n,
                        first_comp=first_comp,
                        last_comp=prev_comp,
                    )
                )
                return first_comp, prev_comp

            if header.page_type != PageType.BPLUS_INTERNAL:
                return None
            if depth >= self._height or (header.flags & PageFlags.IS_LEAF):
                return None
            n = header.slot_count
            if n < 1 or n > self._m_int:
                return None
            stride = self._int_stride
            if header.free_space_ptr != _INTERNAL_ENTRIES_OFFSET + n * stride:
                return None

            children: list[int] = [
                _U32_STRUCT.unpack_from(buf, _INTERNAL_CHILDREN_OFFSET)[0]
            ]
            seps: list[bytes] = []
            prev_sep: bytes | None = None
            for i in range(n):
                off = _INTERNAL_ENTRIES_OFFSET + i * stride
                sep = bytes(buf[off : off + eks])
                if prev_sep is not None and not (prev_sep < sep):
                    return None
                prev_sep = sep
                seps.append(sep)
                children.append(_U32_STRUCT.unpack_from(buf, off + eks)[0])
        finally:
            buf.release()
            self._bpm.unpin_page(page_id, is_dirty=False)

        subtree_min: bytes | None = None
        subtree_max: bytes | None = None
        for idx, child_pid in enumerate(children):
            c_low = seps[idx - 1] if idx > 0 else lower_bound
            c_high = seps[idx] if idx < len(seps) else upper_bound
            res = self._validate_subtree(
                child_pid,
                depth=depth + 1,
                lower_bound=c_low,
                upper_bound=c_high,
                visited=visited,
                leaf_audits=leaf_audits,
            )
            if res is None:
                return None
            c_min, c_max = res
            if idx > 0 and c_min < seps[idx - 1]:
                return None
            if idx < len(seps) and c_max >= seps[idx]:
                return None
            if subtree_min is None:
                subtree_min = c_min
            subtree_max = c_max

        if subtree_min is None or subtree_max is None:
            return None
        return subtree_min, subtree_max

    def flush(self) -> None:
        """Flush Page 0 metadata and all dirty node pages durably via ``os.fsync``.

        Raises:
            StorageError: If the index is closed or disk flushing/syncing fails.

        Complexity:
            ``O(pool_size * PAGE_SIZE)`` worst case plus one ``os.fsync``.
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            self._persist_metadata(flush_to_disk=False)
            self._bpm.flush_all_pages()

    def close(self) -> None:
        """Flush all dirty buffers and close the underlying ``DiskManager`` idempotently.

        Raises:
            StorageError: If flushing or syncing fails prior to closing.

        Complexity:
            ``O(pool_size * PAGE_SIZE)`` worst case plus ``os.fsync``.
            Thread-safe under ``self._lock``.
        """
        with self._lock:
            if self._closed:
                return
            try:
                self._persist_metadata(flush_to_disk=False)
                self._bpm.flush_all_pages()
            finally:
                self._disk_manager.close()
                self._closed = True

    def get_stats(self) -> dict[str, Any]:
        """Return B+ Tree structural and buffer pool I/O telemetry counters.

        Returns:
            Dictionary containing ``index_type``, ``file_path``, ``root_page_id``,
            ``height``, ``entry_count``, ``key_dtype``, ``key_size``,
            ``effective_key_size``, ``unique``, ``m_internal``, ``m_leaf``,
            ``allocated_pages``, ``disk_reads``, ``disk_writes``, and
            ``buffer_pool`` stats.

        Raises:
            StorageError: If the index is closed.

        Complexity:
            ``O(FSM_BYTES + pool_size)`` under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            bpm_stats = self._bpm.get_stats()
            return {
                "index_type": "BPLUS_TREE",
                "file_path": str(self.file_path),
                "root_page_id": self._root_page_id,
                "height": self._height,
                "entry_count": self._entry_count,
                "key_dtype": self._key_dtype.name,
                "key_size": self._key_size,
                "effective_key_size": self._effective_key_size,
                "unique": self._unique,
                "m_internal": self._m_int,
                "m_leaf": self._m_leaf,
                "allocated_pages": len(self._disk_manager.allocated_pages()),
                "disk_reads": bpm_stats["disk_reads"],
                "disk_writes": bpm_stats["disk_writes"],
                "buffer_pool": bpm_stats,
            }

    def __enter__(self) -> BPlusTreeIndex:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


__all__ = ["BPlusTreeIndex"]
