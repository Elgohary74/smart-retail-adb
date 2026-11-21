"""Persistent page-backed B+ Tree index connected to StorageManager and BufferPool."""
from __future__ import annotations

import tempfile
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Callable

from ...core import BPLUS_INLINE_RID_CAP, RID, DataType, decode_key, encode_key
from ...storage import BufferPool, StorageManager
from ...storage.pages import HEADER_SIZE, PageType, read_header, write_header
from ..base import key_in_range
from .buckets import bucket_append, bucket_read, bucket_remove
from .pages import (
    LeafEntry,
    read_internal,
    read_leaf,
    read_meta,
    serialize_internal,
    serialize_leaf,
    write_internal,
    write_leaf,
    write_meta,
)


class BPlusTree:
    """Page-backed B+ Tree supporting equality, range, delete, and duplicate RID buckets."""

    def __init__(
        self,
        order: int | None = None,
        storage_manager: StorageManager | None = None,
        pool: BufferPool | None = None,
        name: str = "bplus",
        key_dtype: DataType = DataType.ANY,
    ):
        if order is not None and order < 3:
            raise ValueError("order must be at least 3")
        self.key_dtype = key_dtype
        self._max_keys = (order - 1) if order is not None else 0
        self._owns_storage = storage_manager is None or pool is None
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None

        if self._owns_storage:
            self._tmp_dir = tempfile.TemporaryDirectory()
            storage_manager = StorageManager(Path(self._tmp_dir.name))
            pool = BufferPool(capacity=32, storage_manager=storage_manager)
        if storage_manager is None or pool is None:
            raise RuntimeError("BPlusTree requires both storage_manager and pool")
        self.storage_manager, self.pool, self.name = storage_manager, pool, name
        self._logical_name = f"index:{name}:bplus"
        self._file_id, self._backend = storage_manager.open_file(
            self._logical_name, f"indexes/{name.replace(':', '_')}.bplus.pages"
        )
        self._usable = self._backend.usable_page_size - HEADER_SIZE
        self._init_or_load()

    def _alloc_page(self, writer: Callable[[bytearray], None]) -> int:
        pid, frame = self.pool.new_page(file_id=self._file_id)
        try:
            writer(frame.data)
            return pid
        finally:
            self.pool.unpin_page(pid, is_dirty=True, file_id=self._file_id)

    def _init_or_load(self) -> None:
        if self._backend.page_count > 0:
            with self.pool.pin_page(0, file_id=self._file_id) as frame:
                self.root_page_id, self.first_leaf_page_id, self.size, max_k = read_meta(frame.data)
                if self._max_keys == 0:
                    self._max_keys = max_k
            return
        meta_pid = self._alloc_page(lambda buf: write_meta(buf, 1, 1, 0, self._max_keys))
        if meta_pid != 0:
            raise IOError(f"Expected B+ Tree meta page at 0, got {meta_pid}")
        root_pid = self._alloc_page(lambda buf: write_leaf(buf, [], next_page=-1, prev_page=-1))
        self.root_page_id, self.first_leaf_page_id, self.size = root_pid, root_pid, 0
        self._save_meta()

    @property
    def page_count(self) -> int:
        """Number of physical/logical pages currently allocated to this tree."""
        return self._backend.page_count

    def disk_bytes(self) -> int:
        """Total physical bytes occupied by this B+ Tree file."""
        return self._backend.page_count * self._backend.physical_page_size

    def _save_meta(self) -> None:
        with self.pool.pin_page(0, file_id=self._file_id, dirty=True) as frame:
            write_meta(frame.data, self.root_page_id, self.first_leaf_page_id, self.size, self._max_keys)

    def _needs_split(self, payload_len: int, count: int) -> bool:
        return payload_len > self._usable or (self._max_keys > 0 and count > self._max_keys)

    def insert(self, key: Any, rid: RID) -> None:
        """Insert a (key, RID) entry into the B+ Tree."""
        split = self._insert_recursive(self.root_page_id, encode_key(self.key_dtype, key), rid)
        if split is not None:
            sep_key, right_pid = split
            self.root_page_id = self._alloc_page(lambda buf: write_internal(buf, [sep_key], [self.root_page_id, right_pid]))
            self._save_meta()

    def _insert_recursive(self, pid: int, key_bytes: bytes, rid: RID) -> tuple[bytes, int] | None:
        with self.pool.pin_page(pid, file_id=self._file_id) as frame:
            is_leaf = read_header(frame.data).page_type == int(PageType.BPLUS_LEAF)
            if not is_leaf:
                keys, children = read_internal(frame.data)
        if is_leaf:
            return self._insert_into_leaf(pid, key_bytes, rid)

        child_idx = bisect_right(keys, key_bytes)
        split = self._insert_recursive(children[child_idx], key_bytes, rid)
        if split is None:
            return None
        sep_key, right_pid = split
        keys.insert(child_idx, sep_key)
        children.insert(child_idx + 1, right_pid)
        if not self._needs_split(len(serialize_internal(keys, children)), len(keys)):
            with self.pool.pin_page(pid, file_id=self._file_id, dirty=True) as frame:
                write_internal(frame.data, keys, children)
            return None
        if len(keys) < 2:
            raise ValueError("Separator key exceeds B+ Tree internal page capacity")
        mid = len(keys) // 2
        with self.pool.pin_page(pid, file_id=self._file_id, dirty=True) as frame:
            write_internal(frame.data, keys[:mid], children[:mid + 1])
        new_pid = self._alloc_page(lambda buf: write_internal(buf, keys[mid + 1:], children[mid + 1:]))
        return keys[mid], new_pid

    def _insert_into_leaf(self, pid: int, key_bytes: bytes, rid: RID) -> tuple[bytes, int] | None:
        with self.pool.pin_page(pid, file_id=self._file_id) as frame:
            entries, hdr = read_leaf(frame.data)
        pos = bisect_left([e[0] for e in entries], key_bytes)
        if pos < len(entries) and entries[pos][0] == key_bytes:
            k, rids, bpid = entries[pos]
            if bpid == -1 and len(rids) < BPLUS_INLINE_RID_CAP:
                rids.append(rid)
            else:
                entries[pos] = (k, rids, bucket_append(self.pool, self._file_id, self._usable, bpid, rid))
        else:
            entries.insert(pos, (key_bytes, [rid], -1))
            self.size += 1
            self._save_meta()
        if not self._needs_split(len(serialize_leaf(entries)), len(entries)):
            with self.pool.pin_page(pid, file_id=self._file_id, dirty=True) as frame:
                write_leaf(frame.data, entries, hdr.next_page, hdr.prev_page)
            return None
        return self._split_leaf(pid, entries, hdr.next_page, hdr.prev_page)

    def _split_leaf(
        self, pid: int, entries: list[LeafEntry], old_next: int, prev_page: int
    ) -> tuple[bytes, int]:
        if len(entries) < 2:
            raise ValueError("Entry exceeds B+ Tree leaf page capacity")
        mid = len(entries) // 2
        left_e, right_e = entries[:mid], entries[mid:]
        right_pid = self._alloc_page(lambda buf: write_leaf(buf, right_e, next_page=old_next, prev_page=pid))
        with self.pool.pin_page(pid, file_id=self._file_id, dirty=True) as frame:
            write_leaf(frame.data, left_e, next_page=right_pid, prev_page=prev_page)
        if old_next != -1:
            with self.pool.pin_page(old_next, file_id=self._file_id, dirty=True) as nxt:
                nhdr = read_header(nxt.data)
                nhdr.prev_page = right_pid
                write_header(nxt.data, nhdr)
        return right_e[0][0], right_pid

    def _find_leaf(self, key_bytes: bytes) -> int:
        pid = self.root_page_id
        while True:
            with self.pool.pin_page(pid, file_id=self._file_id) as frame:
                if read_header(frame.data).page_type == int(PageType.BPLUS_LEAF):
                    return pid
                keys, children = read_internal(frame.data)
            pid = children[bisect_right(keys, key_bytes)]

    def _all_rids(self, inline: list[RID], bucket_pid: int) -> list[RID]:
        return list(inline) if bucket_pid == -1 else list(inline) + bucket_read(self.pool, self._file_id, bucket_pid)

    def search(self, key: Any) -> list[RID]:
        """Return all RIDs indexed under key."""
        kb = encode_key(self.key_dtype, key)
        with self.pool.pin_page(self._find_leaf(kb), file_id=self._file_id) as frame:
            entries, _ = read_leaf(frame.data)
        pos = bisect_left([e[0] for e in entries], kb)
        if pos < len(entries) and entries[pos][0] == kb:
            return self._all_rids(entries[pos][1], entries[pos][2])
        return []

    def delete(self, key: Any, rid: RID | None = None) -> int:
        """Remove a matching RID (or all RIDs if rid is None) for key; returns deleted count."""
        kb = encode_key(self.key_dtype, key)
        leaf_pid = self._find_leaf(kb)
        with self.pool.pin_page(leaf_pid, file_id=self._file_id) as frame:
            entries, hdr = read_leaf(frame.data)
        pos = bisect_left([e[0] for e in entries], kb)
        if pos >= len(entries) or entries[pos][0] != kb:
            return 0
        k, rids, bpid = entries[pos]
        new_bpid, b_rem = bucket_remove(self.pool, self._file_id, bpid, rid)
        if rid is None:
            removed = len(rids) + b_rem
            kept: list[RID] = []
        else:
            kept = [r for r in rids if r != rid]
            removed = (len(rids) - len(kept)) + b_rem
        if not kept and new_bpid == -1:
            entries.pop(pos)
            self.size -= 1
            self._save_meta()
        else:
            entries[pos] = (k, kept, new_bpid)
        with self.pool.pin_page(leaf_pid, file_id=self._file_id, dirty=True) as frame:
            write_leaf(frame.data, entries, hdr.next_page, hdr.prev_page)
        return removed

    def range_search(
        self, low: Any | None = None, high: Any | None = None,
        include_low: bool = True, include_high: bool = True,
    ) -> list[tuple[Any, RID]]:
        """Return ordered (key, RID) pairs within optional lower/upper bounds."""
        low_b = encode_key(self.key_dtype, low) if low is not None else None
        high_b = encode_key(self.key_dtype, high) if high is not None else None
        leaf_pid = self.first_leaf_page_id if low_b is None else self._find_leaf(low_b)
        found: list[tuple[Any, RID]] = []
        while leaf_pid != -1:
            with self.pool.pin_page(leaf_pid, file_id=self._file_id) as frame:
                entries, hdr = read_leaf(frame.data)
                leaf_pid = hdr.next_page
            keys = [e[0] for e in entries]
            start = 0 if low_b is None else (bisect_left(keys, low_b) if include_low else bisect_right(keys, low_b))
            for kb, inline, bpid in entries[start:]:
                if high_b is not None and (kb > high_b or (not include_high and kb == high_b)):
                    return found
                if key_in_range(kb, low_b, high_b, include_low, include_high):
                    decoded = decode_key(self.key_dtype, kb)
                    found.extend((decoded, r) for r in self._all_rids(inline, bpid))
        return found

    def flush(self) -> None:
        """Flush all dirty B+ Tree pages in BufferPool to backend storage."""
        self.pool.flush_file(self._file_id)

    def destroy(self) -> None:
        """Unregister B+ Tree pages from the buffer pool and delete backing storage."""
        self.pool.unregister_file(self._file_id, drop_dirty=True)
        self.storage_manager.drop_file(self._logical_name)

    def close(self) -> None:
        """Flush dirty pages and release owned temporary storage if standalone."""
        self.flush()
        if self._owns_storage:
            self.pool.close()
            self.storage_manager.close_all()
            if self._tmp_dir is not None:
                self._tmp_dir.cleanup()
                self._tmp_dir = None

    def __len__(self) -> int:
        return self.size

    def __enter__(self) -> "BPlusTree":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
