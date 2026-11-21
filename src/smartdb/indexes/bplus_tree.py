"""Persistent page-backed B+ Tree index connected to StorageManager and BufferPool."""
from __future__ import annotations

import struct
import tempfile
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

from ..schema import RID, DataType, decode_key, encode_key
from ..storage.buffer_pool import BufferPool
from ..storage.manager import StorageManager
from ..storage.slotted_page import HEADER_SIZE, PageHeader, PageType, read_header, write_header

_META_STRUCT = "<iiii"
_INLINE_RID_CAP = 8
_RID_STRUCT = "<II"
_RID_SIZE = 8


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

        assert storage_manager is not None and pool is not None
        self.storage_manager = storage_manager
        self.pool = pool
        self.name = name
        safe_name = name.replace(":", "_")
        self._file_id, self._backend = storage_manager.open_file(
            f"index:{name}:bplus", f"indexes/{safe_name}.bplus.pages"
        )
        self._usable = self._backend.usable_page_size - HEADER_SIZE

        if self._backend.page_count == 0:
            meta_pid, meta_frame = self.pool.new_page(file_id=self._file_id)
            assert meta_pid == 0
            self.pool.unpin_page(0, is_dirty=True, file_id=self._file_id)

            root_pid, root_frame = self.pool.new_page(file_id=self._file_id)
            self._write_leaf(root_frame.data, [], next_page=-1, prev_page=-1)
            self.pool.unpin_page(root_pid, is_dirty=True, file_id=self._file_id)

            self.root_page_id = root_pid
            self.first_leaf_page_id = root_pid
            self.size = 0
            self._save_meta()
        else:
            self._load_meta()

    @property
    def page_count(self) -> int:
        return self._backend.page_count

    def disk_bytes(self) -> int:
        return self._backend.page_count * self._backend.physical_page_size

    def _load_meta(self) -> None:
        with self.pool.pin_page(0, file_id=self._file_id) as frame:
            root_pid, first_leaf, size, max_keys = struct.unpack_from(_META_STRUCT, frame.data, HEADER_SIZE)
            self.root_page_id = root_pid
            self.first_leaf_page_id = first_leaf
            self.size = size
            if self._max_keys == 0:
                self._max_keys = max_keys

    def _save_meta(self) -> None:
        with self.pool.pin_page(0, file_id=self._file_id, dirty=True) as frame:
            write_header(frame.data, PageHeader(int(PageType.BPLUS_META), 0, 1, HEADER_SIZE + 16, len(frame.data)))
            struct.pack_into(_META_STRUCT, frame.data, HEADER_SIZE, self.root_page_id, self.first_leaf_page_id, self.size, self._max_keys)

    def _encode(self, key: Any) -> bytes:
        return encode_key(self.key_dtype, key)

    def _decode(self, raw: bytes) -> Any:
        return decode_key(self.key_dtype, raw)

    # --- Node serialization ---

    @staticmethod
    def _serialize_internal(keys: list[bytes], children: list[int]) -> bytes:
        parts = [struct.pack("<I", children[0])]
        for k, child in zip(keys, children[1:]):
            parts.append(struct.pack("<H", len(k)) + k + struct.pack("<I", child))
        return b"".join(parts)

    def _read_internal(self, buf: bytearray) -> tuple[list[bytes], list[int]]:
        hdr = read_header(buf)
        pos = HEADER_SIZE
        children = [struct.unpack_from("<I", buf, pos)[0]]
        pos += 4
        keys: list[bytes] = []
        for _ in range(hdr.item_count):
            klen = struct.unpack_from("<H", buf, pos)[0]
            pos += 2
            keys.append(bytes(buf[pos:pos + klen]))
            pos += klen
            children.append(struct.unpack_from("<I", buf, pos)[0])
            pos += 4
        return keys, children

    def _write_internal(self, buf: bytearray, keys: list[bytes], children: list[int]) -> None:
        payload = self._serialize_internal(keys, children)
        write_header(buf, PageHeader(int(PageType.BPLUS_INTERNAL), 0, len(keys), HEADER_SIZE + len(payload), len(buf)))
        buf[HEADER_SIZE:HEADER_SIZE + len(payload)] = payload

    @staticmethod
    def _serialize_leaf(entries: list[tuple[bytes, list[RID], int]]) -> bytes:
        parts: list[bytes] = []
        for key_bytes, rids, bucket_pid in entries:
            parts.append(struct.pack("<HHi", len(key_bytes), len(rids), bucket_pid))
            parts.append(key_bytes)
            for pid, sid in rids:
                parts.append(struct.pack(_RID_STRUCT, pid, sid))
        return b"".join(parts)

    def _read_leaf(self, buf: bytearray) -> tuple[list[tuple[bytes, list[RID], int]], PageHeader]:
        hdr = read_header(buf)
        pos = HEADER_SIZE
        entries: list[tuple[bytes, list[RID], int]] = []
        for _ in range(hdr.item_count):
            klen, rcount, bucket_pid = struct.unpack_from("<HHi", buf, pos)
            pos += 8
            key_bytes = bytes(buf[pos:pos + klen])
            pos += klen
            rids: list[RID] = []
            for _ in range(rcount):
                rids.append(struct.unpack_from(_RID_STRUCT, buf, pos))
                pos += _RID_SIZE
            entries.append((key_bytes, rids, bucket_pid))
        return entries, hdr

    def _write_leaf(
        self,
        buf: bytearray,
        entries: list[tuple[bytes, list[RID], int]],
        next_page: int,
        prev_page: int,
    ) -> None:
        payload = self._serialize_leaf(entries)
        write_header(
            buf,
            PageHeader(
                int(PageType.BPLUS_LEAF),
                0,
                len(entries),
                HEADER_SIZE + len(payload),
                len(buf),
                next_page=next_page,
                prev_page=prev_page,
            ),
        )
        buf[HEADER_SIZE:HEADER_SIZE + len(payload)] = payload

    # --- Duplicate RID bucket pages ---

    def _bucket_append(self, first_bucket_pid: int, rid: RID) -> int:
        max_rids = self._usable // _RID_SIZE
        if first_bucket_pid == -1:
            pid, frame = self.pool.new_page(file_id=self._file_id)
            write_header(frame.data, PageHeader(int(PageType.OVERFLOW), 0, 1, HEADER_SIZE + _RID_SIZE, len(frame.data)))
            struct.pack_into(_RID_STRUCT, frame.data, HEADER_SIZE, rid[0], rid[1])
            self.pool.unpin_page(pid, is_dirty=True, file_id=self._file_id)
            return pid
        cur = first_bucket_pid
        while True:
            with self.pool.pin_page(cur, file_id=self._file_id, dirty=True) as frame:
                hdr = read_header(frame.data)
                if hdr.next_page == -1 and hdr.item_count < max_rids:
                    struct.pack_into(_RID_STRUCT, frame.data, HEADER_SIZE + hdr.item_count * _RID_SIZE, rid[0], rid[1])
                    hdr.item_count += 1
                    write_header(frame.data, hdr)
                    return first_bucket_pid
                if hdr.next_page == -1:
                    npid, nframe = self.pool.new_page(file_id=self._file_id)
                    write_header(nframe.data, PageHeader(int(PageType.OVERFLOW), 0, 1, HEADER_SIZE + _RID_SIZE, len(nframe.data)))
                    struct.pack_into(_RID_STRUCT, nframe.data, HEADER_SIZE, rid[0], rid[1])
                    self.pool.unpin_page(npid, is_dirty=True, file_id=self._file_id)
                    hdr.next_page = npid
                    write_header(frame.data, hdr)
                    return first_bucket_pid
                cur = hdr.next_page

    def _bucket_read(self, first_bucket_pid: int) -> list[RID]:
        out: list[RID] = []
        cur = first_bucket_pid
        while cur != -1:
            with self.pool.pin_page(cur, file_id=self._file_id) as frame:
                hdr = read_header(frame.data)
                for i in range(hdr.item_count):
                    out.append(struct.unpack_from(_RID_STRUCT, frame.data, HEADER_SIZE + i * _RID_SIZE))
                cur = hdr.next_page
        return out

    def _bucket_remove(self, first_bucket_pid: int, target: RID | None) -> tuple[int, int]:
        if first_bucket_pid == -1:
            return -1, 0
        if target is None:
            removed = 0
            cur = first_bucket_pid
            while cur != -1:
                with self.pool.pin_page(cur, file_id=self._file_id, dirty=True) as frame:
                    hdr = read_header(frame.data)
                    removed += hdr.item_count
                    nxt = hdr.next_page
                    hdr.page_type = int(PageType.FREE)
                    hdr.next_page = -1
                    write_header(frame.data, hdr)
                    cur = nxt
            return -1, removed
        cur = first_bucket_pid
        removed = 0
        while cur != -1:
            with self.pool.pin_page(cur, file_id=self._file_id, dirty=True) as frame:
                hdr = read_header(frame.data)
                kept = [
                    struct.unpack_from(_RID_STRUCT, frame.data, HEADER_SIZE + i * _RID_SIZE)
                    for i in range(hdr.item_count)
                ]
                new_kept = [r for r in kept if r != target]
                if len(new_kept) != len(kept):
                    removed += len(kept) - len(new_kept)
                    hdr.item_count = len(new_kept)
                    for i, r in enumerate(new_kept):
                        struct.pack_into(_RID_STRUCT, frame.data, HEADER_SIZE + i * _RID_SIZE, r[0], r[1])
                    write_header(frame.data, hdr)
                cur = hdr.next_page
        return first_bucket_pid, removed

    # --- Core B+ Tree operations ---

    def _needs_split(self, payload_len: int, count: int) -> bool:
        if payload_len > self._usable:
            return True
        return self._max_keys > 0 and count > self._max_keys

    def insert(self, key: Any, rid: RID) -> None:
        key_bytes = self._encode(key)
        split = self._insert_recursive(self.root_page_id, key_bytes, rid)
        if split is not None:
            sep_key, right_pid = split
            new_root_pid, new_root_frame = self.pool.new_page(file_id=self._file_id)
            self._write_internal(new_root_frame.data, [sep_key], [self.root_page_id, right_pid])
            self.pool.unpin_page(new_root_pid, is_dirty=True, file_id=self._file_id)
            self.root_page_id = new_root_pid
            self._save_meta()

    def _insert_recursive(self, pid: int, key_bytes: bytes, rid: RID) -> tuple[bytes, int] | None:
        with self.pool.pin_page(pid, file_id=self._file_id) as frame:
            hdr = read_header(frame.data)
            is_leaf = hdr.page_type == int(PageType.BPLUS_LEAF)

        if is_leaf:
            return self._insert_into_leaf(pid, key_bytes, rid)

        with self.pool.pin_page(pid, file_id=self._file_id) as frame:
            keys, children = self._read_internal(frame.data)
        child_idx = bisect_right(keys, key_bytes)
        split = self._insert_recursive(children[child_idx], key_bytes, rid)
        if split is None:
            return None

        sep_key, right_pid = split
        keys.insert(child_idx, sep_key)
        children.insert(child_idx + 1, right_pid)
        payload = self._serialize_internal(keys, children)
        if not self._needs_split(len(payload), len(keys)):
            with self.pool.pin_page(pid, file_id=self._file_id, dirty=True) as frame:
                self._write_internal(frame.data, keys, children)
            return None

        mid = len(keys) // 2
        promoted = keys[mid]
        left_keys, left_children = keys[:mid], children[:mid + 1]
        right_keys, right_children = keys[mid + 1:], children[mid + 1:]
        with self.pool.pin_page(pid, file_id=self._file_id, dirty=True) as frame:
            self._write_internal(frame.data, left_keys, left_children)
        new_pid, new_frame = self.pool.new_page(file_id=self._file_id)
        self._write_internal(new_frame.data, right_keys, right_children)
        self.pool.unpin_page(new_pid, is_dirty=True, file_id=self._file_id)
        return promoted, new_pid

    def _insert_into_leaf(self, pid: int, key_bytes: bytes, rid: RID) -> tuple[bytes, int] | None:
        with self.pool.pin_page(pid, file_id=self._file_id, dirty=True) as frame:
            entries, hdr = self._read_leaf(frame.data)
            keys = [e[0] for e in entries]
            pos = bisect_left(keys, key_bytes)
            if pos < len(entries) and entries[pos][0] == key_bytes:
                k, rids, bucket_pid = entries[pos]
                if bucket_pid == -1 and len(rids) < _INLINE_RID_CAP:
                    rids.append(rid)
                else:
                    bucket_pid = self._bucket_append(bucket_pid, rid)
                    entries[pos] = (k, rids, bucket_pid)
                # Adding one inline RID could push leaf over byte budget.
                payload = self._serialize_leaf(entries)
                if len(payload) <= self._usable:
                    self._write_leaf(frame.data, entries, hdr.next_page, hdr.prev_page)
                    return None
            else:
                entries.insert(pos, (key_bytes, [rid], -1))
                self.size += 1
                self._save_meta()

            payload = self._serialize_leaf(entries)
            if not self._needs_split(len(payload), len(entries)):
                self._write_leaf(frame.data, entries, hdr.next_page, hdr.prev_page)
                return None

            mid = len(entries) // 2
            left_entries, right_entries = entries[:mid], entries[mid:]
            right_pid, right_frame = self.pool.new_page(file_id=self._file_id)
            old_next = hdr.next_page
            self._write_leaf(right_frame.data, right_entries, next_page=old_next, prev_page=pid)
            self.pool.unpin_page(right_pid, is_dirty=True, file_id=self._file_id)
            self._write_leaf(frame.data, left_entries, next_page=right_pid, prev_page=hdr.prev_page)
            if old_next != -1:
                with self.pool.pin_page(old_next, file_id=self._file_id, dirty=True) as nxt_frame:
                    nxt_hdr = read_header(nxt_frame.data)
                    nxt_hdr.prev_page = right_pid
                    write_header(nxt_frame.data, nxt_hdr)
            return right_entries[0][0], right_pid

    def _find_leaf(self, key_bytes: bytes) -> int:
        pid = self.root_page_id
        while True:
            with self.pool.pin_page(pid, file_id=self._file_id) as frame:
                hdr = read_header(frame.data)
                if hdr.page_type == int(PageType.BPLUS_LEAF):
                    return pid
                keys, children = self._read_internal(frame.data)
            pid = children[bisect_right(keys, key_bytes)]

    def _all_rids(self, inline: list[RID], bucket_pid: int) -> list[RID]:
        if bucket_pid == -1:
            return list(inline)
        return list(inline) + self._bucket_read(bucket_pid)

    def search(self, key: Any) -> list[RID]:
        key_bytes = self._encode(key)
        leaf_pid = self._find_leaf(key_bytes)
        with self.pool.pin_page(leaf_pid, file_id=self._file_id) as frame:
            entries, _ = self._read_leaf(frame.data)
        keys = [e[0] for e in entries]
        pos = bisect_left(keys, key_bytes)
        if pos < len(entries) and entries[pos][0] == key_bytes:
            return self._all_rids(entries[pos][1], entries[pos][2])
        return []

    def delete(self, key: Any, rid: RID | None = None) -> int:
        key_bytes = self._encode(key)
        leaf_pid = self._find_leaf(key_bytes)
        with self.pool.pin_page(leaf_pid, file_id=self._file_id, dirty=True) as frame:
            entries, hdr = self._read_leaf(frame.data)
            keys = [e[0] for e in entries]
            pos = bisect_left(keys, key_bytes)
            if pos >= len(entries) or entries[pos][0] != key_bytes:
                return 0
            k, rids, bucket_pid = entries[pos]
            if rid is None:
                removed = len(rids)
                _, bucket_removed = self._bucket_remove(bucket_pid, None)
                removed += bucket_removed
                entries.pop(pos)
                self.size -= 1
                self._save_meta()
            else:
                new_inline = [r for r in rids if r != rid]
                removed = len(rids) - len(new_inline)
                bucket_pid, b_rem = self._bucket_remove(bucket_pid, rid)
                removed += b_rem
                if not new_inline and bucket_pid == -1:
                    entries.pop(pos)
                    self.size -= 1
                    self._save_meta()
                elif not new_inline and bucket_pid != -1:
                    # Check if bucket still has any RIDs
                    rem_rids = self._bucket_read(bucket_pid)
                    if not rem_rids:
                        self._bucket_remove(bucket_pid, None)
                        entries.pop(pos)
                        self.size -= 1
                        self._save_meta()
                    else:
                        entries[pos] = (k, [], bucket_pid)
                else:
                    entries[pos] = (k, new_inline, bucket_pid)
            self._write_leaf(frame.data, entries, hdr.next_page, hdr.prev_page)
            return removed

    def range_search(
        self,
        low: Any | None = None,
        high: Any | None = None,
        include_low: bool = True,
        include_high: bool = True,
    ) -> list[tuple[Any, RID]]:
        low_bytes = self._encode(low) if low is not None else None
        high_bytes = self._encode(high) if high is not None else None
        leaf_pid = self.first_leaf_page_id if low_bytes is None else self._find_leaf(low_bytes)

        found: list[tuple[Any, RID]] = []
        while leaf_pid != -1:
            with self.pool.pin_page(leaf_pid, file_id=self._file_id) as frame:
                entries, hdr = self._read_leaf(frame.data)
                next_pid = hdr.next_page
            start = 0
            if low_bytes is not None:
                keys = [e[0] for e in entries]
                start = bisect_left(keys, low_bytes) if include_low else bisect_right(keys, low_bytes)
            for key_bytes, inline, bucket_pid in entries[start:]:
                if high_bytes is not None:
                    if key_bytes > high_bytes or (not include_high and key_bytes == high_bytes):
                        return found
                decoded = self._decode(key_bytes)
                for r in self._all_rids(inline, bucket_pid):
                    found.append((decoded, r))
            leaf_pid = next_pid
        return found

    def flush(self) -> None:
        self.pool.flush_file(self._file_id)

    def close(self) -> None:
        self.flush()
        if self._owns_storage:
            self.pool.close()
            self.storage_manager.close_all()
            if self._tmp_dir is not None:
                self._tmp_dir.cleanup()
                self._tmp_dir = None

    def __len__(self) -> int:
        return self.size
