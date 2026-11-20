"""
Module: src/index/lsm_tree.py
Layer: Layer 2 — Write-Optimized LSM Tree Access Method
Lock Domain: IndexLock (``self._lock = threading.RLock()``)

Purpose:
    Implements a persistent, three-tier Log-Structured Merge Tree index
    (``LSMTreeIndex``) backed by a CRC32-protected Write-Ahead Log
    (``lsm.wal``), an in-memory sorted ``MemTable``, immutable 8,192-byte
    block-aligned ``SSTable`` files with per-table ``BloomFilter`` bitmaps
    and sparse fence indexes, atomic ``manifest.json`` updates, and
    size-tiered compaction.

Key Invariants:
    1. Every ``insert`` and ``delete`` appends a CRC32-framed WAL record to
       ``lsm.wal`` via unbuffered OS file descriptor writes before mutating
       ``MemTable``; startup recovery replays valid WAL records and truncates
       any torn or corrupt tail with ``os.ftruncate``.
    2. ``BloomFilter`` hashes exclusively the fixed-width user key
       (``encode_key(...)`` without any ``RID`` suffix) so point lookups for
       both unique and non-unique indexes never suffer false negatives.
    3. ``unique=True`` indexes key entries by ``user_key`` (upsert/overwrite
       semantics retaining the newest ``RID``), while ``unique=False`` indexes
       key entries by composite ``(user_key, >II(page_id, slot_id))``.
    4. Size-tiered compaction merges all ``SSTable`` files in level ``L_i``
       into ``L_{i+1}`` via ``heapq.merge`` in strict recency order, retaining
       only the newest version per entry key and dropping ``is_tombstone=1``
       records only when no older ``SSTable`` files exist in ``L_{i+1}`` or
       deeper levels.
    5. ``manifest.json`` and new ``SSTable`` files are written via
       ``<file>.tmp`` -> ``os.fsync`` -> ``os.replace``; ``lsm.wal`` is
       truncated only after ``manifest.json`` atomically commits the flushed
       ``SSTable``.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import struct
import threading
from typing import Any
import zlib

from src.common.config import (
    BLOOM_FPR,
    DEFAULT_MEMTABLE_BYTES,
    LSM_COMPACTION_TRIGGER,
    MAGIC_SSTABLE,
    PAGE_SIZE,
)
from src.common.exceptions import PageCorruptedError, RecordTooLargeError, StorageError
from src.common.types import (
    NULL_RID,
    RID,
    DataType,
    decode_key,
    encode_key,
    make_composite_key,
)
from src.index.base_index import BaseIndex

_HAS_PREAD: bool = hasattr(os, "pread")
_HAS_PWRITE: bool = hasattr(os, "pwrite")

_WAL_OP_PUT: int = 1
_WAL_OP_DELETE: int = 2
_WAL_HDR_STRUCT: struct.Struct = struct.Struct("<IBH")
_WAL_PAYLOAD_HDR_STRUCT: struct.Struct = struct.Struct("<BH")
_WAL_RID_STRUCT: struct.Struct = struct.Struct("<II")

_BLOCK_HDR_STRUCT: struct.Struct = struct.Struct("<HH")
_ENTRY_KEY_LEN_STRUCT: struct.Struct = struct.Struct("<H")
_ENTRY_VAL_STRUCT: struct.Struct = struct.Struct("<BII")
_FENCE_COUNT_STRUCT: struct.Struct = struct.Struct("<I")
_FENCE_OFF_STRUCT: struct.Struct = struct.Struct("<Q")

_FOOTER_PREFIX_STRUCT: struct.Struct = struct.Struct("<8sQIIQIQQ")
_FOOTER_TAIL_STRUCT: struct.Struct = struct.Struct("<I8s")
_FOOTER_STRUCT: struct.Struct = struct.Struct("<8sQIIQIQQI8s")

_BLOCK_HDR_SIZE: int = _BLOCK_HDR_STRUCT.size  # 4 bytes
_ENTRY_FIXED_OVERHEAD: int = _ENTRY_KEY_LEN_STRUCT.size + _ENTRY_VAL_STRUCT.size  # 11 bytes
_MAX_ENTRY_KEY_SIZE: int = PAGE_SIZE - _BLOCK_HDR_SIZE - _ENTRY_FIXED_OVERHEAD  # 8177 bytes
_FOOTER_SIZE: int = _FOOTER_STRUCT.size  # 64 bytes
_FOOTER_CRC_OFFSET: int = _FOOTER_PREFIX_STRUCT.size  # 52 bytes
_FOOTER_TAIL_SIZE: int = _FOOTER_TAIL_STRUCT.size  # 12 bytes
_ZERO_PADDING_8: bytes = b"\x00" * 8
_ZERO_RID_SUFFIX: bytes = b"\x00" * 8
_MAX_RID_SUFFIX: bytes = b"\xff" * 8
_UINT32_MAX: int = 0xFFFFFFFF
_CRC_CHUNK_SIZE: int = 65_536

_MEMTABLE_BUCKET_TARGET: int = 512
_MEMTABLE_BUCKET_SPLIT: int = 1024


def _pread_exact(fd: int, size: int, offset: int, io_lock: threading.Lock | None = None) -> bytes:
    """Read ``size`` bytes at ``offset`` from ``fd``, raising ``StorageError`` on short reads."""
    chunks: list[bytes] = []
    remaining = size
    pos = offset
    while remaining > 0:
        try:
            if _HAS_PREAD:
                chunk = os.pread(fd, remaining, pos)
            else:
                if io_lock is not None:
                    with io_lock:
                        os.lseek(fd, pos, os.SEEK_SET)
                        chunk = os.read(fd, remaining)
                else:
                    os.lseek(fd, pos, os.SEEK_SET)
                    chunk = os.read(fd, remaining)
        except OSError as exc:
            raise StorageError(f"Failed reading {size} bytes at offset {offset}: {exc}") from exc
        if not chunk:
            raise StorageError(
                f"Unexpected EOF reading {size} bytes at offset {offset} (remaining={remaining})"
            )
        chunks.append(chunk)
        read_len = len(chunk)
        remaining -= read_len
        pos += read_len
    return b"".join(chunks)


def _write_all_fd(fd: int, data: bytes | bytearray | memoryview) -> None:
    """Write the full buffer sequentially to ``fd`` handling partial OS writes."""
    mv = memoryview(data)
    written = 0
    total = len(mv)
    while written < total:
        try:
            n = os.write(fd, mv[written:])
        except OSError as exc:
            raise StorageError(
                f"Failed writing {total - written} bytes to fd {fd}: {exc}"
            ) from exc
        if n <= 0:
            raise StorageError(f"Short write on fd {fd}: wrote {written} of {total} bytes")
        written += n


def _pwrite_all_fd(fd: int, data: bytes | bytearray | memoryview, offset: int) -> None:
    """Write the full buffer at ``offset`` on ``fd`` handling partial OS writes."""
    mv = memoryview(data)
    written = 0
    total = len(mv)
    while written < total:
        try:
            if _HAS_PWRITE:
                n = os.pwrite(fd, mv[written:], offset + written)
            else:
                os.lseek(fd, offset + written, os.SEEK_SET)
                n = os.write(fd, mv[written:])
        except OSError as exc:
            raise StorageError(
                f"Failed positional write at offset {offset + written}: {exc}"
            ) from exc
        if n <= 0:
            raise StorageError(
                f"Short positional write at offset {offset}: wrote {written} of {total} bytes"
            )
        written += n


def _tag_stream(
    entries: Iterable[tuple[bytes, bool, RID]], priority: int
) -> Iterator[tuple[bytes, int, bool, RID]]:
    """Bind ``priority`` per iterator without loop-variable late-binding hazards."""
    for ekey, is_tomb, rid in entries:
        yield ekey, priority, is_tomb, rid


class BloomFilter:
    """Space-efficient SHA-256 double-hashing Bloom filter for SSTable point lookups.

    Binary Representation:
    ```text
    Raw packed bitset bytearray of length ceil(num_bits / 8):
      Bit i resides at byte (i >> 3), mask (1 << (i & 7)).
      Hash probes: bit_i = (h1 + i * h2) mod num_bits, for i in [0, num_hashes - 1],
      where (h1, h2) = struct.unpack("<QQ", hashlib.sha256(user_key_bytes).digest()[:16]).
    ```
    """

    def __init__(
        self,
        expected_items: int,
        fpr: float = BLOOM_FPR,
        *,
        _num_bits: int | None = None,
        _num_hashes: int | None = None,
        _bitmap: bytearray | None = None,
    ) -> None:
        """Initialize a Bloom filter sized for ``expected_items`` at target ``fpr``.

        Args:
            expected_items: Estimated distinct user keys to insert (``>= 0``).
            fpr: Target false-positive probability in ``(0, 1)``.

        Raises:
            ValueError: If ``expected_items < 0`` or ``fpr`` is outside ``(0, 1)``.

        Complexity:
            $O(m)$ bytes allocated where $m = \\lceil -n \\ln p / (\\ln 2)^2 \\rceil$.
        """
        if _bitmap is not None:
            if _num_bits is None or _num_hashes is None or _num_bits < 1 or _num_hashes < 1:
                raise ValueError("Internal BloomFilter reconstruction requires valid bits and hashes")
            self._num_bits: int = _num_bits
            self._num_hashes: int = _num_hashes
            self._bitmap: bytearray = _bitmap
            return

        if isinstance(expected_items, bool) or not isinstance(expected_items, int) or expected_items < 0:
            raise ValueError(f"expected_items must be a non-negative int, got {expected_items!r}")
        if isinstance(fpr, bool) or not isinstance(fpr, (int, float)) or not (0.0 < float(fpr) < 1.0):
            raise ValueError(f"fpr must be in (0, 1), got {fpr!r}")

        if expected_items == 0:
            # All-zero 8-bit bitmap guarantees 100% negative rejection for empty SSTables.
            self._num_bits = 8
            self._num_hashes = 1
            self._bitmap = bytearray(1)
            return

        ln2 = math.log(2.0)
        m = max(8, math.ceil(-(expected_items * math.log(float(fpr))) / (ln2 * ln2)))
        k = max(1, round((m / expected_items) * ln2))
        self._num_bits = m
        self._num_hashes = k
        self._bitmap = bytearray((m + 7) // 8)

    @property
    def num_bits(self) -> int:
        """Return total bit capacity ``m`` of the filter."""
        return self._num_bits

    @property
    def num_hashes(self) -> int:
        """Return number of probe functions ``k``."""
        return self._num_hashes

    @property
    def byte_length(self) -> int:
        """Return serialized byte length ``ceil(num_bits / 8)``."""
        return len(self._bitmap)

    def add(self, key: bytes) -> None:
        """Record a user key byte sequence into the Bloom filter bitset.

        Args:
            key: Fixed-width encoded user key bytes (without any ``RID`` suffix).

        Complexity:
            $O(k)$ bit sets with one SHA-256 hash evaluation.
        """
        digest = hashlib.sha256(key).digest()
        h1, h2 = struct.unpack_from("<QQ", digest, 0)
        m = self._num_bits
        bitmap = self._bitmap
        for i in range(self._num_hashes):
            bit_idx = (h1 + i * h2) % m
            bitmap[bit_idx >> 3] |= 1 << (bit_idx & 7)

    def might_contain(self, key: bytes) -> bool:
        """Test whether ``key`` may be present (never produces false negatives).

        Args:
            key: Fixed-width encoded user key bytes (without any ``RID`` suffix).

        Returns:
            ``False`` if ``key`` is definitely absent; ``True`` if possibly present.

        Complexity:
            $O(k)$ bit probes with zero disk I/O.
        """
        digest = hashlib.sha256(key).digest()
        h1, h2 = struct.unpack_from("<QQ", digest, 0)
        m = self._num_bits
        bitmap = self._bitmap
        for i in range(self._num_hashes):
            bit_idx = (h1 + i * h2) % m
            if not (bitmap[bit_idx >> 3] & (1 << (bit_idx & 7))):
                return False
        return True

    def __contains__(self, key: bytes) -> bool:
        return self.might_contain(key)

    def to_bytes(self) -> bytes:
        """Return an immutable ``bytes`` copy of the Bloom filter bitmap."""
        return bytes(self._bitmap)

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        num_bits: int,
        num_hashes: int,
    ) -> BloomFilter:
        """Reconstruct a ``BloomFilter`` from persisted bytes and metadata.

        Args:
            data: Raw Bloom filter bitmap bytes.
            num_bits: Bit count ``m`` recorded in the ``SSTable`` footer.
            num_hashes: Probe count ``k`` recorded in the ``SSTable`` footer.

        Returns:
            Reconstructed ``BloomFilter`` ready for ``might_contain`` checks.

        Raises:
            PageCorruptedError: If bit/hash counts or bitmap byte length mismatch.
        """
        if num_bits < 1 or num_hashes < 1:
            raise PageCorruptedError(
                f"Invalid BloomFilter parameters: num_bits={num_bits}, num_hashes={num_hashes}"
            )
        expected_len = (num_bits + 7) // 8
        raw = bytearray(data)
        if len(raw) != expected_len:
            raise PageCorruptedError(
                f"BloomFilter byte length mismatch: expected {expected_len}, got {len(raw)}"
            )
        return cls(
            0,
            _num_bits=num_bits,
            _num_hashes=num_hashes,
            _bitmap=raw,
        )


class MemTable:
    """In-memory sorted write buffer pairing O(1) point lookups with O(log M) range scans.

    Stores entry keys mapped to ``(is_tombstone, RID)`` in a dictionary for
    constant-time point reads and in-place overwrites, alongside chunked sorted
    key buckets so sequential flushes and range queries avoid global re-sorting.
    """

    def __init__(self) -> None:
        """Initialize an empty ``MemTable`` with zero tracked bytes."""
        self._entries: dict[bytes, tuple[bool, RID]] = {}
        self._buckets: list[list[bytes]] = []
        self._maxes: list[bytes] = []
        self._byte_size: int = 0

    @property
    def byte_size(self) -> int:
        """Return serialized entry footprint in bytes across active keys."""
        return self._byte_size

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, entry_key: bytes, rid: RID, is_tombstone: bool = False) -> None:
        """Insert or overwrite ``entry_key`` with ``(is_tombstone, rid)``.

        Args:
            entry_key: Canonical encoded key (or composite key for non-unique).
            rid: Associated ``RID`` (``NULL_RID`` for tombstones).
            is_tombstone: ``True`` for a deletion marker, ``False`` for live entry.

        Complexity:
            $O(1)$ on overwrite of an existing key; $O(\\log M)$ on new key insertion.
        """
        if entry_key in self._entries:
            self._entries[entry_key] = (is_tombstone, rid)
            return

        self._entries[entry_key] = (is_tombstone, rid)
        self._byte_size += _ENTRY_FIXED_OVERHEAD + len(entry_key)

        if not self._buckets:
            self._buckets.append([entry_key])
            self._maxes.append(entry_key)
            return

        b_idx = bisect_left(self._maxes, entry_key)
        if b_idx == len(self._buckets):
            b_idx -= 1
            bucket = self._buckets[b_idx]
            bucket.append(entry_key)
            self._maxes[b_idx] = entry_key
        else:
            bucket = self._buckets[b_idx]
            pos = bisect_left(bucket, entry_key)
            bucket.insert(pos, entry_key)

        if len(bucket) > _MEMTABLE_BUCKET_SPLIT:
            mid = _MEMTABLE_BUCKET_TARGET
            right = bucket[mid:]
            del bucket[mid:]
            self._maxes[b_idx] = bucket[-1]
            self._buckets.insert(b_idx + 1, right)
            self._maxes.insert(b_idx + 1, right[-1])

    def get(self, entry_key: bytes) -> tuple[bool, RID] | None:
        """Return ``(is_tombstone, rid)`` for ``entry_key``, or ``None`` if absent."""
        return self._entries.get(entry_key)

    def range_items(self, low_key: bytes, high_key: bytes) -> Iterator[tuple[bytes, bool, RID]]:
        """Yield ``(entry_key, is_tombstone, rid)`` in ascending key order over ``[low_key, high_key]``.

        Args:
            low_key: Inclusive lower ``entry_key`` bound.
            high_key: Inclusive upper ``entry_key`` bound.

        Yields:
            Sorted ``(entry_key, is_tombstone, rid)`` tuples within bounds.

        Complexity:
            $O(\\log M + k)$ where $k$ is the number of yielded entries.
        """
        if not self._buckets or low_key > high_key:
            return
        b_idx = bisect_left(self._maxes, low_key)
        if b_idx == len(self._buckets):
            return
        pos = bisect_left(self._buckets[b_idx], low_key)
        while b_idx < len(self._buckets):
            bucket = self._buckets[b_idx]
            for i in range(pos, len(bucket)):
                ekey = bucket[i]
                if ekey > high_key:
                    return
                is_tomb, rid = self._entries[ekey]
                yield ekey, is_tomb, rid
            b_idx += 1
            pos = 0

    def iter_sorted(self) -> Iterator[tuple[bytes, bool, RID]]:
        """Yield all entries in ascending ``entry_key`` order."""
        for bucket in self._buckets:
            for ekey in bucket:
                is_tomb, rid = self._entries[ekey]
                yield ekey, is_tomb, rid

    def clear(self) -> None:
        """Remove all entries and reset tracked byte size to zero."""
        self._entries.clear()
        self._buckets.clear()
        self._maxes.clear()
        self._byte_size = 0


class SSTable:
    """Immutable sorted-string table stored in 8 KB data blocks with Bloom filter and fence index.

    Binary File Layout:
    ```text
    0               8192            16384           bloom_offset     index_offset      file_size-64      file_size
    +---------------+---------------+---------------+----------------+-----------------+-----------------+
    | Data Block 0  | Data Block 1  | ... Block B-1 | Bloom Filter   | Sparse Fence    | 64-Byte Footer  |
    | (8,192 Bytes) | (8,192 Bytes) | (8,192 Bytes) | (bloom_len B)  | Index (idx_len) | (<8sQIIQIQQI8s) |
    +---------------+---------------+---------------+----------------+-----------------+-----------------+

    8,192-Byte Data Block:
    0        2           4                                                   used_bytes              8192
    +--------+-----------+---------------------------------------------------+-----------------------+
    | count  |used_bytes | [key_len:u16 | key_bytes | tomb:u8 | rid_p | rid_s]... | Zero Padding (\x00)   |
    | (<H)   | (<H)      | Entry 0, Entry 1, ..., Entry count-1              |                       |
    +--------+-----------+---------------------------------------------------+-----------------------+
      Each entry spans 11 + key_len bytes: <H + key_bytes + <BII.

    Sparse Fence Index Section (index_len Bytes):
    0        4                                                                               index_len
    +--------+-------------------------------------------------------------------------------+
    | num_   | [first_key_len: u16 | first_key_bytes | block_offset: u64] * num_blocks        |
    | blocks | (<H + bytes + <Q per 8,192-byte data block)                                    |
    +--------+-------------------------------------------------------------------------------+

    64-Byte EOF Footer (<8sQIIQIQQI8s):
    0      8            16         20           24           32         36          44             52     56       64
    +------+------------+----------+------------+------------+----------+-----------+--------------+------+--------+
    |magic |bloom_offset|bloom_len | num_hashes |index_offset|index_len |entry_count|bloom_num_bits|crc32 |padding |
    |"CSAI |   uint64   |  uint32  |   uint32   |   uint64   | uint32   |  uint64   |    uint64    |uint32| 8 x 00 |
    |_SST" |            |          |            |            |          |           |              |      |        |
    +------+------------+----------+------------+------------+----------+-----------+--------------+------+--------+
      CRC32 covers bytes [0 : file_size - 12] (Data Blocks + Bloom + Fence Index + Footer[0:52]).
    ```
    """

    def __init__(
        self,
        file_path: Path,
        fd: int,
        bloom: BloomFilter,
        fence_keys: list[bytes],
        block_offsets: list[int],
        entry_count: int,
        file_size: int,
        entry_key_size: int | None = None,
    ) -> None:
        self.file_path: Path = file_path
        self._fd: int | None = fd
        self._io_lock: threading.Lock = threading.Lock()
        self.bloom: BloomFilter = bloom
        self._fence_keys: list[bytes] = fence_keys
        self._block_offsets: list[int] = block_offsets
        self.entry_count: int = entry_count
        self.file_size: int = file_size
        self._entry_key_size: int | None = entry_key_size

    @property
    def num_blocks(self) -> int:
        """Return number of 8,192-byte data blocks in this ``SSTable``."""
        return len(self._block_offsets)

    @property
    def is_closed(self) -> bool:
        """Return ``True`` once the underlying read descriptor is closed."""
        return self._fd is None

    def _ensure_open(self) -> int:
        if self._fd is None:
            raise StorageError(f"SSTable {self.file_path} is closed")
        return self._fd

    @classmethod
    def build(
        cls,
        file_path: str | Path,
        entries: Iterable[tuple[bytes, bool, RID]],
        key_size: int,
        *,
        fpr: float = BLOOM_FPR,
        entry_key_size: int | None = None,
    ) -> SSTable:
        """Write a sorted entry stream atomically into an immutable ``SSTable`` file.

        Args:
            file_path: Destination ``*.sst`` file path.
            entries: Sorted iterable of ``(entry_key, is_tombstone, rid)`` with
                strictly ascending distinct ``entry_key`` values.
            key_size: Fixed byte width of the user key prefix hashed into ``BloomFilter``.
            fpr: Target Bloom filter false-positive rate.
            entry_key_size: Optional fixed ``entry_key`` width enabling $O(\\log B)$
                fixed-stride binary search inside data blocks.

        Returns:
            Opened ``SSTable`` handle backed by the newly written file.

        Raises:
            RecordTooLargeError: If an entry key exceeds single-block capacity.
            ValueError: If entries are unsorted or duplicate, or ``key_size <= 0``.
            StorageError: If writing or syncing the SSTable file fails.

        Complexity:
            $O(N)$ sequential write where $N$ is the number of entries.
        """
        if key_size <= 0:
            raise ValueError(f"key_size must be positive, got {key_size}")

        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f"{target.name}.tmp")
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)

        try:
            fd = os.open(os.fspath(tmp_path), open_flags, 0o644)
        except OSError as exc:
            raise StorageError(f"Failed to create SSTable {tmp_path}: {exc}") from exc

        running_crc = 0
        block_buf = bytearray(PAGE_SIZE)
        block_pos = _BLOCK_HDR_SIZE
        block_entry_count = 0
        block_offset = 0
        first_key_in_block: bytes | None = None

        fence_keys: list[bytes] = []
        block_offsets: list[int] = []
        distinct_user_keys: list[bytes] = []
        prev_entry_key: bytes | None = None
        prev_user_key: bytes | None = None
        total_entries = 0

        def _flush_block() -> None:
            nonlocal running_crc, block_pos, block_entry_count, block_offset, first_key_in_block
            if block_entry_count == 0 or first_key_in_block is None:
                return
            _BLOCK_HDR_STRUCT.pack_into(block_buf, 0, block_entry_count, block_pos)
            # Zero-pad trailing bytes cleanly so unused block slack stays deterministic.
            if block_pos < PAGE_SIZE:
                block_buf[block_pos:PAGE_SIZE] = bytes(PAGE_SIZE - block_pos)
            _write_all_fd(fd, block_buf)
            running_crc = zlib.crc32(block_buf, running_crc)
            fence_keys.append(first_key_in_block)
            block_offsets.append(block_offset)
            block_offset += PAGE_SIZE
            block_pos = _BLOCK_HDR_SIZE
            block_entry_count = 0
            first_key_in_block = None

        try:
            for entry_key, is_tombstone, rid in entries:
                klen = len(entry_key)
                if klen < key_size:
                    raise ValueError(
                        f"entry_key length {klen} is smaller than user key_size {key_size}"
                    )
                if klen > _MAX_ENTRY_KEY_SIZE:
                    raise RecordTooLargeError(
                        f"SSTable entry key length {klen} exceeds block limit {_MAX_ENTRY_KEY_SIZE}"
                    )
                if prev_entry_key is not None and not (prev_entry_key < entry_key):
                    raise ValueError("SSTable entries must be strictly sorted and deduplicated")
                prev_entry_key = entry_key

                user_key = entry_key[:key_size]
                if user_key != prev_user_key:
                    distinct_user_keys.append(user_key)
                    prev_user_key = user_key

                rec_size = _ENTRY_FIXED_OVERHEAD + klen
                if block_pos + rec_size > PAGE_SIZE:
                    _flush_block()

                if first_key_in_block is None:
                    first_key_in_block = entry_key

                _ENTRY_KEY_LEN_STRUCT.pack_into(block_buf, block_pos, klen)
                k_end = block_pos + 2 + klen
                block_buf[block_pos + 2 : k_end] = entry_key
                _ENTRY_VAL_STRUCT.pack_into(
                    block_buf,
                    k_end,
                    1 if is_tombstone else 0,
                    rid.page_id,
                    rid.slot_id,
                )
                block_pos += rec_size
                block_entry_count += 1
                total_entries += 1

            _flush_block()

            bloom = BloomFilter(len(distinct_user_keys), fpr=fpr)
            for ukey in distinct_user_keys:
                bloom.add(ukey)
            bloom_bytes = bloom.to_bytes()
            bloom_offset = block_offset
            bloom_len = len(bloom_bytes)
            _write_all_fd(fd, bloom_bytes)
            running_crc = zlib.crc32(bloom_bytes, running_crc)

            fence_chunks: list[bytes] = [_FENCE_COUNT_STRUCT.pack(len(fence_keys))]
            for fkey, boff in zip(fence_keys, block_offsets):
                fence_chunks.append(_ENTRY_KEY_LEN_STRUCT.pack(len(fkey)))
                fence_chunks.append(fkey)
                fence_chunks.append(_FENCE_OFF_STRUCT.pack(boff))
            index_bytes = b"".join(fence_chunks)
            index_offset = bloom_offset + bloom_len
            index_len = len(index_bytes)
            _write_all_fd(fd, index_bytes)
            running_crc = zlib.crc32(index_bytes, running_crc)

            footer_prefix = _FOOTER_PREFIX_STRUCT.pack(
                MAGIC_SSTABLE,
                bloom_offset,
                bloom_len,
                bloom.num_hashes,
                index_offset,
                index_len,
                total_entries,
                bloom.num_bits,
            )
            final_crc = zlib.crc32(footer_prefix, running_crc) & _UINT32_MAX
            footer_bytes = footer_prefix + _FOOTER_TAIL_STRUCT.pack(final_crc, _ZERO_PADDING_8)
            _write_all_fd(fd, footer_bytes)
            try:
                os.fsync(fd)
            except OSError as exc:
                raise StorageError(f"Failed syncing SSTable {tmp_path}: {exc}") from exc
        except Exception:
            os.close(fd)
            tmp_path.unlink(missing_ok=True)
            raise

        os.close(fd)
        try:
            os.replace(tmp_path, target)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise StorageError(f"Failed finalizing SSTable {target}: {exc}") from exc

        return cls.open(target, entry_key_size=entry_key_size)

    @classmethod
    def open(cls, file_path: str | Path, *, entry_key_size: int | None = None) -> SSTable:
        """Open and CRC-verify an existing ``SSTable`` file, loading Bloom & fence index into RAM.

        Args:
            file_path: Path to an existing ``*.sst`` file.
            entry_key_size: Optional fixed ``entry_key`` width for fast in-block search.

        Returns:
            Verified ``SSTable`` handle.

        Raises:
            StorageError: If the file cannot be opened or read.
            PageCorruptedError: If footer magic, section bounds, fence index, or
                whole-file CRC32 checksum validation fails.

        Complexity:
            $O(F)$ checksum verification on open where $F$ is file size, plus $O(B)$
            RAM for the Bloom filter and sparse block fence index.
        """
        target = Path(file_path)
        open_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(os.fspath(target), open_flags)
        except OSError as exc:
            raise StorageError(f"Failed to open SSTable {target}: {exc}") from exc

        try:
            file_size = os.fstat(fd).st_size
            if file_size < _FOOTER_SIZE:
                raise PageCorruptedError(
                    f"SSTable {target} is truncated ({file_size} < {_FOOTER_SIZE} bytes)"
                )

            footer_bytes = _pread_exact(fd, _FOOTER_SIZE, file_size - _FOOTER_SIZE)
            (
                magic,
                bloom_offset,
                bloom_len,
                num_hashes,
                index_offset,
                index_len,
                entry_count,
                bloom_num_bits,
                stored_crc,
                _padding,
            ) = _FOOTER_STRUCT.unpack(footer_bytes)

            if magic != MAGIC_SSTABLE:
                raise PageCorruptedError(
                    f"Invalid SSTable magic in {target}: expected {MAGIC_SSTABLE!r}, got {magic!r}"
                )
            if (
                bloom_offset % PAGE_SIZE != 0
                or index_offset != bloom_offset + bloom_len
                or index_offset + index_len != file_size - _FOOTER_SIZE
                or bloom_len != (bloom_num_bits + 7) // 8
            ):
                raise PageCorruptedError(
                    f"Corrupt SSTable section layout in {target}: "
                    f"bloom_off={bloom_offset}, bloom_len={bloom_len}, "
                    f"idx_off={index_offset}, idx_len={index_len}, size={file_size}"
                )

            # Verify whole-file CRC32 across Data Blocks, Bloom Filter, Fence Index, and Footer[0:52].
            crc = 0
            remaining = file_size - _FOOTER_TAIL_SIZE
            pos = 0
            while remaining > 0:
                chunk_len = min(_CRC_CHUNK_SIZE, remaining)
                chunk = _pread_exact(fd, chunk_len, pos)
                crc = zlib.crc32(chunk, crc)
                pos += chunk_len
                remaining -= chunk_len
            if (crc & _UINT32_MAX) != stored_crc:
                raise PageCorruptedError(
                    f"SSTable CRC32 mismatch in {target}: "
                    f"expected {stored_crc:#010x}, computed {(crc & _UINT32_MAX):#010x}"
                )

            bloom_raw = _pread_exact(fd, bloom_len, bloom_offset)
            bloom = BloomFilter.from_bytes(bloom_raw, bloom_num_bits, num_hashes)

            index_raw = _pread_exact(fd, index_len, index_offset)
            if len(index_raw) < 4:
                raise PageCorruptedError(f"Truncated fence index header in {target}")
            num_blocks = _FENCE_COUNT_STRUCT.unpack_from(index_raw, 0)[0]
            expected_blocks = bloom_offset // PAGE_SIZE
            if num_blocks != expected_blocks:
                raise PageCorruptedError(
                    f"Fence index block count mismatch in {target}: "
                    f"header={num_blocks}, expected={expected_blocks}"
                )

            fence_keys: list[bytes] = []
            block_offsets: list[int] = []
            idx_pos = 4
            for b_i in range(num_blocks):
                if idx_pos + 2 > index_len:
                    raise PageCorruptedError(f"Truncated fence entry key length in {target}")
                fkey_len = _ENTRY_KEY_LEN_STRUCT.unpack_from(index_raw, idx_pos)[0]
                idx_pos += 2
                if idx_pos + fkey_len + 8 > index_len:
                    raise PageCorruptedError(f"Truncated fence entry payload in {target}")
                fkey = index_raw[idx_pos : idx_pos + fkey_len]
                idx_pos += fkey_len
                boff = _FENCE_OFF_STRUCT.unpack_from(index_raw, idx_pos)[0]
                idx_pos += 8
                if boff != b_i * PAGE_SIZE:
                    raise PageCorruptedError(
                        f"Invalid fence block offset {boff} at block {b_i} in {target}"
                    )
                fence_keys.append(fkey)
                block_offsets.append(boff)

            if idx_pos != index_len:
                raise PageCorruptedError(
                    f"Trailing bytes in fence index of {target}: {idx_pos} != {index_len}"
                )

            return cls(
                file_path=target,
                fd=fd,
                bloom=bloom,
                fence_keys=fence_keys,
                block_offsets=block_offsets,
                entry_count=entry_count,
                file_size=file_size,
                entry_key_size=entry_key_size,
            )
        except Exception:
            os.close(fd)
            raise

    def _read_block(
        self,
        block_idx: int,
        on_block_read: Callable[[], None] | None = None,
    ) -> bytes:
        """Read and validate one 8,192-byte data block from disk."""
        fd = self._ensure_open()
        block = _pread_exact(fd, PAGE_SIZE, self._block_offsets[block_idx], self._io_lock)
        if on_block_read is not None:
            on_block_read()
        entry_count, used_bytes = _BLOCK_HDR_STRUCT.unpack_from(block, 0)
        if used_bytes < _BLOCK_HDR_SIZE or used_bytes > PAGE_SIZE:
            raise PageCorruptedError(
                f"Corrupt SSTable block header in {self.file_path} block {block_idx}: "
                f"entry_count={entry_count}, used_bytes={used_bytes}"
            )
        return block

    def _parse_block_entries(self, block: bytes) -> list[tuple[bytes, bool, RID]]:
        """Decode all sorted entries from an 8,192-byte data block."""
        entry_count, used_bytes = _BLOCK_HDR_STRUCT.unpack_from(block, 0)
        mv = memoryview(block)
        pos = _BLOCK_HDR_SIZE
        entries: list[tuple[bytes, bool, RID]] = []
        for _ in range(entry_count):
            if pos + _ENTRY_FIXED_OVERHEAD > used_bytes:
                raise PageCorruptedError(f"Truncated SSTable entry in {self.file_path}")
            klen = _ENTRY_KEY_LEN_STRUCT.unpack_from(mv, pos)[0]
            pos += 2
            if pos + klen + 9 > used_bytes:
                raise PageCorruptedError(f"Truncated SSTable entry body in {self.file_path}")
            ekey = bytes(mv[pos : pos + klen])
            pos += klen
            is_tomb, rid_page, rid_slot = _ENTRY_VAL_STRUCT.unpack_from(mv, pos)
            pos += 9
            entries.append((ekey, bool(is_tomb), RID(page_id=rid_page, slot_id=rid_slot)))
        if pos != used_bytes:
            raise PageCorruptedError(
                f"SSTable block payload mismatch in {self.file_path}: {pos} != {used_bytes}"
            )
        return entries

    def _bisect_fixed_block_left(
        self,
        block: bytes,
        entry_count: int,
        used_bytes: int,
        target_key: bytes,
    ) -> int | None:
        """Return first entry index in fixed-stride block with ``key >= target_key``, or ``None`` if variable-stride."""
        ek_size = self._entry_key_size
        if ek_size is None:
            return None
        stride = _ENTRY_FIXED_OVERHEAD + ek_size
        if _BLOCK_HDR_SIZE + entry_count * stride != used_bytes:
            return None
        mv = memoryview(block)
        lo = 0
        hi = entry_count
        while lo < hi:
            mid = (lo + hi) >> 1
            off = _BLOCK_HDR_SIZE + mid * stride
            klen = _ENTRY_KEY_LEN_STRUCT.unpack_from(mv, off)[0]
            if klen != ek_size:
                raise PageCorruptedError(
                    f"Unexpected key_len {klen} (expected {ek_size}) in {self.file_path}"
                )
            mid_key = bytes(mv[off + 2 : off + 2 + ek_size])
            if mid_key < target_key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _read_fixed_entry(self, block: bytes, idx: int) -> tuple[bytes, bool, RID]:
        """Unpack entry ``idx`` from a validated fixed-stride block."""
        ek_size = self._entry_key_size
        assert ek_size is not None
        stride = _ENTRY_FIXED_OVERHEAD + ek_size
        off = _BLOCK_HDR_SIZE + idx * stride
        klen = _ENTRY_KEY_LEN_STRUCT.unpack_from(block, off)[0]
        if klen != ek_size:
            raise PageCorruptedError(
                f"Unexpected key_len {klen} (expected {ek_size}) in {self.file_path}"
            )
        k_start = off + 2
        k_end = k_start + ek_size
        is_tomb, rid_page, rid_slot = _ENTRY_VAL_STRUCT.unpack_from(block, k_end)
        return block[k_start:k_end], bool(is_tomb), RID(page_id=rid_page, slot_id=rid_slot)

    def get_exact(
        self,
        entry_key: bytes,
        *,
        on_block_read: Callable[[], None] | None = None,
    ) -> tuple[bool, RID] | None:
        """Look up one exact ``entry_key`` using the fence index and at most 1 block read.

        Args:
            entry_key: Exact ``entry_key`` bytes (``user_key`` for unique,
                composite ``user_key + RID`` for non-unique).
            on_block_read: Optional callback invoked when an 8 KB data block is read.

        Returns:
            ``(is_tombstone, rid)`` if found, else ``None``.

        Complexity:
            $O(\\log B + \\log E_{\\text{block}})$ with at most 1 disk block read.
        """
        if not self._fence_keys:
            return None
        block_idx = bisect_right(self._fence_keys, entry_key) - 1
        if block_idx < 0:
            return None

        block = self._read_block(block_idx, on_block_read=on_block_read)
        entry_count, used_bytes = _BLOCK_HDR_STRUCT.unpack_from(block, 0)
        if entry_count == 0:
            return None

        pos_idx = self._bisect_fixed_block_left(block, entry_count, used_bytes, entry_key)
        if pos_idx is not None:
            if pos_idx >= entry_count:
                return None
            found_key, is_tomb, rid = self._read_fixed_entry(block, pos_idx)
            if found_key == entry_key:
                return is_tomb, rid
            return None

        entries = self._parse_block_entries(block)
        idx = bisect_left(entries, entry_key, key=lambda item: item[0])
        if idx < len(entries) and entries[idx][0] == entry_key:
            _, is_tomb, rid = entries[idx]
            return is_tomb, rid
        return None

    def _iter_block_from(
        self, block: bytes, seek_key: bytes | None
    ) -> Iterator[tuple[bytes, bool, RID]]:
        """Yield entries in ``block`` starting from first key ``>= seek_key`` (or all if ``None``)."""
        entry_count, used_bytes = _BLOCK_HDR_STRUCT.unpack_from(block, 0)
        if entry_count == 0:
            return
        pos_idx = (
            0
            if seek_key is None
            and self._entry_key_size is not None
            and _BLOCK_HDR_SIZE + entry_count * (_ENTRY_FIXED_OVERHEAD + self._entry_key_size)
            == used_bytes
            else (
                None
                if seek_key is None
                else self._bisect_fixed_block_left(block, entry_count, used_bytes, seek_key)
            )
        )
        if pos_idx is not None:
            for i in range(pos_idx, entry_count):
                yield self._read_fixed_entry(block, i)
            return

        entries = self._parse_block_entries(block)
        start_idx = (
            0 if seek_key is None else bisect_left(entries, seek_key, key=lambda item: item[0])
        )
        for i in range(start_idx, len(entries)):
            yield entries[i]

    def get_by_user_key(
        self,
        user_key: bytes,
        *,
        on_block_read: Callable[[], None] | None = None,
    ) -> list[tuple[bytes, bool, RID]]:
        """Return all ``(entry_key, is_tombstone, rid)`` matching ``user_key`` prefix.

        Used for non-unique indexes where multiple ``(user_key, RID)`` entries
        share the same user key and may span adjacent 8,192-byte blocks.
        """
        if not self._fence_keys:
            return []
        ksize = len(user_key)
        low_comp = user_key + _ZERO_RID_SUFFIX
        high_comp = user_key + _MAX_RID_SUFFIX

        block_idx = bisect_right(self._fence_keys, low_comp) - 1
        if block_idx < 0:
            # Block 0's first composite key always exceeds user_key + 0x00*8 even when matching user_key.
            if self._fence_keys[0][:ksize] > user_key:
                return []
            block_idx = 0

        matches: list[tuple[bytes, bool, RID]] = []
        num_blocks = len(self._fence_keys)
        first_block = True
        while block_idx < num_blocks:
            block = self._read_block(block_idx, on_block_read=on_block_read)
            for ekey, is_tomb, rid in self._iter_block_from(
                block, low_comp if first_block else None
            ):
                prefix = ekey[:ksize]
                if prefix > user_key:
                    return matches
                if prefix == user_key:
                    matches.append((ekey, is_tomb, rid))

            first_block = False
            block_idx += 1
            if block_idx >= num_blocks or self._fence_keys[block_idx] > high_comp:
                break
        return matches

    def range_iter(
        self,
        low_bytes: bytes,
        high_bytes: bytes,
        *,
        key_size: int,
        inclusive: bool = True,
        on_block_read: Callable[[], None] | None = None,
    ) -> Iterator[tuple[bytes, bool, RID]]:
        """Yield ``(entry_key, is_tombstone, rid)`` within ``[low_bytes, high_bytes]`` (or open).

        Args:
            low_bytes: Fixed-width encoded lower user-key bound.
            high_bytes: Fixed-width encoded upper user-key bound.
            key_size: User key prefix width in bytes.
            inclusive: Include endpoints when ``True``; exclude both when ``False``.
            on_block_read: Optional callback invoked on each 8 KB block read.

        Yields:
            Sorted ``(entry_key, is_tombstone, rid)`` records matching the range.
        """
        if not self._fence_keys:
            return
        if low_bytes > high_bytes or (low_bytes == high_bytes and not inclusive):
            return

        first_fence_user = self._fence_keys[0][:key_size]
        if first_fence_user > high_bytes or (not inclusive and first_fence_user >= high_bytes):
            return

        is_composite = (
            self._entry_key_size > key_size
            if self._entry_key_size is not None
            else len(self._fence_keys[0]) > key_size
        )
        if is_composite:
            low_seek = low_bytes + (_ZERO_RID_SUFFIX if inclusive else _MAX_RID_SUFFIX)
        else:
            low_seek = low_bytes

        block_idx = bisect_right(self._fence_keys, low_seek) - 1
        if block_idx < 0:
            block_idx = 0

        num_blocks = len(self._fence_keys)
        first_block = True
        while block_idx < num_blocks:
            fence_user = self._fence_keys[block_idx][:key_size]
            if fence_user > high_bytes or (not inclusive and fence_user >= high_bytes):
                return

            block = self._read_block(block_idx, on_block_read=on_block_read)
            for ekey, is_tomb, rid in self._iter_block_from(
                block, low_seek if first_block else None
            ):
                ukey = ekey[:key_size]
                if inclusive:
                    if ukey > high_bytes:
                        return
                    if ukey >= low_bytes:
                        yield ekey, is_tomb, rid
                else:
                    if ukey >= high_bytes:
                        return
                    if ukey > low_bytes:
                        yield ekey, is_tomb, rid

            first_block = False
            block_idx += 1

    def iter_entries(
        self,
        *,
        on_block_read: Callable[[], None] | None = None,
    ) -> Iterator[tuple[bytes, bool, RID]]:
        """Stream every entry sequentially across all 8 KB data blocks."""
        for block_idx in range(len(self._block_offsets)):
            block = self._read_block(block_idx, on_block_read=on_block_read)
            yield from self._parse_block_entries(block)

    def close(self) -> None:
        """Close the read-only ``SSTable`` file descriptor idempotently."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class LSMTreeIndex(BaseIndex):
    """Three-tier write-optimized LSM Tree index (``lsm.wal`` + ``MemTable`` + ``SSTable``s).

    Directory & Binary Formats:
    ```text
    data/indexes/<index_name>/
    ├── manifest.json         (atomic JSON registry: schema, next_seq, levels[0..L])
    ├── lsm.wal               (CRC32-framed MemTable write-ahead log)
    ├── sst_0_000001.sst      (Level 0 SSTables — overlapping key ranges)
    └── sst_1_000005.sst      (Level 1+ SSTables — compacted & deduplicated)

    lsm.wal Record Layout (<IBH + entry_key + <II = 15 + len(entry_key) Bytes):
    0       4         5         7                 7+K          11+K          15+K
    +-------+---------+---------+-----------------+------------+-------------+
    | crc32 | op_type | key_len | entry_key_bytes |  rid_page  |  rid_slot   |
    | (<I)  | 1=PUT   |  (<H)   |   (K Bytes)     |    (<I)    |    (<I)     |
    |       | 2=DEL   |         |                 |            |             |
    +-------+---------+---------+-----------------+------------+-------------+
            |<------------------ CRC32 Input Scope ------------------------->|
    ```
    """

    def __init__(
        self,
        directory: str | Path,
        key_dtype: DataType,
        key_size: int = 8,
        unique: bool = True,
        memtable_size_bytes: int = DEFAULT_MEMTABLE_BYTES,
        compaction_trigger: int = LSM_COMPACTION_TRIGGER,
        fsync_on_write: bool = False,
    ) -> None:
        """Open or initialize a persistent three-tier ``LSMTreeIndex`` in ``directory``.

        Args:
            directory: Folder storing ``lsm.wal``, ``manifest.json``, and ``*.sst`` files.
            key_dtype: Logical ``DataType`` of the indexed key column.
            key_size: Fixed byte width of encoded user keys (``4`` for ``INT32``,
                ``8`` for 64-bit numerics, ``max_len`` for ``VARCHAR``).
            unique: When ``True``, overwrites replace older ``RID``s per user key;
                when ``False``, composite keys store multiple ``RID``s per key.
            memtable_size_bytes: MemTable byte threshold triggering flush to ``L0``.
            compaction_trigger: Number of SSTables in level ``L_i`` triggering
                size-tiered compaction into ``L_{i+1}``.
            fsync_on_write: If ``True``, call ``os.fsync`` after every WAL append.

        Raises:
            TypeError: If ``key_dtype`` is not a ``DataType`` or boolean flags have
                invalid types.
            ValueError: If ``key_size``, ``memtable_size_bytes``, or
                ``compaction_trigger`` is invalid.
            StorageError: If existing ``manifest.json`` schema mismatches or files
                cannot be opened.
            PageCorruptedError: If ``manifest.json`` or any ``SSTable`` is corrupted.

        Complexity:
            $O(W + \\sum F_i)$ on open to replay ``lsm.wal`` ($W$ bytes) and verify
            active ``SSTable`` files ($F_i$ bytes).
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
        if (
            isinstance(memtable_size_bytes, bool)
            or not isinstance(memtable_size_bytes, int)
            or memtable_size_bytes <= 0
        ):
            raise ValueError(
                f"memtable_size_bytes must be a positive int, got {memtable_size_bytes!r}"
            )
        if (
            isinstance(compaction_trigger, bool)
            or not isinstance(compaction_trigger, int)
            or compaction_trigger < 1
        ):
            raise ValueError(
                f"compaction_trigger must be a positive int, got {compaction_trigger!r}"
            )
        if not isinstance(fsync_on_write, bool):
            raise TypeError(
                f"fsync_on_write must be a bool, got {type(fsync_on_write).__name__}"
            )

        entry_key_size = key_size if unique else key_size + 8
        if entry_key_size > _MAX_ENTRY_KEY_SIZE:
            raise ValueError(
                f"entry_key_size={entry_key_size} exceeds SSTable block limit {_MAX_ENTRY_KEY_SIZE}"
            )

        self.directory: Path = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._key_dtype: DataType = key_dtype
        self._key_size: int = key_size
        self._unique: bool = unique
        self._entry_key_size: int = entry_key_size
        self._memtable_size_bytes: int = memtable_size_bytes
        self._compaction_trigger: int = compaction_trigger
        self._fsync_on_write: bool = fsync_on_write

        self._lock: threading.RLock = threading.RLock()
        self._closed: bool = False

        self._manifest_path: Path = self.directory / "manifest.json"
        self._wal_path: Path = self.directory / "lsm.wal"
        self._wal_fd: int | None = None
        self._wal_offset: int = 0

        self._memtable: MemTable = MemTable()
        self._levels: dict[int, list[SSTable]] = {}
        self._next_seq: int = 1

        self._bloom_checks: int = 0
        self._bloom_rejections: int = 0
        self._block_reads: int = 0
        self._compaction_block_reads: int = 0
        self._flushes: int = 0
        self._compactions: int = 0

        try:
            self._cleanup_stale_temps()
            self._load_or_init_manifest()
            self._open_and_replay_wal()
        except Exception:
            self._close_all_fds_quietly()
            self._closed = True
            raise

    def _close_all_fds_quietly(self) -> None:
        """Release WAL and SSTable file descriptors without raising during cleanup."""
        if self._wal_fd is not None:
            try:
                os.close(self._wal_fd)
            except OSError:
                pass
            self._wal_fd = None
        for ssts in self._levels.values():
            for sst in ssts:
                try:
                    sst.close()
                except OSError:
                    pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError(f"LSMTreeIndex in {self.directory} is closed")

    def _cleanup_stale_temps(self) -> None:
        """Remove orphaned ``*.tmp`` files left by an interrupted flush/compaction."""
        for tmp in self.directory.glob("*.tmp"):
            tmp.unlink(missing_ok=True)

    def _persist_manifest(self, levels: dict[int, list[SSTable]] | None = None) -> None:
        """Atomically persist ``manifest.json`` via temp-write, ``os.fsync``, and ``os.replace``."""
        target_levels = self._levels if levels is None else levels
        payload = {
            "version": 1,
            "key_dtype": self._key_dtype.name,
            "key_size": self._key_size,
            "unique": self._unique,
            "next_seq": self._next_seq,
            "levels": {
                str(lvl): [sst.file_path.name for sst in ssts]
                for lvl, ssts in sorted(target_levels.items())
                if ssts
            },
        }
        raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        tmp_path = self.directory / "manifest.json.tmp"
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(os.fspath(tmp_path), open_flags, 0o644)
        except OSError as exc:
            raise StorageError(
                f"Failed opening manifest temp file {tmp_path}: {exc}"
            ) from exc
        try:
            try:
                _write_all_fd(fd, raw)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_path, self._manifest_path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise StorageError(
                f"Failed persisting LSM manifest {self._manifest_path}: {exc}"
            ) from exc
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _load_or_init_manifest(self) -> None:
        """Load and validate ``manifest.json`` on reopen, or initialize a new manifest."""
        if not self._manifest_path.exists():
            self._persist_manifest()
            return

        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PageCorruptedError(f"Corrupt LSM manifest {self._manifest_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise PageCorruptedError(f"Invalid manifest object in {self._manifest_path}")
        if raw.get("key_dtype") != self._key_dtype.name:
            raise StorageError(
                f"LSM key_dtype mismatch in {self.directory}: "
                f"expected {self._key_dtype.name}, found {raw.get('key_dtype')!r}"
            )
        if raw.get("key_size") != self._key_size:
            raise StorageError(
                f"LSM key_size mismatch in {self.directory}: "
                f"expected {self._key_size}, found {raw.get('key_size')!r}"
            )
        if raw.get("unique") != self._unique:
            raise StorageError(
                f"LSM unique flag mismatch in {self.directory}: "
                f"expected {self._unique}, found {raw.get('unique')!r}"
            )

        next_seq = raw.get("next_seq", 1)
        if isinstance(next_seq, bool) or not isinstance(next_seq, int) or next_seq < 1:
            raise PageCorruptedError(f"Invalid next_seq in {self._manifest_path}: {next_seq!r}")
        self._next_seq = next_seq

        levels_raw = raw.get("levels", {})
        if not isinstance(levels_raw, dict):
            raise PageCorruptedError(f"Invalid levels table in {self._manifest_path}")

        loaded_levels: dict[int, list[SSTable]] = {}
        try:
            for lvl_str, filenames in levels_raw.items():
                try:
                    lvl = int(lvl_str)
                except (ValueError, TypeError) as exc:
                    raise PageCorruptedError(
                        f"Invalid level entry {lvl_str!r} in manifest"
                    ) from exc
                if lvl < 0 or not isinstance(filenames, list):
                    raise PageCorruptedError(f"Invalid level entry {lvl_str!r} in manifest")
                ssts: list[SSTable] = []
                for fname in filenames:
                    sst_path = self.directory / str(fname)
                    if not sst_path.exists():
                        raise StorageError(
                            f"Missing SSTable {sst_path} referenced by {self._manifest_path}"
                        )
                    ssts.append(SSTable.open(sst_path, entry_key_size=self._entry_key_size))
                if ssts:
                    loaded_levels[lvl] = ssts
        except Exception:
            for ssts in loaded_levels.values():
                for sst in ssts:
                    sst.close()
            raise
        self._levels = loaded_levels

    def _open_and_replay_wal(self) -> None:
        """Open ``lsm.wal``, replay valid records into ``MemTable``, and truncate torn tails."""
        open_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(os.fspath(self._wal_path), open_flags, 0o644)
        except OSError as exc:
            raise StorageError(f"Failed to open LSM WAL {self._wal_path}: {exc}") from exc

        self._wal_fd = fd
        try:
            file_size = os.fstat(fd).st_size
        except OSError as exc:
            raise StorageError(f"Failed stating LSM WAL {self._wal_path}: {exc}") from exc
        if file_size == 0:
            self._wal_offset = 0
            return

        wal_bytes = _pread_exact(fd, file_size, 0)
        pos = 0
        valid_end = 0
        expected_klen = self._entry_key_size

        while pos + 7 <= file_size:
            crc_stored, op_type, key_len = _WAL_HDR_STRUCT.unpack_from(wal_bytes, pos)
            if op_type not in (_WAL_OP_PUT, _WAL_OP_DELETE) or key_len != expected_klen:
                break
            rec_len = 7 + key_len + 8
            if pos + rec_len > file_size:
                break
            actual_crc = zlib.crc32(wal_bytes[pos + 4 : pos + rec_len]) & _UINT32_MAX
            if actual_crc != crc_stored:
                break
            k_start = pos + 7
            k_end = k_start + key_len
            entry_key = wal_bytes[k_start:k_end]
            rid_page, rid_slot = _WAL_RID_STRUCT.unpack_from(wal_bytes, k_end)
            self._memtable.put(
                entry_key,
                RID(page_id=rid_page, slot_id=rid_slot),
                is_tombstone=(op_type == _WAL_OP_DELETE),
            )
            pos += rec_len
            valid_end = pos

        try:
            if valid_end < file_size:
                # Truncate corrupt/incomplete tail so subsequent WAL appends stay contiguous.
                os.ftruncate(fd, valid_end)
                os.fsync(fd)
            os.lseek(fd, valid_end, os.SEEK_SET)
        except OSError as exc:
            raise StorageError(f"Failed positioning LSM WAL {self._wal_path}: {exc}") from exc
        self._wal_offset = valid_end

        if self._memtable.byte_size >= self._memtable_size_bytes:
            self._flush_memtable_unlocked()

    def _append_wal(self, op_type: int, entry_key: bytes, rid: RID) -> None:
        """Pack and write one CRC32-framed record to ``lsm.wal`` immediately."""
        assert self._wal_fd is not None
        payload = (
            _WAL_PAYLOAD_HDR_STRUCT.pack(op_type, len(entry_key))
            + entry_key
            + _WAL_RID_STRUCT.pack(rid.page_id, rid.slot_id)
        )
        crc = zlib.crc32(payload) & _UINT32_MAX
        record = struct.pack("<I", crc) + payload
        _pwrite_all_fd(self._wal_fd, record, self._wal_offset)
        self._wal_offset += len(record)
        if self._fsync_on_write:
            try:
                os.fsync(self._wal_fd)
            except OSError as exc:
                raise StorageError(f"Failed syncing LSM WAL {self._wal_path}: {exc}") from exc

    def _record_query_block_read(self) -> None:
        """Track 8 KB SSTable block reads performed during lookups and range scans."""
        self._block_reads += 1

    def _record_compaction_block_read(self) -> None:
        """Track 8 KB SSTable block reads performed during compaction merges."""
        self._compaction_block_reads += 1

    def _iter_sstables_newest_first(self) -> Iterator[SSTable]:
        """Yield active ``SSTable`` handles from newest (L0 top) to oldest."""
        for lvl in sorted(self._levels.keys()):
            for sst in reversed(self._levels[lvl]):
                yield sst

    @staticmethod
    def _validate_rid(rid: RID) -> None:
        if not isinstance(rid, RID):
            raise TypeError(f"rid must be a RID, got {type(rid).__name__}")
        if (
            isinstance(rid.page_id, bool)
            or isinstance(rid.slot_id, bool)
            or not (0 <= rid.page_id <= _UINT32_MAX)
            or not (0 <= rid.slot_id <= _UINT32_MAX)
        ):
            raise ValueError(f"RID fields must fit in uint32, got {rid!r}")

    def _next_sst_path(self, level: int) -> Path:
        seq = self._next_seq
        self._next_seq += 1
        return self.directory / f"sst_{level}_{seq:06d}.sst"

    def _flush_memtable_unlocked(self) -> None:
        """Freeze active ``MemTable``, flush to ``L0``, atomically commit manifest, and reset WAL."""
        if len(self._memtable) == 0:
            if self._wal_fd is not None and self._wal_offset > 0:
                try:
                    os.fsync(self._wal_fd)
                except OSError as exc:
                    raise StorageError(
                        f"Failed syncing LSM WAL {self._wal_path}: {exc}"
                    ) from exc
            return

        prev_next_seq = self._next_seq
        sst_path = self._next_sst_path(0)
        new_sst: SSTable | None = None
        try:
            new_sst = SSTable.build(
                sst_path,
                self._memtable.iter_sorted(),
                key_size=self._key_size,
                entry_key_size=self._entry_key_size,
            )
            candidate_levels: dict[int, list[SSTable]] = {
                lvl: list(ssts) for lvl, ssts in self._levels.items()
            }
            candidate_levels.setdefault(0, []).append(new_sst)
            self._persist_manifest(candidate_levels)
        except Exception:
            if new_sst is not None:
                try:
                    new_sst.close()
                finally:
                    new_sst.file_path.unlink(missing_ok=True)
            self._next_seq = prev_next_seq
            raise
        self._levels = candidate_levels

        # Truncate WAL only after manifest.json points durably to the synced L0 SSTable.
        assert self._wal_fd is not None
        try:
            os.ftruncate(self._wal_fd, 0)
            os.lseek(self._wal_fd, 0, os.SEEK_SET)
            os.fsync(self._wal_fd)
        except OSError as exc:
            raise StorageError(f"Failed resetting LSM WAL {self._wal_path}: {exc}") from exc
        self._wal_offset = 0

        self._memtable.clear()
        self._flushes += 1
        self._maybe_auto_compact_unlocked()

    def _maybe_auto_compact_unlocked(self) -> None:
        """Cascade size-tiered compaction upward whenever any level reaches ``compaction_trigger``."""
        lvl = 0
        while True:
            ssts = self._levels.get(lvl)
            if not ssts or len(ssts) < self._compaction_trigger:
                break
            self._compact_level_unlocked(lvl)
            lvl += 1

    def _merge_sstables_to_level(
        self,
        sources_newest_first: list[SSTable],
        target_level: int,
        *,
        drop_tombstones: bool,
        old_ssts_to_remove: list[tuple[int, SSTable]],
    ) -> None:
        """Merge ``sources_newest_first`` into ``target_level``, commit manifest, and delete sources."""
        if not sources_newest_first:
            return

        tagged_iters = [
            _tag_stream(
                sst.iter_entries(on_block_read=self._record_compaction_block_read),
                priority,
            )
            for priority, sst in enumerate(sources_newest_first)
        ]

        def _dedup_stream() -> Iterator[tuple[bytes, bool, RID]]:
            prev_key: bytes | None = None
            for ekey, _priority, is_tomb, rid in heapq.merge(*tagged_iters):
                if ekey == prev_key:
                    continue
                prev_key = ekey
                if drop_tombstones and is_tomb:
                    continue
                yield ekey, is_tomb, rid

        prev_next_seq = self._next_seq
        sst_path = self._next_sst_path(target_level)
        new_sst: SSTable | None = None
        new_sst_closed = False
        created_path_to_cleanup: Path | None = None
        try:
            new_sst = SSTable.build(
                sst_path,
                _dedup_stream(),
                key_size=self._key_size,
                entry_key_size=self._entry_key_size,
            )

            # Build candidate level map without mutating self._levels until manifest commits.
            remove_ids = {id(s) for _, s in old_ssts_to_remove}
            updated_levels: dict[int, list[SSTable]] = {}
            for lvl, ssts in self._levels.items():
                kept = [s for s in ssts if id(s) not in remove_ids]
                if kept:
                    updated_levels[lvl] = kept

            if new_sst.entry_count > 0:
                updated_levels.setdefault(target_level, []).append(new_sst)
            else:
                new_sst.close()
                new_sst_closed = True
                created_path_to_cleanup = new_sst.file_path

            self._persist_manifest(updated_levels)
        except Exception:
            if new_sst is not None:
                try:
                    if not new_sst_closed:
                        new_sst.close()
                finally:
                    new_sst.file_path.unlink(missing_ok=True)
            self._next_seq = prev_next_seq
            raise

        self._levels = updated_levels

        for _, old_sst in old_ssts_to_remove:
            old_sst.close()
            old_sst.file_path.unlink(missing_ok=True)
        if created_path_to_cleanup is not None:
            created_path_to_cleanup.unlink(missing_ok=True)

        self._compactions += 1

    def _compact_level_unlocked(self, level: int) -> None:
        """Compact all SSTables in ``level`` into ``level + 1``."""
        ssts = self._levels.get(level)
        if not ssts:
            return
        target_level = level + 1
        # Only purge tombstones when no older SSTables remain in target_level or deeper levels.
        has_older = any(
            bool(self._levels.get(lvl)) for lvl in self._levels if lvl >= target_level
        )
        sources_newest_first = list(reversed(ssts))
        old_pairs = [(level, sst) for sst in ssts]
        self._merge_sstables_to_level(
            sources_newest_first,
            target_level,
            drop_tombstones=not has_older,
            old_ssts_to_remove=old_pairs,
        )

    def _compact_all_unlocked(self) -> None:
        """Merge all existing SSTables across all levels into a single clean ``L1`` SSTable."""
        all_pairs: list[tuple[int, SSTable]] = []
        sources_newest_first: list[SSTable] = []
        for lvl in sorted(self._levels.keys()):
            lvl_ssts = self._levels[lvl]
            for sst in reversed(lvl_ssts):
                sources_newest_first.append(sst)
            for sst in lvl_ssts:
                all_pairs.append((lvl, sst))

        if not sources_newest_first:
            return

        # Full compaction leaves zero older SSTables on disk, so tombstones can be dropped safely.
        self._merge_sstables_to_level(
            sources_newest_first,
            target_level=1,
            drop_tombstones=True,
            old_ssts_to_remove=all_pairs,
        )

    def _get_unique_unlocked(self, user_key: bytes) -> RID | None:
        """Return active ``RID`` for ``user_key`` in a unique index, or ``None`` if absent/tombstoned."""
        mem_hit = self._memtable.get(user_key)
        if mem_hit is not None:
            is_tomb, rid = mem_hit
            return None if is_tomb else rid

        for sst in self._iter_sstables_newest_first():
            self._bloom_checks += 1
            if not sst.bloom.might_contain(user_key):
                self._bloom_rejections += 1
                continue
            hit = sst.get_exact(user_key, on_block_read=self._record_query_block_read)
            if hit is not None:
                is_tomb, rid = hit
                return None if is_tomb else rid
        return None

    def _get_non_unique_unlocked(self, user_key: bytes) -> list[RID]:
        """Return sorted active ``RID`` list for ``user_key`` in a non-unique index."""
        low_comp = user_key + _ZERO_RID_SUFFIX
        high_comp = user_key + _MAX_RID_SUFFIX
        seen: dict[bytes, tuple[bool, RID]] = {}

        for ekey, is_tomb, rid in self._memtable.range_items(low_comp, high_comp):
            seen[ekey] = (is_tomb, rid)

        for sst in self._iter_sstables_newest_first():
            self._bloom_checks += 1
            if not sst.bloom.might_contain(user_key):
                self._bloom_rejections += 1
                continue
            for ekey, is_tomb, rid in sst.get_by_user_key(
                user_key, on_block_read=self._record_query_block_read
            ):
                if ekey not in seen:
                    seen[ekey] = (is_tomb, rid)

        return [
            rid
            for ekey, (is_tomb, rid) in sorted(seen.items(), key=lambda item: item[0])
            if not is_tomb
        ]

    def _is_comp_key_active_unlocked(self, user_key: bytes, comp_key: bytes) -> bool:
        """Check whether one specific composite ``(user_key, rid)`` entry is currently active."""
        mem_hit = self._memtable.get(comp_key)
        if mem_hit is not None:
            return not mem_hit[0]

        for sst in self._iter_sstables_newest_first():
            self._bloom_checks += 1
            if not sst.bloom.might_contain(user_key):
                self._bloom_rejections += 1
                continue
            hit = sst.get_exact(comp_key, on_block_read=self._record_query_block_read)
            if hit is not None:
                return not hit[0]
        return False

    def _write_entry_unlocked(self, op_type: int, entry_key: bytes, rid: RID) -> None:
        """Durably append to ``lsm.wal``, mutate ``MemTable``, and auto-flush on threshold."""
        self._append_wal(op_type, entry_key, rid)
        self._memtable.put(entry_key, rid, is_tombstone=(op_type == _WAL_OP_DELETE))
        if self._memtable.byte_size >= self._memtable_size_bytes:
            self._flush_memtable_unlocked()

    def insert(self, key: Any, rid: RID) -> None:
        """Append a ``(key, rid)`` PUT to ``lsm.wal`` and insert into ``MemTable``.

        For ``unique=True`` LSM indexes, inserting an existing key acts as an
        upsert that overwrites the previous ``RID`` without performing a slow
        read-before-write uniqueness check.

        Args:
            key: Logical key value matching ``key_dtype``.
            rid: Physical ``RID`` locating the tuple in heap storage.

        Raises:
            TypeError: If ``rid`` is not a ``RID`` or ``key`` mismatches ``key_dtype``.
            ValueError: If ``key`` overflows ``key_size``, is ``NaN``, or ``rid`` overflows uint32.
            StorageError: If the index is closed or WAL/SSTable I/O fails.

        Complexity:
            $O(\\log M)$ amortized ($O(1)$ on duplicate key overwrite); triggers
            sequential ``L0`` flush when ``memtable_bytes >= memtable_size_bytes``.
            Thread-safe under ``self._lock``.
        """
        self._validate_rid(rid)
        user_key = encode_key(key, self._key_dtype, self._key_size)
        entry_key = user_key if self._unique else make_composite_key(user_key, rid)

        with self._lock:
            self._ensure_open()
            self._write_entry_unlocked(_WAL_OP_PUT, entry_key, rid)

    def get(self, key: Any) -> list[RID]:
        """Look up all active ``RID`` values associated with ``key``.

        Searches ``MemTable`` first, then ``L0`` SSTables newest-to-oldest, then
        ``L1+``, consulting each SSTable's user-key ``BloomFilter`` before any
        disk block read.

        Args:
            key: Logical key value to search.

        Returns:
            Sorted list of active ``RID`` instances (empty list if absent or deleted).

        Raises:
            TypeError: If ``key`` has an incompatible Python type.
            ValueError: If ``key`` overflows ``key_size`` or is ``NaN``.
            StorageError: If the index is closed or an SSTable block cannot be read.
            PageCorruptedError: If an accessed SSTable block header is corrupt.

        Complexity:
            $O(1)$ MemTable lookup + $O(T)$ Bloom checks across $T$ SSTables +
            at most 1 disk block read per Bloom positive (for ``unique=True``).
            Thread-safe under ``self._lock``.
        """
        user_key = encode_key(key, self._key_dtype, self._key_size)

        with self._lock:
            self._ensure_open()
            if self._unique:
                rid = self._get_unique_unlocked(user_key)
                return [] if rid is None else [rid]
            return self._get_non_unique_unlocked(user_key)

    def range_scan(
        self, low_key: Any, high_key: Any, inclusive: bool = True
    ) -> list[tuple[Any, RID]]:
        """Merge-scan active ``(key, RID)`` entries between ``low_key`` and ``high_key``.

        Uses ``heapq.merge`` across ``MemTable`` and all active ``SSTable``s
        tagged with recency priority so newer updates and tombstones suppress
        older SSTable versions.

        Args:
            low_key: Lower bound key.
            high_key: Upper bound key.
            inclusive: Include endpoints when ``True``; exclude both when ``False``.

        Returns:
            Sorted list of active ``(decoded_key, RID)`` tuples within range.

        Raises:
            TypeError: If ``low_key`` or ``high_key`` has an incompatible type.
            ValueError: If either bound overflows ``key_size`` or is ``NaN``.
            StorageError: If the index is closed or an SSTable block cannot be read.
            PageCorruptedError: If an accessed SSTable block header is corrupt.

        Complexity:
            $O((K + T) \\log T)$ across $T$ active tiers and $K$ scanned entries.
            Thread-safe under ``self._lock``.
        """
        low_bytes = encode_key(low_key, self._key_dtype, self._key_size)
        high_bytes = encode_key(high_key, self._key_dtype, self._key_size)

        with self._lock:
            self._ensure_open()
            if low_bytes > high_bytes or (low_bytes == high_bytes and not inclusive):
                return []

            if self._unique:
                mem_low = low_bytes
                mem_high = high_bytes
            else:
                mem_low = low_bytes + _ZERO_RID_SUFFIX
                mem_high = high_bytes + _MAX_RID_SUFFIX

            def _mem_iter() -> Iterator[tuple[bytes, int, bool, RID]]:
                for ekey, is_tomb, rid in self._memtable.range_items(mem_low, mem_high):
                    if not inclusive:
                        ukey = ekey[: self._key_size]
                        if ukey <= low_bytes or ukey >= high_bytes:
                            continue
                    yield ekey, 0, is_tomb, rid

            iterators: list[Iterator[tuple[bytes, int, bool, RID]]] = [_mem_iter()]
            priority = 1
            for sst in self._iter_sstables_newest_first():
                iterators.append(
                    _tag_stream(
                        sst.range_iter(
                            low_bytes,
                            high_bytes,
                            key_size=self._key_size,
                            inclusive=inclusive,
                            on_block_read=self._record_query_block_read,
                        ),
                        priority,
                    )
                )
                priority += 1

            results: list[tuple[Any, RID]] = []
            prev_key: bytes | None = None
            prev_user: bytes | None = None
            prev_decoded: Any = None
            ksize = self._key_size

            for ekey, _prio, is_tomb, rid in heapq.merge(*iterators):
                if ekey == prev_key:
                    continue
                prev_key = ekey
                if is_tomb:
                    continue
                ukey = ekey[:ksize]
                if ukey != prev_user:
                    prev_user = ukey
                    prev_decoded = decode_key(ukey, self._key_dtype)
                results.append((prev_decoded, rid))
            return results

    def delete(self, key: Any, rid: RID | None = None) -> bool:
        """Append tombstone record(s) suppressing ``key`` (or a specific ``(key, rid)``).

        Args:
            key: Logical key value to delete.
            rid: Optional specific ``RID`` for non-unique indexes (or verified
                against the active ``RID`` when ``unique=True``). When ``None``
                on a non-unique index, tombstones all currently active ``RID``s
                for ``key``.

        Returns:
            ``True`` if at least one active entry was tombstoned, ``False`` otherwise.

        Raises:
            TypeError: If ``key`` or ``rid`` has an incompatible type.
            ValueError: If ``key`` overflows ``key_size``, is ``NaN``, or ``rid`` overflows uint32.
            StorageError: If the index is closed or WAL/SSTable I/O fails.

        Complexity:
            Point lookup cost plus $O(\\log M)$ WAL + MemTable tombstone write(s).
            Thread-safe under ``self._lock``.
        """
        if rid is not None:
            self._validate_rid(rid)
        user_key = encode_key(key, self._key_dtype, self._key_size)

        with self._lock:
            self._ensure_open()
            if self._unique:
                current_rid = self._get_unique_unlocked(user_key)
                if current_rid is None:
                    return False
                if rid is not None and current_rid != rid:
                    return False
                self._write_entry_unlocked(_WAL_OP_DELETE, user_key, NULL_RID)
                return True

            if rid is not None:
                comp_key = make_composite_key(user_key, rid)
                if not self._is_comp_key_active_unlocked(user_key, comp_key):
                    return False
                self._write_entry_unlocked(_WAL_OP_DELETE, comp_key, NULL_RID)
                return True

            active_rids = self._get_non_unique_unlocked(user_key)
            if not active_rids:
                return False
            for active_rid in active_rids:
                comp_key = make_composite_key(user_key, active_rid)
                self._write_entry_unlocked(_WAL_OP_DELETE, comp_key, NULL_RID)
            return True

    def flush(self) -> None:
        """Flush active ``MemTable`` entries durably into a new ``L0`` ``SSTable`` and reset WAL.

        Raises:
            StorageError: If the index is closed or disk write/sync fails.

        Complexity:
            $O(M)$ sequential write plus ``os.fsync``; thread-safe under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            self._flush_memtable_unlocked()

    def compact(self, level: int | None = None) -> None:
        """Compact SSTables either for a single ``level`` or across all levels when ``level is None``.

        Args:
            level: Specific source level ``L_i`` to merge into ``L_{i+1}``, or
                ``None`` to perform a full multi-tier compaction down to a single
                clean ``L1`` SSTable with all tombstones purged.

        Raises:
            ValueError: If ``level`` is negative or not an integer.
            StorageError: If the index is closed or compaction I/O fails.

        Complexity:
            $O(N \\log T)$ merge over all entries in compacted SSTables; thread-safe
            under ``self._lock``.
        """
        if level is not None and (
            isinstance(level, bool) or not isinstance(level, int) or level < 0
        ):
            raise ValueError(f"level must be None or a non-negative int, got {level!r}")

        with self._lock:
            self._ensure_open()
            if level is None:
                self._compact_all_unlocked()
                return
            self._compact_level_unlocked(level)
            cur = level + 1
            while len(self._levels.get(cur, [])) >= self._compaction_trigger:
                self._compact_level_unlocked(cur)
                cur += 1

    def close(self) -> None:
        """Sync ``lsm.wal`` and close all open WAL and ``SSTable`` file descriptors idempotently.

        Raises:
            StorageError: If syncing ``lsm.wal`` fails prior to closing.

        Complexity:
            $O(T)$ descriptor closes; thread-safe under ``self._lock``.
        """
        with self._lock:
            if self._closed:
                return
            try:
                if self._wal_fd is not None:
                    os.fsync(self._wal_fd)
            except OSError as exc:
                raise StorageError(f"Failed syncing LSM WAL on close: {exc}") from exc
            finally:
                self._close_all_fds_quietly()
                self._closed = True

    def get_stats(self) -> dict[str, Any]:
        """Return LSM structural, disk footprint, Bloom filter, and compaction telemetry.

        Returns:
            Dictionary containing ``index_type``, ``directory``, ``key_dtype``,
            ``key_size``, ``unique``, ``memtable_entries``, ``memtable_bytes``,
            ``memtable_size_bytes``, ``sstable_count``, ``levels``, ``disk_bytes``,
            ``bloom_checks``, ``bloom_rejections``, ``bloom_rejection_rate``,
            ``block_reads``, ``compaction_block_reads``, ``flushes``, and
            ``compactions``.

        Raises:
            StorageError: If the index is closed.

        Complexity:
            $O(T)$ across active SSTable counts under ``self._lock``.
        """
        with self._lock:
            self._ensure_open()
            level_counts = {
                lvl: len(ssts) for lvl, ssts in sorted(self._levels.items()) if ssts
            }
            sstable_bytes = sum(
                sst.file_size for ssts in self._levels.values() for sst in ssts
            )
            wal_bytes = self._wal_offset
            manifest_bytes = (
                self._manifest_path.stat().st_size if self._manifest_path.exists() else 0
            )
            rejection_rate = (
                (self._bloom_rejections / self._bloom_checks)
                if self._bloom_checks > 0
                else 0.0
            )
            return {
                "index_type": "LSM_TREE",
                "directory": str(self.directory),
                "key_dtype": self._key_dtype.name,
                "key_size": self._key_size,
                "unique": self._unique,
                "memtable_entries": len(self._memtable),
                "memtable_bytes": self._memtable.byte_size,
                "memtable_size_bytes": self._memtable_size_bytes,
                "sstable_count": sum(level_counts.values()),
                "levels": level_counts,
                "disk_bytes": sstable_bytes + wal_bytes + manifest_bytes,
                "bloom_checks": self._bloom_checks,
                "bloom_rejections": self._bloom_rejections,
                "bloom_rejection_rate": rejection_rate,
                "block_reads": self._block_reads,
                "compaction_block_reads": self._compaction_block_reads,
                "flushes": self._flushes,
                "compactions": self._compactions,
            }

    def __enter__(self) -> LSMTreeIndex:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


__all__ = ["BloomFilter", "MemTable", "SSTable", "LSMTreeIndex"]
