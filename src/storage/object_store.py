"""
Module: src/storage/object_store.py
Layer: Layer 1 — Extent-Based Encrypted Binary Object Store
Lock Domain: ObjectStoreLock (``self._lock = threading.RLock()``; leaf lock)

Purpose:
    Persists variable-sized binary large objects (product images, rich media
    blobs) inside two consolidated extent files (``objects.dat`` and
    ``objects.dir``) using pluggable AEAD encryption hooks (``NoOpCipher`` or
    ``AES256GCMHook``) with per-object ID AAD binding and CRC32 framing.

Key Invariants:
    1. Never creates one OS file per object; all objects are appended as
       self-describing extent records in ``objects.dat`` and indexed in
       ``objects.dir``.
    2. Every record binds ``aad = struct.pack("<Q", oid)`` inside its AEAD
       authentication tag so cross-OID ciphertext swaps fail with
       ``SecurityViolationError`` even if the outer CRC32 trailer is forged.
    3. ``objects.dir`` is persisted atomically via ``objects.dir.tmp`` ->
       ``os.fsync`` -> ``os.replace`` on ``flush()`` and ``close()`` so
       directory state and deletions survive restarts cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import threading
from typing import Any
import zlib

from src.common.config import MAGIC_OBJECT_DIR, MAGIC_OBJECT_RECORD
from src.common.exceptions import KeyNotFoundError, PageCorruptedError, StorageError
from src.storage.cipher_hooks import CipherHook, CipherPayload, NoOpCipher

_HAS_PREAD: bool = hasattr(os, "pread")
_HAS_PWRITE: bool = hasattr(os, "pwrite")

_RECORD_HEADER_STRUCT: struct.Struct = struct.Struct("<4sQII12s16s")
_RECORD_CRC_STRUCT: struct.Struct = struct.Struct("<I")
_RECORD_HEADER_SIZE: int = _RECORD_HEADER_STRUCT.size  # 48 bytes
_RECORD_TRAILER_SIZE: int = _RECORD_CRC_STRUCT.size  # 4 bytes

_DIR_HEADER_STRUCT: struct.Struct = struct.Struct("<4sI")
_DIR_ENTRY_STRUCT: struct.Struct = struct.Struct("<QQII")
_DIR_HEADER_SIZE: int = _DIR_HEADER_STRUCT.size  # 8 bytes
_DIR_ENTRY_SIZE: int = _DIR_ENTRY_STRUCT.size  # 24 bytes

_UINT32_MAX: int = 0xFFFFFFFF
_UINT64_MAX: int = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class ObjectMeta:
    """In-memory and on-disk directory entry locating one extent in ``objects.dat``."""

    oid: int
    file_offset: int
    cipher_len: int
    raw_len: int


class ObjectStore:
    """Thread-safe extent-based encrypted object store backed by ``objects.dat`` and ``objects.dir``.

    Binary Extent Formats:
    ```text
    objects.dat — Variable-Length Encrypted Object Record Envelope:
    0      4       12        16           20        32       48                  48+C              52+C
    +------+-------+---------+------------+---------+--------+-------------------+-----------------+
    |magic |  oid  | raw_len | cipher_len |  nonce  |  tag   |    ciphertext     |     crc32       |
    |"OBJ1"|uint64 | uint32  |   uint32   | 12 Bytes|16 Bytes|    (C Bytes)      | uint32 (<I)     |
    +------+-------+---------+------------+---------+--------+-------------------+-----------------+
    |<-------------- 48-Byte Header (<4sQII12s16s) --------->|                   |                 |
    |<----------------- CRC32 Input Scope (48 + C Bytes) ----------------------->|                 |

    objects.dir — Atomic Object Directory Snapshot:
    0      4             8                       32                       32+24*(N-1)        8+24*N
    +------+-------------+-----------------------+------------------------+------------------------+
    |magic | entry_count | Entry 0 (<QQII)       | Entry 1 (<QQII)        | ... Entry N-1 (<QQII)  |
    |"ODIR"|   uint32    | oid|off|clen|rlen     | oid|off|clen|rlen      | oid|off|clen|rlen      |
    +------+-------------+-----------------------+------------------------+------------------------+
    Each Directory Entry (24 Bytes, <QQII):
      0..7   (Q): oid         (uint64)
      8..15  (Q): file_offset (uint64)
      16..19 (I): cipher_len  (uint32)
      20..23 (I): raw_len     (uint32)
    ```
    """

    def __init__(
        self, directory: str | Path, cipher: CipherHook | None = None
    ) -> None:
        """Open or initialize an extent-based ``ObjectStore`` in ``directory``.

        Args:
            directory: Folder holding ``objects.dat`` and ``objects.dir``.
            cipher: Pluggable ``CipherHook`` instance; defaults to ``NoOpCipher()``.

        Raises:
            StorageError: If opening ``objects.dat`` or reading ``objects.dir`` fails.
            PageCorruptedError: If an existing ``objects.dir`` has invalid magic or
                truncated/mismatched byte length.

        Complexity:
            $O(N)$ on reopen where $N$ is the number of live objects in ``objects.dir``.
        """
        self.directory: Path = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cipher: CipherHook = NoOpCipher() if cipher is None else cipher
        self._lock: threading.RLock = threading.RLock()

        self._dat_path: Path = self.directory / "objects.dat"
        self._dir_path: Path = self.directory / "objects.dir"
        self._directory: dict[int, ObjectMeta] = {}
        self._write_offset: int = 0

        open_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        try:
            self._fd: int | None = os.open(
                os.fspath(self._dat_path), open_flags, 0o644
            )
        except OSError as exc:
            raise StorageError(
                f"Failed to open object data file {self._dat_path}: {exc}"
            ) from exc

        try:
            self._load_directory()
        except Exception:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise

    def _ensure_open(self) -> int:
        """Return the open ``objects.dat`` descriptor or raise ``StorageError``."""
        if self._fd is None:
            raise StorageError(f"ObjectStore in {self.directory} is closed")
        return self._fd

    @staticmethod
    def _validate_oid(oid: int) -> None:
        """Validate that ``oid`` fits in an unsigned 64-bit integer."""
        if not isinstance(oid, int) or isinstance(oid, bool) or oid < 0 or oid > _UINT64_MAX:
            raise ValueError(f"oid must be a uint64 integer, got {oid!r}")

    def _pread_exact(self, offset: int, size: int) -> bytes:
        """Read exactly ``size`` bytes at ``offset`` from ``objects.dat``."""
        fd = self._ensure_open()
        chunks: list[bytes] = []
        remaining = size
        pos = offset
        while remaining > 0:
            try:
                if _HAS_PREAD:
                    chunk = os.pread(fd, remaining, pos)
                else:
                    os.lseek(fd, pos, os.SEEK_SET)
                    chunk = os.read(fd, remaining)
            except OSError as exc:
                raise StorageError(
                    f"Object read failed at offset {pos} in {self._dat_path}: {exc}"
                ) from exc
            if not chunk:
                raise PageCorruptedError(
                    f"Unexpected EOF reading {size} bytes at offset {offset} in {self._dat_path}"
                )
            chunks.append(chunk)
            pos += len(chunk)
            remaining -= len(chunk)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def _pwrite_exact(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        """Write all bytes of ``data`` at ``offset`` in ``objects.dat``."""
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
                    f"Object write failed at offset {pos} in {self._dat_path}: {exc}"
                ) from exc
            if n <= 0:
                raise StorageError(
                    f"Short write ({n} bytes) at offset {pos} in {self._dat_path}"
                )
            written += n

    def _load_directory(self) -> None:
        """Reload ``objects.dir`` if present, or scan ``objects.dat`` on crash recovery."""
        fd = self._ensure_open()
        dat_size = os.fstat(fd).st_size
        self._write_offset = dat_size

        if self._dir_path.exists():
            try:
                dir_bytes = self._dir_path.read_bytes()
            except OSError as exc:
                raise StorageError(
                    f"Failed to read object directory {self._dir_path}: {exc}"
                ) from exc

            if len(dir_bytes) < _DIR_HEADER_SIZE:
                raise PageCorruptedError(
                    f"Truncated object directory header in {self._dir_path}"
                )
            magic, entry_count = _DIR_HEADER_STRUCT.unpack_from(dir_bytes, 0)
            if magic != MAGIC_OBJECT_DIR:
                raise PageCorruptedError(
                    f"Invalid object directory magic {magic!r} in {self._dir_path}"
                )
            expected_len = _DIR_HEADER_SIZE + entry_count * _DIR_ENTRY_SIZE
            if len(dir_bytes) != expected_len:
                raise PageCorruptedError(
                    f"Object directory size mismatch in {self._dir_path}: "
                    f"expected {expected_len}, got {len(dir_bytes)}"
                )
            # Trust objects.dir strictly when present so flushed deletions remain deleted.
            for idx in range(entry_count):
                oid, file_offset, cipher_len, raw_len = _DIR_ENTRY_STRUCT.unpack_from(
                    dir_bytes, _DIR_HEADER_SIZE + idx * _DIR_ENTRY_SIZE
                )
                self._directory[oid] = ObjectMeta(
                    oid=oid,
                    file_offset=file_offset,
                    cipher_len=cipher_len,
                    raw_len=raw_len,
                )
            return

        # Crash-recovery fallback: rebuild directory if objects.dat exists without objects.dir.
        pos = 0
        min_record = _RECORD_HEADER_SIZE + _RECORD_TRAILER_SIZE
        while pos + min_record <= dat_size:
            header = self._pread_exact(pos, _RECORD_HEADER_SIZE)
            magic, oid, raw_len, cipher_len, _, _ = _RECORD_HEADER_STRUCT.unpack(header)
            record_size = _RECORD_HEADER_SIZE + cipher_len + _RECORD_TRAILER_SIZE
            if magic != MAGIC_OBJECT_RECORD or pos + record_size > dat_size:
                break
            tail = self._pread_exact(
                pos + _RECORD_HEADER_SIZE, cipher_len + _RECORD_TRAILER_SIZE
            )
            ciphertext = tail[:cipher_len]
            (stored_crc,) = _RECORD_CRC_STRUCT.unpack_from(tail, cipher_len)
            calc_crc = zlib.crc32(ciphertext, zlib.crc32(header)) & _UINT32_MAX
            if calc_crc != stored_crc:
                break
            self._directory[oid] = ObjectMeta(
                oid=oid,
                file_offset=pos,
                cipher_len=cipher_len,
                raw_len=raw_len,
            )
            pos += record_size

    def put_object(self, oid: int, data: bytes) -> ObjectMeta:
        """Encrypt and append an object extent for ``oid`` into ``objects.dat``.

        Args:
            oid: Unsigned 64-bit object identifier used as AAD for authentication.
            data: Raw plaintext object bytes.

        Returns:
            ``ObjectMeta`` describing the appended extent offset and lengths.

        Raises:
            ValueError: If ``oid`` is outside ``uint64`` bounds or ``data`` is
                invalid / exceeds ``uint32`` length.
            StorageError: If the store is closed or writing to ``objects.dat`` fails.

        Complexity:
            $O(N)$ encryption, CRC32, and sequential append where $N$ is ``len(data)``;
            thread-safe under ``self._lock``.
        """
        self._validate_oid(oid)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        raw = bytes(data)
        raw_len = len(raw)
        if raw_len > _UINT32_MAX:
            raise ValueError(f"Object raw_len {raw_len} exceeds uint32 maximum")

        with self._lock:
            self._ensure_open()
            aad = struct.pack("<Q", oid)
            payload = self._cipher.encrypt(raw, aad)
            cipher_len = len(payload.ciphertext)
            if cipher_len > _UINT32_MAX:
                raise ValueError(f"Ciphertext length {cipher_len} exceeds uint32 maximum")
            if len(payload.nonce) != 12 or len(payload.tag) != 16:
                raise StorageError(
                    f"CipherHook returned invalid nonce/tag sizes: "
                    f"nonce={len(payload.nonce)}, tag={len(payload.tag)}"
                )

            header = _RECORD_HEADER_STRUCT.pack(
                MAGIC_OBJECT_RECORD,
                oid,
                raw_len,
                cipher_len,
                payload.nonce,
                payload.tag,
            )
            crc = zlib.crc32(payload.ciphertext, zlib.crc32(header)) & _UINT32_MAX
            record = header + payload.ciphertext + _RECORD_CRC_STRUCT.pack(crc)

            offset = self._write_offset
            self._pwrite_exact(offset, record)
            self._write_offset = offset + len(record)

            meta = ObjectMeta(
                oid=oid,
                file_offset=offset,
                cipher_len=cipher_len,
                raw_len=raw_len,
            )
            self._directory[oid] = meta
            return meta

    def get_object(self, oid: int) -> bytes:
        """Read, CRC-verify, and AEAD-decrypt the object identified by ``oid``.

        Args:
            oid: Unsigned 64-bit object identifier.

        Returns:
            Verified and decrypted plaintext object bytes.

        Raises:
            ValueError: If ``oid`` is outside ``uint64`` bounds.
            KeyNotFoundError: If ``oid`` is not present in the directory.
            PageCorruptedError: If the record header, lengths, or CRC32 trailer
                fails validation.
            SecurityViolationError: If AEAD tag verification or ``oid`` AAD binding
                fails (tamper or cross-OID swap attack).
            StorageError: If the store is closed or disk I/O fails.

        Complexity:
            $O(N)$ positioned read, CRC32 check, and AEAD decryption where $N$
            is the stored object size; thread-safe under ``self._lock``.
        """
        self._validate_oid(oid)
        with self._lock:
            self._ensure_open()
            meta = self._directory.get(oid)
            if meta is None:
                raise KeyNotFoundError(f"Object {oid} not found in ObjectStore")

            total_size = _RECORD_HEADER_SIZE + meta.cipher_len + _RECORD_TRAILER_SIZE
            record = self._pread_exact(meta.file_offset, total_size)

            magic, rec_oid, rec_raw_len, rec_cipher_len, nonce, tag = (
                _RECORD_HEADER_STRUCT.unpack_from(record, 0)
            )
            if (
                magic != MAGIC_OBJECT_RECORD
                or rec_oid != oid
                or rec_cipher_len != meta.cipher_len
                or rec_raw_len != meta.raw_len
            ):
                raise PageCorruptedError(
                    f"Corrupted object header for oid={oid} at offset {meta.file_offset}"
                )

            body_end = _RECORD_HEADER_SIZE + meta.cipher_len
            (stored_crc,) = _RECORD_CRC_STRUCT.unpack_from(record, body_end)
            calc_crc = zlib.crc32(memoryview(record)[:body_end]) & _UINT32_MAX
            if calc_crc != stored_crc:
                raise PageCorruptedError(
                    f"CRC32 mismatch for oid={oid} at offset {meta.file_offset}"
                )

            ciphertext = record[_RECORD_HEADER_SIZE:body_end]
            aad = struct.pack("<Q", oid)
            return self._cipher.decrypt(
                CipherPayload(nonce=nonce, tag=tag, ciphertext=ciphertext),
                aad=aad,
            )

    def delete_object(self, oid: int) -> bool:
        """Remove ``oid`` from the live object directory.

        Deleted extents remain unreferenced in ``objects.dat`` until compaction,
        while ``flush()`` and ``close()`` persist the removal atomically to
        ``objects.dir``.

        Args:
            oid: Unsigned 64-bit object identifier to delete.

        Returns:
            ``True`` if ``oid`` existed and was removed, ``False`` otherwise.

        Raises:
            ValueError: If ``oid`` is outside ``uint64`` bounds.
            StorageError: If the store is closed.

        Complexity:
            $O(1)$ hash-map deletion; thread-safe under ``self._lock``.
        """
        self._validate_oid(oid)
        with self._lock:
            self._ensure_open()
            return self._directory.pop(oid, None) is not None

    def exists(self, oid: int) -> bool:
        """Return whether ``oid`` is currently present in the live object directory.

        Args:
            oid: Unsigned 64-bit object identifier.

        Returns:
            ``True`` if ``oid`` is indexed, else ``False``.

        Raises:
            ValueError: If ``oid`` is outside ``uint64`` bounds.
            StorageError: If the store is closed.

        Complexity:
            $O(1)$ under ``self._lock``.
        """
        self._validate_oid(oid)
        with self._lock:
            self._ensure_open()
            return oid in self._directory

    def flush(self) -> None:
        """Sync ``objects.dat`` and atomically persist ``objects.dir``.

        Writes the directory snapshot to ``objects.dir.tmp``, calls ``os.fsync``
        on the temporary file, and atomically replaces ``objects.dir`` via
        ``os.replace``.

        Raises:
            StorageError: If the store is closed or syncing/writing fails.

        Complexity:
            ``O(K log K)`` where ``K`` is the number of live objects; thread-safe
            under ``self._lock``.
        """
        with self._lock:
            fd = self._ensure_open()
            try:
                os.fsync(fd)
            except OSError as exc:
                raise StorageError(
                    f"fsync failed for {self._dat_path}: {exc}"
                ) from exc

            entries = sorted(self._directory.values(), key=lambda m: m.oid)
            buf = bytearray(_DIR_HEADER_SIZE + len(entries) * _DIR_ENTRY_SIZE)
            _DIR_HEADER_STRUCT.pack_into(buf, 0, MAGIC_OBJECT_DIR, len(entries))
            for idx, meta in enumerate(entries):
                _DIR_ENTRY_STRUCT.pack_into(
                    buf,
                    _DIR_HEADER_SIZE + idx * _DIR_ENTRY_SIZE,
                    meta.oid,
                    meta.file_offset,
                    meta.cipher_len,
                    meta.raw_len,
                )

            tmp_path = self.directory / "objects.dir.tmp"
            tmp_flags = (
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
            )
            tmp_fd: int | None = None
            try:
                tmp_fd = os.open(os.fspath(tmp_path), tmp_flags, 0o644)
                mv = memoryview(buf)
                written = 0
                while written < len(mv):
                    n = os.write(tmp_fd, mv[written:])
                    if n <= 0:
                        raise StorageError(
                            f"Short write persisting temporary directory {tmp_path}"
                        )
                    written += n
                os.fsync(tmp_fd)
                os.close(tmp_fd)
                tmp_fd = None
                os.replace(tmp_path, self._dir_path)
            except OSError as exc:
                if tmp_fd is not None:
                    os.close(tmp_fd)
                raise StorageError(
                    f"Failed to persist object directory {self._dir_path}: {exc}"
                ) from exc

    def close(self) -> None:
        """Flush ``objects.dat`` and ``objects.dir``, then close the file descriptor.

        Idempotent: calling ``close()`` on an already-closed store is a no-op.

        Raises:
            StorageError: If flushing prior to close fails.

        Complexity:
            ``O(K log K)`` plus OS fsync; thread-safe under ``self._lock``.
        """
        with self._lock:
            if self._fd is None:
                return
            try:
                self.flush()
            finally:
                if self._fd is not None:
                    fd = self._fd
                    self._fd = None
                    os.close(fd)

    def get_stats(self) -> dict[str, Any]:
        """Return summary storage and cipher telemetry for the object store.

        Returns:
            Dictionary with keys ``object_count``, ``data_bytes``,
            ``raw_bytes``, ``cipher_bytes``, and ``algorithm_id``.

        Raises:
            StorageError: If the store is closed or ``fstat`` fails.

        Complexity:
            $O(K)$ summation over live directory entries under ``self._lock``.
        """
        with self._lock:
            fd = self._ensure_open()
            try:
                data_bytes = os.fstat(fd).st_size
            except OSError as exc:
                raise StorageError(
                    f"fstat failed for {self._dat_path}: {exc}"
                ) from exc
            return {
                "object_count": len(self._directory),
                "data_bytes": data_bytes,
                "raw_bytes": sum(m.raw_len for m in self._directory.values()),
                "cipher_bytes": sum(m.cipher_len for m in self._directory.values()),
                "algorithm_id": self._cipher.algorithm_id,
            }

    def __enter__(self) -> ObjectStore:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


__all__ = ["ObjectMeta", "ObjectStore"]
