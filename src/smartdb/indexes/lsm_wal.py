"""Buffered framed Write-Ahead Log for LSMTree memtable durability."""
from __future__ import annotations

import os
import struct
from pathlib import Path

from ..schema import RID
from ..storage.manager import StorageManager
from .sstable import OpList

_WAL_OP_STRUCT = "<BQiiH"
_WAL_OP_SIZE = struct.calcsize(_WAL_OP_STRUCT)


class LSMWal:
    """Appends, replays, and truncates encrypted or plain memtable ops."""

    def __init__(self, path: Path, storage_manager: StorageManager):
        self.path = path
        self.storage_manager = storage_manager
        self._file = self.path.open("ab")

    def replay(self) -> tuple[dict[bytes, OpList], int, int]:
        memtable: dict[bytes, OpList] = {}
        entries = 0
        max_seq = 0
        if not self.path.exists():
            return memtable, entries, max_seq
        with self.path.open("rb") as f:
            while len(hdr := f.read(4)) == 4:
                length = struct.unpack("<I", hdr)[0]
                enc = f.read(length)
                if len(enc) < length:
                    break
                raw = self.storage_manager.decrypt_record(enc)
                op_flag, seq, pid, sid, klen = struct.unpack_from(_WAL_OP_STRUCT, raw, 0)
                kb = raw[_WAL_OP_SIZE:_WAL_OP_SIZE + klen]
                rid: RID | None = None if pid == -1 else (pid, sid)
                memtable.setdefault(kb, []).append((rid, seq, op_flag == 1))
                entries += 1
                max_seq = max(max_seq, seq)
        return memtable, entries, max_seq

    def append(self, key_bytes: bytes, rid: RID | None, seq: int, is_deleted: bool) -> None:
        pid = -1 if rid is None else rid[0]
        sid = 0 if rid is None else rid[1]
        raw = struct.pack(_WAL_OP_STRUCT, 1 if is_deleted else 0, seq, pid, sid, len(key_bytes)) + key_bytes
        enc = self.storage_manager.encrypt_record(raw)
        self._file.write(struct.pack("<I", len(enc)) + enc)
        self._file.flush()

    def truncate(self) -> None:
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self.path.write_bytes(b"")
        self._file = self.path.open("ab")

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def closed(self) -> bool:
        return self._file.closed
