"""Buffered framed Write-Ahead Log for LSMTree memtable durability."""
from __future__ import annotations

import os
import struct
from pathlib import Path

from ...core import RID
from ...storage import StorageManager
from .sstable import OpList

_FRAME_LEN_STRUCT = "<I"
_FRAME_LEN_SIZE = struct.calcsize(_FRAME_LEN_STRUCT)
_WAL_OP_STRUCT = "<BQiiH"
_WAL_OP_SIZE = struct.calcsize(_WAL_OP_STRUCT)


class LSMWal:
    """Appends, replays, truncates, and deletes encrypted or plain memtable ops."""

    def __init__(self, path: Path, storage_manager: StorageManager):
        self.path = path
        self.storage_manager = storage_manager
        self._file = self.path.open("ab")

    def replay(self) -> tuple[dict[bytes, OpList], int, int]:
        """Replay valid WAL frames into (memtable, entry_count, max_seq)."""
        memtable: dict[bytes, OpList] = {}
        entries = 0
        max_seq = 0
        with self.path.open("rb") as f:
            while len(hdr := f.read(_FRAME_LEN_SIZE)) == _FRAME_LEN_SIZE:
                length = struct.unpack(_FRAME_LEN_STRUCT, hdr)[0]
                enc = f.read(length)
                if len(enc) < length:
                    break
                raw = self.storage_manager.decrypt_record(enc)
                if len(raw) < _WAL_OP_SIZE:
                    raise IOError("Corrupt LSM WAL record payload")
                op_flag, seq, pid, sid, klen = struct.unpack_from(_WAL_OP_STRUCT, raw, 0)
                kb = raw[_WAL_OP_SIZE:_WAL_OP_SIZE + klen]
                rid: RID | None = None if pid == -1 else (pid, sid)
                memtable.setdefault(kb, []).append((rid, seq, op_flag == 1))
                entries += 1
                max_seq = max(max_seq, seq)
        return memtable, entries, max_seq

    def append(self, key_bytes: bytes, rid: RID | None, seq: int, is_deleted: bool) -> None:
        """Append one framed mutation record to the WAL."""
        pid = -1 if rid is None else rid[0]
        sid = 0 if rid is None else rid[1]
        raw = struct.pack(_WAL_OP_STRUCT, 1 if is_deleted else 0, seq, pid, sid, len(key_bytes)) + key_bytes
        enc = self.storage_manager.encrypt_record(raw)
        self._file.write(struct.pack(_FRAME_LEN_STRUCT, len(enc)) + enc)
        self._file.flush()

    def truncate(self) -> None:
        """Fsync and reset the WAL after memtable state has been persisted to an SSTable."""
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self.path.write_bytes(b"")
        self._file = self.path.open("ab")

    def close(self) -> None:
        """Flush and close the underlying WAL file handle."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def destroy(self) -> None:
        """Close the WAL handle and unlink its file from disk."""
        self.close()
        self.path.unlink(missing_ok=True)

    @property
    def closed(self) -> bool:
        """Return True if the WAL file handle is closed."""
        return self._file.closed
