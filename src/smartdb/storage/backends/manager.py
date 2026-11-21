"""Unified StorageManager mapping logical DB files to PageBackends."""
from __future__ import annotations

from pathlib import Path

from ...core.config import DEFAULT_BACKEND_KIND, PAGE_SIZE
from .encryption import PageTransform, RecordTransform, TransformingBackend
from .object_store import LocalObjectStore
from .page_file import PageBackend, PageFile


class StorageManager:
    """Opens, tracks, wraps (encryption/object store), and drops PageBackends."""

    def __init__(
        self,
        root: str | Path,
        backend_kind: str = DEFAULT_BACKEND_KIND,
        encrypt_page: PageTransform | None = None,
        decrypt_page: PageTransform | None = None,
        crypto_overhead: int = 0,
        encrypt_record: RecordTransform | None = None,
        decrypt_record: RecordTransform | None = None,
        physical_page_size: int = PAGE_SIZE,
    ):
        if backend_kind not in {"page", "objects"}:
            raise ValueError("backend_kind must be 'page' or 'objects'")
        if (encrypt_page is None) != (decrypt_page is None):
            raise ValueError("Provide both encrypt_page and decrypt_page, or neither")
        if (encrypt_record is None) != (decrypt_record is None):
            raise ValueError("Provide both encrypt_record and decrypt_record, or neither")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend_kind = backend_kind
        self._encrypt_page = encrypt_page
        self._decrypt_page = decrypt_page
        self.crypto_overhead = crypto_overhead if encrypt_page is not None else 0
        self._encrypt_record = encrypt_record
        self._decrypt_record = decrypt_record
        self.physical_page_size = physical_page_size
        self.usable_page_size = physical_page_size - self.crypto_overhead

        self._next_file_id = 0
        self._name_to_id: dict[str, int] = {}
        self._backends: dict[int, PageBackend] = {}

    def open_file(self, logical_name: str, rel_path: str, backend_kind: str | None = None) -> tuple[int, PageBackend]:
        if logical_name in self._name_to_id:
            fid = self._name_to_id[logical_name]
            return fid, self._backends[fid]
        kind = backend_kind or self.backend_kind
        target = self.root / rel_path
        if kind == "page":
            raw: PageBackend = PageFile(target, page_size=self.physical_page_size)
        elif kind == "objects":
            obj_dir = target.with_suffix(target.suffix + ".objects")
            raw = LocalObjectStore(obj_dir, page_size=self.physical_page_size)
        else:
            raise ValueError("backend_kind must be 'page' or 'objects'")

        backend: PageBackend = (
            TransformingBackend(
                raw,
                encrypt=self._encrypt_page,
                decrypt=self._decrypt_page,
                overhead_bytes=self.crypto_overhead,
            )
            if self._encrypt_page is not None
            else raw
        )
        fid = self._next_file_id
        self._next_file_id += 1
        self._name_to_id[logical_name] = fid
        self._backends[fid] = backend
        return fid, backend

    def get_backend(self, file_id: int) -> PageBackend:
        if file_id not in self._backends:
            raise KeyError(f"Unknown file_id: {file_id}")
        return self._backends[file_id]

    def file_id_for(self, logical_name: str) -> int | None:
        return self._name_to_id.get(logical_name)

    def drop_file(self, logical_name: str) -> None:
        fid = self._name_to_id.pop(logical_name, None)
        if fid is None:
            return
        backend = self._backends.pop(fid)
        backend.destroy()

    def encrypt_record(self, data: bytes) -> bytes:
        return self._encrypt_record(data) if self._encrypt_record is not None else data

    def decrypt_record(self, data: bytes) -> bytes:
        return self._decrypt_record(data) if self._decrypt_record is not None else data

    def close_all(self) -> None:
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        self._name_to_id.clear()
