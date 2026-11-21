"""Unified StorageManager mapping logical DB files to PageBackends."""
from __future__ import annotations

from pathlib import Path

from ...core.config import DEFAULT_BACKEND_KIND, PAGE_SIZE
from .base import PageBackend
from .encryption import CryptoHooks, TransformingBackend
from .object_store import LocalObjectStore
from .page_file import PageFile


class StorageManager:
    """Opens, tracks, wraps (encryption/object store), and drops PageBackends."""

    def __init__(
        self,
        root: str | Path,
        backend_kind: str = DEFAULT_BACKEND_KIND,
        crypto: CryptoHooks | None = None,
        physical_page_size: int = PAGE_SIZE,
    ):
        if backend_kind not in {"page", "objects"}:
            raise ValueError("backend_kind must be 'page' or 'objects'")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend_kind = backend_kind
        self.crypto = crypto or CryptoHooks()
        self.crypto_overhead = self.crypto.overhead_bytes if self.crypto.page_encrypt is not None else 0
        self.physical_page_size = physical_page_size
        self.usable_page_size = physical_page_size - self.crypto_overhead

        self._next_file_id = 0
        self._name_to_id: dict[str, int] = {}
        self._backends: dict[int, PageBackend] = {}

    def _create_raw_backend(self, rel_path: str, kind: str) -> PageBackend:
        target = self.root / rel_path
        if kind == "page":
            return PageFile(target, page_size=self.physical_page_size)
        if kind == "objects":
            return LocalObjectStore(target.with_suffix(target.suffix + ".objects"), page_size=self.physical_page_size)
        raise ValueError("backend_kind must be 'page' or 'objects'")

    def open_file(self, logical_name: str, rel_path: str, backend_kind: str | None = None) -> tuple[int, PageBackend]:
        if logical_name in self._name_to_id:
            fid = self._name_to_id[logical_name]
            return fid, self._backends[fid]
        raw = self._create_raw_backend(rel_path, backend_kind or self.backend_kind)
        backend: PageBackend = (
            TransformingBackend(
                raw,
                encrypt=self.crypto.page_encrypt,
                decrypt=self.crypto.page_decrypt,
                overhead_bytes=self.crypto_overhead,
            )
            if self.crypto.page_encrypt is not None
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
        self._backends.pop(fid).destroy()

    def encrypt_record(self, data: bytes) -> bytes:
        return self.crypto.record_encrypt(data) if self.crypto.record_encrypt is not None else data

    def decrypt_record(self, data: bytes) -> bytes:
        return self.crypto.record_decrypt(data) if self.crypto.record_decrypt is not None else data

    def close_all(self) -> None:
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        self._name_to_id.clear()
