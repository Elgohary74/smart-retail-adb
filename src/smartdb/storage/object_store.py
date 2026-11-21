"""A local object-store-like backend, with one fixed-size object per page.

This is an adapter for demonstrating the M2 object-storage interface. It is
not an S3 client; the interface can later be implemented by an S3-compatible
backend without changing the buffer pool or table layer.
"""
from __future__ import annotations

from pathlib import Path
from .page_file import PAGE_SIZE


class LocalObjectStore:
    def __init__(self, directory: str | Path, page_size: int = PAGE_SIZE):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.page_size = page_size
        self._closed = False
        self._page_count = sum(1 for p in self.directory.glob("page_*.obj") if p.is_file())

    def _path(self, page_id: int) -> Path:
        if not isinstance(page_id, int) or isinstance(page_id, bool) or page_id < 0:
            raise ValueError("page_id must be a non-negative integer")
        return self.directory / f"page_{page_id:012d}.obj"

    @property
    def page_count(self) -> int:
        return self._page_count

    def allocate_page(self) -> int:
        page_id = self._page_count
        # Normally objects are contiguous because this implementation never deletes pages.
        while self._path(page_id).exists():
            page_id += 1
        self._path(page_id).write_bytes(b"\x00" * self.page_size)
        self._page_count = max(self._page_count + 1, page_id + 1)
        return page_id

    def read_page(self, page_id: int) -> bytes:
        path = self._path(page_id)
        if not path.exists():
            raise IndexError(f"Page object {page_id} does not exist")
        data = path.read_bytes()
        if len(data) != self.page_size:
            raise IOError(f"Page object {page_id} has invalid size {len(data)}")
        return data

    def write_page(self, page_id: int, data: bytes) -> None:
        path = self._path(page_id)
        if not path.exists():
            raise IndexError(f"Page object {page_id} does not exist")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("page data must be bytes-like")
        if len(data) != self.page_size:
            raise ValueError(f"Page must be exactly {self.page_size} bytes; got {len(data)}")
        path.write_bytes(bytes(data))

    def close(self) -> None:
        self._closed = True
