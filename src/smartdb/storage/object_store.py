"""Local object-store-like backend storing one fixed-size object per page."""
from __future__ import annotations

import shutil
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
        max_id = -1
        for path in self.directory.glob("page_*.obj"):
            if path.is_file():
                try:
                    max_id = max(max_id, int(path.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        self._page_count = max_id + 1

    @property
    def physical_page_size(self) -> int:
        return self.page_size

    @property
    def usable_page_size(self) -> int:
        return self.page_size

    @property
    def page_count(self) -> int:
        return self._page_count

    def _path(self, page_id: int) -> Path:
        if not isinstance(page_id, int) or isinstance(page_id, bool) or page_id < 0:
            raise ValueError("page_id must be a non-negative integer")
        return self.directory / f"page_{page_id:012d}.obj"

    def allocate_page(self) -> int:
        page_id = self._page_count
        while self._path(page_id).exists():
            page_id += 1
        self._path(page_id).write_bytes(b"\x00" * self.page_size)
        self._page_count = page_id + 1
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

    def destroy(self) -> None:
        self.close()
        if self.directory.exists():
            shutil.rmtree(self.directory, ignore_errors=True)
        self._page_count = 0
