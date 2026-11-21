"""Fixed-size persistent page files for the Smart Retail DB storage engine."""
from __future__ import annotations

from pathlib import Path

from ...core.config import PAGE_SIZE


class PageFile:
    """A contiguous file of fixed-size pages. Page IDs start at zero."""

    def __init__(self, path: str | Path, page_size: int = PAGE_SIZE):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.page_size = page_size
        self._file = self.path.open("r+b" if self.path.exists() else "w+b")
        self._file.seek(0, 2)
        if self._file.tell() % self.page_size:
            self._file.close()
            raise ValueError(f"File size is not a multiple of page size: {self.path}")

    @property
    def physical_page_size(self) -> int:
        return self.page_size

    @property
    def usable_page_size(self) -> int:
        return self.page_size

    @property
    def page_count(self) -> int:
        self._file.seek(0, 2)
        return self._file.tell() // self.page_size

    def _validate_existing_page(self, page_id: int) -> None:
        if not isinstance(page_id, int) or isinstance(page_id, bool) or page_id < 0:
            raise ValueError("page_id must be a non-negative integer")
        if page_id >= self.page_count:
            raise IndexError(f"Page {page_id} does not exist; page_count={self.page_count}")

    def allocate_page(self) -> int:
        page_id = self.page_count
        self._file.seek(page_id * self.page_size)
        self._file.write(b"\x00" * self.page_size)
        self._file.flush()
        return page_id

    def read_page(self, page_id: int) -> bytes:
        self._validate_existing_page(page_id)
        self._file.seek(page_id * self.page_size)
        data = self._file.read(self.page_size)
        if len(data) != self.page_size:
            raise IOError(f"Short read for page {page_id}")
        return data

    def write_page(self, page_id: int, data: bytes) -> None:
        self._validate_existing_page(page_id)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("page data must be bytes-like")
        if len(data) != self.page_size:
            raise ValueError(f"Page must be exactly {self.page_size} bytes; got {len(data)}")
        self._file.seek(page_id * self.page_size)
        self._file.write(data)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def destroy(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> "PageFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
