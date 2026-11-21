"""Page-backed table with stable (page_id, row_offset) record locators."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator
from .buffer_pool import BufferPool
from .layouts import decode_records, encode_records, empty_page
from .page_file import PageBackend, PageFile
from .object_store import LocalObjectStore
from .encryption import TransformingBackend


class StorageTable:
    def __init__(self, path: str | Path, fields: list[str], layout: str = "row",
                 buffer_frames: int = 16, replacement_policy: str = "LRU",
                 backend_kind: str = "page", encrypt_page=None, decrypt_page=None):
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("fields must be a non-empty list of unique names")
        if layout not in {"row", "column"}:
            raise ValueError("layout must be 'row' or 'column'")
        self.fields = list(fields)
        self.layout = layout
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if backend_kind == "page":
            backend: PageBackend = PageFile(self.path)
        elif backend_kind == "objects":
            # Use a sibling directory because objects are separate files.
            backend = LocalObjectStore(self.path.with_suffix(self.path.suffix + ".objects"))
        else:
            raise ValueError("backend_kind must be 'page' or 'objects'")
        self.backend_kind = backend_kind
        if encrypt_page is not None or decrypt_page is not None:
            backend = TransformingBackend(backend, encrypt=encrypt_page, decrypt=decrypt_page)
        self.pool = BufferPool(backend, capacity=buffer_frames, policy=replacement_policy)

    def insert(self, record: dict[str, Any]) -> tuple[int, int]:
        unknown = set(record) - set(self.fields)
        missing = set(self.fields) - set(record)
        if unknown or missing:
            raise ValueError(f"Record fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        normalized = {field: record[field] for field in self.fields}
        if self.pool.page_count == 0:
            self.pool.allocate_page()
        page_id = self.pool.page_count - 1
        current = decode_records(self.pool.read_page(page_id))
        try:
            encoded = encode_records(current + [normalized], self.layout)
            row_offset = len(current)
            self.pool.write_page(page_id, encoded)
            return page_id, row_offset
        except OverflowError:
            page_id = self.pool.allocate_page()
            self.pool.write_page(page_id, encode_records([normalized], self.layout))
            return page_id, 0

    def get_record(self, locator: tuple[int, int] | list[int]) -> dict[str, Any]:
        page_id, row_offset = int(locator[0]), int(locator[1])
        records = decode_records(self.pool.read_page(page_id))
        if row_offset < 0 or row_offset >= len(records):
            raise IndexError(f"Record locator ({page_id}, {row_offset}) does not exist")
        return records[row_offset]

    def iter_records(self) -> Iterator[tuple[tuple[int, int], dict[str, Any]]]:
        for page_id in range(self.pool.page_count):
            records = decode_records(self.pool.read_page(page_id))
            for row_offset, record in enumerate(records):
                yield (page_id, row_offset), record

    def scan(self, field: str | None = None, value: Any = None) -> list[dict[str, Any]]:
        if field is not None and field not in self.fields:
            raise KeyError(f"Unknown field: {field}")
        results = []
        for _, record in self.iter_records():
            if field is None or record.get(field) == value:
                results.append(record)
        return results

    def flush(self) -> None:
        self.pool.flush()

    def close(self) -> None:
        self.pool.close()
