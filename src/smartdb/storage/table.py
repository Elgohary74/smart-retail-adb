"""Page-backed table facade coordinating StorageManager and LayoutManager."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

from .buffer_pool import BufferPool
from .encryption import TransformingBackend
from .layouts import ColumnLayout, RowLayout
from .object_store import LocalObjectStore
from .page_file import PageBackend, PageFile
from .schema import ColumnSpec, TableSchema


class StorageTable:
    """High-level table API over a shared or local BufferPool and Row/Column layout."""

    def __init__(
        self,
        path: str | Path,
        fields: TableSchema | Sequence[ColumnSpec],
        layout: str = "row",
        buffer_frames: int = 16,
        replacement_policy: str = "LRU",
        backend_kind: str = "page",
        encrypt_page=None,
        decrypt_page=None,
        pool: BufferPool | None = None,
        stream_prefix: str | None = None,
    ):
        self.schema = fields if isinstance(fields, TableSchema) else TableSchema.from_spec(fields)
        self.fields = self.schema.field_names
        normalized_layout = layout.lower()
        if normalized_layout not in {"row", "column"}:
            raise ValueError("layout must be 'row' or 'column'")
        if backend_kind not in {"page", "objects"}:
            raise ValueError("backend_kind must be 'page' or 'objects'")

        self.layout = normalized_layout
        self.backend_kind = backend_kind
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._encrypt_page = encrypt_page
        self._decrypt_page = decrypt_page
        self._owns_pool = pool is None
        self.pool = pool if pool is not None else BufferPool(capacity=buffer_frames, policy=replacement_policy)
        self._prefix = stream_prefix or str(self.path)
        self._streams: list[str] = []

        if self.layout == "row":
            stream = f"{self._prefix}.data"
            self._register_stream(stream, self.path)
            self._layout_impl: RowLayout | ColumnLayout = RowLayout(self.pool, self._prefix, self.schema)
        else:
            col_dir = self.path.with_suffix("")
            col_dir.mkdir(parents=True, exist_ok=True)
            for col in self.fields:
                self._register_stream(f"{self._prefix}.col.{col}", col_dir / f"{col}.pages")
            self._register_stream(f"{self._prefix}._tombstones", col_dir / "_tombstones.pages")
            self._layout_impl = ColumnLayout(self.pool, self._prefix, self.schema)

    def _open_backend(self, target: Path) -> PageBackend:
        if self.backend_kind == "page":
            backend: PageBackend = PageFile(target)
        else:
            backend = LocalObjectStore(target.with_suffix(target.suffix + ".objects"))
        if self._encrypt_page is not None or self._decrypt_page is not None:
            backend = TransformingBackend(backend, encrypt=self._encrypt_page, decrypt=self._decrypt_page)
        return backend

    def _register_stream(self, stream_id: str, target: Path) -> None:
        backend = self._open_backend(target)
        self.pool.register_backend(stream_id, backend)
        self._streams.append(stream_id)

    def insert(self, record: dict[str, Any]) -> tuple[int, int]:
        return self._layout_impl.insert(record)

    def get_record(self, locator: tuple[int, int] | list[int]) -> dict[str, Any]:
        return self._layout_impl.get_record(locator)

    def update(self, locator: tuple[int, int] | list[int], record: dict[str, Any]) -> tuple[int, int]:
        return self._layout_impl.update(locator, record)

    def delete(self, locator: tuple[int, int] | list[int]) -> None:
        self._layout_impl.delete(locator)

    def iter_records(
        self, columns: Sequence[str] | None = None
    ) -> Iterator[tuple[tuple[int, int], dict[str, Any]]]:
        return self._layout_impl.iter_records(columns=columns)

    def iter_batches(
        self, columns: Sequence[str] | None = None, batch_size: int = 256
    ) -> Iterator[dict[str, list[Any]]]:
        return self._layout_impl.iter_batches(columns=columns, batch_size=batch_size)

    def scan(
        self,
        field: str | None = None,
        value: Any = None,
        columns: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if field is not None and field not in self.fields:
            raise KeyError(f"Unknown field: {field}")
        if field is None:
            return [record for _, record in self.iter_records(columns=columns)]

        # Ensure filter field is read during scan, then project requested output columns.
        read_cols = None
        if columns is not None:
            read_cols = list(columns) if field in columns else list(columns) + [field]
        results: list[dict[str, Any]] = []
        for _, record in self.iter_records(columns=read_cols):
            if record.get(field) == value:
                if columns is not None and field not in columns:
                    results.append({col: record[col] for col in columns})
                else:
                    results.append(record)
        return results

    def flush(self) -> None:
        for stream in self._streams:
            self.pool.flush(stream_id=stream)

    def close(self) -> None:
        if self._owns_pool:
            self.pool.close()
        else:
            for stream in list(self._streams):
                self.pool.unregister_backend(stream, close_backend=True)
            self._streams.clear()
