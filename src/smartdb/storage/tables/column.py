"""True per-column page storage with projection pushdown and vectorized batches."""
from __future__ import annotations

import bisect
import struct
from typing import Any, Callable, Iterator

from ..backends import PageBackend, StorageManager
from ..buffer_pool import BufferPool
from ..pages import HEADER_SIZE, PageHeader, PageType, SlottedPage, read_header, write_header
from ...core import RID, ColumnSpec, TableSchema
from .base import scan_records
from .column_codec import (
    ColumnStream,
    append_fixed,
    append_var,
    decode_var_payload,
    fixed_elem_size,
    fixed_slots_per_page,
    is_deleted_bit,
    read_fixed_cell,
    read_fixed_slice,
    read_live_rowids,
    read_var_slice,
    set_deleted_bit,
)


class ColumnTable:
    """Columnar table storing one PageBackend stream per column + deleted bitmap."""

    def __init__(self, storage_manager: StorageManager, pool: BufferPool, name: str, schema: TableSchema) -> None:
        self.storage_manager, self.pool, self.name, self.schema = storage_manager, pool, name, schema
        self.fields = schema.field_names
        usable = storage_manager.usable_page_size

        self._meta_fid, self._meta_backend = storage_manager.open_file(
            f"table:{name}:col_meta", f"tables/{name}.col_meta.pages"
        )
        self._del_fid, self._del_backend = storage_manager.open_file(
            f"table:{name}:deleted", f"tables/{name}.deleted.pages"
        )
        self._ovf_fid, _ = storage_manager.open_file(
            f"table:{name}:overflow", f"tables/{name}.overflow.pages"
        )
        self._bits_per_page = (usable - HEADER_SIZE) * 8
        self._max_var_inline = usable - HEADER_SIZE - 8
        self._streams: dict[str, ColumnStream] = {
            col.name: self._open_column_stream(col, usable) for col in schema.columns
        }
        self._row_count = self._load_row_count()

    def _load_var_starts(self, fid: int, backend: PageBackend) -> list[int]:
        starts: list[int] = []
        for pid in range(backend.page_count):
            with self.pool.pin_page(pid, file_id=fid) as frame:
                starts.append(read_header(frame.data).prev_page)
        return starts

    def _open_column_stream(self, col: ColumnSpec, usable: int) -> ColumnStream:
        fid, backend = self.storage_manager.open_file(
            f"table:{self.name}:col:{col.name}", f"tables/{self.name}.col_{col.name}.pages"
        )
        elem_size = fixed_elem_size(col, usable)
        if elem_size is not None:
            spp = fixed_slots_per_page(usable, elem_size)
            return ColumnStream(col, fid, backend, self._ovf_fid, self._max_var_inline, fixed_spec=(elem_size, spp, (spp + 7) // 8))
        return ColumnStream(col, fid, backend, self._ovf_fid, self._max_var_inline, var_starts=self._load_var_starts(fid, backend))

    def _load_row_count(self) -> int:
        if self._meta_backend.page_count == 0:
            return 0
        with self.pool.pin_page(0, file_id=self._meta_fid) as frame:
            return struct.unpack_from("<Q", frame.data, HEADER_SIZE)[0]

    def _save_row_count(self, count: int) -> None:
        if self._meta_backend.page_count == 0:
            pid, frame = self.pool.new_page(file_id=self._meta_fid)
            write_header(frame.data, PageHeader(int(PageType.COL_META), 0, 1, HEADER_SIZE + 8, len(frame.data)))
            struct.pack_into("<Q", frame.data, HEADER_SIZE, count)
            self.pool.unpin_page(pid, is_dirty=True, file_id=self._meta_fid)
            return
        with self.pool.pin_page(0, file_id=self._meta_fid, dirty=True) as frame:
            struct.pack_into("<Q", frame.data, HEADER_SIZE, count)

    @property
    def page_count(self) -> int:
        return sum(s.backend.page_count for s in self._streams.values()) + self._del_backend.page_count

    def column_page_count(self, field: str) -> int:
        return self._streams[field].backend.page_count

    def insert(self, record: dict[str, Any]) -> RID:
        validated = self.schema.validate_record(record)
        rowid = self._row_count
        set_deleted_bit(self.pool, self._del_fid, self._del_backend, self._bits_per_page, rowid, False)
        for col in self.schema.columns:
            stream = self._streams[col.name]
            if stream.fixed_spec is not None:
                append_fixed(self.pool, stream, rowid, validated[col.name])
            else:
                append_var(self.pool, stream, rowid, validated[col.name])
        self._row_count = rowid + 1
        self._save_row_count(self._row_count)
        return 0, rowid

    def _read_cell(self, stream: ColumnStream, rowid: int) -> Any:
        if stream.fixed_spec is not None:
            _, spp, _ = stream.fixed_spec
            with self.pool.pin_page(rowid // spp, file_id=stream.file_id) as frame:
                return read_fixed_cell(frame.data, stream.col, stream.fixed_spec, rowid)
        pid = bisect.bisect_right(stream.var_starts, rowid) - 1
        with self.pool.pin_page(pid, file_id=stream.file_id) as frame:
            payload = SlottedPage.get_payload(frame.data, rowid - stream.var_starts[pid])
        if payload is None:
            raise IOError(f"Corrupt COL_VAR page {pid}: missing slot for rowid {rowid}")
        return decode_var_payload(self.pool, self._ovf_fid, stream.col, payload)

    def _live_rowid(self, rid: RID) -> int | None:
        chunk_id, rowid = int(rid[0]), int(rid[1])
        if chunk_id != 0 or rowid < 0 or rowid >= self._row_count:
            return None
        if is_deleted_bit(self.pool, self._del_fid, self._del_backend, self._bits_per_page, rowid):
            return None
        return rowid

    def try_get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any] | None:
        rowid = self._live_rowid(rid)
        if rowid is None:
            return None
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        return {col.name: self._read_cell(self._streams[col.name], rowid) for col in cols}

    def get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any]:
        rec = self.try_get_record(rid, fields=fields)
        if rec is None:
            raise IndexError(f"Record locator {rid} does not exist")
        return rec

    def delete(self, rid: RID) -> bool:
        rowid = self._live_rowid(rid)
        if rowid is None:
            return False
        set_deleted_bit(self.pool, self._del_fid, self._del_backend, self._bits_per_page, rowid, True)
        return True

    def update(self, rid: RID, updates: dict[str, Any]) -> RID:
        merged = {**self.get_record(rid), **updates}
        self.schema.validate_record(merged)
        self.delete(rid)
        return self.insert(merged)

    def _read_column_slice(self, col_name: str, rowids: list[int]) -> list[Any]:
        if not rowids:
            return []
        stream = self._streams[col_name]
        if stream.fixed_spec is not None:
            return read_fixed_slice(self.pool, stream, rowids)
        return read_var_slice(self.pool, stream, rowids)

    def scan_batches(
        self, fields: list[str] | None = None, batch_size: int = 1024
    ) -> Iterator[dict[str, list[Any]]]:
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        for start in range(0, self._row_count, batch_size):
            live = read_live_rowids(self.pool, self._del_fid, self._bits_per_page, start, min(start + batch_size, self._row_count))
            if live:
                yield {col.name: self._read_column_slice(col.name, live) for col in cols}

    def iter_records(self, fields: list[str] | None = None) -> Iterator[tuple[RID, dict[str, Any]]]:
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        for start in range(0, self._row_count, 1024):
            live = read_live_rowids(self.pool, self._del_fid, self._bits_per_page, start, min(start + 1024, self._row_count))
            if not live:
                continue
            col_data = {col.name: self._read_column_slice(col.name, live) for col in cols}
            for i, rowid in enumerate(live):
                yield (0, rowid), {col.name: col_data[col.name][i] for col in cols}

    def scan(
        self, fields: list[str] | None = None, predicate: Callable[[dict[str, Any]], bool] | None = None
    ) -> list[dict[str, Any]]:
        return scan_records(self.iter_records, fields, predicate)

    def flush(self) -> None:
        for fid in (self._meta_fid, self._del_fid, self._ovf_fid, *(s.file_id for s in self._streams.values())):
            self.pool.flush_file(fid)
