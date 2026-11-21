"""True per-column page storage with projection pushdown and vectorized batches."""
from __future__ import annotations

import bisect
import struct
from typing import Any, Callable, Iterator

from ..schema import RID, ColumnSpec, TableSchema, deserialize_value
from .buffer_pool import BufferPool
from .column_codec import (
    append_fixed,
    append_var,
    decode_var_payload,
    fixed_elem_size,
    fixed_slots_per_page,
    is_deleted_bit,
    read_fixed_slice,
    read_live_rowids,
    read_var_slice,
    set_deleted_bit,
)
from .manager import StorageManager
from .slotted_page import HEADER_SIZE, PageHeader, PageType, SlottedPage, read_header, write_header


class ColumnTable:
    """Columnar table storing one PageBackend stream per column + deleted bitmap."""

    def __init__(self, storage_manager: StorageManager, pool: BufferPool, name: str, schema: TableSchema):
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
        self._col_fids: dict[str, int] = {}
        self._col_backends: dict[str, Any] = {}
        self._fixed_info: dict[str, tuple[int, int, int]] = {}
        self._var_dirs: dict[str, list[int]] = {}
        self._init_columns(usable)
        self._row_count = self._load_row_count()

    def _init_columns(self, usable: int) -> None:
        for col in self.schema.columns:
            fid, backend = self.storage_manager.open_file(
                f"table:{self.name}:col:{col.name}", f"tables/{self.name}.col_{col.name}.pages"
            )
            self._col_fids[col.name], self._col_backends[col.name] = fid, backend
            elem_size = fixed_elem_size(col, usable)
            if elem_size is not None:
                spp = fixed_slots_per_page(usable, elem_size)
                self._fixed_info[col.name] = (elem_size, spp, (spp + 7) // 8)
            else:
                starts: list[int] = []
                for pid in range(backend.page_count):
                    with self.pool.pin_page(pid, file_id=fid) as frame:
                        starts.append(read_header(frame.data).prev_page)
                self._var_dirs[col.name] = starts

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
        else:
            with self.pool.pin_page(0, file_id=self._meta_fid, dirty=True) as frame:
                struct.pack_into("<Q", frame.data, HEADER_SIZE, count)

    @property
    def page_count(self) -> int:
        return sum(b.page_count for b in self._col_backends.values()) + self._del_backend.page_count

    def column_page_count(self, field: str) -> int:
        return self._col_backends[field].page_count

    def insert(self, record: dict[str, Any]) -> RID:
        validated = self.schema.validate_record(record)
        rowid = self._row_count
        set_deleted_bit(self.pool, self._del_fid, self._del_backend, self._bits_per_page, rowid, False)
        for col in self.schema.columns:
            fid, backend = self._col_fids[col.name], self._col_backends[col.name]
            if col.name in self._fixed_info:
                append_fixed(self.pool, fid, backend, col, self._fixed_info[col.name], rowid, validated[col.name])
            else:
                append_var(
                    self.pool, fid, backend, self._ovf_fid,
                    self._max_var_inline, self._var_dirs[col.name], col, rowid, validated[col.name],
                )
        self._row_count = rowid + 1
        self._save_row_count(self._row_count)
        return 0, rowid

    def _read_cell(self, col: ColumnSpec, rowid: int) -> Any:
        fid = self._col_fids[col.name]
        if col.name in self._fixed_info:
            elem_size, spp, bmap_len = self._fixed_info[col.name]
            with self.pool.pin_page(rowid // spp, file_id=fid) as frame:
                slot = rowid % spp
                if frame.data[HEADER_SIZE + (slot // 8)] & (1 << (slot % 8)):
                    return None
                return deserialize_value(col.dtype, frame.data, HEADER_SIZE + bmap_len + slot * elem_size, col.dim)[0]
        starts = self._var_dirs[col.name]
        pid = bisect.bisect_right(starts, rowid) - 1
        with self.pool.pin_page(pid, file_id=fid) as frame:
            payload = SlottedPage.get_payload(frame.data, rowid - starts[pid])
        assert payload is not None
        return decode_var_payload(self.pool, self._ovf_fid, col, payload)

    def try_get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any] | None:
        chunk_id, rowid = int(rid[0]), int(rid[1])
        if chunk_id != 0 or rowid < 0 or rowid >= self._row_count:
            return None
        if is_deleted_bit(self.pool, self._del_fid, self._del_backend, self._bits_per_page, rowid):
            return None
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        return {col.name: self._read_cell(col, rowid) for col in cols}

    def get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any]:
        rec = self.try_get_record(rid, fields=fields)
        if rec is None:
            raise IndexError(f"Record locator {rid} does not exist")
        return rec

    def delete(self, rid: RID) -> bool:
        chunk_id, rowid = int(rid[0]), int(rid[1])
        if chunk_id != 0 or rowid < 0 or rowid >= self._row_count:
            return False
        if is_deleted_bit(self.pool, self._del_fid, self._del_backend, self._bits_per_page, rowid):
            return False
        set_deleted_bit(self.pool, self._del_fid, self._del_backend, self._bits_per_page, rowid, True)
        return True

    def update(self, rid: RID, updates: dict[str, Any]) -> RID:
        merged = {**self.get_record(rid), **updates}
        self.schema.validate_record(merged)
        self.delete(rid)
        return self.insert(merged)

    def _read_column_slice(self, col: ColumnSpec, rowids: list[int]) -> list[Any]:
        if not rowids:
            return []
        fid = self._col_fids[col.name]
        if col.name in self._fixed_info:
            return read_fixed_slice(self.pool, fid, col, self._fixed_info[col.name], rowids)
        return read_var_slice(self.pool, fid, self._ovf_fid, col, self._var_dirs[col.name], rowids)

    def scan_batches(
        self, fields: list[str] | None = None, batch_size: int = 1024
    ) -> Iterator[dict[str, list[Any]]]:
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        for start in range(0, self._row_count, batch_size):
            live = read_live_rowids(self.pool, self._del_fid, self._bits_per_page, start, min(start + batch_size, self._row_count))
            if live:
                yield {col.name: self._read_column_slice(col, live) for col in cols}

    def iter_records(self, fields: list[str] | None = None) -> Iterator[tuple[RID, dict[str, Any]]]:
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        for start in range(0, self._row_count, 1024):
            live = read_live_rowids(self.pool, self._del_fid, self._bits_per_page, start, min(start + 1024, self._row_count))
            if not live:
                continue
            col_data = {col.name: self._read_column_slice(col, live) for col in cols}
            for i, rowid in enumerate(live):
                yield (0, rowid), {col.name: col_data[col.name][i] for col in cols}

    def scan(
        self, fields: list[str] | None = None, predicate: Callable[[dict[str, Any]], bool] | None = None
    ) -> list[dict[str, Any]]:
        return [
            (rec if fields is None else {f: rec[f] for f in fields})
            for _, rec in self.iter_records(fields=fields if predicate is None else None)
            if predicate is None or predicate(rec)
        ]

    def flush(self) -> None:
        for fid in (self._meta_fid, self._del_fid, self._ovf_fid, *self._col_fids.values()):
            self.pool.flush_file(fid)
