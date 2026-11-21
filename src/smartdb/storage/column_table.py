"""True per-column page storage with projection pushdown and vectorized batches."""
from __future__ import annotations

import bisect
import struct
from typing import Any, Callable, Iterator

from ..schema import (
    RID,
    ColumnSpec,
    DataType,
    TableSchema,
    deserialize_value,
    serialize_value,
)
from .buffer_pool import BufferPool
from .manager import StorageManager
from .overflow import read_overflow_chain, write_overflow_chain
from .slotted_page import (
    HEADER_SIZE,
    PageHeader,
    PageType,
    SlottedPage,
    read_header,
    write_header,
)

_OVF_MAGIC = b"\xffOVF"
_OVF_STUB_STRUCT = "<4sIi"
_OVF_STUB_SIZE = struct.calcsize(_OVF_STUB_STRUCT)


def _fixed_elem_size(col: ColumnSpec, usable_page_size: int) -> int | None:
    if col.dtype == DataType.INT or col.dtype == DataType.FLOAT:
        return 8
    if col.dtype == DataType.BOOL:
        return 1
    if col.dtype == DataType.VECTOR and col.dim is not None:
        size = 4 * col.dim
        if size <= usable_page_size - HEADER_SIZE - 64:
            return size
    return None


def _fixed_slots_per_page(usable_page_size: int, elem_size: int) -> int:
    avail = usable_page_size - HEADER_SIZE
    # Solve ceil(s/8) + s * elem_size <= avail
    s = (avail * 8) // (8 * elem_size + 1)
    while ((s + 7) // 8) + s * elem_size > avail:
        s -= 1
    return max(1, s)


class ColumnTable:
    """Columnar table storing one PageBackend stream per column + deleted bitmap."""

    def __init__(self, storage_manager: StorageManager, pool: BufferPool, name: str, schema: TableSchema):
        self.storage_manager = storage_manager
        self.pool = pool
        self.name = name
        self.schema = schema
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

        self._col_fids: dict[str, int] = {}
        self._col_backends: dict[str, Any] = {}
        self._fixed_info: dict[str, tuple[int, int, int]] = {}  # col -> (elem_size, slots_per_page, bmap_bytes)
        self._var_dirs: dict[str, list[int]] = {}  # col -> [first_rowid per page]
        self._max_var_inline = usable - HEADER_SIZE - 8

        for col in schema.columns:
            fid, backend = storage_manager.open_file(
                f"table:{name}:col:{col.name}", f"tables/{name}.col_{col.name}.pages"
            )
            self._col_fids[col.name] = fid
            self._col_backends[col.name] = backend
            elem_size = _fixed_elem_size(col, usable)
            if elem_size is not None:
                spp = _fixed_slots_per_page(usable, elem_size)
                self._fixed_info[col.name] = (elem_size, spp, (spp + 7) // 8)
            else:
                starts: list[int] = []
                for pid in range(backend.page_count):
                    with pool.pin_page(pid, file_id=fid) as frame:
                        starts.append(read_header(frame.data).prev_page)
                self._var_dirs[col.name] = starts

        self._row_count = self._load_row_count()

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

    def _is_deleted(self, rowid: int) -> bool:
        pid = rowid // self._bits_per_page
        bit = rowid % self._bits_per_page
        if pid >= self._del_backend.page_count:
            return True
        with self.pool.pin_page(pid, file_id=self._del_fid) as frame:
            return bool(frame.data[HEADER_SIZE + (bit // 8)] & (1 << (bit % 8)))

    def _set_deleted_bit(self, rowid: int, deleted: bool) -> None:
        pid = rowid // self._bits_per_page
        bit = rowid % self._bits_per_page
        while self._del_backend.page_count <= pid:
            npid, frame = self.pool.new_page(file_id=self._del_fid)
            write_header(frame.data, PageHeader(int(PageType.BITMAP), 0, 0, HEADER_SIZE, len(frame.data)))
            self.pool.unpin_page(npid, is_dirty=True, file_id=self._del_fid)
        with self.pool.pin_page(pid, file_id=self._del_fid, dirty=True) as frame:
            byte_idx = HEADER_SIZE + (bit // 8)
            if deleted:
                frame.data[byte_idx] |= 1 << (bit % 8)
            else:
                frame.data[byte_idx] &= ~(1 << (bit % 8))

    def _append_fixed(self, col: ColumnSpec, rowid: int, value: Any) -> None:
        fid = self._col_fids[col.name]
        backend = self._col_backends[col.name]
        elem_size, spp, bmap_len = self._fixed_info[col.name]
        pid = rowid // spp
        slot = rowid % spp
        while backend.page_count <= pid:
            npid, frame = self.pool.new_page(file_id=fid)
            write_header(frame.data, PageHeader(int(PageType.COL_FIXED), 0, 0, HEADER_SIZE + bmap_len, len(frame.data)))
            self.pool.unpin_page(npid, is_dirty=True, file_id=fid)
        with self.pool.pin_page(pid, file_id=fid, dirty=True) as frame:
            if value is None:
                frame.data[HEADER_SIZE + (slot // 8)] |= 1 << (slot % 8)
            else:
                frame.data[HEADER_SIZE + (slot // 8)] &= ~(1 << (slot % 8))
                raw = serialize_value(col.dtype, value, col.dim)
                off = HEADER_SIZE + bmap_len + slot * elem_size
                frame.data[off:off + elem_size] = raw
            hdr = read_header(frame.data)
            hdr.item_count = max(hdr.item_count, slot + 1)
            write_header(frame.data, hdr)

    def _append_var(self, col: ColumnSpec, rowid: int, value: Any) -> None:
        fid = self._col_fids[col.name]
        backend = self._col_backends[col.name]
        if value is None:
            payload = b"\x00"
        else:
            raw = b"\x01" + serialize_value(col.dtype, value, col.dim)
            if len(raw) <= self._max_var_inline:
                payload = raw
            else:
                first_ovf = write_overflow_chain(self.pool, self._ovf_fid, raw)
                payload = b"\x02" + struct.pack(_OVF_STUB_STRUCT, _OVF_MAGIC, len(raw), first_ovf)
        if backend.page_count > 0:
            last_pid = backend.page_count - 1
            with self.pool.pin_page(last_pid, file_id=fid, dirty=True) as frame:
                if SlottedPage.insert_payload(frame.data, payload) is not None:
                    return
        npid, frame = self.pool.new_page(file_id=fid)
        try:
            SlottedPage.init(frame.data, PageType.COL_VAR)
            hdr = read_header(frame.data)
            hdr.prev_page = rowid
            write_header(frame.data, hdr)
            if SlottedPage.insert_payload(frame.data, payload) is None:
                raise OverflowError("Column value stub too large")
            self._var_dirs[col.name].append(rowid)
        finally:
            self.pool.unpin_page(npid, is_dirty=True, file_id=fid)

    def _read_fixed(self, col: ColumnSpec, rowid: int) -> Any:
        fid = self._col_fids[col.name]
        elem_size, spp, bmap_len = self._fixed_info[col.name]
        pid = rowid // spp
        slot = rowid % spp
        with self.pool.pin_page(pid, file_id=fid) as frame:
            if frame.data[HEADER_SIZE + (slot // 8)] & (1 << (slot % 8)):
                return None
            off = HEADER_SIZE + bmap_len + slot * elem_size
            val, _ = deserialize_value(col.dtype, frame.data, off, col.dim)
            return val

    def _decode_var_payload(self, col: ColumnSpec, payload: bytes) -> Any:
        if payload[0] == 0:
            return None
        if payload[0] == 2:
            _, total_len, first_ovf = struct.unpack(_OVF_STUB_STRUCT, payload[1:])
            payload = read_overflow_chain(self.pool, self._ovf_fid, first_ovf, total_len)
        val, _ = deserialize_value(col.dtype, payload, 1, col.dim)
        return val

    def _read_var(self, col: ColumnSpec, rowid: int) -> Any:
        starts = self._var_dirs[col.name]
        pid = bisect.bisect_right(starts, rowid) - 1
        slot = rowid - starts[pid]
        with self.pool.pin_page(pid, file_id=self._col_fids[col.name]) as frame:
            payload = SlottedPage.get_payload(frame.data, slot)
        assert payload is not None
        return self._decode_var_payload(col, payload)

    def insert(self, record: dict[str, Any]) -> RID:
        validated = self.schema.validate_record(record)
        rowid = self._row_count
        self._set_deleted_bit(rowid, False)
        for col in self.schema.columns:
            if col.name in self._fixed_info:
                self._append_fixed(col, rowid, validated[col.name])
            else:
                self._append_var(col, rowid, validated[col.name])
        self._row_count = rowid + 1
        self._save_row_count(self._row_count)
        return 0, rowid

    def try_get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any] | None:
        chunk_id, rowid = int(rid[0]), int(rid[1])
        if chunk_id != 0 or rowid < 0 or rowid >= self._row_count or self._is_deleted(rowid):
            return None
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        out: dict[str, Any] = {}
        for col in cols:
            if col.name in self._fixed_info:
                out[col.name] = self._read_fixed(col, rowid)
            else:
                out[col.name] = self._read_var(col, rowid)
        return out

    def get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any]:
        rec = self.try_get_record(rid, fields=fields)
        if rec is None:
            raise IndexError(f"Record locator {rid} does not exist")
        return rec

    def delete(self, rid: RID) -> bool:
        chunk_id, rowid = int(rid[0]), int(rid[1])
        if chunk_id != 0 or rowid < 0 or rowid >= self._row_count or self._is_deleted(rowid):
            return False
        self._set_deleted_bit(rowid, True)
        return True

    def update(self, rid: RID, updates: dict[str, Any]) -> RID:
        current = self.get_record(rid)
        merged = {**current, **updates}
        self.schema.validate_record(merged)
        self.delete(rid)
        return self.insert(merged)

    def _read_live_rowids(self, start: int, end: int) -> list[int]:
        live: list[int] = []
        cur = start
        while cur < end:
            pid = cur // self._bits_per_page
            with self.pool.pin_page(pid, file_id=self._del_fid) as frame:
                page_end = min(end, (pid + 1) * self._bits_per_page)
                while cur < page_end:
                    bit = cur % self._bits_per_page
                    if not (frame.data[HEADER_SIZE + (bit // 8)] & (1 << (bit % 8))):
                        live.append(cur)
                    cur += 1
        return live

    def _read_column_slice(self, col: ColumnSpec, rowids: list[int]) -> list[Any]:
        if not rowids:
            return []
        fid = self._col_fids[col.name]
        values: list[Any] = []
        if col.name in self._fixed_info:
            elem_size, spp, bmap_len = self._fixed_info[col.name]
            idx = 0
            while idx < len(rowids):
                pid = rowids[idx] // spp
                with self.pool.pin_page(pid, file_id=fid) as frame:
                    while idx < len(rowids) and rowids[idx] // spp == pid:
                        slot = rowids[idx] % spp
                        if frame.data[HEADER_SIZE + (slot // 8)] & (1 << (slot % 8)):
                            values.append(None)
                        else:
                            off = HEADER_SIZE + bmap_len + slot * elem_size
                            val, _ = deserialize_value(col.dtype, frame.data, off, col.dim)
                            values.append(val)
                        idx += 1
            return values
        starts = self._var_dirs[col.name]
        idx = 0
        while idx < len(rowids):
            pid = bisect.bisect_right(starts, rowids[idx]) - 1
            first_rid = starts[pid]
            with self.pool.pin_page(pid, file_id=fid) as frame:
                hdr = read_header(frame.data)
                page_end = first_rid + hdr.item_count
                while idx < len(rowids) and rowids[idx] < page_end:
                    payload = SlottedPage.get_payload(frame.data, rowids[idx] - first_rid)
                    assert payload is not None
                    values.append(self._decode_var_payload(col, payload))
                    idx += 1
        return values

    def scan_batches(
        self,
        fields: list[str] | None = None,
        batch_size: int = 1024,
    ) -> Iterator[dict[str, list[Any]]]:
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        for start in range(0, self._row_count, batch_size):
            live = self._read_live_rowids(start, min(start + batch_size, self._row_count))
            if not live:
                continue
            yield {col.name: self._read_column_slice(col, live) for col in cols}

    def iter_records(self, fields: list[str] | None = None) -> Iterator[tuple[RID, dict[str, Any]]]:
        cols = [self.schema.column(f) for f in fields] if fields is not None else self.schema.columns
        for start in range(0, self._row_count, 1024):
            live = self._read_live_rowids(start, min(start + 1024, self._row_count))
            if not live:
                continue
            col_data = {col.name: self._read_column_slice(col, live) for col in cols}
            for i, rowid in enumerate(live):
                yield (0, rowid), {col.name: col_data[col.name][i] for col in cols}

    def scan(
        self,
        fields: list[str] | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _, rec in self.iter_records(fields=fields if predicate is None else None):
            if predicate is None or predicate(rec):
                results.append(rec if fields is None else {f: rec[f] for f in fields})
        return results

    def flush(self) -> None:
        self.pool.flush_file(self._meta_fid)
        self.pool.flush_file(self._del_fid)
        self.pool.flush_file(self._ovf_fid)
        for fid in self._col_fids.values():
            self.pool.flush_file(fid)
