"""Row-oriented and column-oriented page layout managers with projection support."""
from __future__ import annotations

from bisect import bisect_right
import json
import struct
from typing import Any, Callable, Iterator, Sequence

from .buffer_pool import BufferPool
from .page_file import PAGE_SIZE
from .schema import TableSchema

_ROW_HEADER_SIZE = 4
_COL_HEADER_SIZE = 8


def _pack_json_page(obj: Any, page_size: int) -> bytes:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    max_payload = page_size - _ROW_HEADER_SIZE
    if len(payload) > max_payload:
        raise OverflowError(f"Encoded records need {len(payload) + _ROW_HEADER_SIZE} bytes; page size is {page_size}")
    return struct.pack(">I", len(payload)) + payload + b"\x00" * (page_size - _ROW_HEADER_SIZE - len(payload))


def _unpack_json_page(page: bytes, page_size: int | None = None) -> Any | None:
    expected_size = len(page) if page_size is None else page_size
    if len(page) != expected_size:
        raise ValueError(f"Expected a {expected_size}-byte page")
    payload_len = struct.unpack(">I", page[:_ROW_HEADER_SIZE])[0]
    if payload_len == 0:
        return None
    if payload_len > len(page) - _ROW_HEADER_SIZE:
        raise ValueError("Corrupt page: invalid payload length")
    return json.loads(page[_ROW_HEADER_SIZE:_ROW_HEADER_SIZE + payload_len].decode("utf-8"))


def _pack_column_page(values: list[Any], page_size: int) -> bytes:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    max_payload = page_size - _COL_HEADER_SIZE
    if len(payload) > max_payload:
        raise OverflowError(f"Encoded column needs {len(payload) + _COL_HEADER_SIZE} bytes; page size is {page_size}")
    header = struct.pack(">II", len(payload), len(values))
    return header + payload + b"\x00" * (page_size - _COL_HEADER_SIZE - len(payload))


def _unpack_column_page(page: bytes, page_size: int | None = None) -> list[Any]:
    expected_size = len(page) if page_size is None else page_size
    if len(page) != expected_size:
        raise ValueError(f"Expected a {expected_size}-byte page")
    payload_len, item_count = struct.unpack(">II", page[:_COL_HEADER_SIZE])
    if payload_len == 0:
        return []
    if payload_len > len(page) - _COL_HEADER_SIZE:
        raise ValueError("Corrupt column page: invalid payload length")
    values = json.loads(page[_COL_HEADER_SIZE:_COL_HEADER_SIZE + payload_len].decode("utf-8"))
    if len(values) != item_count:
        raise ValueError("Corrupt column page: item_count mismatch")
    return values


def _read_column_item_count(page: bytes) -> int:
    payload_len, item_count = struct.unpack(">II", page[:_COL_HEADER_SIZE])
    return 0 if payload_len == 0 else item_count


def encode_records(records: list[dict[str, Any]], layout: str, page_size: int = PAGE_SIZE) -> bytes:
    """Encode a list of row dicts into a single page using 'row' or 'column' JSON shape."""
    normalized = layout.lower()
    if normalized == "row":
        payload_obj: dict[str, Any] = {"layout": "row", "records": records}
    elif normalized == "column":
        keys = list(records[0].keys()) if records else []
        payload_obj = {
            "layout": "column",
            "columns": {key: [row.get(key) for row in records] for key in keys},
        }
    else:
        raise ValueError("layout must be 'row' or 'column'")
    return _pack_json_page(payload_obj, page_size)


def decode_records(page: bytes, page_size: int | None = None) -> list[dict[str, Any]]:
    """Decode a single row- or column-encoded JSON page into a list of row dicts."""
    obj = _unpack_json_page(page, page_size=page_size)
    if obj is None:
        return []
    if obj.get("layout") == "row":
        return obj["records"]
    if obj.get("layout") == "column":
        columns = obj["columns"]
        if not columns:
            return []
        names = list(columns)
        count = len(columns[names[0]])
        if any(len(columns[name]) != count for name in names):
            raise ValueError("Corrupt columnar page: column lengths differ")
        return [{name: columns[name][i] for name in names} for i in range(count)]
    raise ValueError("Corrupt page: unknown layout")


def empty_page(page_size: int = PAGE_SIZE) -> bytes:
    return b"\x00" * page_size


def _resolve_columns(schema: TableSchema, columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return schema.field_names
    valid = set(schema.field_names)
    resolved = list(columns)
    for col in resolved:
        if col not in valid:
            raise KeyError(f"Unknown field: {col}")
    return resolved


def _project_record(record: dict[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    return {col: record[col] for col in columns}


class RowLayout:
    """Stores full records in slotted JSON pages with stable (page_id, slot_id) locators."""

    def __init__(self, pool: BufferPool, stream_prefix: str, schema: TableSchema):
        self.pool = pool
        self.stream = f"{stream_prefix}.data"
        self.schema = schema
        self.page_size = pool.stream_page_size(self.stream)

    def _read_slots(self, page_id: int) -> list[dict[str, Any] | None]:
        obj = _unpack_json_page(self.pool.read_page(page_id, stream_id=self.stream), self.page_size)
        return [] if obj is None else obj["records"]

    def _write_slots(self, page_id: int, slots: list[dict[str, Any] | None]) -> None:
        encoded = _pack_json_page({"layout": "row", "records": slots}, self.page_size)
        self.pool.write_page(page_id, encoded, stream_id=self.stream)

    def insert(self, record: dict[str, Any]) -> tuple[int, int]:
        normalized = self.schema.validate_record(record)
        page_count = self.pool.stream_page_count(self.stream)
        if page_count > 0:
            last_page = page_count - 1
            slots = self._read_slots(last_page)
            try:
                self._write_slots(last_page, slots + [normalized])
                return last_page, len(slots)
            except OverflowError:
                pass
        # Pre-encode before allocating so an oversized single record leaves no orphan page.
        encoded = _pack_json_page({"layout": "row", "records": [normalized]}, self.page_size)
        new_page = self.pool.allocate_page(self.stream)
        self.pool.write_page(new_page, encoded, stream_id=self.stream)
        return new_page, 0

    def _read_live_slots(self, locator: tuple[int, int] | list[int]) -> tuple[int, int, list[dict[str, Any] | None]]:
        page_id, slot_id = int(locator[0]), int(locator[1])
        slots = self._read_slots(page_id)
        if slot_id < 0 or slot_id >= len(slots):
            raise IndexError(f"Record locator ({page_id}, {slot_id}) does not exist")
        if slots[slot_id] is None:
            raise KeyError(f"Record locator ({page_id}, {slot_id}) was deleted")
        return page_id, slot_id, slots

    def get_record(self, locator: tuple[int, int] | list[int]) -> dict[str, Any]:
        _, slot_id, slots = self._read_live_slots(locator)
        return dict(slots[slot_id])  # type: ignore[arg-type]

    def delete(self, locator: tuple[int, int] | list[int]) -> None:
        page_id, slot_id, slots = self._read_live_slots(locator)
        slots[slot_id] = None
        self._write_slots(page_id, slots)

    def update(self, locator: tuple[int, int] | list[int], record: dict[str, Any]) -> tuple[int, int]:
        normalized = self.schema.validate_record(record)
        page_id, slot_id, slots = self._read_live_slots(locator)
        updated_slots = list(slots)
        updated_slots[slot_id] = normalized
        try:
            self._write_slots(page_id, updated_slots)
            return page_id, slot_id
        except OverflowError:
            slots[slot_id] = None
            self._write_slots(page_id, slots)
            return self.insert(normalized)

    def iter_records(
        self, columns: Sequence[str] | None = None
    ) -> Iterator[tuple[tuple[int, int], dict[str, Any]]]:
        selected = _resolve_columns(self.schema, columns)
        for page_id in range(self.pool.stream_page_count(self.stream)):
            slots = self._read_slots(page_id)
            for slot_id, record in enumerate(slots):
                if record is not None:
                    yield (page_id, slot_id), _project_record(record, selected)

    def iter_batches(
        self, columns: Sequence[str] | None = None, batch_size: int = 256
    ) -> Iterator[dict[str, list[Any]]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        selected = _resolve_columns(self.schema, columns)
        batch: dict[str, list[Any]] = {col: [] for col in selected}
        count = 0
        for _, record in self.iter_records(columns=selected):
            for col in selected:
                batch[col].append(record[col])
            count += 1
            if count == batch_size:
                yield batch
                batch = {col: [] for col in selected}
                count = 0
        if count > 0:
            yield batch


class ColumnLayout:
    """Stores each column in its own page stream for projection-only I/O."""

    def __init__(self, pool: BufferPool, stream_prefix: str, schema: TableSchema):
        self.pool = pool
        self.schema = schema
        self._col_streams: dict[str, str] = {
            col: f"{stream_prefix}.col.{col}" for col in schema.field_names
        }
        self._tombstone_stream = f"{stream_prefix}._tombstones"
        self.page_size = pool.stream_page_size(self._col_streams[schema.field_names[0]])
        self._page_starts: dict[str, list[int]] = {}
        self._total_rows = 0
        self._deleted: set[int] = set()
        self._load_metadata()

    def _load_metadata(self) -> None:
        for col, stream in self._col_streams.items():
            starts: list[int] = []
            running = 0
            for page_id in range(self.pool.stream_page_count(stream)):
                starts.append(running)
                running += _read_column_item_count(self.pool.read_page(page_id, stream_id=stream))
            self._page_starts[col] = starts
        first_col = self.schema.field_names[0]
        first_stream = self._col_streams[first_col]
        first_count = self.pool.stream_page_count(first_stream)
        if first_count == 0:
            self._total_rows = 0
        else:
            last_page = first_count - 1
            last_items = _read_column_item_count(self.pool.read_page(last_page, stream_id=first_stream))
            self._total_rows = self._page_starts[first_col][last_page] + last_items

        for page_id in range(self.pool.stream_page_count(self._tombstone_stream)):
            page = self.pool.read_page(page_id, stream_id=self._tombstone_stream)
            self._deleted.update(_unpack_column_page(page, self.page_size))

    def _locate(self, col: str, row_id: int) -> tuple[int, int]:
        starts = self._page_starts[col]
        page_id = bisect_right(starts, row_id) - 1
        return page_id, row_id - starts[page_id]

    def _plan_append(self, stream: str, starts: list[int], value: Any) -> tuple[bool, int, bytes]:
        page_count = self.pool.stream_page_count(stream)
        if page_count > 0:
            last_page = page_count - 1
            values = _unpack_column_page(self.pool.read_page(last_page, stream_id=stream), self.page_size)
            try:
                return False, last_page, _pack_column_page(values + [value], self.page_size)
            except OverflowError:
                pass
        return True, page_count, _pack_column_page([value], self.page_size)

    def _apply_append(self, stream: str, starts: list[int], row_id: int, needs_alloc: bool, page_id: int, data: bytes) -> None:
        if needs_alloc:
            page_id = self.pool.allocate_page(stream)
            starts.append(row_id)
        self.pool.write_page(page_id, data, stream_id=stream)

    def _record_tombstone(self, row_id: int) -> None:
        needs_alloc, page_id, data = self._plan_append(self._tombstone_stream, [], row_id)
        if needs_alloc:
            page_id = self.pool.allocate_page(self._tombstone_stream)
        self.pool.write_page(page_id, data, stream_id=self._tombstone_stream)
        self._deleted.add(row_id)

    def _check_live_row_id(self, locator: tuple[int, int] | list[int]) -> int:
        bucket, row_id = int(locator[0]), int(locator[1])
        if bucket != 0 or row_id < 0 or row_id >= self._total_rows:
            raise IndexError(f"Record locator ({bucket}, {row_id}) does not exist")
        if row_id in self._deleted:
            raise KeyError(f"Record locator ({bucket}, {row_id}) was deleted")
        return row_id

    def insert(self, record: dict[str, Any]) -> tuple[int, int]:
        normalized = self.schema.validate_record(record)
        row_id = self._total_rows
        # Plan all column writes before mutating so an oversized field fails atomically.
        planned = [
            (col, self._plan_append(self._col_streams[col], self._page_starts[col], normalized[col]))
            for col in self.schema.field_names
        ]
        for col, (needs_alloc, page_id, data) in planned:
            self._apply_append(self._col_streams[col], self._page_starts[col], row_id, needs_alloc, page_id, data)
        self._total_rows += 1
        return 0, row_id

    def get_record(self, locator: tuple[int, int] | list[int]) -> dict[str, Any]:
        row_id = self._check_live_row_id(locator)
        result: dict[str, Any] = {}
        for col, stream in self._col_streams.items():
            page_id, local_idx = self._locate(col, row_id)
            values = _unpack_column_page(self.pool.read_page(page_id, stream_id=stream), self.page_size)
            result[col] = values[local_idx]
        return result

    def delete(self, locator: tuple[int, int] | list[int]) -> None:
        row_id = self._check_live_row_id(locator)
        self._record_tombstone(row_id)

    def update(self, locator: tuple[int, int] | list[int], record: dict[str, Any]) -> tuple[int, int]:
        row_id = self._check_live_row_id(locator)
        normalized = self.schema.validate_record(record)
        planned_writes: list[tuple[str, int, bytes]] = []
        try:
            for col, stream in self._col_streams.items():
                page_id, local_idx = self._locate(col, row_id)
                values = _unpack_column_page(self.pool.read_page(page_id, stream_id=stream), self.page_size)
                if values[local_idx] != normalized[col]:
                    values[local_idx] = normalized[col]
                    planned_writes.append((stream, page_id, _pack_column_page(values, self.page_size)))
        except OverflowError:
            # Verify the replacement fits before tombstoning the old row.
            new_loc = self.insert(normalized)
            self._record_tombstone(row_id)
            return new_loc
        for stream, page_id, data in planned_writes:
            self.pool.write_page(page_id, data, stream_id=stream)
        return 0, row_id

    def _iter_column_values(self, col: str) -> Iterator[Any]:
        stream = self._col_streams[col]
        for page_id in range(self.pool.stream_page_count(stream)):
            yield from _unpack_column_page(self.pool.read_page(page_id, stream_id=stream), self.page_size)

    def iter_batches(
        self, columns: Sequence[str] | None = None, batch_size: int = 256
    ) -> Iterator[dict[str, list[Any]]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        selected = _resolve_columns(self.schema, columns)
        if self._total_rows == 0:
            return
        if not selected:
            live = sum(1 for row_id in range(self._total_rows) if row_id not in self._deleted)
            while live > 0:
                chunk = min(batch_size, live)
                yield {}
                live -= chunk
            return

        iterators = [self._iter_column_values(col) for col in selected]
        batch: dict[str, list[Any]] = {col: [] for col in selected}
        count = 0
        for row_id, row_tuple in enumerate(zip(*iterators)):
            if row_id in self._deleted:
                continue
            for col, val in zip(selected, row_tuple):
                batch[col].append(val)
            count += 1
            if count == batch_size:
                yield batch
                batch = {col: [] for col in selected}
                count = 0
        if count > 0:
            yield batch

    def iter_records(
        self, columns: Sequence[str] | None = None
    ) -> Iterator[tuple[tuple[int, int], dict[str, Any]]]:
        selected = _resolve_columns(self.schema, columns)
        if self._total_rows == 0:
            return
        if not selected:
            for row_id in range(self._total_rows):
                if row_id not in self._deleted:
                    yield (0, row_id), {}
            return
        iterators = [self._iter_column_values(col) for col in selected]
        for row_id, row_tuple in enumerate(zip(*iterators)):
            if row_id not in self._deleted:
                yield (0, row_id), dict(zip(selected, row_tuple))
