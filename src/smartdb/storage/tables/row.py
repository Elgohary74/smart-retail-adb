"""Binary slotted-page RowTable with stable (page_id, slot_id) RIDs and overflow."""
from __future__ import annotations

from typing import Any, Callable, Iterator

from ...core import RID, TableSchema, deserialize_row, serialize_row
from ..backends import StorageManager
from ..buffer_pool import BufferPool
from ..pages import (
    HEADER_SIZE,
    PageType,
    SlottedPage,
    free_overflow_chain,
    is_overflow_stub,
    pack_overflow_stub,
    read_overflow_chain,
    unpack_overflow_stub,
    write_overflow_chain,
)


class RowTable:
    """Slotted-page heap table with stable (page_id, slot_id) RIDs and overflow."""

    def __init__(self, storage_manager: StorageManager, pool: BufferPool, name: str, schema: TableSchema):
        self.storage_manager = storage_manager
        self.pool = pool
        self.name = name
        self.schema = schema
        self.fields = schema.field_names
        self._data_file_id, self._data_backend = storage_manager.open_file(
            f"table:{name}:row", f"tables/{name}.pages"
        )
        self._ovf_file_id, _ = storage_manager.open_file(
            f"table:{name}:overflow", f"tables/{name}.overflow.pages"
        )
        self._max_inline = self._data_backend.usable_page_size - HEADER_SIZE - 8

    @property
    def page_count(self) -> int:
        return self._data_backend.page_count

    def _encode_payload(self, record: dict[str, Any]) -> bytes:
        validated = self.schema.validate_record(record)
        raw = serialize_row(self.schema, validated)
        if len(raw) <= self._max_inline:
            return raw
        first_ovf = write_overflow_chain(self.pool, self._ovf_file_id, raw)
        return pack_overflow_stub(len(raw), first_ovf)

    def _decode_payload(self, payload: bytes, fields: list[str] | None = None) -> dict[str, Any]:
        if is_overflow_stub(payload):
            total_len, first_ovf = unpack_overflow_stub(payload)
            payload = read_overflow_chain(self.pool, self._ovf_file_id, first_ovf, total_len)
        return deserialize_row(self.schema, payload, projected=fields)

    def _free_payload_overflow(self, payload: bytes) -> None:
        if is_overflow_stub(payload):
            _, first_ovf = unpack_overflow_stub(payload)
            free_overflow_chain(self.pool, self._ovf_file_id, first_ovf)

    def insert(self, record: dict[str, Any]) -> RID:
        payload = self._encode_payload(record)
        count = self._data_backend.page_count
        if count > 0:
            last_pid = count - 1
            with self.pool.pin_page(last_pid, file_id=self._data_file_id, dirty=True) as frame:
                slot_id = SlottedPage.insert_payload(frame.data, payload)
                if slot_id is not None:
                    return last_pid, slot_id
        pid, frame = self.pool.new_page(file_id=self._data_file_id)
        try:
            SlottedPage.init(frame.data, PageType.ROW_SLOTTED)
            slot_id = SlottedPage.insert_payload(frame.data, payload)
            if slot_id is None:
                raise OverflowError("Payload too large for slotted page")
            return pid, slot_id
        finally:
            self.pool.unpin_page(pid, is_dirty=True, file_id=self._data_file_id)

    def try_get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any] | None:
        page_id, slot_id = int(rid[0]), int(rid[1])
        if page_id < 0 or page_id >= self._data_backend.page_count:
            return None
        with self.pool.pin_page(page_id, file_id=self._data_file_id) as frame:
            payload = SlottedPage.get_payload(frame.data, slot_id)
        return None if payload is None else self._decode_payload(payload, fields=fields)

    def get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any]:
        rec = self.try_get_record(rid, fields=fields)
        if rec is None:
            raise IndexError(f"Record locator {rid} does not exist")
        return rec

    def update(self, rid: RID, updates: dict[str, Any]) -> RID:
        current = self.get_record(rid)
        merged = {**current, **updates}
        new_payload = self._encode_payload(merged)
        page_id, slot_id = int(rid[0]), int(rid[1])
        old_payload: bytes | None = None
        updated_in_place = False
        with self.pool.pin_page(page_id, file_id=self._data_file_id, dirty=True) as frame:
            old_payload = SlottedPage.get_payload(frame.data, slot_id)
            if old_payload is None:
                raise IndexError(f"Record locator {rid} does not exist")
            if len(new_payload) <= len(old_payload):
                updated_in_place = SlottedPage.update_payload(frame.data, slot_id, new_payload)
        if updated_in_place:
            if old_payload is not None:
                self._free_payload_overflow(old_payload)
            return rid
        self.delete(rid)
        return self.insert(merged)

    def delete(self, rid: RID) -> bool:
        page_id, slot_id = int(rid[0]), int(rid[1])
        if page_id < 0 or page_id >= self._data_backend.page_count:
            return False
        with self.pool.pin_page(page_id, file_id=self._data_file_id, dirty=True) as frame:
            payload = SlottedPage.get_payload(frame.data, slot_id)
            if payload is None:
                return False
            SlottedPage.delete_slot(frame.data, slot_id)
        self._free_payload_overflow(payload)
        return True

    def iter_records(self, fields: list[str] | None = None) -> Iterator[tuple[RID, dict[str, Any]]]:
        for page_id in range(self._data_backend.page_count):
            with self.pool.pin_page(page_id, file_id=self._data_file_id) as frame:
                slots = list(SlottedPage.live_slots(frame.data))
            for slot_id, payload in slots:
                yield (page_id, slot_id), self._decode_payload(payload, fields=fields)

    def scan(
        self,
        fields: list[str] | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            (rec if fields is None else {f: rec[f] for f in fields})
            for _, rec in self.iter_records(fields=fields if predicate is None else None)
            if predicate is None or predicate(rec)
        ]

    def scan_batches(
        self,
        fields: list[str] | None = None,
        batch_size: int = 1024,
    ) -> Iterator[dict[str, list[Any]]]:
        cols = list(fields) if fields is not None else list(self.fields)
        batch: dict[str, list[Any]] = {c: [] for c in cols}
        count = 0
        for _, rec in self.iter_records(fields=cols):
            for c in cols:
                batch[c].append(rec[c])
            count += 1
            if count >= batch_size:
                yield batch
                batch = {c: [] for c in cols}
                count = 0
        if count > 0:
            yield batch

    def flush(self) -> None:
        self.pool.flush_file(self._data_file_id)
        self.pool.flush_file(self._ovf_file_id)
