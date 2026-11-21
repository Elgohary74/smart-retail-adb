"""Shared TableStorage protocol implemented by RowTable and ColumnTable."""
from __future__ import annotations

from typing import Any, Callable, Iterator, Protocol

from ...core import RID, TableSchema


class TableStorage(Protocol):
    """Shared storage interface implemented by RowTable and ColumnTable."""

    schema: TableSchema

    def insert(self, record: dict[str, Any]) -> RID: ...
    def get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any]: ...
    def try_get_record(self, rid: RID, fields: list[str] | None = None) -> dict[str, Any] | None: ...
    def update(self, rid: RID, updates: dict[str, Any]) -> RID: ...
    def delete(self, rid: RID) -> bool: ...
    def iter_records(self, fields: list[str] | None = None) -> Iterator[tuple[RID, dict[str, Any]]]: ...
    def scan(
        self,
        fields: list[str] | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]: ...
    def scan_batches(
        self,
        fields: list[str] | None = None,
        batch_size: int = 1024,
    ) -> Iterator[dict[str, list[Any]]]: ...
    @property
    def page_count(self) -> int: ...
    def flush(self) -> None: ...


def scan_records(
    iter_records: Callable[[list[str] | None], Iterator[tuple[RID, dict[str, Any]]]],
    fields: list[str] | None = None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Shared scan projection/filtering helper over a table's iter_records() stream."""
    return [
        (rec if fields is None else {f: rec[f] for f in fields})
        for _, rec in iter_records(fields if predicate is None else None)
        if predicate is None or predicate(rec)
    ]
