"""High-level Smart Retail DB engine integrating custom storage and indexes."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .storage.table import StorageTable
from .indexes.manager import IndexManager
from .query import parse_select


class SmartRetailDB:
    def __init__(self, directory: str | Path, buffer_frames: int = 16,
                 replacement_policy: str = "LRU", backend_kind: str = "page",
                 encrypt_page=None, decrypt_page=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer_frames = buffer_frames
        self.replacement_policy = replacement_policy
        self.backend_kind = backend_kind
        self.encrypt_page = encrypt_page
        self.decrypt_page = decrypt_page
        self.catalog_path = self.directory / "catalog.json"
        self.index_catalog_path = self.directory / "indexes.json"
        self.tables: dict[str, StorageTable] = {}
        self.schemas: dict[str, dict[str, Any]] = {}
        self.index_manager = IndexManager(self.directory / "index_data")
        self._load_catalog()
        self._open_tables()
        self._load_indexes()

    @staticmethod
    def _normal_name(name: str) -> str:
        if not name or not name.isidentifier():
            raise ValueError(f"Invalid identifier: {name!r}")
        return name

    def _load_catalog(self) -> None:
        if self.catalog_path.exists():
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            self.schemas = data.get("tables", {})
        else:
            self.schemas = {}

    def _save_catalog(self) -> None:
        temp = self.catalog_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"tables": self.schemas}, indent=2), encoding="utf-8")
        temp.replace(self.catalog_path)

    def _open_tables(self) -> None:
        for name, schema in self.schemas.items():
            self.tables[name] = StorageTable(
                self.directory / "tables" / f"{name}.pages",
                fields=schema["fields"], layout=schema["layout"],
                buffer_frames=self.buffer_frames,
                replacement_policy=self.replacement_policy,
                backend_kind=schema.get("backend_kind", self.backend_kind),
                encrypt_page=self.encrypt_page, decrypt_page=self.decrypt_page,
            )

    def create_table(self, name: str, fields: list[str], layout: str = "row") -> None:
        self._normal_name(name)
        if name in self.tables:
            raise ValueError(f"Table {name} already exists")
        if layout not in {"row", "column"}:
            raise ValueError("layout must be 'row' or 'column'")
        if not fields or any(not f.isidentifier() for f in fields) or len(fields) != len(set(fields)):
            raise ValueError("fields must be unique Python-style identifiers")
        schema = {"fields": list(fields), "layout": layout, "backend_kind": self.backend_kind}
        table = StorageTable(
            self.directory / "tables" / f"{name}.pages", fields=fields,
            layout=layout, buffer_frames=self.buffer_frames,
            replacement_policy=self.replacement_policy,
            backend_kind=self.backend_kind,
            encrypt_page=self.encrypt_page, decrypt_page=self.decrypt_page,
        )
        self.tables[name] = table
        self.schemas[name] = schema
        self._save_catalog()

    def insert(self, table_name: str, record: dict[str, Any]) -> tuple[int, int]:
        table = self._get_table(table_name)
        locator = table.insert(record)
        for field in table.fields:
            self.index_manager.add(table_name, field, record[field], locator)
        return locator

    def create_index(self, table_name: str, field: str, index_type: str = "auto",
                     reads: int = 100, writes: int = 10) -> str:
        table = self._get_table(table_name)
        if field not in table.fields:
            raise KeyError(f"Unknown field {field!r} in table {table_name!r}")
        kind, index = self.index_manager.create(table_name, field, index_type, reads, writes)
        # B+ Tree indexes are rebuilt from the table whenever the DB opens.
        # For an LSM index with existing sorted runs, retain the persisted runs
        # to avoid duplicating entries; otherwise build it for current records.
        has_persisted_lsm_data = kind == "LSM" and bool(index.runs)
        if not has_persisted_lsm_data:
            for locator, record in table.iter_records():
                index.insert(record[field], locator)
        self._save_indexes_catalog()
        return kind

    def _save_indexes_catalog(self) -> None:
        items = []
        for (table, field), (kind, _) in self.index_manager.indexes.items():
            items.append({"table": table, "field": field, "type": kind})
        temp = self.index_catalog_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"indexes": items}, indent=2), encoding="utf-8")
        temp.replace(self.index_catalog_path)

    def _load_indexes(self) -> None:
        if not self.index_catalog_path.exists():
            return
        data = json.loads(self.index_catalog_path.read_text(encoding="utf-8"))
        for item in data.get("indexes", []):
            table_name, field, kind = item["table"], item["field"], item["type"]
            if table_name not in self.tables or field not in self.tables[table_name].fields:
                continue
            _, index = self.index_manager.create(table_name, field, kind)
            if kind == "LSM" and index.runs:
                continue
            for locator, record in self.tables[table_name].iter_records():
                index.insert(record[field], locator)

    def execute(self, sql: str) -> list[dict[str, Any]]:
        query = parse_select(sql)
        table = self._get_table(query.table)
        if query.field is None:
            return table.scan()
        if query.field not in table.fields:
            raise KeyError(f"Unknown field {query.field!r} in table {query.table!r}")
        locators = self.index_manager.search(query.table, query.field, query.value)
        if locators is None:
            return table.scan(query.field, query.value)
        # An index can contain stale locators only if a data file was changed
        # outside this engine; verify the predicate before returning rows.
        return [record for loc in locators if (record := table.get_record(loc)).get(query.field) == query.value]

    def explain(self, sql: str) -> str:
        query = parse_select(sql)
        table = self._get_table(query.table)
        if query.field is None:
            return f"TABLE SCAN {query.table} (layout={table.layout})"
        kind = self.index_manager.kind(query.table, query.field)
        if kind:
            return f"INDEX LOOKUP {kind} ON {query.table}.{query.field} -> FETCH PAGE/ROW"
        return f"TABLE SCAN {query.table} FILTER {query.field} = {query.value!r}"

    def buffer_stats(self) -> dict[str, dict[str, Any]]:
        return {name: table.pool.snapshot() for name, table in self.tables.items()}

    def _get_table(self, name: str) -> StorageTable:
        if name not in self.tables:
            raise KeyError(f"Unknown table {name!r}")
        return self.tables[name]

    def close(self) -> None:
        self.index_manager.close()
        for table in self.tables.values():
            table.close()

    def __enter__(self) -> "SmartRetailDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
