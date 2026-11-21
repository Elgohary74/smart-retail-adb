"""High-level Smart Retail DB engine integrating unified storage, buffer pool, and indexes."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .indexes.manager import IndexManager
from .query import (
    CreateIndexStmt,
    CreateTableStmt,
    DeleteStmt,
    InsertStmt,
    Predicate,
    SelectStmt,
    UpdateStmt,
    parse_sql,
)
from .schema import RID, ColumnSpec, DataType, TableSchema
from .storage.buffer_pool import BufferPool
from .storage.column_table import ColumnTable
from .storage.encryption import PageTransform, RecordTransform
from .storage.manager import StorageManager
from .storage.row_table import RowTable, TableStorage


def _matches_predicate(schema: TableSchema, record: dict[str, Any], pred: Predicate) -> bool:
    val = record.get(pred.field)
    if pred.op == "IS_NULL":
        return val is None
    if pred.op == "IS_NOT_NULL":
        return val is not None
    target = schema.coerce_literal(pred.field, pred.value)
    if pred.op == "=":
        if target is None:
            return val is None
        if val is None or isinstance(val, bool) != isinstance(target, bool):
            return False
        return val == target
    # SQL three-valued logic: NULL never satisfies <, <=, >, >=, BETWEEN
    if val is None or target is None:
        return False
    if isinstance(val, bool) != isinstance(target, bool):
        return False
    if pred.op == "<":
        return val < target
    if pred.op == "<=":
        return val <= target
    if pred.op == ">":
        return val > target
    if pred.op == ">=":
        return val >= target
    if pred.op == "BETWEEN":
        high = schema.coerce_literal(pred.field, pred.high)
        if high is None or isinstance(val, bool) != isinstance(high, bool):
            return False
        return target <= val <= high
    raise ValueError(f"Unsupported predicate operator: {pred.op}")


class SmartRetailDB:
    """Custom database engine with shared BufferPool, Row/Column tables, and B+/LSM indexes."""

    def __init__(
        self,
        directory: str | Path,
        buffer_frames: int = 64,
        replacement_policy: str = "LRU",
        backend_kind: str = "page",
        encrypt_page: PageTransform | None = None,
        decrypt_page: PageTransform | None = None,
        crypto_overhead: int = 0,
        encrypt_record: RecordTransform | None = None,
        decrypt_record: RecordTransform | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer_frames = buffer_frames
        self.replacement_policy = replacement_policy
        self.backend_kind = backend_kind

        self.catalog_path = self.directory / "catalog.json"
        self.index_catalog_path = self.directory / "indexes.json"

        self.storage_manager = StorageManager(
            self.directory,
            backend_kind=backend_kind,
            encrypt_page=encrypt_page,
            decrypt_page=decrypt_page,
            crypto_overhead=crypto_overhead,
            encrypt_record=encrypt_record,
            decrypt_record=decrypt_record,
        )
        self.pool = BufferPool(
            capacity=buffer_frames,
            policy=replacement_policy,
            storage_manager=self.storage_manager,
        )
        self.index_manager = IndexManager(self.directory / "index_data", self.storage_manager, self.pool)

        self.tables: dict[str, TableStorage] = {}
        self.schemas: dict[str, dict[str, Any]] = {}
        self._table_stats: dict[str, dict[str, int]] = {}
        self._clean_shutdown = True

        self._load_catalog()
        self._open_tables()
        self._load_indexes()
        if not self._clean_shutdown:
            self._reconcile_indexes()
        self._save_catalog(clean_shutdown=False)

    @staticmethod
    def _normal_name(name: str) -> str:
        if not name or not name.isidentifier():
            raise ValueError(f"Invalid identifier: {name!r}")
        return name

    def _load_catalog(self) -> None:
        if self.catalog_path.exists():
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            self.schemas = data.get("tables", {})
            self._clean_shutdown = bool(data.get("clean_shutdown", True))
            for tname, meta in self.schemas.items():
                self._table_stats[tname] = dict(meta.get("stats", {"reads": 0, "writes": 0, "range_reads": 0}))
        else:
            self.schemas = {}
            self._clean_shutdown = True

    def _save_catalog(self, clean_shutdown: bool = False) -> None:
        for tname, stats in self._table_stats.items():
            if tname in self.schemas:
                self.schemas[tname]["stats"] = stats
        tmp = self.catalog_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"clean_shutdown": clean_shutdown, "tables": self.schemas}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.catalog_path)

    def _open_tables(self) -> None:
        for name, meta in self.schemas.items():
            schema = TableSchema.from_dict(meta)
            layout = meta["layout"]
            if layout == "row":
                self.tables[name] = RowTable(self.storage_manager, self.pool, name, schema)
            elif layout == "column":
                self.tables[name] = ColumnTable(self.storage_manager, self.pool, name, schema)
            else:
                raise ValueError(f"Unknown layout: {layout}")

    def _save_indexes_catalog(self) -> None:
        items = [
            {"table": t, "field": f, "type": kind}
            for (t, f), (kind, _) in self.index_manager.indexes.items()
        ]
        tmp = self.index_catalog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"indexes": items}, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_catalog_path)

    def _load_indexes(self) -> None:
        if not self.index_catalog_path.exists():
            return
        data = json.loads(self.index_catalog_path.read_text(encoding="utf-8"))
        for item in data.get("indexes", []):
            tname, field, kind = item["table"], item["field"], item["type"]
            if tname not in self.tables or field not in self.tables[tname].schema.field_names:
                continue
            dtype = self.tables[tname].schema.column(field).dtype
            self.index_manager.create(tname, field, kind, key_dtype=dtype, replace=False)

    def _reconcile_indexes(self) -> None:
        for (tname, field), (kind, _) in list(self.index_manager.indexes.items()):
            table = self.tables[tname]
            dtype = table.schema.column(field).dtype
            _, idx = self.index_manager.create(tname, field, kind, key_dtype=dtype, replace=True)
            for rid, rec in table.iter_records():
                idx.insert(rec[field], rid)

    def _get_table(self, name: str) -> TableStorage:
        if name not in self.tables:
            raise KeyError(f"Unknown table {name!r}")
        return self.tables[name]

    def create_table(
        self,
        name: str,
        schema_or_fields: list[str] | list[ColumnSpec] | dict[str, Any] | TableSchema,
        layout: str = "row",
    ) -> None:
        self._normal_name(name)
        if name in self.tables:
            raise ValueError(f"Table {name} already exists")
        if layout not in {"row", "column"}:
            raise ValueError("layout must be 'row' or 'column'")
        schema = (
            schema_or_fields
            if isinstance(schema_or_fields, TableSchema)
            else TableSchema.from_spec(schema_or_fields)
        )
        self.schemas[name] = {
            **schema.to_dict(),
            "layout": layout,
            "backend_kind": self.backend_kind,
            "stats": {"reads": 0, "writes": 0, "range_reads": 0},
        }
        self._table_stats[name] = {"reads": 0, "writes": 0, "range_reads": 0}
        self.tables[name] = (
            RowTable(self.storage_manager, self.pool, name, schema)
            if layout == "row"
            else ColumnTable(self.storage_manager, self.pool, name, schema)
        )
        self._save_catalog(clean_shutdown=False)

    def insert(self, table_name: str, record: dict[str, Any]) -> RID:
        table = self._get_table(table_name)
        validated = table.schema.validate_record(record)
        rid = table.insert(validated)
        for field in table.schema.field_names:
            self.index_manager.add(table_name, field, validated[field], rid)
        self._table_stats[table_name]["writes"] += 1
        return rid

    def update(self, table_name: str, updates: dict[str, Any], where: Predicate | None = None) -> int:
        table = self._get_table(table_name)
        for col_name, val in updates.items():
            col = table.schema.column(col_name)
            table.schema.validate_value(col, val)
        targets = [
            (rid, rec)
            for rid, rec in table.iter_records()
            if where is None or _matches_predicate(table.schema, rec, where)
        ]
        for old_rid, old_rec in targets:
            new_rec = {**old_rec, **updates}
            new_rid = table.update(old_rid, updates)
            for field in table.schema.field_names:
                if old_rid != new_rid or old_rec[field] != new_rec[field]:
                    self.index_manager.remove(table_name, field, old_rec[field], old_rid)
                    self.index_manager.add(table_name, field, new_rec[field], new_rid)
        if targets:
            self._table_stats[table_name]["writes"] += len(targets)
        return len(targets)

    def delete(self, table_name: str, where: Predicate | None = None) -> int:
        table = self._get_table(table_name)
        targets = [
            (rid, rec)
            for rid, rec in table.iter_records()
            if where is None or _matches_predicate(table.schema, rec, where)
        ]
        for rid, rec in targets:
            if table.delete(rid):
                for field in table.schema.field_names:
                    self.index_manager.remove(table_name, field, rec[field], rid)
        if targets:
            self._table_stats[table_name]["writes"] += len(targets)
        return len(targets)

    def create_index(
        self,
        table_name: str,
        field: str,
        index_type: str = "auto",
        reads: int | None = None,
        writes: int | None = None,
        range_reads: int | None = None,
    ) -> str:
        table = self._get_table(table_name)
        col = table.schema.column(field)
        if col.dtype == DataType.VECTOR:
            raise TypeError("Cannot build scalar B+/LSM index on VECTOR column")
        stats = self._table_stats[table_name]
        if reads is None and writes is None:
            total = stats["reads"] + stats["writes"] + stats["range_reads"]
            r = stats["reads"] if total > 0 else 100
            w = stats["writes"] if total > 0 else 10
            rr = stats["range_reads"] if total > 0 else 0
        else:
            r = 100 if reads is None else reads
            w = 10 if writes is None else writes
            rr = 0 if range_reads is None else range_reads
        kind, idx = self.index_manager.create(
            table_name, field, index_type, key_dtype=col.dtype, reads=r, writes=w, range_reads=rr, replace=True
        )
        for rid, rec in table.iter_records(fields=[field]):
            idx.insert(rec[field], rid)
        self._save_indexes_catalog()
        return kind

    def _lookup_rids(self, table: TableStorage, table_name: str, where: Predicate) -> list[RID] | None:
        if where.op == "=":
            target = table.schema.coerce_literal(where.field, where.value)
            return self.index_manager.search(table_name, where.field, target)
        if where.op == "IS_NULL":
            return self.index_manager.search(table_name, where.field, None)
        if where.op in {"<", "<=", ">", ">=", "BETWEEN"}:
            low = table.schema.coerce_literal(where.field, where.value) if where.op in {">", ">=", "BETWEEN"} else None
            high = (
                table.schema.coerce_literal(where.field, where.high)
                if where.op == "BETWEEN"
                else (table.schema.coerce_literal(where.field, where.value) if where.op in {"<", "<="} else None)
            )
            pairs = self.index_manager.range_search(
                table_name,
                where.field,
                low=low,
                high=high,
                include_low=where.op in {">=", "BETWEEN"},
                include_high=where.op in {"<=", "BETWEEN"},
            )
            if pairs is None:
                return None
            return [rid for key, rid in pairs if key is not None]
        return None

    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: Predicate | None = None,
    ) -> list[dict[str, Any]]:
        table = self._get_table(table_name)
        proj = columns if columns is not None else table.schema.field_names
        for c in proj:
            table.schema.column(c)
        if where is None:
            self._table_stats[table_name]["reads"] += 1
            return table.scan(fields=proj)

        table.schema.column(where.field)
        if where.op in {"<", "<=", ">", ">=", "BETWEEN"}:
            self._table_stats[table_name]["range_reads"] += 1
        else:
            self._table_stats[table_name]["reads"] += 1

        rids = self._lookup_rids(table, table_name, where)
        if rids is not None:
            fetch_cols = list(dict.fromkeys([*proj, where.field]))
            out: list[dict[str, Any]] = []
            for rid in rids:
                rec = table.try_get_record(rid, fields=fetch_cols)
                if rec is not None and _matches_predicate(table.schema, rec, where):
                    out.append({c: rec[c] for c in proj})
            return out

        needed = list(dict.fromkeys([*proj, where.field]))
        return [
            {c: rec[c] for c in proj}
            for rec in table.scan(fields=needed)
            if _matches_predicate(table.schema, rec, where)
        ]

    def execute(self, sql: str) -> list[dict[str, Any]] | int | str:
        stmt = parse_sql(sql)
        if isinstance(stmt, SelectStmt):
            return self.select(stmt.table, columns=stmt.columns, where=stmt.where)
        if isinstance(stmt, CreateTableStmt):
            self.create_table(stmt.table, stmt.columns, layout=stmt.layout)
            return f"Created table {stmt.table}"
        if isinstance(stmt, InsertStmt):
            table = self._get_table(stmt.table)
            fields = table.schema.field_names
            if len(stmt.values) != len(fields):
                raise ValueError(f"Expected {len(fields)} values, got {len(stmt.values)}")
            self.insert(stmt.table, dict(zip(fields, stmt.values)))
            return 1
        if isinstance(stmt, CreateIndexStmt):
            kind = self.create_index(
                stmt.table, stmt.field, stmt.index_type, reads=stmt.reads, writes=stmt.writes
            )
            return f"Created {kind} index on {stmt.table}.{stmt.field}"
        if isinstance(stmt, UpdateStmt):
            return self.update(stmt.table, stmt.updates, where=stmt.where)
        if isinstance(stmt, DeleteStmt):
            return self.delete(stmt.table, where=stmt.where)
        raise ValueError("Unsupported SQL statement")

    def explain(self, sql: str) -> str:
        stmt = parse_sql(sql)
        if not isinstance(stmt, SelectStmt):
            return f"COMMAND {type(stmt).__name__}"
        table = self._get_table(stmt.table)
        layout = self.schemas[stmt.table]["layout"]
        cols = ",".join(stmt.columns) if stmt.columns is not None else "*"
        if stmt.where is None:
            return f"{layout.upper()} TABLE SCAN {stmt.table} [cols={cols}]"
        kind = self.index_manager.kind(stmt.table, stmt.where.field)
        if kind and stmt.where.op in {"=", "IS_NULL"}:
            return f"INDEX POINT LOOKUP {kind} ON {stmt.table}.{stmt.where.field} [cols={cols}]"
        if kind and stmt.where.op in {"<", "<=", ">", ">=", "BETWEEN"}:
            return f"INDEX RANGE SCAN {kind} ON {stmt.table}.{stmt.where.field} [cols={cols}]"
        return f"{layout.upper()} TABLE SCAN {stmt.table} FILTER {stmt.where.field} {stmt.where.op} [cols={cols}]"

    def buffer_stats(self) -> dict[str, int | float | str]:
        return self.pool.snapshot()

    def close(self) -> None:
        self.index_manager.close()
        self.pool.close()
        self.storage_manager.close_all()
        self._save_catalog(clean_shutdown=True)

    def __enter__(self) -> "SmartRetailDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
