"""High-level Smart Retail DB engine integrating unified storage, buffer pool, and indexes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalog import CatalogStore
from ..core import (
    AUTO_DEFAULT_READS,
    AUTO_DEFAULT_WRITES,
    RID,
    ColumnSpec,
    DataType,
    TableSchema,
    load_settings,
)
from ..indexes import IndexManager, WorkloadProfile
from ..query import (
    CreateIndexStmt,
    CreateTableStmt,
    DeleteStmt,
    InsertStmt,
    Predicate,
    SelectStmt,
    UpdateStmt,
    matches_predicate,
    parse_sql,
)
from ..storage import (
    BufferPool,
    ColumnTable,
    CryptoHooks,
    RowTable,
    StorageManager,
    TableStorage,
)


class SmartRetailDB:
    """Custom database engine with shared BufferPool, Row/Column tables, and B+/LSM indexes."""

    def __init__(
        self,
        directory: str | Path | None = None,
        buffer_frames: int | None = None,
        replacement_policy: str | None = None,
        backend_kind: str | None = None,
        crypto: CryptoHooks | None = None,
    ):
        settings = load_settings()
        self.directory = Path(directory if directory is not None else settings.db_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer_frames = settings.buffer_frames if buffer_frames is None else buffer_frames
        self.replacement_policy = settings.replacement_policy if replacement_policy is None else replacement_policy
        self.backend_kind = settings.backend_kind if backend_kind is None else backend_kind
        self.storage_manager = StorageManager(
            self.directory,
            backend_kind=self.backend_kind,
            crypto=crypto,
            physical_page_size=settings.page_size,
        )
        self.pool = BufferPool(capacity=self.buffer_frames, policy=self.replacement_policy, storage_manager=self.storage_manager)
        self.index_manager = IndexManager(
            self.directory / "index_data",
            self.storage_manager,
            self.pool,
            lsm_flush_threshold=settings.lsm_flush_threshold,
            lsm_compaction_threshold=settings.lsm_compaction_threshold,
        )
        self._catalog = CatalogStore(self.directory)
        self.schemas = self._catalog.schemas
        self.tables: dict[str, TableStorage] = {}

        self._open_tables()
        self._load_indexes()
        if not self._catalog.clean_shutdown:
            self._reconcile_indexes()
        self._catalog.save_catalog(clean_shutdown=False)

    def _open_tables(self) -> None:
        for name, meta in self.schemas.items():
            schema = TableSchema.from_dict(meta)
            cls = RowTable if meta["layout"] == "row" else ColumnTable
            self.tables[name] = cls(self.storage_manager, self.pool, name, schema)

    def _load_indexes(self) -> None:
        for item in self._catalog.load_indexes():
            tname, field, kind = item["table"], item["field"], item["type"]
            if tname in self.tables and field in self.tables[tname].schema.field_names:
                dtype = self.tables[tname].schema.column(field).dtype
                self.index_manager.create(tname, field, kind, key_dtype=dtype, replace=False)

    def _reconcile_indexes(self) -> None:
        for (tname, field), (kind, _) in list(self.index_manager.indexes.items()):
            table = self.tables[tname]
            _, idx = self.index_manager.create(
                tname, field, kind, key_dtype=table.schema.column(field).dtype, replace=True
            )
            for rid, rec in table.iter_records(fields=[field]):
                idx.insert(rec[field], rid)

    def _get_table(self, name: str) -> TableStorage:
        if name not in self.tables:
            raise KeyError(f"Unknown table {name!r}")
        return self.tables[name]

    def create_table(
        self, name: str, schema_or_fields: list[str] | list[ColumnSpec] | dict[str, Any] | TableSchema,
        layout: str = "row",
    ) -> None:
        if not name or not name.isidentifier():
            raise ValueError(f"Invalid identifier: {name!r}")
        if name in self.tables:
            raise ValueError(f"Table {name} already exists")
        if layout not in {"row", "column"}:
            raise ValueError("layout must be 'row' or 'column'")
        schema = schema_or_fields if isinstance(schema_or_fields, TableSchema) else TableSchema.from_spec(schema_or_fields)
        stats = {"reads": 0, "writes": 0, "range_reads": 0}
        self.schemas[name] = {**schema.to_dict(), "layout": layout, "backend_kind": self.backend_kind, "stats": stats}
        self._catalog.table_stats[name] = dict(stats)
        cls = RowTable if layout == "row" else ColumnTable
        self.tables[name] = cls(self.storage_manager, self.pool, name, schema)
        self._catalog.save_catalog(clean_shutdown=False)

    def insert(self, table_name: str, record: dict[str, Any]) -> RID:
        table = self._get_table(table_name)
        validated = table.schema.validate_record(record)
        rid = table.insert(validated)
        for field in table.schema.field_names:
            self.index_manager.add(table_name, field, validated[field], rid)
        self._catalog.table_stats[table_name]["writes"] += 1
        return rid

    def _matching_records(self, table: TableStorage, table_name: str, where: Predicate | None) -> list[tuple[RID, dict[str, Any]]]:
        if where is not None:
            table.schema.column(where.field)
            rids = self._lookup_rids(table, table_name, where)
            if rids is not None:
                matched: list[tuple[RID, dict[str, Any]]] = []
                for rid in dict.fromkeys(rids):
                    rec = table.try_get_record(rid)
                    if rec is not None and matches_predicate(table.schema, rec, where):
                        matched.append((rid, rec))
                return matched
        return [
            (rid, rec) for rid, rec in table.iter_records()
            if where is None or matches_predicate(table.schema, rec, where)
        ]

    def update(self, table_name: str, updates: dict[str, Any], where: Predicate | None = None) -> int:
        table = self._get_table(table_name)
        for col_name, val in updates.items():
            table.schema.validate_value(table.schema.column(col_name), val)
        targets = self._matching_records(table, table_name, where)
        for old_rid, old_rec in targets:
            new_rec = {**old_rec, **updates}
            new_rid = table.update(old_rid, updates)
            for field in table.schema.field_names:
                if old_rid != new_rid or old_rec[field] != new_rec[field]:
                    self.index_manager.remove(table_name, field, old_rec[field], old_rid)
                    self.index_manager.add(table_name, field, new_rec[field], new_rid)
        if targets:
            self._catalog.table_stats[table_name]["writes"] += len(targets)
        return len(targets)

    def delete(self, table_name: str, where: Predicate | None = None) -> int:
        table = self._get_table(table_name)
        targets = self._matching_records(table, table_name, where)
        for rid, rec in targets:
            if table.delete(rid):
                for field in table.schema.field_names:
                    self.index_manager.remove(table_name, field, rec[field], rid)
        if targets:
            self._catalog.table_stats[table_name]["writes"] += len(targets)
        return len(targets)

    def create_index(
        self, table_name: str, field: str, index_type: str = "auto",
        reads: int | None = None, writes: int | None = None, range_reads: int | None = None,
    ) -> str:
        table = self._get_table(table_name)
        col = table.schema.column(field)
        if col.dtype == DataType.VECTOR:
            raise TypeError("Cannot build scalar B+/LSM index on VECTOR column")
        stats = self._catalog.table_stats[table_name]
        has_ops = (stats["reads"] + stats["writes"] + stats["range_reads"]) > 0
        workload = WorkloadProfile(
            reads=(stats["reads"] if has_ops else AUTO_DEFAULT_READS) if reads is None else reads,
            writes=(stats["writes"] if has_ops else AUTO_DEFAULT_WRITES) if writes is None else writes,
            range_reads=(stats["range_reads"] if has_ops else 0) if range_reads is None else range_reads,
        )
        kind, idx = self.index_manager.create(
            table_name, field, index_type, key_dtype=col.dtype, workload=workload, replace=True
        )
        for rid, rec in table.iter_records(fields=[field]):
            idx.insert(rec[field], rid)
        self._catalog.save_indexes([(t, f, k) for (t, f), (k, _) in self.index_manager.indexes.items()])
        return kind

    def _lookup_rids(self, table: TableStorage, table_name: str, where: Predicate) -> list[RID] | None:
        if where.op == "=":
            return self.index_manager.search(table_name, where.field, table.schema.coerce_literal(where.field, where.value))
        if where.op == "IS_NULL":
            return self.index_manager.search(table_name, where.field, None)
        if where.op in {"<", "<=", ">", ">=", "BETWEEN"}:
            low = table.schema.coerce_literal(where.field, where.value) if where.op in {">", ">=", "BETWEEN"} else None
            raw_hi = where.high if where.op == "BETWEEN" else (where.value if where.op in {"<", "<="} else None)
            high = table.schema.coerce_literal(where.field, raw_hi) if raw_hi is not None else None
            pairs = self.index_manager.range_search(
                table_name, where.field, low=low, high=high,
                include_low=where.op in {">=", "BETWEEN"}, include_high=where.op in {"<=", "BETWEEN"},
            )
            return None if pairs is None else [rid for key, rid in pairs if key is not None]
        return None

    def select(self, table_name: str, columns: list[str] | None = None, where: Predicate | None = None) -> list[dict[str, Any]]:
        table = self._get_table(table_name)
        proj = columns if columns is not None else table.schema.field_names
        for c in proj:
            table.schema.column(c)
        if where is None:
            self._catalog.table_stats[table_name]["reads"] += 1
            return table.scan(fields=proj)
        table.schema.column(where.field)
        stat_key = "range_reads" if where.op in {"<", "<=", ">", ">=", "BETWEEN"} else "reads"
        self._catalog.table_stats[table_name][stat_key] += 1

        rids = self._lookup_rids(table, table_name, where)
        needed = list(dict.fromkeys([*proj, where.field]))
        if rids is not None:
            rows = (table.try_get_record(rid, fields=needed) for rid in rids)
            return [{c: r[c] for c in proj} for r in rows if r is not None and matches_predicate(table.schema, r, where)]
        return [{c: r[c] for c in proj} for r in table.scan(fields=needed) if matches_predicate(table.schema, r, where)]

    def execute(self, sql: str) -> list[dict[str, Any]] | int | str:
        stmt = parse_sql(sql)
        if isinstance(stmt, SelectStmt):
            return self.select(stmt.table, columns=stmt.columns, where=stmt.where)
        if isinstance(stmt, CreateTableStmt):
            self.create_table(stmt.table, stmt.columns, layout=stmt.layout)
            return f"Created table {stmt.table}"
        if isinstance(stmt, InsertStmt):
            fields = self._get_table(stmt.table).schema.field_names
            if len(stmt.values) != len(fields):
                raise ValueError(f"Expected {len(fields)} values, got {len(stmt.values)}")
            self.insert(stmt.table, dict(zip(fields, stmt.values)))
            return 1
        if isinstance(stmt, CreateIndexStmt):
            kind = self.create_index(stmt.table, stmt.field, stmt.index_type, reads=stmt.reads, writes=stmt.writes)
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
        self._get_table(stmt.table)
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
        self._catalog.save_catalog(clean_shutdown=True)

    def __enter__(self) -> "SmartRetailDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
