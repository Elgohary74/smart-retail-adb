"""
Module: src/query/simple_executor.py
Layer: Layer 3 — Parameterized Query Parser & Access-Path Router
Lock Domain: CatalogLock (``self._lock = threading.RLock()``; acquires ``CatalogLock -> IndexLock -> TableHeapLock``)

Purpose:
    Implements ``SimpleExecutor``, a parameterized, SQL-injection-proof query
    shell supporting index DDL (``BTREE``, ``LSM``, and ``WorkloadAdvisor``
    ``AUTO`` selection), DML (``INSERT``, ``UPDATE`` with cross-page ``RID``
    relocation index maintenance, ``DELETE``), point/range/full-scan ``SELECT``
    routing, ``EXPLAIN`` plan inspection, and persistent index reopening across
    engine restarts.

Key Invariants:
    1. All literal values must be passed via ``params`` using ``?``
       placeholders; raw quotes, comments, stacked semicolons, tautologies
       (e.g., ``OR 1=1``), ``UNION``, and unparameterized literals raise
       ``SecurityViolationError`` before execution.
    2. Lock acquisition strictly flows top-down:
       ``CatalogLock -> IndexLock -> TableHeapLock -> BufferPoolLock -> DiskManagerLock``.
    3. Uniqueness of primary-key / unique indexes is verified before heap
       mutation so duplicate-key violations never leave orphaned heap tuples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Any

from src.common.exceptions import (
    DuplicateKeyError,
    QuerySyntaxError,
    SecurityViolationError,
    StorageError,
)
from src.common.types import RID, ColumnDef, Schema, encode_key
from src.index.base_index import BaseIndex
from src.index.bplus_tree import BPlusTreeIndex
from src.index.index_advisor import IndexStrategy, WorkloadAdvisor
from src.index.lsm_tree import LSMTreeIndex
from src.query.catalog import CatalogManager
from src.storage.storage_engine import StorageEngine, TableHeap
from src.storage.table_heap import RowTableHeap

_QUOTE_OR_NULL_RE = re.compile(r"['\"`\x00]")
_COMMENT_RE = re.compile(r"--|/\*|\*/")
_FORBIDDEN_SQL_KEYWORDS_RE = re.compile(
    r"\b(?:UNION|DROP|ALTER|TRUNCATE|GRANT|REVOKE|EXEC|EXECUTE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)
_FORBIDDEN_OR_RE = re.compile(r"\bOR\b", re.IGNORECASE)
_TAUTOLOGY_RE = re.compile(r"\b(?:1\s*=\s*1|0\s*=\s*0)\b", re.IGNORECASE)

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

_CREATE_INDEX_RE = re.compile(
    rf"^CREATE\s+INDEX\s+(?P<idx>{_IDENT})\s+ON\s+(?P<table>{_IDENT})\s*"
    rf"\(\s*(?P<col>{_IDENT})\s*\)"
    rf"(?:\s+USING\s+(?P<using>{_IDENT}))?$",
    re.IGNORECASE,
)
_INSERT_RE = re.compile(
    rf"^INSERT\s+INTO\s+(?P<table>{_IDENT})\s+VALUES\s*\((?P<vals>.*)\)$",
    re.IGNORECASE,
)
_SELECT_ALL_RE = re.compile(
    rf"^SELECT\s+\*\s+FROM\s+(?P<table>{_IDENT})$",
    re.IGNORECASE,
)
_SELECT_WHERE_RE = re.compile(
    rf"^SELECT\s+\*\s+FROM\s+(?P<table>{_IDENT})\s+WHERE\s+(?P<where>.+)$",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    rf"^UPDATE\s+(?P<table>{_IDENT})\s+SET\s+(?P<set_part>.+?)\s+WHERE\s+(?P<where>.+)$",
    re.IGNORECASE,
)
_DELETE_RE = re.compile(
    rf"^DELETE\s+FROM\s+(?P<table>{_IDENT})\s+WHERE\s+(?P<where>.+)$",
    re.IGNORECASE,
)

_WHERE_EQ_RE = re.compile(rf"^(?P<col>{_IDENT})\s*=\s*\?$")
_WHERE_BETWEEN_RE = re.compile(
    rf"^(?P<col>{_IDENT})\s+BETWEEN\s+\?\s+AND\s+\?$",
    re.IGNORECASE,
)
_SET_ITEM_RE = re.compile(rf"^(?P<col>{_IDENT})\s*=\s*(?P<rhs>.+)$")


@dataclass(frozen=True)
class _CreateIndexStmt:
    index_name: str
    table_name: str
    column_name: str
    using: str  # "BTREE", "LSM", or "AUTO"


@dataclass(frozen=True)
class _InsertStmt:
    table_name: str
    param_count: int


@dataclass(frozen=True)
class _SelectAllStmt:
    table_name: str


@dataclass(frozen=True)
class _SelectEqStmt:
    table_name: str
    column_name: str


@dataclass(frozen=True)
class _SelectBetweenStmt:
    table_name: str
    column_name: str


@dataclass(frozen=True)
class _UpdateStmt:
    table_name: str
    set_columns: tuple[str, ...]
    where_column: str


@dataclass(frozen=True)
class _DeleteStmt:
    table_name: str
    where_column: str


_ParsedStmt = (
    _CreateIndexStmt
    | _InsertStmt
    | _SelectAllStmt
    | _SelectEqStmt
    | _SelectBetweenStmt
    | _UpdateStmt
    | _DeleteStmt
)


def _guard_sql_security(sql: str) -> str:
    """Reject raw quotes, comments, stacked semicolons, and injection keywords."""
    if not isinstance(sql, str):
        raise QuerySyntaxError(f"SQL statement must be a str, got {type(sql).__name__}")
    stripped = sql.strip()
    if not stripped:
        raise QuerySyntaxError("SQL statement cannot be empty")

    if _QUOTE_OR_NULL_RE.search(sql):
        raise SecurityViolationError(
            "Embedded quotes or null bytes are forbidden; pass literals via params with '?'"
        )
    if _COMMENT_RE.search(sql):
        raise SecurityViolationError("SQL comments ('--', '/*', '*/') are forbidden")

    semi_count = sql.count(";")
    if semi_count > 1 or (semi_count == 1 and not stripped.endswith(";")):
        raise SecurityViolationError(
            "Stacked SQL statements after ';' are forbidden"
        )

    clean_sql = stripped[:-1].strip() if semi_count == 1 else stripped
    if not clean_sql:
        raise QuerySyntaxError("SQL statement cannot be empty")

    if _FORBIDDEN_SQL_KEYWORDS_RE.search(clean_sql):
        raise SecurityViolationError("Forbidden SQL keyword detected")
    if _FORBIDDEN_OR_RE.search(clean_sql):
        raise SecurityViolationError(
            "Boolean OR clauses / tautology injections are forbidden"
        )
    if _TAUTOLOGY_RE.search(clean_sql):
        raise SecurityViolationError("Boolean tautology expression detected")

    return clean_sql


def _parse_where_predicate(where_clause: str, *, allow_between: bool) -> tuple[str, str]:
    """Validate and parse a parameterized ``WHERE`` predicate."""
    where = where_clause.strip()
    m_eq = _WHERE_EQ_RE.match(where)
    if m_eq is not None:
        return "EQ", m_eq.group("col")

    m_btw = _WHERE_BETWEEN_RE.match(where)
    if m_btw is not None:
        if not allow_between:
            raise QuerySyntaxError(
                "BETWEEN predicates are only supported on SELECT statements"
            )
        return "BETWEEN", m_btw.group("col")

    # Reject unparameterized literals (e.g. col = 1, 1 = 1, col BETWEEN 1 AND 2) as security violations.
    if re.match(rf"^{_IDENT}\s+BETWEEN\b", where, re.IGNORECASE):
        raise SecurityViolationError(
            "Unparameterized literal or trailing injection in BETWEEN clause; use 'BETWEEN ? AND ?'"
        )
    if re.search(r"=\s*[^?\s]", where) or re.match(r"^\d+\s*=", where):
        raise SecurityViolationError(
            "Unparameterized literal in WHERE predicate; use '= ?'"
        )
    if re.search(r"\bBETWEEN\b", where, re.IGNORECASE):
        raise SecurityViolationError("Malformed or unparameterized BETWEEN predicate")

    raise QuerySyntaxError(f"Unsupported WHERE predicate: {where!r}")


def _parse_sql(sql: str) -> tuple[_ParsedStmt, int]:
    """Validate SQL security/syntax and return ``(parsed_statement, expected_param_count)``."""
    clean_sql = _guard_sql_security(sql)
    upper_head = clean_sql.split(None, 1)[0].upper()

    if upper_head == "CREATE":
        m = _CREATE_INDEX_RE.match(clean_sql)
        if m is None:
            raise QuerySyntaxError(f"Invalid CREATE INDEX syntax: {clean_sql!r}")
        raw_using = (m.group("using") or "BTREE").upper()
        using_map = {
            "BTREE": "BTREE",
            "BPLUS_TREE": "BTREE",
            "LSM": "LSM",
            "LSM_TREE": "LSM",
            "AUTO": "AUTO",
        }
        using = using_map.get(raw_using)
        if using is None:
            raise QuerySyntaxError(
                f"Unsupported index strategy {raw_using!r}; expected BTREE, LSM, or AUTO"
            )
        return (
            _CreateIndexStmt(
                index_name=m.group("idx"),
                table_name=m.group("table"),
                column_name=m.group("col"),
                using=using,
            ),
            0,
        )

    if upper_head == "INSERT":
        m = _INSERT_RE.match(clean_sql)
        if m is None:
            if re.search(r"\bVALUES\s*\([^)]*[^?\s,()][^)]*\)", clean_sql, re.IGNORECASE):
                raise SecurityViolationError(
                    "Unparameterized literal in INSERT VALUES; use '?' placeholders"
                )
            raise QuerySyntaxError(f"Invalid INSERT syntax: {clean_sql!r}")

        raw_vals = m.group("vals").strip()
        if not raw_vals:
            raise QuerySyntaxError("INSERT VALUES list cannot be empty")
        items = [part.strip() for part in raw_vals.split(",")]
        for item in items:
            if not item:
                raise QuerySyntaxError(f"Malformed INSERT VALUES list: {raw_vals!r}")
            if item != "?":
                raise SecurityViolationError(
                    f"Unparameterized literal {item!r} in INSERT VALUES; use '?'"
                )
        return _InsertStmt(table_name=m.group("table"), param_count=len(items)), len(items)

    if upper_head == "SELECT":
        m_all = _SELECT_ALL_RE.match(clean_sql)
        if m_all is not None:
            return _SelectAllStmt(table_name=m_all.group("table")), 0

        m_where = _SELECT_WHERE_RE.match(clean_sql)
        if m_where is not None:
            kind, col_name = _parse_where_predicate(m_where.group("where"), allow_between=True)
            if kind == "EQ":
                return _SelectEqStmt(table_name=m_where.group("table"), column_name=col_name), 1
            return _SelectBetweenStmt(table_name=m_where.group("table"), column_name=col_name), 2

        if re.search(r"\bWHERE\b", clean_sql, re.IGNORECASE):
            raise SecurityViolationError("Malformed or unparameterized SELECT WHERE clause")
        raise QuerySyntaxError(f"Unsupported SELECT syntax: {clean_sql!r}")

    if upper_head == "UPDATE":
        m_upd = _UPDATE_RE.match(clean_sql)
        if m_upd is None:
            if re.search(r"\bSET\b.*=\s*[^?\s]", clean_sql, re.IGNORECASE):
                raise SecurityViolationError(
                    "Unparameterized literal in UPDATE SET clause; use '= ?'"
                )
            raise QuerySyntaxError(f"Invalid UPDATE syntax: {clean_sql!r}")

        set_cols: list[str] = []
        seen_cols: set[str] = set()
        for raw_assign in m_upd.group("set_part").split(","):
            assign = raw_assign.strip()
            m_item = _SET_ITEM_RE.match(assign)
            if m_item is None:
                raise QuerySyntaxError(f"Malformed UPDATE SET assignment: {assign!r}")
            rhs = m_item.group("rhs").strip()
            if rhs != "?":
                raise SecurityViolationError(
                    f"Unparameterized literal {rhs!r} in UPDATE SET clause; use '= ?'"
                )
            col_name = m_item.group("col")
            if col_name in seen_cols:
                raise QuerySyntaxError(f"Duplicate column {col_name!r} in UPDATE SET clause")
            seen_cols.add(col_name)
            set_cols.append(col_name)

        _, where_col = _parse_where_predicate(m_upd.group("where"), allow_between=False)
        return (
            _UpdateStmt(
                table_name=m_upd.group("table"),
                set_columns=tuple(set_cols),
                where_column=where_col,
            ),
            len(set_cols) + 1,
        )

    if upper_head == "DELETE":
        m_del = _DELETE_RE.match(clean_sql)
        if m_del is None:
            raise QuerySyntaxError(f"Invalid DELETE syntax: {clean_sql!r}")
        _, where_col = _parse_where_predicate(m_del.group("where"), allow_between=False)
        return _DeleteStmt(table_name=m_del.group("table"), where_column=where_col), 1

    raise QuerySyntaxError(f"Unsupported SQL statement: {clean_sql!r}")


def _validate_col_key(col_def: ColumnDef, value: Any) -> bytes:
    """Validate and encode a query/index parameter against ``col_def``."""
    return encode_key(value, col_def.dtype, col_def.key_size)


def _normalize_index_type(raw_type: str) -> str:
    """Map persisted/input index type aliases to canonical ``BPLUS_TREE`` / ``LSM_TREE``."""
    upper = raw_type.upper()
    if upper in {"BPLUS_TREE", "BTREE"}:
        return "BPLUS_TREE"
    if upper in {"LSM_TREE", "LSM"}:
        return "LSM_TREE"
    raise StorageError(f"Unsupported persisted index type {raw_type!r}")


class SimpleExecutor:
    """Parameterized SQL-like executor coordinating catalog, indexes, and storage.

    Maintains active ``BPlusTreeIndex`` and ``LSMTreeIndex`` instances across
    DDL/DML operations and reopens persisted indexes automatically on startup.
    """

    def __init__(
        self,
        storage_engine: StorageEngine,
        catalog: CatalogManager,
        advisor: WorkloadAdvisor | None = None,
        indexes_dir: str | Path | None = None,
    ) -> None:
        """Initialize the query executor and reopen persisted catalog indexes.

        Args:
            storage_engine: Open ``StorageEngine`` managing table heaps.
            catalog: Persistent ``CatalogManager`` tracking schemas, stats, and indexes.
            advisor: Optional ``WorkloadAdvisor`` for telemetry and ``USING AUTO``
                index selection (defaults to a fresh ``WorkloadAdvisor()``).
            indexes_dir: Directory storing ``*.bpt`` files and LSM directories
                (defaults to ``storage_engine.data_dir / "indexes"``).

        Raises:
            TypeError: If ``storage_engine``, ``catalog``, or ``advisor`` has an
                invalid type.
            StorageError: If reopening persisted index files fails.
            PageCorruptedError: If a persisted index fails checksum/manifest validation.
        """
        if not isinstance(storage_engine, StorageEngine):
            raise TypeError(
                f"storage_engine must be a StorageEngine, got {type(storage_engine).__name__}"
            )
        if not isinstance(catalog, CatalogManager):
            raise TypeError(
                f"catalog must be a CatalogManager, got {type(catalog).__name__}"
            )
        if advisor is not None and not isinstance(advisor, WorkloadAdvisor):
            raise TypeError(
                f"advisor must be a WorkloadAdvisor or None, got {type(advisor).__name__}"
            )

        self._storage: StorageEngine = storage_engine
        self._catalog: CatalogManager = catalog
        self._advisor: WorkloadAdvisor = advisor if advisor is not None else WorkloadAdvisor()
        self.indexes_dir: Path = (
            Path(indexes_dir)
            if indexes_dir is not None
            else self._storage.data_dir / "indexes"
        )
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self._lock: threading.RLock = threading.RLock()
        self._closed: bool = False

        self._indexes_by_col: dict[tuple[str, str], BaseIndex] = {}
        self._indexes_by_name: dict[str, tuple[str, str, BaseIndex]] = {}
        self._table_indexes: dict[str, list[tuple[str, str, BaseIndex]]] = {}

        try:
            self._reopen_persisted_indexes()
        except Exception:
            self._close_indexes_quietly()
            self._closed = True
            raise

    def _ensure_open(self) -> None:
        """Raise ``StorageError`` if this executor has already been closed."""
        if self._closed:
            raise StorageError("SimpleExecutor is closed")

    def _close_indexes_quietly(self) -> None:
        """Close all opened index handles during failed initialization."""
        for _, _, idx in self._indexes_by_name.values():
            try:
                idx.close()
            except Exception:
                pass
        self._indexes_by_col.clear()
        self._indexes_by_name.clear()
        self._table_indexes.clear()

    def _resolve_index_path(self, index_name: str, index_type: str, raw_path: Any) -> Path:
        """Resolve an index path from catalog metadata with fallback to ``indexes_dir``."""
        if raw_path:
            candidate = Path(str(raw_path))
            if candidate.is_absolute() or candidate.exists():
                return candidate
            local_fallback = self.indexes_dir / candidate.name
            if local_fallback.exists():
                return local_fallback
            return candidate
        if index_type == "BPLUS_TREE":
            return self.indexes_dir / f"{index_name}.bpt"
        return self.indexes_dir / index_name

    def _open_index(
        self,
        index_type: str,
        idx_path: Path,
        col_def: ColumnDef,
        unique: bool,
    ) -> BaseIndex:
        """Instantiate either a ``BPlusTreeIndex`` or ``LSMTreeIndex`` handle."""
        if index_type == "BPLUS_TREE":
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            return BPlusTreeIndex(
                idx_path,
                key_dtype=col_def.dtype,
                key_size=col_def.key_size,
                unique=unique,
            )
        if index_type == "LSM_TREE":
            return LSMTreeIndex(
                idx_path,
                key_dtype=col_def.dtype,
                key_size=col_def.key_size,
                unique=unique,
            )
        raise StorageError(f"Unsupported index type {index_type!r}")

    def _register_active_index(
        self,
        table_name: str,
        index_name: str,
        column_name: str,
        idx: BaseIndex,
    ) -> None:
        """Track an open index handle across executor lookup tables."""
        self._indexes_by_col[(table_name, column_name)] = idx
        self._indexes_by_name[index_name] = (table_name, column_name, idx)
        self._table_indexes.setdefault(table_name, []).append(
            (index_name, column_name, idx)
        )

    def _reopen_persisted_indexes(self) -> None:
        """Reload every persisted index already registered in ``self._catalog``."""
        for table_name in self._catalog.list_tables():
            schema = self._catalog.get_schema(table_name)
            indexes = self._catalog.get_indexes(table_name)
            for index_name, meta in indexes.items():
                col_name = str(meta["column"])
                col_def = schema.get_column(col_name)
                index_type = _normalize_index_type(str(meta["type"]))
                unique = bool(meta.get("unique", col_name == schema.primary_key))
                idx_path = self._resolve_index_path(index_name, index_type, meta.get("path"))
                idx = self._open_index(index_type, idx_path, col_def, unique)
                self._register_active_index(table_name, index_name, col_name, idx)

    def _require_column(self, schema: Schema, column_name: str) -> ColumnDef:
        """Return ``ColumnDef`` for ``column_name`` or raise ``QuerySyntaxError``."""
        try:
            return schema.get_column(column_name)
        except KeyError as exc:
            raise QuerySyntaxError(
                f"Unknown column {column_name!r} on table {schema.table_name!r}"
            ) from exc

    def _get_table_and_schema(self, table_name: str) -> tuple[Schema, TableHeap]:
        """Resolve ``(Schema, TableHeap)`` from catalog and storage engine."""
        schema = self._catalog.get_schema(table_name)
        heap = self._storage.get_table(table_name)
        return schema, heap

    def _find_column_index(
        self, table_name: str, column_name: str
    ) -> tuple[str, BaseIndex] | None:
        """Return ``(index_name, BaseIndex)`` for ``(table_name, column_name)`` if indexed."""
        for idx_name, col, idx in self._table_indexes.get(table_name, []):
            if col == column_name:
                return idx_name, idx
        return None

    @staticmethod
    def _cleanup_new_index_path(idx_path: Path, index_type: str) -> None:
        """Remove newly created index artifacts if DDL backfill or catalog registration fails."""
        if index_type == "BPLUS_TREE":
            idx_path.unlink(missing_ok=True)
            return
        if idx_path.is_dir():
            for child in idx_path.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
            try:
                idx_path.rmdir()
            except OSError:
                pass

    def _locate_matching_rows(
        self,
        table_name: str,
        heap: TableHeap,
        where_col: str,
        where_val: Any,
    ) -> list[tuple[RID, dict[str, Any]]]:
        """Find matching ``(RID, row_dict)`` pairs using an active index or full scan."""
        idx = self._indexes_by_col.get((table_name, where_col))
        if idx is not None:
            matches: list[tuple[RID, dict[str, Any]]] = []
            for rid in idx.get(where_val):
                row = heap.get(rid)
                if row is not None:
                    matches.append((rid, row))
            return matches

        return [
            (rid, row)
            for rid, row in heap.full_scan()
            if row[where_col] == where_val
        ]

    def _execute_create_index(self, stmt: _CreateIndexStmt) -> dict[str, Any]:
        """Execute ``CREATE INDEX ... [USING BTREE | LSM | AUTO]`` with backfill."""
        schema, heap = self._get_table_and_schema(stmt.table_name)
        col_def = self._require_column(schema, stmt.column_name)

        if stmt.index_name in self._indexes_by_name or stmt.index_name in self._catalog.get_indexes(
            stmt.table_name
        ):
            raise StorageError(f"Index {stmt.index_name!r} already exists")
        if (stmt.table_name, stmt.column_name) in self._indexes_by_col:
            raise StorageError(
                f"Column {stmt.table_name}.{stmt.column_name} already has an active index"
            )

        if stmt.using == "AUTO":
            rec = self._advisor.recommend(
                stmt.table_name,
                stmt.column_name,
                table_pages=heap.page_count(),
            )
            index_type = (
                "LSM_TREE"
                if rec.strategy == IndexStrategy.LSM_TREE
                else "BPLUS_TREE"
            )
        elif stmt.using == "LSM":
            index_type = "LSM_TREE"
        else:
            index_type = "BPLUS_TREE"

        unique = stmt.column_name == schema.primary_key
        idx_path = (
            self.indexes_dir / f"{stmt.index_name}.bpt"
            if index_type == "BPLUS_TREE"
            else self.indexes_dir / stmt.index_name
        )
        existed_before = idx_path.exists()

        idx = self._open_index(index_type, idx_path, col_def, unique)
        try:
            for rid, row in heap.full_scan():
                key_val = row[stmt.column_name]
                # LSMTreeIndex uses upsert semantics for unique=True, so enforce uniqueness explicitly.
                if unique and index_type == "LSM_TREE" and idx.get(key_val):
                    raise DuplicateKeyError(
                        f"Duplicate key {key_val!r} found while building unique index {stmt.index_name!r}"
                    )
                idx.insert(key_val, rid)
            idx.flush()
            self._catalog.register_index(
                stmt.table_name,
                stmt.index_name,
                stmt.column_name,
                index_type,
                unique,
                idx_path,
            )
        except Exception:
            try:
                idx.close()
            except Exception:
                pass
            if not existed_before:
                self._cleanup_new_index_path(idx_path, index_type)
            raise

        self._register_active_index(stmt.table_name, stmt.index_name, stmt.column_name, idx)
        return {
            "index_name": stmt.index_name,
            "table_name": stmt.table_name,
            "column": stmt.column_name,
            "index_type": index_type,
            "unique": unique,
            "path": str(idx_path),
        }

    def _execute_insert(self, stmt: _InsertStmt, params: Sequence[Any]) -> RID:
        """Execute parameterized ``INSERT`` while preserving index uniqueness and stats."""
        schema, heap = self._get_table_and_schema(stmt.table_name)
        if len(params) != len(schema.columns):
            raise QuerySyntaxError(
                f"Table {stmt.table_name!r} expects {len(schema.columns)} column values, "
                f"got {len(params)}"
            )

        row_dict = {col.name: val for col, val in zip(schema.columns, params)}
        # Validate row payload and indexed key encodings before mutating heap or indexes.
        schema.encode_row(row_dict)

        active_indexes = self._table_indexes.get(stmt.table_name, [])
        for _idx_name, idx_col, idx in active_indexes:
            col_def = schema.get_column(idx_col)
            key_val = row_dict[idx_col]
            _validate_col_key(col_def, key_val)
            is_unique = bool(idx.get_stats().get("unique", idx_col == schema.primary_key))
            if is_unique and idx.get(key_val):
                raise DuplicateKeyError(
                    f"Duplicate key {key_val!r} for unique index on "
                    f"{stmt.table_name}.{idx_col}"
                )

        rid = heap.insert(row_dict)
        inserted_indexes: list[tuple[str, BaseIndex, Any]] = []
        try:
            for _idx_name, idx_col, idx in active_indexes:
                key_val = row_dict[idx_col]
                idx.insert(key_val, rid)
                inserted_indexes.append((idx_col, idx, key_val))
        except Exception:
            for _icol, ins_idx, ins_key in inserted_indexes:
                try:
                    ins_idx.delete(ins_key, rid)
                except Exception:
                    pass
            if isinstance(heap, RowTableHeap):
                try:
                    heap.delete(rid)
                except Exception:
                    pass
            raise

        self._catalog.update_table_stats(
            stmt.table_name,
            row_delta=1,
            page_count=heap.page_count(),
            row_values=row_dict,
        )
        for col in schema.columns:
            self._advisor.record("WRITE", stmt.table_name, col.name)
        return rid

    def _execute_select_all(self, stmt: _SelectAllStmt) -> list[dict[str, Any]]:
        """Execute full table scan ``SELECT * FROM <table>``."""
        _schema, heap = self._get_table_and_schema(stmt.table_name)
        return [row for _rid, row in heap.full_scan()]

    def _execute_select_eq(
        self, stmt: _SelectEqStmt, params: Sequence[Any]
    ) -> list[dict[str, Any]]:
        """Execute parameterized equality lookup ``SELECT * FROM <table> WHERE <col> = ?``."""
        schema, heap = self._get_table_and_schema(stmt.table_name)
        col_def = self._require_column(schema, stmt.column_name)
        target_val = params[0]
        _validate_col_key(col_def, target_val)

        self._advisor.record("POINT_READ", stmt.table_name, stmt.column_name)
        matches = self._locate_matching_rows(
            stmt.table_name, heap, stmt.column_name, target_val
        )
        return [row for _rid, row in matches]

    def _execute_select_between(
        self, stmt: _SelectBetweenStmt, params: Sequence[Any]
    ) -> list[dict[str, Any]]:
        """Execute parameterized inclusive range scan ``WHERE <col> BETWEEN ? AND ?``."""
        schema, heap = self._get_table_and_schema(stmt.table_name)
        col_def = self._require_column(schema, stmt.column_name)
        low_val, high_val = params[0], params[1]
        _validate_col_key(col_def, low_val)
        _validate_col_key(col_def, high_val)

        self._advisor.record("RANGE_READ", stmt.table_name, stmt.column_name)

        idx = self._indexes_by_col.get((stmt.table_name, stmt.column_name))
        if idx is not None:
            results: list[dict[str, Any]] = []
            for _k, rid in idx.range_scan(low_val, high_val, inclusive=True):
                row = heap.get(rid)
                if row is not None:
                    results.append(row)
            return results

        return [
            row
            for _rid, row in heap.full_scan()
            if low_val <= row[stmt.column_name] <= high_val
        ]

    @staticmethod
    def _rollback_single_update_row(
        heap: RowTableHeap,
        active_indexes: list[tuple[str, str, BaseIndex]],
        orig_rid: RID,
        new_rid: RID,
        old_row: dict[str, Any],
        new_row: dict[str, Any],
        idx_ops: dict[str, tuple[bool, bool]],
    ) -> None:
        """Revert a single updated row and keep every active index aligned with the live RID."""
        try:
            reverted_rid = heap.update(new_rid, old_row)
        except Exception:
            if new_rid != orig_rid and heap.delete(new_rid):
                reverted_rid = heap.insert(old_row)
            else:
                raise

        for idx_name, idx_col, idx in active_indexes:
            did_del, did_ins = idx_ops.get(idx_name, (False, False))
            old_val = old_row[idx_col]
            new_val = new_row[idx_col]
            try:
                if did_ins:
                    idx.delete(new_val, new_rid)
                if reverted_rid == orig_rid:
                    if did_del:
                        idx.insert(old_val, reverted_rid)
                else:
                    # Cross-page relocation tombstoned orig_rid; move untouched or partially
                    # rolled-back entries onto the live reverted_rid.
                    if not did_del:
                        idx.delete(old_val, orig_rid)
                    idx.insert(old_val, reverted_rid)
            except Exception:
                pass

    def _execute_update(
        self, stmt: _UpdateStmt, params: Sequence[Any]
    ) -> int:
        """Execute parameterized ``UPDATE`` and maintain all active indexes on ``RID`` relocation."""
        schema, heap = self._get_table_and_schema(stmt.table_name)
        if not isinstance(heap, RowTableHeap):
            raise StorageError(
                f"Table {stmt.table_name!r} uses append-only COLUMN layout and does not support UPDATE"
            )

        for col_name in stmt.set_columns:
            self._require_column(schema, col_name)
        where_col_def = self._require_column(schema, stmt.where_column)

        set_vals = params[:-1]
        where_val = params[-1]
        _validate_col_key(where_col_def, where_val)

        for col_name in dict.fromkeys([*stmt.set_columns, stmt.where_column]):
            self._advisor.record("WRITE", stmt.table_name, col_name)

        matches = self._locate_matching_rows(
            stmt.table_name, heap, stmt.where_column, where_val
        )
        if not matches:
            return 0

        active_indexes = self._table_indexes.get(stmt.table_name, [])
        prepared_updates: list[tuple[RID, dict[str, Any], dict[str, Any]]] = []
        for rid, old_row in matches:
            new_row = dict(old_row)
            for col_name, new_val in zip(stmt.set_columns, set_vals):
                new_row[col_name] = new_val
            schema.encode_row(new_row)

            if schema.primary_key is not None and schema.primary_key in stmt.set_columns:
                pk_col = schema.primary_key
                pk_def = schema.get_column(pk_col)
                old_pk_enc = _validate_col_key(pk_def, old_row[pk_col])
                new_pk_enc = _validate_col_key(pk_def, new_row[pk_col])
                if old_pk_enc != new_pk_enc:
                    if len(matches) > 1:
                        raise DuplicateKeyError(
                            f"Duplicate key {new_row[pk_col]!r} for primary key "
                            f"{stmt.table_name}.{pk_col}"
                        )
                    pk_idx = self._indexes_by_col.get((stmt.table_name, pk_col))
                    if pk_idx is not None:
                        if pk_idx.get(new_row[pk_col]):
                            raise DuplicateKeyError(
                                f"Duplicate key {new_row[pk_col]!r} for primary key "
                                f"{stmt.table_name}.{pk_col}"
                            )
                    elif any(
                        other_rid != rid
                        and _validate_col_key(pk_def, other_row[pk_col]) == new_pk_enc
                        for other_rid, other_row in heap.full_scan()
                    ):
                        raise DuplicateKeyError(
                            f"Duplicate key {new_row[pk_col]!r} for primary key "
                            f"{stmt.table_name}.{pk_col}"
                        )

            for _idx_name, idx_col, idx in active_indexes:
                col_def = schema.get_column(idx_col)
                old_encoded = _validate_col_key(col_def, old_row[idx_col])
                new_encoded = _validate_col_key(col_def, new_row[idx_col])
                is_unique = bool(idx.get_stats().get("unique", idx_col == schema.primary_key))
                if is_unique and old_encoded != new_encoded:
                    if len(matches) > 1 or idx.get(new_row[idx_col]):
                        raise DuplicateKeyError(
                            f"Duplicate key {new_row[idx_col]!r} for unique index on "
                            f"{stmt.table_name}.{idx_col}"
                        )
            prepared_updates.append((rid, old_row, new_row))

        completed_rows: list[
            tuple[RID, RID, dict[str, Any], dict[str, Any], dict[str, tuple[bool, bool]]]
        ] = []
        for rid, old_row, new_row in prepared_updates:
            new_rid = heap.update(rid, new_row)
            rid_changed = new_rid != rid
            idx_ops: dict[str, tuple[bool, bool]] = {}

            try:
                for idx_name, idx_col, idx in active_indexes:
                    col_def = schema.get_column(idx_col)
                    old_val = old_row[idx_col]
                    new_val = new_row[idx_col]
                    key_changed = _validate_col_key(col_def, old_val) != _validate_col_key(
                        col_def, new_val
                    )
                    # Update index when indexed value changes OR cross-page relocation altered RID.
                    if key_changed or rid_changed:
                        idx.delete(old_val, rid)
                        idx_ops[idx_name] = (True, False)
                        idx.insert(new_val, new_rid)
                        idx_ops[idx_name] = (True, True)
            except Exception:
                self._rollback_single_update_row(
                    heap, active_indexes, rid, new_rid, old_row, new_row, idx_ops
                )
                for comp_rid, comp_new_rid, comp_old, comp_new, comp_ops in reversed(
                    completed_rows
                ):
                    self._rollback_single_update_row(
                        heap,
                        active_indexes,
                        comp_rid,
                        comp_new_rid,
                        comp_old,
                        comp_new,
                        comp_ops,
                    )
                raise

            completed_rows.append((rid, new_rid, old_row, new_row, idx_ops))

        cur_page_count = heap.page_count()
        for _rid, _new_rid, _old_row, new_row, _ops in completed_rows:
            self._catalog.update_table_stats(
                stmt.table_name,
                row_delta=0,
                page_count=cur_page_count,
                row_values=new_row,
            )

        return len(prepared_updates)

    def _execute_delete(
        self, stmt: _DeleteStmt, params: Sequence[Any]
    ) -> int:
        """Execute parameterized ``DELETE`` and purge deleted ``(key, RID)`` from all indexes."""
        schema, heap = self._get_table_and_schema(stmt.table_name)
        if not isinstance(heap, RowTableHeap):
            raise StorageError(
                f"Table {stmt.table_name!r} uses append-only COLUMN layout and does not support DELETE"
            )

        where_col_def = self._require_column(schema, stmt.where_column)
        where_val = params[0]
        _validate_col_key(where_col_def, where_val)

        for col in schema.columns:
            self._advisor.record("WRITE", stmt.table_name, col.name)

        matches = self._locate_matching_rows(
            stmt.table_name, heap, stmt.where_column, where_val
        )
        if not matches:
            return 0

        active_indexes = self._table_indexes.get(stmt.table_name, [])
        deleted_count = 0
        completed_deletes: list[tuple[RID, dict[str, Any], bool]] = []
        for rid, old_row in matches:
            deleted_from_indexes: list[tuple[BaseIndex, Any]] = []
            try:
                for _idx_name, idx_col, idx in active_indexes:
                    key_val = old_row[idx_col]
                    if idx.delete(key_val, rid):
                        deleted_from_indexes.append((idx, key_val))
                was_deleted = heap.delete(rid)
                if was_deleted:
                    deleted_count += 1
                completed_deletes.append((rid, old_row, was_deleted))
            except Exception:
                for d_idx, d_key in reversed(deleted_from_indexes):
                    try:
                        d_idx.insert(d_key, rid)
                    except Exception:
                        pass
                for comp_rid, comp_row, comp_was_deleted in reversed(completed_deletes):
                    try:
                        restored_rid = (
                            heap.insert(comp_row) if comp_was_deleted else comp_rid
                        )
                        for _idx_name, idx_col, idx in active_indexes:
                            try:
                                idx.insert(comp_row[idx_col], restored_rid)
                            except Exception:
                                pass
                    except Exception:
                        pass
                raise

        self._catalog.update_table_stats(
            stmt.table_name,
            row_delta=-deleted_count,
            page_count=heap.page_count(),
        )
        return deleted_count

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """Validate, bind, and execute a parameterized SQL-like statement.

        Args:
            sql: SQL-like DDL/DML/query string using ``?`` placeholders for all
                literal values.
            params: Sequence of bound parameter values matching ``?`` positions.

        Returns:
            * ``CREATE INDEX``: metadata dictionary describing the created index.
            * ``INSERT``: newly assigned ``RID``.
            * ``SELECT``: ``list[dict[str, Any]]`` of matching decoded rows.
            * ``UPDATE`` / ``DELETE``: integer count of affected rows.

        Raises:
            SecurityViolationError: If ``sql`` contains embedded quotes, comments,
                stacked semicolons, boolean tautologies, forbidden DDL/DCL, or
                unparameterized literals.
            QuerySyntaxError: If ``sql`` has invalid syntax, unknown columns, or
                parameter count mismatches.
            KeyNotFoundError: If the target table does not exist.
            DuplicateKeyError: If inserting or updating violates a unique index.
            StorageError: If executor/storage is closed or an unsupported
                operation is attempted on a columnar table.
        """
        stmt, expected_params = _parse_sql(sql)
        if isinstance(params, (str, bytes, bytearray)) or not isinstance(params, Sequence):
            raise QuerySyntaxError(
                f"params must be a sequence (tuple/list), got {type(params).__name__}"
            )
        if len(params) != expected_params:
            raise QuerySyntaxError(
                f"Expected {expected_params} bound parameters, got {len(params)}"
            )

        with self._lock:
            self._ensure_open()
            if isinstance(stmt, _CreateIndexStmt):
                return self._execute_create_index(stmt)
            if isinstance(stmt, _InsertStmt):
                return self._execute_insert(stmt, params)
            if isinstance(stmt, _SelectAllStmt):
                return self._execute_select_all(stmt)
            if isinstance(stmt, _SelectEqStmt):
                return self._execute_select_eq(stmt, params)
            if isinstance(stmt, _SelectBetweenStmt):
                return self._execute_select_between(stmt, params)
            if isinstance(stmt, _UpdateStmt):
                return self._execute_update(stmt, params)
            if isinstance(stmt, _DeleteStmt):
                return self._execute_delete(stmt, params)
            raise QuerySyntaxError(f"Unsupported statement {stmt!r}")

    def explain(self, sql: str) -> str:
        """Return a deterministic physical access-plan description for ``sql``.

        Runs the exact same SQL-injection and syntax checks as ``execute()``
        without modifying data or ``WorkloadAdvisor`` state.

        Args:
            sql: Parameterized SQL-like statement to inspect.

        Returns:
            Human-readable execution plan string describing chosen index or scan.

        Raises:
            SecurityViolationError: If ``sql`` violates parameterization rules.
            QuerySyntaxError: If ``sql`` syntax or column references are invalid.
            KeyNotFoundError: If the referenced table is not registered.
            StorageError: If this executor is closed.
        """
        stmt, _ = _parse_sql(sql)

        with self._lock:
            self._ensure_open()
            if isinstance(stmt, _CreateIndexStmt):
                schema = self._catalog.get_schema(stmt.table_name)
                self._require_column(schema, stmt.column_name)
                return (
                    f"CreateIndex(index={stmt.index_name}, table={stmt.table_name}, "
                    f"column={stmt.column_name}, mode={stmt.using})"
                )

            if isinstance(stmt, _InsertStmt):
                schema = self._catalog.get_schema(stmt.table_name)
                idx_names = [name for name, _, _ in self._table_indexes.get(stmt.table_name, [])]
                return (
                    f"InsertRow(table={stmt.table_name}, cols={len(schema.columns)}, "
                    f"maintained_indexes={idx_names})"
                )

            stats = self._catalog.get_table_stats(stmt.table_name)
            schema = self._catalog.get_schema(stmt.table_name)
            row_count = stats["row_count"]
            page_count = stats["page_count"]

            if isinstance(stmt, _SelectAllStmt):
                return (
                    f"FullTableScan(table={stmt.table_name}, "
                    f"est_rows={row_count}, page_count={page_count})"
                )

            if isinstance(stmt, _SelectEqStmt):
                self._require_column(schema, stmt.column_name)
                found = self._find_column_index(stmt.table_name, stmt.column_name)
                if found is not None:
                    idx_name, idx = found
                    idx_type = idx.get_stats().get("index_type", "UNKNOWN")
                    return (
                        f"IndexLookup(table={stmt.table_name}, column={stmt.column_name}, "
                        f"index={idx_name}, type={idx_type}, est_rows={row_count}, "
                        f"page_count={page_count})"
                    )
                return (
                    f"FullTableScan(table={stmt.table_name}, filter=({stmt.column_name} = ?), "
                    f"est_rows={row_count}, page_count={page_count})"
                )

            if isinstance(stmt, _SelectBetweenStmt):
                self._require_column(schema, stmt.column_name)
                found = self._find_column_index(stmt.table_name, stmt.column_name)
                if found is not None:
                    idx_name, idx = found
                    idx_type = idx.get_stats().get("index_type", "UNKNOWN")
                    return (
                        f"IndexRangeScan(table={stmt.table_name}, column={stmt.column_name}, "
                        f"index={idx_name}, type={idx_type}, est_rows={row_count}, "
                        f"page_count={page_count})"
                    )
                return (
                    f"FullTableScan(table={stmt.table_name}, "
                    f"filter=({stmt.column_name} BETWEEN ? AND ?), "
                    f"est_rows={row_count}, page_count={page_count})"
                )

            if isinstance(stmt, _UpdateStmt):
                for col_name in stmt.set_columns:
                    self._require_column(schema, col_name)
                self._require_column(schema, stmt.where_column)
                access = (
                    "IndexLookup"
                    if (stmt.table_name, stmt.where_column) in self._indexes_by_col
                    else "FullTableScan"
                )
                idx_names = [name for name, _, _ in self._table_indexes.get(stmt.table_name, [])]
                return (
                    f"UpdateRows(table={stmt.table_name}, set={list(stmt.set_columns)}, "
                    f"access={access}({stmt.where_column}), maintained_indexes={idx_names})"
                )

            if isinstance(stmt, _DeleteStmt):
                self._require_column(schema, stmt.where_column)
                access = (
                    "IndexLookup"
                    if (stmt.table_name, stmt.where_column) in self._indexes_by_col
                    else "FullTableScan"
                )
                idx_names = [name for name, _, _ in self._table_indexes.get(stmt.table_name, [])]
                return (
                    f"DeleteRows(table={stmt.table_name}, access={access}({stmt.where_column}), "
                    f"maintained_indexes={idx_names})"
                )

            raise QuerySyntaxError(f"Unsupported statement {stmt!r}")

    def get_index(self, table_name: str, column_name: str) -> BaseIndex | None:
        """Return the active ``BaseIndex`` on ``(table_name, column_name)``, or ``None``."""
        with self._lock:
            self._ensure_open()
            return self._indexes_by_col.get((table_name, column_name))

    def flush_all(self) -> None:
        """Flush all active indexes, underlying table heaps/objects, and ``catalog.json``."""
        with self._lock:
            self._ensure_open()
            for _, _, idx in self._indexes_by_name.values():
                idx.flush()
            self._storage.flush_all()
            self._catalog.save()

    def close(self) -> None:
        """Flush and idempotently close all active index handles and persist catalog state."""
        with self._lock:
            if self._closed:
                return
            first_exc: Exception | None = None
            try:
                for _, _, idx in self._indexes_by_name.values():
                    try:
                        idx.close()
                    except Exception as exc:
                        if first_exc is None:
                            first_exc = exc
                try:
                    self._catalog.save()
                except Exception as exc:
                    if first_exc is None:
                        first_exc = exc
            finally:
                self._indexes_by_col.clear()
                self._indexes_by_name.clear()
                self._table_indexes.clear()
                self._closed = True

            if first_exc is not None:
                raise first_exc

    def __enter__(self) -> SimpleExecutor:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


__all__ = ["SimpleExecutor"]
