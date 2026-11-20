"""
Module: src/query/catalog.py
Layer: Layer 3 — Metadata Catalog & Query Layer
Lock Domain: CatalogLock (``self._lock = threading.RLock()``; top-level lock in DBMS hierarchy)

Purpose:
    Implements ``CatalogManager``, the persistent atomic JSON metadata catalog
    (``catalog.json``) storing table schemas, storage layouts, physical file
    paths, active index registrations, and Milestone 3 Cost-Based Optimizer (CBO)
    statistics (``row_count``, ``page_count``, and per-column ``min_val`` /
    ``max_val``).

Key Invariants:
    1. Every persistent write stages UTF-8 JSON to ``catalog.json.tmp``, calls
       ``os.fsync(fd)``, and swaps atomically via ``os.replace()`` so crashes
       never produce a half-written ``catalog.json``.
    2. In-memory mutations roll back cleanly if atomic disk persistence fails,
       preserving exact consistency between memory and disk.
    3. ``CatalogLock`` sits at the top of the global lock ordering hierarchy
       (``CatalogLock -> IndexLock -> TableHeapLock -> BufferPoolLock ->
       DiskManagerLock``) and never acquires lower storage or index locks.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from src.common.exceptions import KeyNotFoundError, PageCorruptedError, StorageError
from src.common.types import DataType, LayoutType, Schema


def _write_all_fd(fd: int, data: bytes) -> None:
    """Write ``data`` completely to ``fd``, retrying partial OS writes."""
    mv = memoryview(data)
    written = 0
    total = len(mv)
    while written < total:
        n = os.write(fd, mv[written:])
        if n <= 0:
            raise StorageError(f"Short write persisting catalog (wrote {written}/{total} bytes)")
        written += n


def _normalize_stat_value(value: Any, dtype: DataType) -> int | float | str | None:
    """Convert Python/NumPy scalar values into JSON-serializable stat bounds."""
    if value is None:
        return None
    if isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.floating):
        value = float(value)

    if dtype == DataType.FLOAT64:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Expected numeric value for FLOAT64 stat, got {type(value).__name__}")
        fval = float(value)
        if math.isnan(fval):
            return None
        # Normalize -0.0 to +0.0 so JSON bounds match encode_key zero canonicalization.
        return 0.0 if fval == 0.0 else fval

    if dtype in (DataType.INT32, DataType.INT64, DataType.UINT64):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Expected int value for {dtype.name} stat, got {type(value).__name__}")
        return int(value)

    if dtype == DataType.VARCHAR:
        if not isinstance(value, str):
            raise TypeError(f"Expected str value for VARCHAR stat, got {type(value).__name__}")
        return value

    raise ValueError(f"Unsupported column DataType {dtype!r}")


class CatalogManager:
    """Thread-safe atomic metadata manager backed by ``catalog.json``.

    Persists table definitions, layout types, row/page counts, per-column
    ``min_val``/``max_val`` histograms for Milestone 3 Cost-Based Optimization,
    and active B+ Tree / LSM Tree index registrations.
    """

    def __init__(self, catalog_path: str | Path) -> None:
        """Load an existing ``catalog.json`` or initialize an empty catalog state.

        Args:
            catalog_path: Filesystem path to the persistent JSON catalog file.

        Raises:
            PageCorruptedError: If an existing catalog file cannot be parsed or
                violates the expected schema structure.
            StorageError: If reading the existing catalog fails due to OS errors.
        """
        self.catalog_path: Path = Path(catalog_path)
        self._lock: threading.RLock = threading.RLock()
        self._data: dict[str, Any] = {"version": 1, "tables": {}}

        if self.catalog_path.exists():
            self._load()

    def _load(self) -> None:
        """Read and validate persisted catalog state from ``self.catalog_path``."""
        try:
            raw_text = self.catalog_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"Failed reading catalog {self.catalog_path}: {exc}") from exc

        try:
            raw = json.loads(raw_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PageCorruptedError(
                f"Corrupt catalog JSON in {self.catalog_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("tables"), dict):
            raise PageCorruptedError(
                f"Invalid catalog metadata structure in {self.catalog_path}"
            )

        for table_name, entry in raw["tables"].items():
            if not isinstance(table_name, str) or not isinstance(entry, dict):
                raise PageCorruptedError(
                    f"Invalid table entry {table_name!r} in {self.catalog_path}"
                )
            try:
                LayoutType(entry["layout"])
                Schema.from_dict(
                    {
                        "table_name": table_name,
                        "primary_key": entry.get("primary_key"),
                        "columns": entry["columns"],
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PageCorruptedError(
                    f"Invalid schema metadata for table {table_name!r} in {self.catalog_path}: {exc}"
                ) from exc

            entry.setdefault("column_stats", {})
            entry.setdefault("indexes", {})
            entry.setdefault("row_count", 0)
            entry.setdefault("page_count", 0)

        self._data = raw

    def _get_table_entry_unlocked(self, table_name: str) -> dict[str, Any]:
        """Return mutable table dictionary while ``self._lock`` is held."""
        tables: dict[str, Any] = self._data["tables"]
        entry = tables.get(table_name)
        if entry is None:
            raise KeyNotFoundError(f"Table {table_name!r} is not registered in catalog")
        return entry

    def register_table(
        self,
        schema: Schema,
        layout: LayoutType,
        file_path: str | Path,
    ) -> None:
        """Register a newly created table in the catalog and persist atomically.

        Args:
            schema: Table ``Schema`` defining columns and optional primary key.
            layout: Physical ``LayoutType`` (``ROW`` or ``COLUMN``).
            file_path: Physical storage path (e.g. ``data/tables/products.db``).

        Raises:
            TypeError: If ``schema`` is not a ``Schema``.
            ValueError: If ``layout`` is invalid.
            StorageError: If the table is already registered or atomic save fails.

        Complexity & Concurrency:
            $O(C + T)$ JSON serialization under ``self._lock`` (``CatalogLock``).
        """
        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema, got {type(schema).__name__}")
        resolved_layout = LayoutType(layout)
        path_str = str(file_path)

        with self._lock:
            tables: dict[str, Any] = self._data["tables"]
            if schema.table_name in tables:
                raise StorageError(
                    f"Table {schema.table_name!r} is already registered in catalog"
                )

            tables[schema.table_name] = {
                "layout": resolved_layout.value,
                "file_path": path_str,
                "primary_key": schema.primary_key,
                "row_count": 0,
                "page_count": 0,
                "columns": [col.to_dict() for col in schema.columns],
                "column_stats": {},
                "indexes": {},
            }
            try:
                self.save()
            except Exception:
                tables.pop(schema.table_name, None)
                raise

    def register_index(
        self,
        table_name: str,
        index_name: str,
        column: str,
        index_type: str,
        unique: bool,
        path: str | Path,
    ) -> None:
        """Register an active index on ``table_name`` and persist atomically.

        Args:
            table_name: Registered table name owning the indexed column.
            index_name: Unique logical index identifier.
            column: Indexed column name present in ``table_name``'s schema.
            index_type: Index backend tag (e.g., ``"BPLUS_TREE"`` or ``"LSM_TREE"``).
            unique: Whether the index enforces key uniqueness.
            path: On-disk path to the ``*.bpt`` file or LSM directory.

        Raises:
            KeyNotFoundError: If ``table_name`` or ``column`` does not exist.
            ValueError: If ``index_name`` or ``index_type`` is empty.
            TypeError: If ``unique`` is not a ``bool``.
            StorageError: If ``index_name`` already exists or atomic save fails.
        """
        if not isinstance(index_name, str) or not index_name.strip():
            raise ValueError(f"index_name must be a non-empty str, got {index_name!r}")
        if not isinstance(index_type, str) or not index_type.strip():
            raise ValueError(f"index_type must be a non-empty str, got {index_type!r}")
        if not isinstance(unique, bool):
            raise TypeError(f"unique must be a bool, got {type(unique).__name__}")

        path_str = str(path)
        with self._lock:
            entry = self._get_table_entry_unlocked(table_name)
            valid_cols = {c["name"] for c in entry["columns"]}
            if column not in valid_cols:
                raise KeyNotFoundError(
                    f"Column {column!r} not found in table {table_name!r}"
                )

            indexes: dict[str, Any] = entry.setdefault("indexes", {})
            if index_name in indexes:
                raise StorageError(
                    f"Index {index_name!r} is already registered on table {table_name!r}"
                )

            indexes[index_name] = {
                "column": column,
                "type": index_type,
                "unique": unique,
                "path": path_str,
            }
            try:
                self.save()
            except Exception:
                indexes.pop(index_name, None)
                raise

    def update_table_stats(
        self,
        table_name: str,
        row_delta: int = 0,
        page_count: int | None = None,
        row_values: Mapping[str, Any] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        """Update table row count, page count, and per-column min/max statistics.

        Args:
            table_name: Registered table name.
            row_delta: Signed adjustment applied to ``row_count`` (clamped >= 0).
            page_count: Optional updated physical page count.
            row_values: Optional mapping of column names to inserted/updated
                values used to expand per-column ``min_val`` / ``max_val`` bounds
                for Milestone 3 Cost-Based Optimization.
            persist: When ``True`` (default), atomically persist ``catalog.json``.

        Raises:
            KeyNotFoundError: If ``table_name`` or a key in ``row_values`` is unknown.
            TypeError: If ``row_delta``, ``page_count``, or a column stat value has
                an invalid type.
            ValueError: If ``page_count`` is negative.
            StorageError: If atomic catalog persistence fails.
        """
        if isinstance(row_delta, bool) or not isinstance(row_delta, int):
            raise TypeError(f"row_delta must be an int, got {type(row_delta).__name__}")
        if page_count is not None:
            if isinstance(page_count, bool) or not isinstance(page_count, int):
                raise TypeError(f"page_count must be an int, got {type(page_count).__name__}")
            if page_count < 0:
                raise ValueError(f"page_count must be >= 0, got {page_count}")

        with self._lock:
            entry = self._get_table_entry_unlocked(table_name)
            col_dtypes: dict[str, DataType] = {
                c["name"]: DataType[c["dtype"]] if isinstance(c["dtype"], str) else DataType(c["dtype"])
                for c in entry["columns"]
            }

            if row_values is not None:
                for col_name in row_values:
                    if col_name not in col_dtypes:
                        raise KeyNotFoundError(
                            f"Unknown column {col_name!r} for table {table_name!r}"
                        )

            prev_row_count = entry.get("row_count", 0)
            prev_page_count = entry.get("page_count", 0)
            prev_col_stats = copy.deepcopy(entry.get("column_stats", {}))

            try:
                entry["row_count"] = max(0, int(prev_row_count) + row_delta)
                if page_count is not None:
                    entry["page_count"] = page_count

                if row_values is not None:
                    col_stats: dict[str, dict[str, Any]] = entry.setdefault("column_stats", {})
                    for col_name, raw_val in row_values.items():
                        val = _normalize_stat_value(raw_val, col_dtypes[col_name])
                        if val is None:
                            continue
                        stat = col_stats.setdefault(
                            col_name, {"min_val": None, "max_val": None}
                        )
                        cur_min = stat.get("min_val")
                        cur_max = stat.get("max_val")
                        if cur_min is None or val < cur_min:
                            stat["min_val"] = val
                        if cur_max is None or val > cur_max:
                            stat["max_val"] = val

                if persist:
                    self.save()
            except Exception:
                entry["row_count"] = prev_row_count
                entry["page_count"] = prev_page_count
                entry["column_stats"] = prev_col_stats
                raise

    def get_table_stats(self, table_name: str) -> dict[str, Any]:
        """Return table statistics, layout, primary key, and index metadata.

        Provides the exact ``row_count``, ``page_count``, ``column_stats``
        (``min_val``/``max_val``), ``indexes``, ``layout``, and ``primary_key``
        counters consumed by Milestone 3's ``CostBasedOptimizer``.

        Args:
            table_name: Registered table name.

        Returns:
            Deep-copied metadata dictionary for ``table_name``.

        Raises:
            KeyNotFoundError: If ``table_name`` is not registered.
        """
        with self._lock:
            entry = self._get_table_entry_unlocked(table_name)
            return {
                "table_name": table_name,
                "layout": entry["layout"],
                "file_path": entry["file_path"],
                "primary_key": entry["primary_key"],
                "row_count": entry["row_count"],
                "page_count": entry["page_count"],
                "column_stats": copy.deepcopy(entry.get("column_stats", {})),
                "indexes": copy.deepcopy(entry.get("indexes", {})),
            }

    def get_schema(self, table_name: str) -> Schema:
        """Reconstruct and return the ``Schema`` for ``table_name``.

        Args:
            table_name: Registered table name.

        Returns:
            Reconstructed ``Schema`` instance.

        Raises:
            KeyNotFoundError: If ``table_name`` is not registered.
        """
        with self._lock:
            entry = self._get_table_entry_unlocked(table_name)
            return Schema.from_dict(
                {
                    "table_name": table_name,
                    "primary_key": entry.get("primary_key"),
                    "columns": entry["columns"],
                }
            )

    def get_indexes(self, table_name: str) -> dict[str, dict[str, Any]]:
        """Return a deep copy of registered index definitions for ``table_name``.

        Args:
            table_name: Registered table name.

        Returns:
            Mapping from index name to ``{column, type, unique, path}`` metadata.

        Raises:
            KeyNotFoundError: If ``table_name`` is not registered.
        """
        with self._lock:
            entry = self._get_table_entry_unlocked(table_name)
            return copy.deepcopy(entry.get("indexes", {}))

    def list_tables(self) -> list[str]:
        """Return a sorted list of all registered table names."""
        with self._lock:
            return sorted(self._data["tables"].keys())

    def has_table(self, table_name: str) -> bool:
        """Return ``True`` if ``table_name`` is registered in the catalog."""
        with self._lock:
            return table_name in self._data["tables"]

    def save(self) -> None:
        """Atomically persist catalog state via ``<file>.tmp -> fsync -> os.replace``.

        Raises:
            StorageError: If creating directories, writing, syncing, or replacing
                ``catalog.json`` fails.
        """
        with self._lock:
            try:
                self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StorageError(
                    f"Failed preparing catalog directory {self.catalog_path.parent}: {exc}"
                ) from exc

            payload_bytes = json.dumps(self._data, indent=2, sort_keys=True).encode("utf-8")
            tmp_path = self.catalog_path.with_name(f"{self.catalog_path.name}.tmp")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
            fd: int | None = None
            try:
                fd = os.open(os.fspath(tmp_path), flags, 0o644)
                _write_all_fd(fd, payload_bytes)
                os.fsync(fd)
                os.close(fd)
                fd = None
                os.replace(tmp_path, self.catalog_path)
            except OSError as exc:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                raise StorageError(
                    f"Failed persisting catalog {self.catalog_path}: {exc}"
                ) from exc


__all__ = ["CatalogManager"]
