"""
Module: src/storage/storage_engine.py
Layer: Layer 1 — Unified Storage Engine Facade
Lock Domain: StorageEngine lock (acquired before TableHeapLock / ObjectStoreLock)

Purpose:
    Coordinates all Milestone 1 and Milestone 2 persistent storage components
    behind a single lifecycle facade: ``RowTableHeap`` (``tables/*.db``),
    ``ColumnTableHeap`` (``tables/*.*.col``), ``ObjectStore``
    (``objects/objects.dat`` + ``objects/objects.dir``), and atomic standalone
    table metadata persistence (``storage_meta.json``).

Key Invariants:
    1. Table registrations persist atomically via ``storage_meta.json.tmp ->
       os.fsync -> os.replace`` so ``StorageEngine(data_dir).open()`` reloads
       all row tables, column tables, and encrypted objects across restarts even
       without an external ``CatalogManager``.
    2. Lock acquisition strictly flows top-down: ``StorageEngine._lock ->
       TableHeapLock -> BufferPoolLock -> DiskManagerLock`` (and
       ``StorageEngine._lock -> ObjectStoreLock``).
    3. Opening, creation, and closing clean up already-opened descriptors on
       failure so partial initialization never leaks OS file descriptors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

from src.common.config import CHUNK_ROWS, DEFAULT_BUFFER_POOL_SIZE
from src.common.exceptions import KeyNotFoundError, PageCorruptedError, StorageError
from src.common.types import LayoutType, Schema
from src.storage.buffer_pool import BufferPoolManager
from src.storage.cipher_hooks import CipherHook
from src.storage.disk_manager import DiskManager
from src.storage.object_store import ObjectStore
from src.storage.replacer import FIFOReplacer, LRUReplacer, ReplacementPolicy
from src.storage.table_heap import ColumnTableHeap, RowTableHeap

TableHeap = RowTableHeap | ColumnTableHeap


def _validate_table_name(name: str) -> None:
    """Reject empty or path-traversing table names before mapping to filenames."""
    if not name or "/" in name or "\\" in name or "\x00" in name or name in {".", ".."}:
        raise ValueError(f"Invalid table name {name!r}")


class StorageEngine:
    """Unified M1 + M2 storage facade managing row heaps, column heaps, and objects.

    Directory Layout under ``data_dir``:
    ```text
    <data_dir>/
    ├── storage_meta.json              # Atomic table registry {name -> layout, schema}
    ├── tables/
    │   ├── <row_table>.db             # 8,192-byte SlottedPage heap + Page 0 FSM
    │   └── <col_table>.<col>.col      # Aligned 8,192-byte ColumnPage files
    └── objects/
        ├── objects.dat                # Extent-based encrypted object stream
        └── objects.dir                # Atomic OID -> extent offset directory
    ```
    """

    def __init__(
        self,
        data_dir: str | Path,
        buffer_pool_size: int = DEFAULT_BUFFER_POOL_SIZE,
        policy: str = "LRU",
        cipher: CipherHook | None = None,
    ) -> None:
        """Configure a ``StorageEngine`` rooted at ``data_dir``.

        Args:
            data_dir: Root directory containing ``tables/``, ``objects/``, and
                ``storage_meta.json``.
            buffer_pool_size: Frame capacity per table/column ``BufferPoolManager``.
            policy: Eviction policy selector (``"LRU"`` or ``"FIFO"``).
            cipher: Optional ``CipherHook`` passed to ``ObjectStore`` (defaults
                to ``NoOpCipher`` inside ``ObjectStore``).

        Raises:
            TypeError: If ``buffer_pool_size``, ``policy``, or ``cipher`` has an
                invalid type.
            ValueError: If ``buffer_pool_size < 1`` or ``policy`` is unknown.
        """
        if isinstance(buffer_pool_size, bool) or not isinstance(buffer_pool_size, int):
            raise TypeError(
                f"buffer_pool_size must be an int, got {type(buffer_pool_size).__name__}"
            )
        if buffer_pool_size < 1:
            raise ValueError(
                f"buffer_pool_size must be >= 1, got {buffer_pool_size}"
            )
        if not isinstance(policy, str):
            raise TypeError(f"policy must be a str, got {type(policy).__name__}")
        norm_policy = policy.upper()
        if norm_policy not in {"LRU", "FIFO"}:
            raise ValueError(
                f"Unsupported replacement policy {policy!r}; expected 'LRU' or 'FIFO'"
            )
        if cipher is not None and not isinstance(cipher, CipherHook):
            raise TypeError(
                f"cipher must be a CipherHook or None, got {type(cipher).__name__}"
            )

        self.data_dir: Path = Path(data_dir)
        self.tables_dir: Path = self.data_dir / "tables"
        self.objects_dir: Path = self.data_dir / "objects"
        self._meta_path: Path = self.data_dir / "storage_meta.json"

        self._buffer_pool_size: int = buffer_pool_size
        self._policy: str = norm_policy
        self._cipher: CipherHook | None = cipher

        self._lock: threading.RLock = threading.RLock()
        self._is_open: bool = False
        self._tables: dict[str, TableHeap] = {}
        self._table_layouts: dict[str, LayoutType] = {}
        self._object_store: ObjectStore | None = None

    def _ensure_open(self) -> None:
        """Raise ``StorageError`` if ``open()`` has not been called or engine is closed."""
        if not self._is_open or self._object_store is None:
            raise StorageError("StorageEngine is not open; call open() first")

    def _make_replacer(self) -> ReplacementPolicy:
        """Instantiate the configured buffer pool replacement policy."""
        if self._policy == "FIFO":
            return FIFOReplacer(self._buffer_pool_size)
        return LRUReplacer(self._buffer_pool_size)

    def _open_table_heap(
        self,
        schema: Schema,
        layout: LayoutType,
        *,
        chunk_rows: int = CHUNK_ROWS,
    ) -> TableHeap:
        """Open or create the underlying ``DiskManager``/``BufferPoolManager``/heap."""
        _validate_table_name(schema.table_name)
        if layout == LayoutType.ROW:
            db_path = self.tables_dir / f"{schema.table_name}.db"
            dm = DiskManager(db_path)
            try:
                bpm = BufferPoolManager(
                    dm,
                    pool_size=self._buffer_pool_size,
                    replacer=self._make_replacer(),
                )
                return RowTableHeap(schema, bpm)
            except Exception:
                dm.close()
                raise

        if layout == LayoutType.COLUMN:
            dms: list[DiskManager] = []
            col_bpms: dict[str, BufferPoolManager] = {}
            try:
                for col in schema.columns:
                    col_path = self.tables_dir / f"{schema.table_name}.{col.name}.col"
                    dm = DiskManager(col_path)
                    dms.append(dm)
                    col_bpms[col.name] = BufferPoolManager(
                        dm,
                        pool_size=self._buffer_pool_size,
                        replacer=self._make_replacer(),
                    )
                return ColumnTableHeap(schema, col_bpms, chunk_rows=chunk_rows)
            except Exception:
                for dm in dms:
                    dm.close()
                raise

        raise ValueError(f"Unsupported LayoutType {layout!r}")

    def _persist_meta(self) -> None:
        """Atomically persist table schemas and layouts to ``storage_meta.json``."""
        tables_payload: dict[str, Any] = {}
        for name, heap in self._tables.items():
            layout = self._table_layouts[name]
            entry: dict[str, Any] = {
                "layout": layout.value,
                "schema": heap.schema.to_dict(),
            }
            if isinstance(heap, ColumnTableHeap):
                entry["chunk_rows"] = heap.chunk_rows
            tables_payload[name] = entry

        payload = {"version": 1, "tables": tables_payload}
        raw_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")

        tmp_path = self.data_dir / "storage_meta.json.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
        fd: int | None = None
        try:
            fd = os.open(os.fspath(tmp_path), flags, 0o644)
            mv = memoryview(raw_bytes)
            written = 0
            while written < len(mv):
                n = os.write(fd, mv[written:])
                if n <= 0:
                    raise StorageError(f"Short write persisting {tmp_path}")
                written += n
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(tmp_path, self._meta_path)
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            raise StorageError(
                f"Failed to persist storage metadata {self._meta_path}: {exc}"
            ) from exc

    def open(self) -> None:
        """Initialize storage directories and reload all persisted tables and objects.

        Idempotent: calling ``open()`` on an already-open engine is a no-op.

        Raises:
            StorageError: If creating directories or opening storage files fails.
            PageCorruptedError: If ``storage_meta.json``, ``objects.dir``, or an
                existing Page 0 fails validation.
        """
        with self._lock:
            if self._is_open:
                return

            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                self.tables_dir.mkdir(parents=True, exist_ok=True)
                self.objects_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StorageError(
                    f"Failed to prepare storage directories in {self.data_dir}: {exc}"
                ) from exc

            obj_store = ObjectStore(self.objects_dir, cipher=self._cipher)
            loaded_tables: dict[str, TableHeap] = {}
            loaded_layouts: dict[str, LayoutType] = {}

            try:
                if self._meta_path.exists():
                    try:
                        raw_meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise PageCorruptedError(
                            f"Corrupt storage metadata in {self._meta_path}: {exc}"
                        ) from exc

                    if not isinstance(raw_meta, dict) or not isinstance(
                        raw_meta.get("tables"), dict
                    ):
                        raise PageCorruptedError(
                            f"Invalid storage metadata structure in {self._meta_path}"
                        )

                    for table_name, meta in raw_meta["tables"].items():
                        try:
                            layout = LayoutType(meta["layout"])
                            schema = Schema.from_dict(meta["schema"])
                            chunk_rows = int(meta.get("chunk_rows", CHUNK_ROWS))
                        except (KeyError, TypeError, ValueError) as exc:
                            raise PageCorruptedError(
                                f"Invalid table metadata for {table_name!r} in "
                                f"{self._meta_path}: {exc}"
                            ) from exc

                        heap = self._open_table_heap(
                            schema, layout, chunk_rows=chunk_rows
                        )
                        loaded_tables[schema.table_name] = heap
                        loaded_layouts[schema.table_name] = layout
            except Exception:
                for heap in loaded_tables.values():
                    try:
                        heap.close()
                    except Exception:
                        pass
                try:
                    obj_store.close()
                except Exception:
                    pass
                raise

            self._object_store = obj_store
            self._tables = loaded_tables
            self._table_layouts = loaded_layouts
            self._is_open = True

    def create_table(
        self, schema: Schema, layout: LayoutType = LayoutType.ROW
    ) -> TableHeap:
        """Create and register a new persistent row or column table heap.

        Args:
            schema: Relational ``Schema`` defining the table name and columns.
            layout: ``LayoutType.ROW`` (``*.db``) or ``LayoutType.COLUMN``
                (``*.*.col``).

        Returns:
            Newly created ``RowTableHeap`` or ``ColumnTableHeap``.

        Raises:
            TypeError: If ``schema`` is not a ``Schema``.
            ValueError: If ``schema.table_name`` or ``layout`` is invalid.
            StorageError: If the engine is not open, the table already exists,
                or creating/persisting table metadata fails.
        """
        if not isinstance(schema, Schema):
            raise TypeError(f"schema must be a Schema, got {type(schema).__name__}")
        _validate_table_name(schema.table_name)
        resolved_layout = LayoutType(layout)

        with self._lock:
            self._ensure_open()
            if schema.table_name in self._tables:
                raise StorageError(f"Table {schema.table_name!r} already exists")

            heap = self._open_table_heap(schema, resolved_layout)
            self._tables[schema.table_name] = heap
            self._table_layouts[schema.table_name] = resolved_layout
            try:
                self._persist_meta()
            except Exception:
                self._tables.pop(schema.table_name, None)
                self._table_layouts.pop(schema.table_name, None)
                try:
                    heap.close()
                except Exception:
                    pass
                raise
            return heap

    def get_table(self, name: str) -> TableHeap:
        """Return the opened ``RowTableHeap`` or ``ColumnTableHeap`` for ``name``.

        Args:
            name: Registered table name.

        Returns:
            Corresponding table heap instance.

        Raises:
            StorageError: If the engine is not open.
            KeyNotFoundError: If ``name`` is not registered.
        """
        with self._lock:
            self._ensure_open()
            heap = self._tables.get(name)
            if heap is None:
                raise KeyNotFoundError(f"Table {name!r} does not exist")
            return heap

    def has_table(self, name: str) -> bool:
        """Return whether table ``name`` is registered in this storage engine."""
        with self._lock:
            self._ensure_open()
            return name in self._tables

    def list_tables(self) -> list[str]:
        """Return a sorted list of all registered table names."""
        with self._lock:
            self._ensure_open()
            return sorted(self._tables.keys())

    @property
    def object_store(self) -> ObjectStore:
        """Return the engine's extent-based encrypted ``ObjectStore``."""
        with self._lock:
            self._ensure_open()
            assert self._object_store is not None
            return self._object_store

    def flush_all(self) -> None:
        """Flush all dirty relational pages, the object store, and ``storage_meta.json``.

        Raises:
            StorageError: If the engine is not open or flushing/fsync fails.
        """
        with self._lock:
            self._ensure_open()
            assert self._object_store is not None
            for heap in self._tables.values():
                heap.flush()
            self._object_store.flush()
            self._persist_meta()

    def close(self) -> None:
        """Flush and idempotently close all open tables and the ``ObjectStore``.

        Raises:
            StorageError: If flushing or closing any resource fails (all
                remaining resources are still closed before re-raising).
        """
        with self._lock:
            if not self._is_open:
                return

            first_exc: Exception | None = None
            try:
                try:
                    self._persist_meta()
                except Exception as exc:
                    first_exc = exc

                for heap in self._tables.values():
                    try:
                        heap.close()
                    except Exception as exc:
                        if first_exc is None:
                            first_exc = exc

                if self._object_store is not None:
                    try:
                        self._object_store.close()
                    except Exception as exc:
                        if first_exc is None:
                            first_exc = exc
            finally:
                self._tables.clear()
                self._table_layouts.clear()
                self._object_store = None
                self._is_open = False

            if first_exc is not None:
                raise first_exc

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate telemetry across all open tables and ``ObjectStore``."""
        with self._lock:
            self._ensure_open()
            assert self._object_store is not None
            return {
                "data_dir": str(self.data_dir),
                "buffer_pool_size": self._buffer_pool_size,
                "policy": self._policy,
                "table_count": len(self._tables),
                "tables": {
                    name: heap.get_stats() for name, heap in self._tables.items()
                },
                "object_store": self._object_store.get_stats(),
            }

    def __enter__(self) -> StorageEngine:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


__all__ = ["StorageEngine", "TableHeap"]
