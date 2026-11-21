# Smart Retail DB M1 & M2 Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Smart Retail DB's M1 storage engine and M2 indexing engine onto a unified page-managed, buffer-pooled, typed binary architecture that passes all M1/M2 rubric criteria and eliminates M3–M5 architectural blockers.

**Architecture:** Replace JSON-per-page storage, isolated per-table buffer pools, in-memory `BPlusTree`, and JSONL-in-RAM `LSMTree` with: (1) typed schemas + order-preserving key encoding, (2) `StorageManager` + `PageBackend` (`PageFile` / `LocalObjectStore`) + `page_id`-aware `TransformingBackend` (supporting AEAD overhead), (3) one shared `BufferPool` (`(file_id, page_id)` with `pin`/`unpin`, `FIFO`/`LRU`, `pageLSN` header), (4) binary slotted `RowTable` (+ overflow) and true per-column `ColumnTable` (projection pushdown + late materialization), (5) page-backed `BPlusTree` and disk-backed SSTable `LSMTree` (Bloom filters, sparse fence pointers, memtable WAL, atomic compaction), and (6) safe unified SQL/DML parser + engine + reproducible 5-run M1/M2 benchmarks.

**Tech Stack:** Python >= 3.10 standard library (`struct`, `hashlib`, `bisect`, `collections`, `ast`, `re`, `os`, `pathlib`, `unittest`) for core `smartdb`; optional `cryptography` (AES-256-GCM helper) and `matplotlib` (benchmark plots).

**Spec:** [2026-11-21-m1-m2-architecture-refactor-design.md](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/docs/superpowers/specs/2026-11-21-m1-m2-architecture-refactor-design.md)

## Global Constraints

- Core `smartdb` package must remain zero-external-runtime-dependency pure Python >= 3.10 (`pyproject.toml` `dependencies = []`); `cryptography` and `matplotlib` are optional imports only.
- Cross-platform Python (no POSIX-only `fcntl`); use `temp.replace(target)` for atomic file swaps.
- Do not use SQLite, PostgreSQL, MySQL, DuckDB, RocksDB, or any external DB engine.
- Every physical page on disk/object store defaults to `4096` bytes (`PAGE_SIZE = 4096`), with a 24-byte `PageHeader` (`<BBHHIqi`) at offset 0 of every usable page.
- `bool` must never equal or coerce to `int` (`True != 1`, `False != 0`); `NaN` and `Inf` floats/vectors are rejected on write and query.
- Benchmarks must run warm-up + 5 measured repeats (`--repeats 5`), declare OS/Python/CPU/seed/dataset specs, and report wall-clock ms, throughput (ops/s), median, mean, stdev, and p95 latency.

## Review Focus

- **`bool` vs `int` collision:** Python evaluates `1 == True` and `hash(1) == hash(True)`. Pinned in Task 1 (`test_schema_rejects_bool_as_int_and_orders_distinctly`) and Task 8 (`test_bool_int_query_isolation`).
- **SQL three-valued `NULL` filtering on open/range predicates:** Since `None` sorts first in `encode_key`, `WHERE price < 10`, `<= 10`, `> 0`, `>= 0`, `BETWEEN a AND b` must never return `NULL` rows. Pinned in Task 8 (`test_null_excluded_from_range_and_inequality_queries`).
- **Unclean shutdown between `LSMTree` WAL and `BufferPool` table pages:** Reopening after an unclosed process where LSM flushed ahead of `BufferPool` must reconcile cleanly without raising `IndexError`. Pinned in Task 8 (`test_unclean_shutdown_reconciles_indexes`).
- **Low-cardinality / high-duplicate key overflow:** Inserting 1,000+ duplicate keys for 3 distinct categories must trigger `LSMTree` flushes (`total_entries >= flush_threshold`) and spill `BPlusTree` RID lists to overflow bucket pages without overflowing 4 KB leaf pages. Pinned in Task 6 (`test_bplus_high_duplicate_rid_overflow`) and Task 7 (`test_lsm_low_cardinality_flushes_by_entry_count`).
- **Oversized record or validation failure leaving orphan pages / partial state:** Failed `insert`/`update` due to >4 KB validation or bad index key must not allocate orphan pages or commit half-updated indexes. Pinned in Task 4 (`test_failed_insert_allocates_no_orphan_pages`) and Task 8 (`test_failed_insert_is_atomic_across_table_and_indexes`).

---

## File Structure Map

- **Create:**
  - `src/smartdb/schema.py` — `DataType`, `ColumnSpec`, `TableSchema`, `RID`, key encoding, binary row/value codecs.
  - `src/smartdb/storage/manager.py` — `StorageManager` registering/opening/dropping `PageBackend`s <-> `file_id`s.
  - `src/smartdb/storage/slotted_page.py` — 24-byte `PageHeader`, `PageType`, `SlottedPage`.
  - `src/smartdb/storage/overflow.py` — `OVERFLOW` page chain read/write/free helpers.
  - `src/smartdb/storage/row_table.py` — `TableStorage` protocol + binary slotted `RowTable`.
  - `src/smartdb/storage/column_table.py` — true per-column page streams `ColumnTable` with projection & vectorized batches.
  - `benchmarks/common.py` — warm-up + repeated-run stats (median, p95, stdev, ops/s), metadata, CSV, optional matplotlib plots.
  - `tests/test_schema.py`, `tests/test_storage_backends.py`, `tests/test_buffer_pool.py`, `tests/test_row_table.py`, `tests/test_column_table.py`, `tests/test_bplus_tree.py`, `tests/test_lsm_tree.py`.
- **Modify:**
  - `src/smartdb/storage/page_file.py`, `object_store.py`, `encryption.py`, `buffer_pool.py`, `storage/__init__.py`
  - `src/smartdb/indexes/bplus_tree.py`, `lsm_tree.py`, `manager.py`, `indexes/__init__.py`
  - `src/smartdb/query.py`, `engine.py`, `cli.py`, `smartdb/__init__.py`
  - `workloads/generate_retail_data.py`, `benchmarks/benchmark_storage.py`, `benchmarks/benchmark_indexes.py`
  - `tests/test_engine.py`, `README.md`, `docs/architecture.md`, `docs/milestone-checklist.md`
- **Delete:**
  - `src/smartdb/storage/layouts.py`, `src/smartdb/storage/table.py` (replaced by `schema.py`, `slotted_page.py`, `overflow.py`, `row_table.py`, `column_table.py`)
  - `tests/test_storage.py`, `tests/test_indexes.py` (split into the focused per-module tests above)
  - Empty legacy directories: `src/common`, `src/index`, `src/query`, `src/storage`, `bench`

---

### Task 1: Workspace Packaging Cleanup & Typed Schema / Binary Codec (`schema.py`)

**Files:**
- Delete empty legacy dirs: `src/common`, `src/index`, `src/query`, `src/storage`, `bench`
- Create: `src/smartdb/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `RID = tuple[int, int]`
  - `class DataType(str, Enum)`: `INT`, `FLOAT`, `BOOL`, `STR`, `BYTES`, `VECTOR`, `ANY`
  - `@dataclass(frozen=True) class ColumnSpec(name: str, dtype: DataType, nullable: bool = True, dim: int | None = None)`
  - `class TableSchema`: `__init__(columns: list[ColumnSpec])`, `.from_spec(spec: list[str] | list[ColumnSpec] | dict[str, Any]) -> TableSchema`, `.to_dict() -> dict`, `.from_dict(data: dict) -> TableSchema`, `.field_names: list[str]`, `.column(name: str) -> ColumnSpec`, `.validate_record(record: dict[str, Any]) -> dict[str, Any]`, `.coerce_literal(field: str, value: Any) -> Any`
  - `encode_key(dtype: DataType, value: Any) -> bytes`
  - `decode_key(dtype: DataType, raw: bytes) -> Any`
  - `serialize_value(dtype: DataType, value: Any, dim: int | None = None) -> bytes`
  - `deserialize_value(dtype: DataType, buf: bytes | memoryview, offset: int = 0, dim: int | None = None) -> tuple[Any, int]`
  - `serialize_row(schema: TableSchema, record: dict[str, Any]) -> bytes`
  - `deserialize_row(schema: TableSchema, buf: bytes | memoryview, projected: list[str] | None = None) -> dict[str, Any]`

- [ ] **Step 1: Remove stale empty legacy directories and ensure editable install in `.venv`**

Run:
```bash
rm -rf src/common src/index src/query src/storage bench
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```
Expected: `Successfully installed smart-retail-db-0.1.0` and `src/` only contains `smartdb/`.

- [ ] **Step 2: Write failing tests for schema validation, `bool`/`int` separation, `NaN`/`Inf` rejection, lossless `encode_key`, and binary row round-trip**

Create `tests/test_schema.py`:
```python
import math
import unittest

from smartdb.schema import (
    ColumnSpec,
    DataType,
    TableSchema,
    decode_key,
    deserialize_row,
    encode_key,
    serialize_row,
)


class SchemaTests(unittest.TestCase):
    def test_schema_rejects_bool_as_int_and_orders_distinctly(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("flag", DataType.BOOL),
            ColumnSpec("price", DataType.FLOAT),
            ColumnSpec("vec", DataType.VECTOR, dim=3),
        ])
        with self.assertRaises(TypeError):
            schema.validate_record({"id": True, "flag": True, "price": 1.0, "vec": [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError):
            schema.validate_record({"id": 1, "flag": True, "price": float("nan"), "vec": [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError):
            schema.validate_record({"id": 1, "flag": True, "price": 1.0, "vec": [1.0, 2.0]})

        self.assertLess(encode_key(DataType.ANY, None), encode_key(DataType.ANY, False))
        self.assertLess(encode_key(DataType.ANY, False), encode_key(DataType.ANY, True))
        self.assertNotEqual(encode_key(DataType.ANY, True), encode_key(DataType.ANY, 1))
        self.assertEqual(encode_key(DataType.ANY, 1), encode_key(DataType.ANY, 1.0))

    def test_lossless_int64_and_float64_key_ordering(self):
        vals = [-2**62, -10, -1.5, -1, -0.25, 0, 0.25, 1, 1.5, 2**53, 2**53 + 1, 2**62]
        encoded = [encode_key(DataType.ANY, v) for v in vals]
        self.assertEqual(encoded, sorted(encoded))
        self.assertEqual(len(set(encoded)), len(vals))
        for v, raw in zip(vals, encoded):
            self.assertEqual(decode_key(DataType.ANY, raw), v)

    def test_binary_row_round_trip_and_projection(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("name", DataType.STR),
            ColumnSpec("price", DataType.FLOAT),
            ColumnSpec("active", DataType.BOOL),
            ColumnSpec("emb", DataType.VECTOR, dim=2),
        ])
        row = {"id": 42, "name": "Keyboard ⌨️", "price": 29.99, "active": False, "emb": [0.5, -1.25]}
        blob = serialize_row(schema, schema.validate_record(row))
        decoded = deserialize_row(schema, blob)
        self.assertEqual(decoded["id"], 42)
        self.assertEqual(decoded["name"], "Keyboard ⌨️")
        self.assertAlmostEqual(decoded["price"], 29.99)
        self.assertIs(decoded["active"], False)
        self.assertEqual(decoded["emb"], [0.5, -1.25])
        proj = deserialize_row(schema, blob, projected=["name", "active"])
        self.assertEqual(proj, {"name": "Keyboard ⌨️", "active": False})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_schema -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'smartdb.schema'`.

- [ ] **Step 4: Implement `src/smartdb/schema.py`**

Key implementation details:
- Use big-endian IEEE-754 order-preserving float encoding (flip sign bit if positive; flip all 64 bits if negative) and signed `int64` XOR `(1 << 63)` for `DataType.INT`.
- For `DataType.ANY` numeric (`int`/`float`): tag byte `2`, bucket byte (`0` for `< -2**63`, `1` for `[-2**63, 2**63)`, `2` for `>= 2**63`), `floor_int64` (bucket `1`) + `frac_float64`. Because `1` and `1.0` both have `floor_int64=1` and `frac_float64=0.0`, `encode_key(ANY, 1) == encode_key(ANY, 1.0)` holds while preserving full 64-bit integer precision (`2**53 != 2**53 + 1`). `decode_key` returns `int` when `frac == 0.0` (which equals both `1` and `1.0` in Python) and `float` otherwise.
- Null bitmap at front of `serialize_row`: `ceil(len(columns) / 8)` bytes.
- Vector encoding: `dim` `float32`s (`struct.pack(f"<{dim}f", *vec)`), validating `isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)`.

- [ ] **Step 5: Run tests to verify `test_schema` passes**

Run: `.venv/bin/python -m unittest tests.test_schema -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/smartdb/schema.py tests/test_schema.py
git commit -m "feat: add typed TableSchema, order-preserving key codec, and binary row serialization"
```

---

### Task 2: Storage Backends, Cached Object Store, AEAD Encryption Hooks & `StorageManager`

**Files:**
- Modify: `src/smartdb/storage/page_file.py`
- Modify: `src/smartdb/storage/object_store.py`
- Modify: `src/smartdb/storage/encryption.py`
- Create: `src/smartdb/storage/manager.py`
- Test: `tests/test_storage_backends.py`

**Interfaces:**
- Produces:
  - `PageBackend` (`physical_page_size: int`, `usable_page_size: int`, `page_count: int`, `read_page(page_id: int) -> bytes`, `write_page(page_id: int, data: bytes) -> None`, `allocate_page() -> int`, `close() -> None`, `destroy() -> None`)
  - `PageFile(path, page_size=PAGE_SIZE)`
  - `LocalObjectStore(directory, page_size=PAGE_SIZE)` (cached `_page_count`)
  - `TransformingBackend(backend, encrypt=None, decrypt=None, overhead_bytes=0)` where `encrypt: Callable[[int, bytes], bytes]`, `decrypt: Callable[[int, bytes], bytes]`
  - `make_aes_gcm_hooks(key: bytes) -> tuple[PageTransform, PageTransform, int, RecordTransform, RecordTransform]` (`overhead_bytes=28`, optional `cryptography` import inside function)
  - `StorageManager(root: str | Path, backend_kind: str = "page", encrypt_page=None, decrypt_page=None, crypto_overhead: int = 0, encrypt_record=None, decrypt_record=None, physical_page_size: int = PAGE_SIZE)`:
    - `.open_file(logical_name: str, rel_path: str) -> tuple[int, PageBackend]`
    - `.get_backend(file_id: int) -> PageBackend`
    - `.usable_page_size: int`
    - `.drop_file(logical_name: str) -> None`
    - `.encrypt_record(data: bytes) -> bytes`, `.decrypt_record(data: bytes) -> bytes`
    - `.close_all() -> None`

- [ ] **Step 1: Write failing tests in `tests/test_storage_backends.py`**

Include tests for:
- `PageFile` with custom `page_size=512` wrapped in `TransformingBackend` (no hardcoded `4096` crash).
- `LocalObjectStore` `page_count` increments without directory globbing on every call (`allocate_page` + reopen consistency + `destroy()` removes object files).
- `TransformingBackend` with 2-arg `(page_id, data)` XOR hook AND with `make_aes_gcm_hooks(os.urandom(32))` verifying `usable_page_size == 4068`, `physical_page_size == 4096`, round-trip, and tampering detection (`InvalidTag` / `ValueError`).
- `StorageManager` registering multiple logical files under both `"page"` and `"objects"` backends and dropping files cleanly.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_storage_backends -v`
Expected: FAIL (`ImportError` / signature mismatch).

- [ ] **Step 3: Update `page_file.py`, `object_store.py`, `encryption.py` and create `storage/manager.py`**

- In `page_file.py`: expose `physical_page_size` and `usable_page_size` (both return `self.page_size`), add `destroy()` closing and unlinking `self.path`.
- In `object_store.py`: compute `self._page_count` once in `__init__` (`max((int(p.stem.split("_")[1]) for p in self.directory.glob("page_*.obj")), default=-1) + 1`), increment in `allocate_page()`, add `destroy()` removing `page_*.obj` and directory if empty.
- In `encryption.py`: validate `0 <= overhead_bytes < backend.physical_page_size`; set `usable_page_size = backend.usable_page_size - overhead_bytes`; pass `(page_id, bytes(data))` to `self._encrypt` (expecting `backend.usable_page_size` bytes) and `(page_id, raw)` to `self._decrypt` (expecting `self.usable_page_size` bytes). In `allocate_page()`, allocate on `self.backend` and write `self._encrypt(page_id, b"\x00" * self.usable_page_size)` so direct `read_page(page_id)` calls remain valid under AEAD.
- In `storage/manager.py`: deterministic `file_id` allocation (monotonic int per registered logical name), delegates `record_encrypt`/`record_decrypt` for WAL framing.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_storage_backends -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartdb/storage/page_file.py src/smartdb/storage/object_store.py src/smartdb/storage/encryption.py src/smartdb/storage/manager.py tests/test_storage_backends.py
git commit -m "feat: upgrade PageBackend, LocalObjectStore, page-aware TransformingBackend, and StorageManager"
```

---

### Task 3: Unified Shared `BufferPool` with Pinning, LSNs & Multi-File Management

**Files:**
- Modify: `src/smartdb/storage/buffer_pool.py`
- Test: `tests/test_buffer_pool.py`

**Interfaces:**
- Produces:
  - `@dataclass class BufferFrame(file_id: int, page_id: int, data: bytearray, pin_count: int = 0, dirty: bool = False, page_lsn: int = 0, rec_lsn: int = 0)`
  - `@dataclass class BufferStats(hits: int = 0, misses: int = 0, disk_reads: int = 0, disk_writes: int = 0, evictions: int = 0, dirty_evictions: int = 0)`
  - `class BufferPool`:
    - `__init__(self, backend: PageBackend | None = None, capacity: int = 64, policy: str = "LRU", storage_manager: StorageManager | None = None, wal_flushed_lsn_fn: Callable[[], int] | None = None)`
    - `.register_backend(backend: PageBackend, file_id: int | None = None) -> int`
    - `.fetch_page(page_id: int, file_id: int = 0) -> BufferFrame`
    - `.new_page(file_id: int = 0) -> tuple[int, BufferFrame]`
    - `.unpin_page(page_id: int, is_dirty: bool = False, lsn: int | None = None, file_id: int = 0) -> None`
    - `.pin_page(page_id: int, file_id: int = 0, dirty: bool = False)` (context manager yielding `BufferFrame` and unpinning on exit)
    - `.flush_page(page_id: int, file_id: int = 0) -> None`
    - `.flush_file(file_id: int) -> None`
    - `.flush_all() -> None` / `.flush() -> None`
    - `.unregister_file(file_id: int, drop_dirty: bool = True) -> None`
    - `.close() -> None` / `.snapshot() -> dict`

- [ ] **Step 1: Write failing tests in `tests/test_buffer_pool.py`**

Test:
- Standalone single-backend FIFO vs LRU exact eviction victims (`[0, 1, 0, 2]` with capacity 2 evicts `0` under FIFO and `1` under LRU).
- `pin_count > 0` blocks eviction; evicting when all frames are pinned raises `RuntimeError`.
- `new_page` allocates and installs a pinned dirty frame without causing an extra `disk_reads` miss, and writes back on eviction/flush.
- Multi-file shared pool (file `0` and file `1` share total `capacity` frames cleanly).
- `wal_flushed_lsn_fn` guard raises `RuntimeError` if evicting a dirty frame with `page_lsn > wal_flushed_lsn_fn()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_buffer_pool -v`
Expected: FAIL.

- [ ] **Step 3: Implement refactored `src/smartdb/storage/buffer_pool.py`**

- Use `OrderedDict[tuple[int, int], BufferFrame]` for frame replacement ordering.
- On `_evict_if_needed()`: iterate frames in order, pick first with `frame.pin_count == 0`, enforce `wal_flushed_lsn_fn` if dirty before calling `backend.write_page(frame.page_id, bytes(frame.data))`.
- Support `new_page(file_id=0)` calling `backend.allocate_page()` to reserve `page_id` and installing a pinned dirty `bytearray(backend.usable_page_size)` frame in the pool (avoiding a redundant `read_page` miss before initialization).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_buffer_pool -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartdb/storage/buffer_pool.py tests/test_buffer_pool.py
git commit -m "feat: implement shared multi-file BufferPool with pinning, LSN hooks, and FIFO/LRU"
```

---

### Task 4: Common Page Header, Binary Slotted Pages, Overflow Chains & `RowTable`

**Files:**
- Create: `src/smartdb/storage/slotted_page.py`
- Create: `src/smartdb/storage/overflow.py`
- Create: `src/smartdb/storage/row_table.py`
- Delete: `src/smartdb/storage/layouts.py`, `src/smartdb/storage/table.py`
- Modify: `src/smartdb/storage/__init__.py`
- Test: `tests/test_row_table.py`

**Interfaces:**
- Produces:
  - `class PageType(IntEnum)`: `FREE = 0`, `ROW_SLOTTED = 1`, `OVERFLOW = 2`, `COL_FIXED = 3`, `COL_VAR = 4`, `BITMAP = 5`, `BPLUS_META = 6`, `BPLUS_INTERNAL = 7`, `BPLUS_LEAF = 8`, `LSM_DATA = 9`, `LSM_META = 10`, `COL_META = 11`
  - `HEADER_SIZE = 24`, `HEADER_STRUCT = "<BBHHHqii"` (`page_type(1), flags(1), item_count(2), free_start(2), free_end(2), lsn(8), next_page(4), prev_page(4)` = 24 bytes)
  - `read_header(buf)` / `write_header(buf, ...)`
  - `SlottedPage`: `init(buf, page_type=PageType.ROW_SLOTTED)`, `insert_payload(buf, payload: bytes) -> int | None` (slot_id), `get_payload(buf, slot_id: int) -> bytes | None`, `update_payload(buf, slot_id: int, payload: bytes) -> bool`, `delete_slot(buf, slot_id: int) -> bool`, `live_slots(buf) -> Iterator[tuple[int, bytes]]`
  - `write_overflow_chain(pool: BufferPool, file_id: int, data: bytes) -> int` (first overflow page_id), `read_overflow_chain(pool: BufferPool, file_id: int, first_page_id: int, total_len: int) -> bytes`, `free_overflow_chain(pool: BufferPool, file_id: int, first_page_id: int) -> None`
  - `TableStorage` Protocol & `RowTable(storage_manager, pool, name, schema)`:
    - `.insert(record: dict[str, Any]) -> RID`
    - `.get_record(rid: RID, fields: list[str] | None = None) -> dict[str, Any]`
    - `.try_get_record(rid: RID, fields: list[str] | None = None) -> dict[str, Any] | None`
    - `.update(rid: RID, updates: dict[str, Any]) -> RID`
    - `.delete(rid: RID) -> bool`
    - `.iter_records(fields: list[str] | None = None) -> Iterator[tuple[RID, dict[str, Any]]]`
    - `.scan(fields: list[str] | None = None, predicate=None) -> list[dict[str, Any]]`
    - `.scan_batches(fields: list[str] | None = None, batch_size: int = 1024) -> Iterator[dict[str, list[Any]]]`
    - `.page_count: int`, `.flush() -> None`

- [ ] **Step 1: Write failing tests in `tests/test_row_table.py`**

Include tests for:
- `test_slotted_page_insert_delete_keeps_stable_slot_ids_and_reclaims_on_compact`
- `test_row_table_insert_get_reopen_and_delete`
- `test_row_table_in_place_and_relocating_update`
- `test_oversized_record_and_1536d_vector_overflow_chain`
- `test_failed_insert_allocates_no_orphan_pages` (Review Focus: invalid field/type or `NaN` raises BEFORE `new_page`, keeping `page_count` unchanged)
- `test_try_get_record_returns_none_on_out_of_bounds_or_deleted`

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_row_table -v`
Expected: FAIL.

- [ ] **Step 3: Implement `slotted_page.py`, `overflow.py`, `row_table.py`, update `storage/__init__.py`, remove `layouts.py` & `table.py`**

- In `RowTable`:
  - Register `"table:{name}:row"` (data) and `"table:{name}:overflow"` (overflow) backends in `StorageManager` + `BufferPool` (`_data_file_id`, `_ovf_file_id`).
  - Maintain `_last_page_id` (initialized from `data_backend.page_count - 1`) for $O(1)$ append inserts.
  - Validate `schema.validate_record(record)` and serialize row bytes **before** touching or allocating any page!
  - If `len(row_bytes) <= max_inline_bytes`, insert into last slotted page (or allocate new page if full).
  - If `len(row_bytes) > max_inline_bytes`, write overflow chain in `_ovf_file_id` and store 12-byte stub `b"\xffOVF" + struct.pack("<Ii", len(row_bytes), first_ovf_page)` in the slotted page.
  - Note: on `RowTable.update(rid, updates)`, validate full merged record first; if updated payload fits in-page (`SlottedPage.update_payload`), update in place keeping `rid` identical; otherwise delete old slot (and free old overflow if any), insert new record, and return `new_rid`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_row_table -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git rm src/smartdb/storage/layouts.py src/smartdb/storage/table.py
git add src/smartdb/storage/slotted_page.py src/smartdb/storage/overflow.py src/smartdb/storage/row_table.py src/smartdb/storage/__init__.py tests/test_row_table.py
git commit -m "feat: implement binary SlottedPage, overflow chains, and RowTable"
```

---

### Task 5: True Columnar Storage with Projection Pushdown & Vectorized Batches (`column_table.py`)

**Files:**
- Create: `src/smartdb/storage/column_table.py`
- Modify: `src/smartdb/storage/__init__.py`
- Test: `tests/test_column_table.py`

**Interfaces:**
- Produces:
  - `class ColumnTable`: implements `TableStorage` protocol (`insert`, `get_record`, `try_get_record`, `update`, `delete`, `iter_records`, `scan`, `scan_batches`, `page_count`, `flush`), plus `.column_page_count(field: str) -> int`
  - Storage streams per table:
    - `"table:{name}:col_meta"` (`COL_META` page 0 holding `row_count: uint64` and var-column page index)
    - `"table:{name}:deleted"` (`BITMAP` pages, 1 bit per `rowid`)
    - `"table:{name}:col:{col_name}"` (`COL_FIXED` or `COL_VAR` pages; large `VECTOR`/`STR`/`BYTES` exceeding 1 page use `"table:{name}:overflow"`)

- [ ] **Step 1: Write failing tests in `tests/test_column_table.py`**

Include tests for:
- Round-trip `insert`, `get_record`, reopen, `update` (tombstone + append), `delete` across `INT`, `FLOAT`, `BOOL`, `STR`, `VECTOR(3)`, `ANY`, and `NULL`s.
- **Projection pushdown I/O verification:** Insert 2,000 rows (`id`, `price`, `big_notes`), flush & reset `pool.stats`, run `scan_batches(fields=["price"])`, and assert:
  - Total `pool.stats.disk_reads` equals ONLY `deleted` + `col:price` + `col_meta` pages (orders of magnitude fewer reads than reading `big_notes`!).
  - Every `batch` is a `dict[str, list]` with exact projected column values and deleted rows omitted.
- **Late materialization:** `scan(fields=["id"], predicate=...)` or filter on `price` verifies low disk reads.
- `try_get_record((0, 999999))` returns `None`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_column_table -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/smartdb/storage/column_table.py`**

Design highlights:
- Fixed-width columns (`INT` 8B, `FLOAT` 8B, `BOOL` 1B, `VECTOR(D)` `4*D` bytes when `<= usable_page_size - 256`): compute `slots_per_page` from `usable_page_size - HEADER_SIZE` accounting for null bitmap (`ceil(slots_per_page / 8)` bytes at offset `24`, followed by fixed value array). Any `rowid` maps directly via `page_id = rowid // K`, `slot = rowid % K` in $O(1)$!
- Variable-width columns (`STR`, `BYTES`, `ANY`, oversized `VECTOR`): packed inside `COL_VAR` slotted pages (same `SlottedPage` primitives with `item_count` and `first_rowid` stored in a compact directory page / memory-cached directory persisted to `col_meta` on flush/close and rebuilt if needed from page headers!).
- `Locator` (`RID`) = `(0, rowid)`.
- Keep `ColumnTable` validated against `schema.validate_record` before writing to any column or bitmap stream.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_column_table -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartdb/storage/column_table.py src/smartdb/storage/__init__.py tests/test_column_table.py
git commit -m "feat: implement columnar storage with projection pushdown and vectorized batch scans"
```

---

### Task 6: Persistent Page-Backed B+ Tree (`bplus_tree.py`)

**Files:**
- Modify: `src/smartdb/indexes/bplus_tree.py`
- Test: `tests/test_bplus_tree.py`

**Interfaces:**
- Produces:
  - `class BPlusTree`:
    - `__init__(self, order: int | None = None, storage_manager: StorageManager | None = None, pool: BufferPool | None = None, name: str = "bplus", key_dtype: DataType = DataType.ANY)`
      - If `storage_manager is None` or `pool is None`, auto-provisions an isolated `TemporaryDirectory` + `PageFile` + `BufferPool` (cleaned up on `.close()`).
      - `order` optionally caps max keys per node (`max_keys = order - 1` if `order` given, default computed from page payload capacity).
    - `.insert(key: Any, rid: RID) -> None`
    - `.delete(key: Any, rid: RID | None = None) -> int` (returns count removed)
    - `.search(key: Any) -> list[RID]`
    - `.range_search(low: Any | None = None, high: Any | None = None, include_low: bool = True, include_high: bool = True) -> list[tuple[Any, RID]]`
    - `.page_count: int`, `.disk_bytes() -> int`, `.flush() -> None`, `.close() -> None`, `__len__() -> int`

- [ ] **Step 1: Write failing tests in `tests/test_bplus_tree.py`**

Include tests for:
- Standalone `BPlusTree(order=4)` insert/search/range across 500 keys (triggers multi-level splits).
- Reopen persistence using explicit `StorageManager` + `BufferPool` (reopen reads meta page 0 in $O(1)$ and returns exact point/range results with zero table scan).
- `test_bplus_high_duplicate_rid_overflow` (Review Focus): insert 1,500 distinct RIDs for the same key `"Electronics"` -> spills cleanly into linked RID overflow bucket pages without overflowing a 4 KB leaf page, and `search("Electronics")` returns all 1,500 RIDs.
- `delete(key, rid)` removes only that RID; `delete(key, None)` removes all RIDs for `key`.
- `range_search` inclusive/exclusive bounds (`include_low=False`, `include_high=False`), empty ranges, `None` keys, and `bool` vs `int` distinctness (`True` != `1`).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_bplus_tree -v`
Expected: FAIL.

- [ ] **Step 3: Implement page-backed `src/smartdb/indexes/bplus_tree.py`**

Design highlights:
- Page 0 (`BPLUS_META`): `root_page_id`, `height`, `first_leaf_page_id`, `distinct_keys`, `total_entries`, `max_keys_override` (0 if uncapped).
- Leaf entries: `(encoded_key, rid_count, inline_rids_or_bucket_page_id)`. When inline RIDs for a key exceed e.g. 16 RIDs (or when a page needs to split), store RIDs in linked `OVERFLOW` RID-bucket pages (each holding `next_bucket_page_id` + array of `(page_id: uint32, slot_id: uint32)`).
- Variable-length encoded keys split cleanly by serialized byte budget (or `max_keys` when `order` is explicitly specified for unit tests).
- Link leaf pages via `next_page` / `prev_page` in the 24-byte `PageHeader`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_bplus_tree -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartdb/indexes/bplus_tree.py tests/test_bplus_tree.py
git commit -m "feat: implement persistent page-backed B+ Tree with duplicate RID buckets and delete"
```

---

### Task 7: Disk-Backed SSTable LSM Tree with Bloom Filters, Fence Pointers & Memtable WAL (`lsm_tree.py`)

**Files:**
- Modify: `src/smartdb/indexes/lsm_tree.py`
- Test: `tests/test_lsm_tree.py`

**Interfaces:**
- Produces:
  - `class BloomFilter`: `__init__(num_bits: int, num_hashes: int)`, `.add(key_bytes: bytes) -> None`, `.might_contain(key_bytes: bytes) -> bool`, `.to_bytes() -> bytes`, `.from_bytes(raw: bytes) -> BloomFilter`
  - `class LSMTree`:
    - `__init__(self, directory: str | Path, flush_threshold: int = 128, compaction_threshold: int = 4, storage_manager: StorageManager | None = None, pool: BufferPool | None = None, name_prefix: str = "lsm", key_dtype: DataType = DataType.ANY)`
    - `.insert(key: Any, rid: RID) -> None`
    - `.delete(key: Any, rid: RID | None = None) -> None`
    - `.search(key: Any) -> list[RID]`
    - `.range_search(low: Any | None = None, high: Any | None = None, include_low: bool = True, include_high: bool = True) -> list[tuple[Any, RID]]`
    - `.flush() -> None`, `.compact() -> None`, `.disk_bytes() -> int`, `.run_count: int`, `.close() -> None`

- [ ] **Step 1: Write failing tests in `tests/test_lsm_tree.py`**

Include tests for:
- `test_lsm_low_cardinality_flushes_by_entry_count` (Review Focus): insert 300 entries across 3 distinct keys with `flush_threshold=100` -> verifies `flushes >= 3` and runs exist on disk BEFORE `close()`.
- Memtable WAL crash recovery: insert 50 entries (`< flush_threshold`), close file handles WITHOUT calling `.close()` / `.flush()`, reopen `LSMTree` -> all 50 entries recovered from `memtable.wal`.
- Atomic compaction across >8 flushes: verify `run_count` drops after compaction, manifest swaps atomically, old run files are cleaned up, and all keys/RIDs remain exact.
- Tombstone semantics: insert `(k, rid1)`, flush to SSTable, delete `(k, rid1)`, insert `(k, rid2)`, verify `search(k) == [rid2]` and `range_search` omits `rid1` both before and after compaction.
- Bloom filter effectiveness: verify `BloomFilter.might_contain` returns `True` for all inserted keys and low false positive rate (~1% at 10 bits/key) for absent keys, and `search(absent_key)` issues 0 `LSM_DATA` `BufferPool` reads.
- `LocalObjectStore` + AES-GCM `TransformingBackend` + WAL record encryption integration round-trip.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_lsm_tree -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/smartdb/indexes/lsm_tree.py`**

Design highlights:
- When `storage_manager` and `pool` are passed (DB mode), store each SSTable run as `"index:{name_prefix}:run_{id:08d}"` through `storage_manager` + `pool` (`LSM_DATA` pages + `LSM_META` page storing Bloom filter bytes + sparse fence pointers `[(first_key_bytes, page_id), ...]`). If omitted (standalone tests), create an internal `StorageManager` + `BufferPool` in `directory`.
- **Memtable:** dict mapping `key_bytes -> list[tuple[RID, int, bool]]` (or `(rid, seq, is_deleted)` plus `total_entries` counter; if `rid is None` on `delete(key, None)`, store wildcard key-level tombstone `(None, seq, True)`!).
- **WAL (`memtable.wal`):** append length-prefixed binary op records `(seq, op_type, key_bytes, rid)` passed through `storage_manager.encrypt_record` if configured (`file.write` + `file.flush()`; `os.fsync()` on SSTable flush/compaction). On `_load_wal()`, replay records safely (ignoring a torn trailing record if process crashed mid-write!).
- **Point & Range Lookup:** track `seq` monotonically (persisted in `manifest.json` `.tmp`-replaced atomically!). For a given key, iterate memtable + runs newest-to-oldest (`run_id` descending), skip runs where `key < min_key or key > max_key or not bloom.might_contain(key)`, binary-search sparse fence pointers to fetch **1** `LSM_DATA` page via `BufferPool`, and resolve final live RIDs (where latest `seq` for `(key, rid)` is not deleted and `seq > latest_key_tombstone_seq`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_lsm_tree -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartdb/indexes/lsm_tree.py tests/test_lsm_tree.py
git commit -m "feat: implement disk-backed SSTable LSMTree with Bloom filter, fence pointers, WAL, and atomic compaction"
```

---

### Task 8: Index Manager, Safe SQL/DML Parser, Engine Reconciliation & CLI

**Files:**
- Modify: `src/smartdb/indexes/manager.py`, `src/smartdb/indexes/__init__.py`
- Modify: `src/smartdb/query.py`, `src/smartdb/engine.py`, `src/smartdb/cli.py`, `src/smartdb/__init__.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces:
  - `choose_index_type(reads: int, writes: int, range_reads: int = 0) -> str`
  - `IndexManager(directory, storage_manager, pool)`:
    - `.create(table, field, index_type="auto", key_dtype=DataType.ANY, reads=100, writes=10, range_reads=0, replace=True) -> tuple[str, BPlusTree | LSMTree]`
    - `.drop(table, field) -> None`
    - `.add(table, field, key, rid)`, `.remove(table, field, key, rid=None)`
    - `.search(table, field, key) -> list[RID] | None`
    - `.range_search(table, field, low=None, high=None, include_low=True, include_high=True) -> list[tuple[Any, RID]] | None`
    - `.disk_bytes(table, field) -> int`
  - `query.py` AST dataclasses (`CreateTableStmt`, `InsertStmt`, `CreateIndexStmt`, `SelectStmt`, `UpdateStmt`, `DeleteStmt`, `Predicate`) + `parse_sql(sql: str)` + `parse_select(sql: str)`
  - `SmartRetailDB`:
    - `__init__(directory, buffer_frames=64, replacement_policy="LRU", backend_kind="page", encrypt_page=None, decrypt_page=None, crypto_overhead=0, encrypt_record=None, decrypt_record=None)`
    - `.create_table(name, schema_or_fields, layout="row") -> None`
    - `.insert(table_name, record) -> RID`
    - `.update(table_name, updates, where=None) -> int`
    - `.delete(table_name, where=None) -> int`
    - `.select(table_name, columns=None, where=None) -> list[dict[str, Any]]`
    - `.create_index(table_name, field, index_type="auto", reads=None, writes=None, range_reads=None) -> str`
    - `.execute(sql: str) -> list[dict[str, Any]] | int | str`
    - `.explain(sql: str) -> str`
    - `.buffer_stats() -> dict[str, int | float | str]`

- [ ] **Step 1: Write failing tests in `tests/test_engine.py`**

Include tests covering:
- `choose_index_type` edge cases (`reads=0, writes=1 -> LSM`, `reads=10, writes=0 -> BPLUS`, `writes=100, reads=10, range_reads=5 -> BPLUS`).
- Full lifecycle (`CREATE TABLE` typed & untyped, `INSERT` including 1-column `(1)` and `VECTOR(3)` `[0.1, 0.2, 0.3]`, `SELECT col1, col2` projection, `WHERE` `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`, `IS NULL`, `IS NOT NULL`, `UPDATE`, `DELETE`, reopen persistence).
- `test_bool_int_query_isolation` (Review Focus): inserting `id=1` then running `SELECT * FROM T WHERE id = TRUE` returns `[]` (or raises/returns empty, never matches `id=1`!).
- `test_null_excluded_from_range_and_inequality_queries` (Review Focus): rows with `price=None` are NOT returned by `WHERE price < 10`, `WHERE price <= 10`, `WHERE price BETWEEN 0 AND 10`, or `WHERE price > 0`; only `WHERE price IS NULL` returns them.
- `test_unclean_shutdown_reconciles_indexes` (Review Focus): insert 150 rows with an LSM index (triggers SSTable flush while table pages sit dirty in `BufferPool`), close raw handles WITHOUT `db.close()`, reopen `SmartRetailDB` -> startup reconciliation runs, queries succeed with zero `IndexError`, and durable rows match indexed lookups.
- `test_failed_insert_is_atomic_across_table_and_indexes` (Review Focus): bad record or unindexed/invalid type raises before mutating table or any index.
- Re-creating / switching `LSM <-> BPLUS` index never leaves stale runs or drops rows.
- Object backend + AES-GCM encryption end-to-end reopen test.
- SQL injection rejection (`OR 1=1`, `; DROP TABLE`, `1+2j`).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_engine -v`
Expected: FAIL.

- [ ] **Step 3: Implement `manager.py`, `query.py`, `engine.py`, `cli.py`, and update `__init__.py` exports**

Key details:
- In `engine.py`:
  - `catalog.json` stores `clean_shutdown: bool`, `tables: {name: {columns, layout, backend_kind, stats: {reads, writes, range_reads}}}`.
  - On open: read `catalog.json`, set `clean_shutdown = False` and save immediately. On `close()`: flush indexes + `BufferPool`, set `clean_shutdown = True` and save.
  - On reopen when `clean_shutdown is False`: iterate indexes, verify or rebuild indexes from durable table records (`iter_records()`), ensuring table and indexes match 100% with no dangling RIDs!
  - In `execute` with indexed `WHERE`:
    - `=` with `None` or `IS NULL` -> scan/lookup `None` explicitly; `IS NOT NULL` -> table/index scan excluding `None`.
    - `<`, `<=`, `>`, `>=`, `BETWEEN`: call `IndexManager.range_search(...)`, filtering out any `None` keys/rows so SQL three-valued logic holds even when `low is None`!
    - Always verify `try_get_record(rid, fields=...)` is not `None` and matches the predicate.
- In `cli.py`: remove duplicated regexes and route SQL statements directly to `db.execute(line)`, printing rows / affected counts / status cleanly, plus `:explain`, `:stats`, `:tables`, `:help`, `:quit`.

- [ ] **Step 4: Run full unit test suite to verify everything passes**

Run: `.venv/bin/python -m unittest discover -s tests -v`
Expected: ALL tests across `test_schema`, `test_storage_backends`, `test_buffer_pool`, `test_row_table`, `test_column_table`, `test_bplus_tree`, `test_lsm_tree`, `test_engine` PASS.

- [ ] **Step 5: Commit**

```bash
git rm tests/test_storage.py tests/test_indexes.py
git add src/smartdb/__init__.py src/smartdb/indexes/__init__.py src/smartdb/indexes/manager.py src/smartdb/query.py src/smartdb/engine.py src/smartdb/cli.py tests/test_engine.py
git commit -m "feat: integrate persistent indexes, safe DML/range SQL parser, crash reconciliation, and CLI"
```

---

### Task 9: Retail Workload Generator, Reproducible M1/M2 Benchmarks & Documentation

**Files:**
- Modify: `workloads/generate_retail_data.py`
- Create: `benchmarks/common.py`
- Modify: `benchmarks/benchmark_storage.py`
- Modify: `benchmarks/benchmark_indexes.py`
- Modify: `README.md`, `docs/architecture.md`, `docs/milestone-checklist.md`

**Interfaces:**
- Produces:
  - `workloads/generate_retail_data.py`: generates typed `Products`, `Customers`, `Orders`, `OrderItems`, `Inventory` tables with `--db`, `--products`, `--customers`, `--orders`, `--seed`, `--backend`.
  - `benchmarks/common.py`: `environment_metadata() -> dict`, `run_repeated(fn, repeats=5, warmup=1) -> tuple[list[float], dict[str, float]]` (returning `mean_ms`, `median_ms`, `p95_ms`, `stdev_ms`), `write_csv(path, rows)`, `save_bar_chart(...)` (optional when `matplotlib` is available & `--plots` passed).
  - `benchmarks/benchmark_storage.py`: benchmarks (1) FIFO vs LRU trace, (2) Row vs Column (insert, 2-col projection scan, full-row lookup, page I/O), (3) `page` vs `objects` (and optional AES-GCM overhead).
  - `benchmarks/benchmark_indexes.py`: benchmarks (1) Indexed (`BPLUS`, `LSM`) vs Full Table Scan (point & range), (2) Write-heavy (80W/20R), Read-heavy (90R/10W), and Range-scan workloads comparing `BPLUS` vs `LSM`, (3) Index build time & on-disk byte size.

- [ ] **Step 1: Update `workloads/generate_retail_data.py` with typed schemas and `OrderItems` + `Inventory` tables**

Ensure schema definitions match `TableSchema` (`INT`, `FLOAT`, `STR`) and deterministic seed generation populates realistic `OrderItems` (1–4 items per order) and `Inventory` (stock per product/warehouse).

- [ ] **Step 2: Create `benchmarks/common.py` and rewrite `benchmarks/benchmark_storage.py` and `benchmarks/benchmark_indexes.py`**

Requirements from `CSAI302_Project_doc.pdf` pp. 5, 11:
- Default `--repeats 5` after 1 warm-up run.
- Print declared machine/OS/Python/seed/dataset specs.
- Include correctness assertions inside every benchmark iteration (e.g., indexed search matches full scan result set!).
- Output CSVs under `benchmark_results/` (and PNG charts under `benchmark_results/plots/` when `--plots` is specified).

- [ ] **Step 3: Update `README.md`, `docs/architecture.md`, and `docs/milestone-checklist.md`**

Document:
- Linux/macOS & Windows quick start (`pip install -e .` and `PYTHONPATH=src`).
- Updated SQL subset (`CREATE TABLE ... (col TYPE)`, `INSERT`, `SELECT cols`, `WHERE` range/`BETWEEN`/`IS NULL`, `UPDATE`, `DELETE`, `CREATE INDEX ... USING AUTO`).
- Updated storage format (24-byte page header, slotted pages, overflow, per-column storage, page-backed B+ Tree, SSTable LSM with Bloom filter & WAL, AEAD encryption hooks).

- [ ] **Step 4: Run full verification suite (tests + generator + M1/M2 benchmarks)**

Run:
```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python workloads/generate_retail_data.py --db /tmp/verify_retail --products 200 --customers 50 --orders 200 --seed 42
.venv/bin/python benchmarks/benchmark_storage.py --pages 64 --accesses 2000 --records 1000 --out /tmp/bench_m1.csv
.venv/bin/python benchmarks/benchmark_indexes.py --records 1000 --out /tmp/bench_m2.csv
rm -rf /tmp/verify_retail /tmp/bench_m1.csv /tmp/bench_m2.csv
```
Expected:
- All unit tests pass (`OK`).
- Generator adds `Products`, `Customers`, `Orders`, `OrderItems`, `Inventory` without error.
- M1 benchmark shows LRU > FIFO hit rate on skewed trace AND Column layout significantly faster with fewer page reads on narrow projection scan vs Row layout.
- M2 benchmark shows Indexed lookup faster than Full Table Scan, LSM faster than B+ Tree on write-heavy workloads, and B+ Tree faster on read/range workloads.

- [ ] **Step 5: Commit**

```bash
git add workloads/generate_retail_data.py benchmarks/common.py benchmarks/benchmark_storage.py benchmarks/benchmark_indexes.py README.md docs/architecture.md docs/milestone-checklist.md
git commit -m "feat: add typed retail workload, reproducible 5-run M1/M2 benchmarks, and updated docs"
```
