# `csai302_db` (M1 + M2) Comprehensive Codebase Refactoring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the legacy prototype in `src/smartdb/` into the spec-compliant, disk-oriented custom relational & object DBMS (`csai302_db`) covering Milestones 1 & 2, complete with M3–M6 forward-compatibility hooks, Gate 1 & Gate 2 adversarial tests, Gate 3 benchmarks & plots, and dual-audience living documentation.

**Architecture:** Replace the 4 KB JSON-payload prototype with a layered binary engine (`src/common/`, `src/storage/`, `src/index/`, `src/query/`) built on universal 8,192-byte checksummed pages with a Page-0 Free Space Map, a thread-safe pin-aware `BufferPoolManager` (`LRU`/`FIFO`), dual `SlottedPage` (with 16-byte MVCC headers) and zero-copy NumPy `ColumnPage` layouts, an extent-based `AES-256-GCM` `ObjectStore`, a buffer-pool-backed `BPlusTreeIndex`, a WAL + MemTable + Bloom-filtered SSTable `LSMTreeIndex`, a sliding-window $(\alpha, \beta)$ `WorkloadAdvisor`, and a parameterized injection-proof `SimpleExecutor`.

**Tech Stack:** Python 3.12+, Python stdlib (`struct`, `memoryview`, `os`, `threading`, `collections`, `heapq`, `bisect`, `hashlib`, `zlib`, `json`, `dataclasses`, `enum`), `numpy`, `cryptography` (`AESGCM`), `pytest`, `matplotlib`.

**Specs:**
- `docs/ai_specs/00_AGENT_GOVERNANCE.md`
- `docs/ai_specs/01_RETAIL_SCHEMA_AND_DATA.md`
- `docs/ai_specs/02_STORAGE_AND_OBJECT_SPEC.md`
- `docs/ai_specs/03_INDEXING_AND_QUERY_SPEC.md`
- `docs/ai_specs/04_TEST_AND_RUBRIC_GATE.md`
- `docs/ai_specs/05_BENCHMARK_PROTOCOL.md`
- `docs/ai_specs/06_M3_TO_M6_HOOKS.md`
- `docs/docify.md`
- `docs/architecture.md` (promoted from `docs/updated_architecture.md`)

---

## Global Constraints

- **Runtime & Annotations:** Target Python 3.12+; all public classes, methods, and functions in `src/` must include complete type annotations and Google-style docstrings.
- **Forbidden Libraries:** Never import `sqlite3`, `psycopg2`, `pymongo`, `mysql`, `rocksdb`, `leveldb`, `lmdb`, `duckdb`, `whoosh`, `faiss`, `pickle`, `dill`, `marshal`, or `shelve` anywhere in `src/`.
- **No OS-File-Per-Record/Object:** Rows live inside 8,192-byte `*.db` files; columns inside 8,192-byte `*.col` files; objects inside `objects.dat` + `objects.dir`.
- **No Test Tampering:** Never modify, weaken, skip, or delete assertions in `tests/` to force a failing test suite to pass.
- **Approved Stack:** Python stdlib + `numpy` (zero-copy `ColumnPage` views & metric math only) + `cryptography` (`AESGCM`) + `pytest` + `matplotlib`.
- **Universal Page Geometry:** Strictly `PAGE_SIZE = 8192` bytes, `HEADER_SIZE = 32` bytes (`<IBBHQIIQ`), 64-bit BLAKE2b checksum over `buf[0:24] + buf[32:8192]`.
- **Resolved Binary Formats:**
  - `BPlusTreeIndex` Page 0 uses `fsm_offset = 64` (reserving bytes `32..63` for B+ Tree metadata and `64..8191` for FSM = 65,024 pages). Metadata at bytes `32..55` packs `<IBHHQ?6x`: `(root_page_id: uint32, key_dtype: uint8, key_size: uint16, height: uint16, entry_count: uint64, unique: bool, 6B padding)`.
  - `SSTable` 64-byte EOF footer packs `<8sQIIQIQQI8s`: `(magic=b"CSAI_SST", bloom_offset: uint64, bloom_len: uint32, num_hashes: uint32, index_offset: uint64, index_len: uint32, entry_count: uint64, bloom_num_bits: uint64, crc32: uint32, padding: 8B)`.
- **Thread-Safety & Lock Ordering:** Synchronize shared mutable structures with `threading.RLock`. Strictly acquire locks top-down: `CatalogLock -> IndexLock -> TableHeapLock -> BufferPoolLock -> DiskManagerLock`.
- **Standardized Exceptions:** All engine errors inherit from `DatabaseError` in `src/common/exceptions.py`.
- **In-Code Documentation (`docs/docify.md`):** Every `.py` under `src/`, `tests/`, `bench/` must start with a structured module header (`Module`, `Layer`, `Lock Domain`, `Purpose`, `Key Invariants`). Every binary-packing class (`PageHeader`, `SlottedPage`, `ColumnPage`, `ObjectStore`, `BPlusTreeIndex`, `LSMTreeIndex`) must include an exact byte-offset ASCII diagram matching its `struct` formats. Inline comments explain *why*, never *what*.
- **Living Documentation:** Maintain `README.md`, `docs/DEV_JOURNAL.md`, `docs/decisions/ADR_M1_M2.md`, `docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md`, and `docs/handoff/M3_M6_HANDOFF_CONTRACT.md` at Wave Gates 2, 4, and 5 with zero `TODO`/`TBD`/`FIXME`.

---

## Review Focus

1. **Negative `FLOAT64` & `-0.0` Key Ordering:** `-1000.5 < -0.5 < -0.0 == +0.0 < 0.5 < 1000.5` must hold in raw byte comparison; `NaN` rejected. (Pinned in Task 0.2 & Task 3A.1)
2. **B+ Tree Exact Separator Lookup & Latch-Crabbing Pin Limit:** Internal routing uses `bisect_right` (since `k == sep_i` lives in `child_{i+1}`) and unpins parent before descending so `<= 2` frames are pinned even on a 3-frame pool. (Pinned in Task 3A.1)
3. **LSM Torn WAL Tail & Multi-Level Tombstone Non-Resurrection:** Abrupt crash mid-WAL write discards corrupt trailing bytes safely; tombstones are dropped ONLY during compaction into the deepest existing level. (Pinned in Task 3B.1)
4. **`SlottedPage` Update Overflow & Immutable `RID`:** `compact()` preserves `slot_id`s; `RowTableHeap.update()` relocates tuples on overflow and `SimpleExecutor` updates indexes cleanly. (Pinned in Task 1.2, 2.2, 4.2)
5. **`ObjectStore` Cross-OID Swap Tampering (`AAD` Binding):** Swapping ciphertext/nonce/tag from `oid=1` to `oid=2` fails `AES256GCMHook.decrypt()` with `SecurityViolationError` due to `aad = pack("<Q", oid)`. (Pinned in Task 1.4)

---

### Task 0.1: Environment Setup, Packaging & Legacy Cleanup

**Files:**
- Modify: `pyproject.toml`, `requirements.txt`, `.gitignore`, `docs/architecture.md`
- Rename: `docs/ai_specs/00_AGENT_GOVERNANCE.md.md` -> `docs/ai_specs/00_AGENT_GOVERNANCE.md`
- Delete: `src/smartdb/`, `src/smart_retail_db.egg-info/`, `retail_benchmark_data/`, `tests/test_storage.py`, `tests/test_indexes.py`, `tests/test_engine.py`, `benchmarks/`, `workloads/`, `docs/updated_architecture.md`
- Create: `README.md`, `docs/DEV_JOURNAL.md`

**Interfaces:**
- Produces: Working `.venv/` with `numpy`, `cryptography`, `pytest`, `matplotlib`; clean repo layout matching `docs/ai_specs/00_AGENT_GOVERNANCE.md`.

- [ ] **Step 1: Create `.venv` and install dependencies**
  Create `.venv` via `python3 -m venv .venv` and install `numpy`, `cryptography`, `pytest`, `matplotlib` inside `.venv`. Verify `.venv/bin/python -c "import numpy, cryptography, pytest, matplotlib; print('OK')"` prints `OK`.
- [ ] **Step 2: Update `pyproject.toml`, `requirements.txt`, and `.gitignore`**
  Configure `pyproject.toml` with `name = "csai302-db"`, `requires-python = ">=3.12"`, `dependencies = ["numpy>=1.26", "cryptography>=42.0"]`, `[project.optional-dependencies] dev = ["pytest>=8.0", "matplotlib>=3.8"]`, and `[tool.pytest.ini_options] pythonpath = ["."]`, `testpaths = ["tests"]`. Update `requirements.txt` with `numpy>=1.26`, `cryptography>=42.0`, `pytest>=8.0`, `matplotlib>=3.8`. Add `data/` and `.superpowers/` to `.gitignore`.
- [ ] **Step 3: Clean up legacy files and align docs**
  Rename `docs/ai_specs/00_AGENT_GOVERNANCE.md.md` to `docs/ai_specs/00_AGENT_GOVERNANCE.md`. Copy `docs/updated_architecture.md` into `docs/architecture.md` with the two struct-format clarifications (`<IBHHQ?6x` with `fsm_offset=64` for B+ Tree Page 0; `<8sQIIQIQQI8s` for SSTable 64B footer), and update the `00_AGENT_GOVERNANCE.md` path reference. Remove `docs/updated_architecture.md`, `src/smartdb/`, `src/smart_retail_db.egg-info/`, `retail_benchmark_data/`, `tests/test_storage.py`, `tests/test_indexes.py`, `tests/test_engine.py`, `benchmarks/`, `workloads/`.
- [ ] **Step 4: Initialize `README.md` and `docs/DEV_JOURNAL.md`**
  Write initial `README.md` (system overview, `.venv` setup, repo map, Gate 1–3 commands, 30-mark rubric table) and `docs/DEV_JOURNAL.md` (logging Wave 0 legacy audit & the two spec struct-format conflicts resolved).
- [ ] **Step 5: Commit**
  `git add -A && git commit -m "chore(wave0): clean legacy prototype, setup venv packaging, and align docs"`

---

### Task 0.2: Common Primitives, ABCs & Complete Gate 1/2 Test Suites

**Files:**
- Create: `src/__init__.py`, `src/common/__init__.py`, `src/common/config.py`, `src/common/exceptions.py`, `src/common/types.py`, `src/storage/__init__.py`, `src/storage/replacer.py`, `src/storage/cipher_hooks.py`, `src/index/__init__.py`, `src/index/base_index.py`, `tests/__init__.py`, `tests/test_m1_storage.py`, `tests/test_m2_indexing.py`

**Interfaces:**
- Produces `src/common/config.py`: `PAGE_SIZE = 8192`, `HEADER_SIZE = 32`, `INVALID_PAGE_ID = 0xFFFFFFFF`, `INVALID_SLOT_ID = 0xFFFFFFFF`, `MVCC_HEADER_SIZE = 16`, `SLOT_ENTRY_SIZE = 4`, `CHUNK_ROWS = 1020`, `DEFAULT_FSM_OFFSET = 32`, `BPLUS_FSM_OFFSET = 64`, `DEFAULT_BUFFER_POOL_SIZE = 64`, `DEFAULT_MEMTABLE_BYTES = 1_048_576`, `LSM_COMPACTION_TRIGGER = 4`, `BLOOM_FPR = 0.01`, `ADVISOR_WINDOW_SIZE = 1000`, `ADVISOR_MIN_OPS = 200`, `ADVISOR_MIN_PAGES = 4`, `ADVISOR_ALPHA_THRESHOLD = 0.45`, `ADVISOR_BETA_THRESHOLD = 0.25`, `MAGIC_OBJECT_RECORD = b"OBJ1"`, `MAGIC_OBJECT_DIR = b"ODIR"`, `MAGIC_SSTABLE = b"CSAI_SST"`.
- Produces `src/common/exceptions.py`: `DatabaseError`, `StorageError`, `PageCorruptedError`, `BufferPoolFullError`, `RecordTooLargeError`, `IndexErrorBase`, `DuplicateKeyError`, `KeyNotFoundError`, `QuerySyntaxError`, `SecurityViolationError`.
- Produces `src/common/types.py`:
  - `DataType(IntEnum)`: `INT32 = 1`, `INT64 = 2`, `UINT64 = 3`, `FLOAT64 = 4`, `VARCHAR = 5` (properties `fixed_size`, `struct_fmt`, `numpy_dtype`)
  - `LayoutType(str, Enum)`: `ROW = "ROW"`, `COLUMN = "COLUMN"`
  - `RID(page_id: int, slot_id: int)` (`frozen=True, order=True`) with `pack() -> bytes`, `unpack(data, offset=0) -> RID`, `NULL_RID = RID(0xFFFFFFFF, 0xFFFFFFFF)`
  - `ColumnDef(name: str, dtype: DataType, max_len: int = 8, nullable: bool = False)` with `to_dict()`, `from_dict()`
  - `Schema(table_name: str, columns: tuple[ColumnDef, ...], primary_key: str | None = None)` with `get_column(name)`, `column_index(name)`, `encode_row(values) -> bytes`, `decode_row(data) -> dict[str, Any]`
  - `encode_key(value: Any, dtype: DataType, key_size: int) -> bytes` and `decode_key(raw: bytes | memoryview, dtype: DataType) -> Any`
  - `make_composite_key(key_bytes: bytes, rid: RID) -> bytes` and `split_composite_key(comp: bytes, key_size: int) -> tuple[bytes, RID]`
- Produces ABCs:
  - `src/storage/replacer.py`: `ReplacementPolicy(ABC)` (`victim`, `pin`, `unpin`, `size`)
  - `src/storage/cipher_hooks.py`: `CipherPayload(nonce, tag, ciphertext)` and `CipherHook(ABC)` (`algorithm_id`, `encrypt`, `decrypt`)
  - `src/index/base_index.py`: `BaseIndex(ABC)` (`insert`, `get`, `range_scan`, `delete`, `flush`, `close`, `get_stats`)

- [ ] **Step 1: Implement `src/common/config.py`, `exceptions.py`, `types.py` and `__init__.py` files**
  Include full `docify.md` module headers, Google docstrings, exact binary formats (`<II` for `RID`, `<H + Ls` for row `VARCHAR`, sign-bit XOR for signed ints, IEEE-754 total-order mapping normalizing `-0.0 -> +0.0` and rejecting `NaN` for `FLOAT64`, `\x00`-padded UTF-8 for fixed-width `VARCHAR` keys).
- [ ] **Step 2: Create ABC interfaces in `replacer.py`, `cipher_hooks.py`, `base_index.py`**
  Define clean abstract base classes per specs.
- [ ] **Step 3: Author `tests/test_m1_storage.py` (11 Gate 1 tests + key codec tests) and `tests/test_m2_indexing.py` (11 Gate 2 tests)**
  Include all 11 Gate 1 tests (`test_m1_page_header_binary_packing`, `test_m1_torn_page_checksum_detection`, `test_m1_fsm_allocation_reuse`, `test_m1_slotted_page_crud_and_compaction`, `test_m1_column_page_zero_copy_numpy`, `test_m1_lru_eviction_order`, `test_m1_fifo_eviction_order`, `test_m1_pin_protection_and_pool_full`, `test_m1_dirty_writeback_accuracy`, `test_m1_hit_miss_telemetry`, `test_m1_persistence_across_restart`) plus `test_common_key_encoding_total_order` (negative floats, `-0.0 == +0.0`, NaN raises `ValueError`, signed ints, VARCHAR).
  Include all 11 Gate 2 tests (`test_m2_object_store_aes256_gcm_and_tamper_detection` including cross-OID swap check, `test_m2_bplus_50k_inserts_and_invariants` including 3-frame latch-crabbing assertion, `test_m2_bplus_sibling_range_scan`, `test_m2_bplus_restart_persistence`, `test_m2_lsm_wal_crash_recovery` including torn WAL tail bytes, `test_m2_lsm_tombstone_compaction`, `test_m2_lsm_bloom_filter_effectiveness`, `test_m2_workload_advisor_decision_rules`, `test_m2_sql_executor_and_injection_defense`, `test_m2_index_vs_fullscan_equivalence`, `test_m2_bplus_delete_and_lsm_range_scan`).
- [ ] **Step 4: Run common codec tests and confirm storage/index tests fail as expected**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py -k "key_encoding"` -> PASS; confirm remaining Gate 1/2 tests fail due to unimplemented modules.
- [ ] **Step 5: Commit**
  `git add src tests && git commit -m "feat(wave0): freeze common contracts, ABCs, and adversarial Gate 1/2 test suites"`

---

### Task 1.1: Universal `PageHeader` & `DiskManager`

**Files:**
- Create: `src/storage/page_header.py`, `src/storage/disk_manager.py`
- Test: `tests/test_m1_storage.py`

**Interfaces:**
- Consumes: `src/common/config.py`, `src/common/exceptions.py`
- Produces `src/storage/page_header.py`: `PageType(IntEnum)`, `PageFlags(IntFlag)`, `PageHeader` (`pack`, `unpack`, `pack_into`, `compute_checksum`, `stamp_checksum`, `verify_checksum`, `get_page_lsn`, `set_page_lsn`)
- Produces `src/storage/disk_manager.py`: `DiskManager(file_path, fsm_offset=DEFAULT_FSM_OFFSET)` (`allocate_page`, `deallocate_page`, `read_page`, `write_page`, `allocated_pages`, `is_allocated`, `num_pages`, `disk_reads`, `disk_writes`, `sync`, `close`)

- [ ] **Step 1: Verify RED on Gate 1 header/disk tests**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py -k "page_header or torn_page or fsm"` -> FAIL (ImportError).
- [ ] **Step 2: Implement `src/storage/page_header.py` and `src/storage/disk_manager.py`**
  Use `<IBBHQIIQ`, 64-bit `BLAKE2b(digest_size=8)` checksum over `buf[0:24] + buf[32:8192]`. In `DiskManager`, use `os.open(..., O_RDWR | O_CREAT)`, `os.pread`/`os.pwrite` (with lock+seek fallback if unsupported), `threading.RLock`, Page 0 FSM bitmap starting at `fsm_offset` (bit 0 always 1), zero-page bypass on `read_page` checksum check, and authoritative FSM bitmap merge on `write_page(0, ...)`. Include exact ASCII byte diagrams.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py -k "page_header or torn_page or fsm"` -> 3/3 PASS.
- [ ] **Step 4: Commit**
  `git add src/storage/page_header.py src/storage/disk_manager.py && git commit -m "feat(storage): implement 8KB PageHeader with BLAKE2b checksum and FSM DiskManager"`

---

### Task 1.2: Row-Oriented `SlottedPage`

**Files:**
- Create: `src/storage/slotted_page.py`
- Test: `tests/test_m1_storage.py`

**Interfaces:**
- Consumes: `PageHeader`, `PageType`, `config.py`, `RecordTooLargeError`
- Produces `SlottedPage`: `init_page(buf, page_id)`, `insert_tuple(payload, xmin=1, xmax=0) -> int | None`, `get_tuple(slot_id, include_mvcc=False) -> bytes | None`, `update_tuple(slot_id, new_payload, xmin=None, xmax=None) -> bool`, `delete_tuple(slot_id) -> bool`, `compact()`, `contiguous_free_space()`, `total_free_space()`, `iter_live_slots()`, `get_tuple_mvcc_header(slot_id) -> tuple[int, int]`, `set_tuple_xmax(slot_id, xmax)`

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py -k "slotted_page"` -> FAIL.
- [ ] **Step 2: Implement `src/storage/slotted_page.py`**
  Include exact ASCII byte layout diagram (`PageHeader 32B | Slot Dir 4B*N | Free | Packed Tuples [16B MVCC + User Data]`). Never remove/shift slot entries on `delete_tuple` or `compact()`. Raise `RecordTooLargeError` if `len(payload) + 16 + 4 > PAGE_SIZE - HEADER_SIZE`.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py -k "slotted_page"` -> PASS.
- [ ] **Step 4: Commit**
  `git add src/storage/slotted_page.py && git commit -m "feat(storage): implement SlottedPage with 16B MVCC headers and RID-preserving compaction"`

---

### Task 1.3: Column-Oriented `ColumnPage`

**Files:**
- Create: `src/storage/column_page.py`
- Test: `tests/test_m1_storage.py`

**Interfaces:**
- Consumes: `PageHeader`, `PageType`, `DataType`, `config.py`, `RecordTooLargeError`, `numpy`
- Produces `ColumnPage(buf, dtype, max_rows=None, stride=None)`: `init_page`, `capacity`, `append_value`, `append_batch`, `get_value`, `as_numpy_array(dtype=None) -> np.ndarray`

- [ ] **Step 1: Implement `src/storage/column_page.py`**
  Support fixed-width types (`INT32`, `INT64`, `UINT64`, `FLOAT64`), variable-length `VARCHAR` (4-byte forward offset entries + backward UTF-8 heap), and arbitrary fixed-byte `stride` (for M5 384-d `float32` vectors -> `stride=1536`, `capacity=5`). Ensure `as_numpy_array()` returns a true zero-copy `np.frombuffer(self.buffer, ...)` view (`arr.flags['OWNDATA'] is False`). Include ASCII layout diagrams.
- [ ] **Step 2: Verify with targeted unit check**
  Run `.venv/bin/python -c "..."` or focused test verifying `OWNDATA is False`, `VARCHAR` round-trip, and `stride=1536` `capacity == 5`.
- [ ] **Step 3: Commit**
  `git add src/storage/column_page.py && git commit -m "feat(storage): implement ColumnPage with zero-copy NumPy views and vector stride hook"`

---

### Task 1.4: Cipher Hooks & Extent-Based `ObjectStore`

**Files:**
- Modify: `src/storage/cipher_hooks.py`
- Create: `src/storage/object_store.py`
- Test: `tests/test_m2_indexing.py`

**Interfaces:**
- Consumes: `config.py`, `SecurityViolationError`, `PageCorruptedError`, `cryptography.hazmat.primitives.ciphers.aead.AESGCM`
- Produces `src/storage/cipher_hooks.py`: `NoOpCipher`, `AES256GCMHook(key: bytes)`
- Produces `src/storage/object_store.py`: `ObjectMeta(oid, file_offset, cipher_len, raw_len)`, `ObjectStore(directory, cipher=None)` (`put_object`, `get_object`, `delete_object`, `exists`, `flush`, `close`, `get_stats`)

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "object_store"` -> FAIL.
- [ ] **Step 2: Implement `NoOpCipher`, `AES256GCMHook`, and `ObjectStore`**
  In `ObjectStore`, write 48 B `<4sQII12s16s` header (`b"OBJ1"`, `oid`, `raw_len`, `cipher_len`, `nonce`, `tag`) + `ciphertext` + 4 B `<I` CRC32 to `objects.dat`. Bind `aad = struct.pack("<Q", oid)`. Persist `objects.dir` (`<4sI` `b"ODIR"`, `entry_count` + 24 B `<QQII` entries) atomically (`write .tmp -> fsync -> os.replace`). Include ASCII diagrams and `threading.RLock`.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "object_store"` -> PASS (including tamper & cross-OID swap detection).
- [ ] **Step 4: Commit**
  `git add src/storage/cipher_hooks.py src/storage/object_store.py && git commit -m "feat(storage): implement AES-256-GCM CipherHook and extent-based ObjectStore"`

---

### Task 2.1: Replacement Policies & `BufferPoolManager`

**Files:**
- Modify: `src/storage/replacer.py`
- Create: `src/storage/buffer_pool.py`
- Test: `tests/test_m1_storage.py`

**Interfaces:**
- Consumes: `ReplacementPolicy`, `DiskManager`, `BufferPoolFullError`, `PAGE_SIZE`
- Produces `src/storage/replacer.py`: `LRUReplacer(num_frames)`, `FIFOReplacer(num_frames)`
- Produces `src/storage/buffer_pool.py`: `FrameHeader`, `BufferPoolManager(disk_manager, pool_size=64, replacer=None)` (`fetch_page`, `unpin_page`, `new_page`, `flush_page`, `flush_all_pages`, `delete_page`, `is_resident`, `get_stats`, `.hit_count`, `.miss_count`, `.eviction_write_count`, `.pool_size`)

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py -k "lru or fifo or pin_protection or dirty_writeback or hit_miss"` -> FAIL.
- [ ] **Step 2: Implement `LRUReplacer`, `FIFOReplacer`, and `BufferPoolManager`**
  Protect all operations with `self._lock = threading.RLock()`. In `new_page()`, secure a frame (from `free_list` or `replacer.victim()`, flushing if dirty) *before* calling `disk_manager.allocate_page()` so `BufferPoolFullError` never leaks an allocated page on disk.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py -k "lru or fifo or pin_protection or dirty_writeback or hit_miss"` -> 5/5 PASS.
- [ ] **Step 4: Commit**
  `git add src/storage/replacer.py src/storage/buffer_pool.py && git commit -m "feat(storage): implement pin-aware BufferPoolManager with LRU and FIFO replacers"`

---

### Task 2.2: `RowTableHeap`, `ColumnTableHeap`, `StorageEngine` & Gate 1 + M1 ADRs

**Files:**
- Create: `src/storage/table_heap.py`, `src/storage/storage_engine.py`, `docs/decisions/ADR_M1_M2.md`
- Modify: `docs/DEV_JOURNAL.md`
- Test: `tests/test_m1_storage.py` (full Gate 1)

**Interfaces:**
- Consumes: `Schema`, `LayoutType`, `RID`, `DiskManager`, `BufferPoolManager`, `SlottedPage`, `ColumnPage`, `ObjectStore`, `CipherHook`
- Produces `src/storage/table_heap.py`: `RowTableHeap(schema, bpm)`, `ColumnTableHeap(schema, column_bpms, chunk_rows=CHUNK_ROWS)`
- Produces `src/storage/storage_engine.py`: `StorageEngine(data_dir, buffer_pool_size=64, policy="LRU", cipher=None)`

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m1_storage.py` -> remaining table/restart tests FAIL.
- [ ] **Step 2: Implement `src/storage/table_heap.py` and `src/storage/storage_engine.py`**
  `RowTableHeap` caches `_last_page_id`, initializes pages with `SlottedPage.init_page`, relocates on `update` overflow, and unpins pages in `try/finally`. `ColumnTableHeap` maintains aligned `chunk_rows=1020` across one `.col` file per column, implements `insert`, `insert_batch`, `get`, `iter_column_pages` (safe `try/finally` unpin), `iter_batches` (M3 hook), and `full_scan`. `StorageEngine` persists lightweight engine table registry (`storage_meta.json` or synced catalog) so `StorageEngine(data_dir)` reopens existing tables and `ObjectStore` automatically after restart.
- [ ] **Step 3: Run Gate 1 (`pytest -v tests/test_m1_storage.py`)**
  Verify 100% PASS across all Gate 1 tests (12/12 tests including key encoding).
- [ ] **Step 4: Complete Wave 2 `docify.md` Deliverables**
  Verify ASCII binary layout diagrams in `src/storage/*`. Create `docs/decisions/ADR_M1_M2.md` with `ADR-001` through `ADR-004` using the 4-part template in `docs/docify.md`. Append Gate 1 findings/resolutions to `docs/DEV_JOURNAL.md`.
- [ ] **Step 5: Commit**
  `git add src/storage docs/ && git commit -m "feat(wave2): complete table heaps, StorageEngine, Gate 1 (100% pass), and M1 ADRs"`

---

### Task 3A.1: Disk-Backed `BPlusTreeIndex` (Track 3A)

**Files:**
- Create: `src/index/bplus_tree.py`
- Test: `tests/test_m2_indexing.py`

**Interfaces:**
- Consumes: `BaseIndex`, `DataType`, `RID`, `encode_key`, `decode_key`, `make_composite_key`, `split_composite_key`, `DiskManager`, `BufferPoolManager`, `PageHeader`, `PageType`, `PageFlags`, `DuplicateKeyError`
- Produces `src/index/bplus_tree.py`: `BPlusTreeIndex(file_path, key_dtype, key_size=8, unique=False, buffer_pool_size=64)` (`insert`, `get`, `range_scan`, `delete`, `flush`, `close`, `get_stats`, `validate_invariants`)

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "bplus"` -> FAIL.
- [ ] **Step 2: Implement `src/index/bplus_tree.py`**
  Use `DiskManager(file_path, fsm_offset=64)`. Persist Metadata Page 0 (`<IBHHQ?6x` at bytes `32..55`) immediately on root/height changes. Use composite key `encode_key(...) + pack(">II", rid.page_id, rid.slot_id)`. Use `bisect_right` on internal node separators (`page_type = 3`) and in-place `memoryview` slice shifts for fast inserts/deletes. Maintain `next_page_id` chain on leaves (`page_type = 4`). Follow latch-crabbing pin discipline (`<= 2` pinned frames during traversal). Include exact ASCII diagrams.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "bplus"` -> 4/4 PASS (including 50k shuffled inserts & structural invariants, float64 range scan, restart persistence, delete, and 3-frame pool stress).
- [ ] **Step 4: Commit**
  `git add src/index/bplus_tree.py && git commit -m "feat(index): implement disk-backed BPlusTreeIndex with order-preserving keys and splits"`

---

### Task 3B.1: Three-Tier Write-Optimized `LSMTreeIndex` (Track 3B)

**Files:**
- Create: `src/index/lsm_tree.py`
- Test: `tests/test_m2_indexing.py`

**Interfaces:**
- Consumes: `BaseIndex`, `DataType`, `RID`, `encode_key`, `decode_key`, `make_composite_key`, `config.py`
- Produces `src/index/lsm_tree.py`: `BloomFilter`, `MemTable`, `SSTable`, `LSMTreeIndex(directory, key_dtype, key_size=8, unique=True, memtable_size_bytes=1_048_576, compaction_trigger=4, fsync_on_write=False)` (`insert`, `get`, `range_scan`, `delete`, `flush`, `compact`, `close`, `get_stats`)

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "lsm"` -> FAIL.
- [ ] **Step 2: Implement `src/index/lsm_tree.py`**
  - `lsm.wal`: CRC32 + op_type + key_len + key_bytes + rid_page + rid_slot; on open, replay valid records and truncate any torn/corrupt tail.
  - `BloomFilter`: SHA-256 double hashing over **user key only** (`encode_key(...)`), target FPR `0.01`.
  - `SSTable`: 8,192 B data blocks (`<HH` header + entries), Bloom filter section, sparse fence index loaded into RAM, exact 64 B EOF footer `<8sQIIQIQQI8s`.
  - `unique=True` keys store `encode_key(key)` so newer PUTs overwrite older RIDs and DELETEs tombstone the key; `unique=False` stores composite `(key, rid)` keys.
  - Size-tiered compaction merges all SSTables in `L_i` into `L_{i+1}` via `heapq.merge`, deduplicating newest versions and dropping tombstones ONLY when `L_{i+1}` is the deepest level; updates `manifest.json` atomically.
  - Include exact ASCII binary layout diagrams.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "lsm"` -> PASS (WAL crash recovery, tombstone compaction, >95% Bloom rejection, multi-tier range scan).
- [ ] **Step 4: Commit**
  `git add src/index/lsm_tree.py && git commit -m "feat(index): implement LSMTreeIndex with CRC32 WAL, Bloom filters, and compaction"`

---

### Task 4.1: Workload-Based Index Auto-Selector (`WorkloadAdvisor`)

**Files:**
- Create: `src/index/index_advisor.py`
- Test: `tests/test_m2_indexing.py`

**Interfaces:**
- Consumes: `config.py`
- Produces `src/index/index_advisor.py`: `IndexStrategy(str, Enum)` (`FULL_SCAN`, `BPLUS_TREE`, `LSM_TREE`), `Recommendation`, `WorkloadAdvisor(window_size=1000, min_ops=200, min_pages=4, alpha_threshold=0.45, beta_threshold=0.25)` (`record`, `recommend`, `reset`, `decision_log`)

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "workload_advisor"` -> FAIL.
- [ ] **Step 2: Implement `src/index/index_advisor.py`**
  Use per-`(table, column)` `deque(maxlen=window_size)` under `threading.RLock`, compute $\alpha$ and $\beta$, enforce Rules 1–3 (`N_ops < 200 or P < 4 -> FULL_SCAN`; `alpha > 0.45 and beta < 0.25 -> LSM_TREE`; else `BPLUS_TREE`), and log `Recommendation` entries in `decision_log`.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "workload_advisor"` -> PASS.
- [ ] **Step 4: Commit**
  `git add src/index/index_advisor.py && git commit -m "feat(index): implement sliding-window WorkloadAdvisor with (alpha, beta) rules"`

---

### Task 4.2: Persistent `CatalogManager` & Parameterized `SimpleExecutor`

**Files:**
- Create: `src/query/__init__.py`, `src/query/catalog.py`, `src/query/simple_executor.py`
- Test: `tests/test_m2_indexing.py`

**Interfaces:**
- Consumes: `StorageEngine`, `RowTableHeap`, `ColumnTableHeap`, `BPlusTreeIndex`, `LSMTreeIndex`, `WorkloadAdvisor`, `Schema`, `LayoutType`, `QuerySyntaxError`, `SecurityViolationError`
- Produces `src/query/catalog.py`: `CatalogManager(catalog_path)` (`register_table`, `register_index`, `update_table_stats`, `get_table_stats`, `get_schema`, `get_indexes`, `list_tables`, `save`)
- Produces `src/query/simple_executor.py`: `SimpleExecutor(storage_engine, catalog, advisor=None, indexes_dir=None)` (`execute`, `explain`, `get_index`, `flush_all`, `close`)

- [ ] **Step 1: Verify RED**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "sql_executor"` -> FAIL.
- [ ] **Step 2: Implement `src/query/catalog.py` and `src/query/simple_executor.py`**
  `CatalogManager` persists `catalog.json` atomically (`.tmp -> fsync -> os.replace`) and maintains `row_count`, `page_count`, and per-column `min_val`/`max_val` for M3 CBO hooks. `SimpleExecutor` rejects unparameterized literals, quotes, SQL comments, and stacked semicolons with `SecurityViolationError`; supports `CREATE INDEX`, `INSERT`, `SELECT * [WHERE col = ?]`, `SELECT * WHERE col BETWEEN ? AND ?`, `UPDATE ... SET ... WHERE ...`, and `DELETE FROM ... WHERE ...`; maintains indexes on inserts, updates (including `RID` relocation), and deletes; persists & reloads existing indexes on reopen.
- [ ] **Step 3: Verify GREEN**
  Run `.venv/bin/pytest -v tests/test_m2_indexing.py -k "sql_executor"` -> PASS.
- [ ] **Step 4: Commit**
  `git add src/query/ && git commit -m "feat(query): implement atomic CatalogManager and injection-proof SimpleExecutor"`

---

### Task 4.3: Deterministic `RetailDataGenerator`, Gate 2 Verification & M2 ADRs

**Files:**
- Create: `bench/__init__.py`, `bench/retail_loader.py`
- Modify: `docs/decisions/ADR_M1_M2.md`, `README.md`, `docs/DEV_JOURNAL.md`
- Test: `tests/test_m1_storage.py`, `tests/test_m2_indexing.py`

**Interfaces:**
- Consumes: `StorageEngine`, `CatalogManager`, `SimpleExecutor`, `Schema`, `ColumnDef`, `DataType`, `LayoutType`, `AES256GCMHook`
- Produces `bench/retail_loader.py`: `CUSTOMERS_SCHEMA`, `PRODUCTS_SCHEMA`, `INVENTORY_SCHEMA`, `ORDERS_SCHEMA`, `ORDER_ITEMS_SCHEMA`, `SCALES`, `RetailDataGenerator(seed=42)`, `load_retail_dataset(data_dir, scale="standard", seed=42, build_indexes=True, encryption_key=None)`

- [ ] **Step 1: Implement `bench/retail_loader.py`**
  Match all 5 table schemas, Zipfian skew ($\theta = 0.9$), PNG-prefixed product images (`16 KB`–`128 KB`), and `"standard"` / `"tiny"` scales from `01_RETAIL_SCHEMA_AND_DATA.md` verbatim. Add CLI entrypoint `python -m bench.retail_loader`.
- [ ] **Step 2: Run Gate 2 (`pytest -v tests/test_m1_storage.py tests/test_m2_indexing.py`)**
  Verify 100% PASS across ALL Gate 1 & Gate 2 tests (23/23 tests).
- [ ] **Step 3: Smoke-test CLI data loader**
  Run `.venv/bin/python -m bench.retail_loader --scale tiny --seed 42 --data-dir /tmp/csai302_tiny_smoke` and verify exit code 0 and clean cleanup.
- [ ] **Step 4: Complete Wave 4 `docify.md` Deliverables**
  Verify ASCII diagrams in `src/index/*`. Add `ADR-005` through `ADR-007` to `docs/decisions/ADR_M1_M2.md`. Update `README.md` with full query/loader usage. Append Gate 2 notes to `docs/DEV_JOURNAL.md`.
- [ ] **Step 5: Commit**
  `git add bench/retail_loader.py docs/ README.md && git commit -m "feat(wave4): add RetailDataGenerator, pass Gate 2 (100%), and complete M2 ADRs"`

---

### Task 5.1: Reproducible Benchmark Suite & 5 High-Res Plots (`bench/run_m1_m2_benchmarks.py`)

**Files:**
- Create: `bench/run_m1_m2_benchmarks.py`
- Generated Outputs: `results/m1_m2/env_metadata.json`, `results/m1_m2/raw_runs.csv`, `results/m1_m2/summary_metrics.csv`, `results/m1_m2/plots/fig1_buffer_pool_hit_ratio.png`..`fig5_encryption_throughput.png`

**Protocol Decisions:**
- 1 untimed warm-up + 5 measured trials (`seed=42`) per configuration point across Experiments 1A, 1B, 2A, 2B, 2C, 2D.
- Every experiment runs correctness assertions before recording timings.
- In Exp 2A/2B, run the full 10,000 / 20,000 operations on `BPlusTreeIndex` and `LSMTreeIndex`, and measure a statistically representative 200-query read sample (+ all writes) for `FullTableScan`, documenting the exact methodology transparently in `env_metadata.json`.
- Generate all 5 300-DPI PNG charts with titles, axis units, legends, and $\pm 1\text{ std}$ error bars (`matplotlib.use("Agg")`).

- [ ] **Step 1: Implement `bench/run_m1_m2_benchmarks.py`**
  Implement all 6 experiments, CSV/JSON exporters, and all 5 plotting functions.
- [ ] **Step 2: Execute Gate 3 benchmark suite**
  Run `.venv/bin/python -m bench.run_m1_m2_benchmarks --runs 5 --seed 42 --out-dir results/m1_m2` and verify `env_metadata.json`, `raw_runs.csv`, `summary_metrics.csv`, and all 5 PNG plots exist and contain valid non-empty numeric data.
- [ ] **Step 3: Commit**
  `git add bench/run_m1_m2_benchmarks.py results/ && git commit -m "feat(bench): implement M1/M2 benchmark suite and generate Gate 3 metrics and plots"`

---

### Task 5.2: Technical Narrative, M3–M6 Handoff Contract & Final 6-Point Audit

**Files:**
- Create: `docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md`, `docs/handoff/M3_M6_HANDOFF_CONTRACT.md`
- Modify: `README.md`, `docs/DEV_JOURNAL.md`

- [ ] **Step 1: Write `docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md`**
  Populate with exact hardware/env specs from `results/m1_m2/env_metadata.json`, exact metrics from `results/m1_m2/summary_metrics.csv`, figure embeds/links for `fig1`–`fig5`, hypothesis vs. empirical discussion, and honest limitations.
- [ ] **Step 2: Write `docs/handoff/M3_M6_HANDOFF_CONTRACT.md`**
  Include copy-ready public API signatures, step-by-step M3 (vectorized batches & CBO stats), M4 (MVCC `xmin`/`xmax` & ARIES `page_lsn`), and M5 (shard routing & `float32[384]` vector column) recipes, plus the Do-Not-Break invariants list.
- [ ] **Step 3: Final 6-Point Docify & Anti-Cheat Verification**
  1. Verify zero forbidden imports in `src/`.
  2. Verify zero `TODO`/`TBD`/`FIXME` placeholders in `README.md` and `docs/`.
  3. Cross-check every numeric figure in `M1_M2_TECHNICAL_NARRATIVE.md` against `results/m1_m2/summary_metrics.csv`.
  4. Verify ASCII byte-offset diagrams match `struct.Struct` format strings across `src/storage/` and `src/index/`.
  5. Re-run `.venv/bin/pytest -v tests/test_m1_storage.py tests/test_m2_indexing.py` -> 100% PASS.
- [ ] **Step 4: Commit**
  `git add docs/ README.md && git commit -m "docs(wave5): complete M1/M2 technical narrative, M3-M6 handoff contract, and final audit"`
