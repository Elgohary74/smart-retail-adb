# Development Journal — `csai302_db` (Milestones 1 & 2)

Chronological engineering log documenting legacy audits, specification conflict resolutions, bugs, bottlenecks, and debugging guardrails across Waves 0–5 (per `docs/docify.md`).

---

### [Wave 0 | Legacy Audit] Replacing 4 KB JSON Prototype with Spec-Compliant 8 KB Binary Engine
* **Symptom / Failing Test:** Pre-refactor audit of `src/smartdb/` revealed fundamental architectural divergence from `docs/ai_specs/00_AGENT_GOVERNANCE.md` through `06_M3_TO_M6_HOOKS.md`.
* **Root Cause Analysis:**
  * `src/smartdb/storage/page_file.py` used `PAGE_SIZE = 4096` with raw file append (`page_count`) rather than `8,192`-byte pages, universal `32`-byte headers, BLAKE2b checksums, or a Page-0 Free Space Map.
  * `src/smartdb/storage/layouts.py` packed rows and columns as UTF-8 JSON (`json.dumps` behind a 4-byte length prefix) instead of binary `SlottedPage` tuples with `16`-byte MVCC headers or zero-copy NumPy `ColumnPage` buffers.
  * `src/smartdb/indexes/lsm_tree.py` persisted text `run_*.jsonl` files without a CRC32 WAL, binary `SSTable` blocks, or Bloom filters; `src/smartdb/storage/encryption.py` only offered identity lambda transforms rather than `AES-256-GCM` extent storage.
  * `pyproject.toml` targeted `smart-retail-db` on Python `>=3.10` with zero dependencies and legacy `benchmarks/` / `workloads/` scripts.
* **Resolution:** Removed the legacy `src/smartdb/` package, legacy `tests/`, `benchmarks/`, `workloads/`, `retail_benchmark_data/`, and `src/smart_retail_db.egg-info/`. Created a clean Python `3.14` `.venv` with `numpy`, `cryptography`, `pytest`, and `matplotlib`, updated `pyproject.toml` (`csai302-db`, `requires-python = ">=3.12"`), and aligned `docs/architecture.md` and `docs/ai_specs/00_AGENT_GOVERNANCE.md`.
* **Lesson / Guardrail Added:** Wave 0 freezes exact binary contracts, standardized exceptions, and all Gate 1 + Gate 2 adversarial tests before any storage or index module is implemented.

### [Wave 0 | Binary Spec Reconciliation] Resolving B+ Tree Page 0 FSM Collision and SSTable Footer Size Conflict
* **Symptom / Failing Test:** Cross-checking `docs/ai_specs/03_INDEXING_AND_QUERY_SPEC.md` against `02_STORAGE_AND_OBJECT_SPEC.md` prior to coding exposed two struct-packing mismatches in `03_INDEXING_AND_QUERY_SPEC.md`.
* **Root Cause Analysis:**
  1. **B+ Tree Page 0 Metadata vs. FSM Offset:** Section 2.1 specified bytes `32..55` (`24` bytes) with format `<IBBHQ` (`16` bytes: `uint32, uint8, uint8, uint16, uint64`), mis-sizing `key_size` (`uint16`, not `uint8`) and leaving no room for the `unique` flag while colliding with `DEFAULT_FSM_OFFSET = 32`.
  2. **SSTable 64-Byte EOF Footer:** Section 3.2 specified a "64-Byte Footer at EOF" with format `<8sQIIQIQQI12s`, which sums to `68` bytes (`8+8+4+4+8+4+8+8+4+12`), and omitted `bloom_num_bits` from the prose field list.
* **Resolution:**
  1. Standardized B+ Tree Page 0 metadata at bytes `32..55` to `<IBHHQ?6x` (`24` bytes: `root_page_id`, `key_dtype`, `key_size`, `height`, `entry_count`, `unique`, 6-byte padding) and set `BPLUS_FSM_OFFSET = 64` so the FSM occupies bytes `64..8191` (`65,024` pages).
  2. Standardized the SSTable EOF footer to `<8sQIIQIQQI8s` (exact `64` bytes: `magic`, `bloom_offset`, `bloom_len`, `num_hashes`, `index_offset`, `index_len`, `entry_count`, `bloom_num_bits`, `crc32`, `8B` padding).
  Both clarifications are documented in `docs/architecture.md`.
* **Lesson / Guardrail Added:** Every `struct.Struct(...).size` in `src/` must match its documented byte span in `docs/architecture.md` and class ASCII diagrams, verified by Gate 1/2 binary packing tests.

### [Wave 0 | src/common/types.py] Python `bool` Subclass Trap in `FLOAT64` Row Encoding
* **Symptom / Failing Test:** Passing `True` or `False` into a `DataType.FLOAT64` column during `Schema.encode_row()` was accepted silently because Python's `bool` subclasses `int` (`isinstance(True, (int, float))` is `True`).
* **Root Cause Analysis:** While `INT32`/`INT64`/`UINT64` checked `isinstance(val, bool)` explicitly, the `FLOAT64` branch omitted the `bool` guard and coerced booleans to `1.0`/`0.0`.
* **Resolution:** Added an explicit `isinstance(val, bool)` rejection before type-checking numeric inputs in `Schema.encode_row()` (`cda45c2`), raising `TypeError` for booleans and strings.
* **Lesson / Guardrail Added:** Added regression assertions in `test_common_key_encoding_total_order` covering `True`, `False`, `"9.99"`, and `np.float64` inputs.

### [Wave 1 | src/storage/disk_manager.py] Page 0 FSM/Metadata Coexistence & Torn-Zero Distinction
* **Symptom / Failing Test:** Page 0 stores both the 32-byte `PageHeader` (plus optional B+ Tree metadata at bytes `32..63` when `fsm_offset = 64`) and the Free Space Map bitmap (`fsm_offset..8191`), creating risk of stale buffer frames overwriting live FSM bits or FSM flushes wiping Page 0 metadata.
* **Root Cause Analysis:**
  1. Calling `write_page(0, buf)` from an older buffer pool copy could overwrite newer `allocate_page()`/`deallocate_page()` bits in `_fsm_bitmap`, whereas writing `_fsm_bitmap` blindly could erase bytes `0..fsm_offset - 1`.
  2. Treating any Page 0 checksum mismatch on reopen as "uninitialized" would mask real Page 0 corruption, and Python `bool` arguments (`True`/`False`) pass `isinstance(..., int)` checks.
* **Resolution:** Hardened `DiskManager` (`3a416e4`) so `_persist_fsm()` reads existing Page 0 bytes `0..fsm_offset - 1` before stamping the checksum, `write_page(0, data)` overlays the authoritative in-memory `_fsm_bitmap` onto `data[fsm_offset:]`, `_init_or_reload_file()` re-initializes only an exact all-zero Page 0 while raising `PageCorruptedError` on torn non-zero headers, and `_validate_page_id()` rejects `bool` values and out-of-range page IDs.
* **Lesson / Guardrail Added:** Verified by `test_m1_torn_page_checksum_detection` and `test_m1_fsm_allocation_reuse` with persistent reopen checks.

### [Wave 2 | src/storage/replacer.py & buffer_pool.py] `FIFOReplacer` Arrival Order vs. Cache Hits & Frame Allocation Atomicity
* **Symptom / Failing Test:** Naive `OrderedDict` FIFO tracking that removes `frame_id` on `pin()` and appends it back on `unpin()` refreshes a frame's queue position on cache hits, violating `test_m1_fifo_eviction_order`; separately, allocating a disk page before reserving a buffer frame risks leaking an orphan FSM bit if all frames are pinned (`BufferPoolFullError`).
* **Root Cause Analysis:**
  1. `FIFOReplacer` must distinguish *temporarily pinned on a cache hit* (keep original arrival position in `_od`) from *permanently deleted from pool* (`delete_page()` clearing history).
  2. In `BufferPoolManager`, failed `read_page()` or `allocate_page()` calls after popping from `_free_list` or evicting a victim require restoring clean frame metadata without dropping resident entries.
* **Resolution:** Implemented `FIFOReplacer` using `_od: OrderedDict[int, None]` plus `_pinned: set[int]` (`_pinned <= _od.keys()`), formalized `ReplacementPolicy.remove(frame_id)` on the ABC (`2046315`), reserved/flushed the replacement frame in `new_page()` *before* calling `disk_manager.allocate_page()`, and reset `pin_count = 0`/`is_dirty = False` inside `_restore_unused_frame()`.
* **Lesson / Guardrail Added:** `test_m1_fifo_eviction_order`, `test_m1_pin_protection_and_pool_full`, `test_m1_dirty_writeback_accuracy`, and `test_m1_hit_miss_telemetry` all pass (`5/5`).

### [Wave 2 | src/storage/table_heap.py & storage_engine.py] Row Update Relocation Safety, Cross-Column Alignment, and Standalone Restart Persistence
* **Symptom / Failing Test:** Gate 1 `test_m1_column_page_zero_copy_numpy` and `test_m1_persistence_across_restart` require coordinated multi-page row/column heaps, safe generator unpinning, and full restart recovery from `StorageEngine(db_dir)` without `CatalogManager`.
* **Root Cause Analysis & Design Subtleties:**
  1. **Row Update Relocation Safety:** In `RowTableHeap.update()`, deleting an overflowing slot prior to inserting the enlarged tuple could lose the original row if `BufferPoolManager.new_page()` raised `BufferPoolFullError`, whereas holding the old page pinned during relocation would fail on 1-frame buffer pools.
  2. **Cross-Column Lockstep Alignment:** In `ColumnTableHeap`, partial `new_page()` failure on column $k$ or early `VARCHAR` page overflow could desynchronize `(page_id, slot_id)` across `.col` files, and `iter_column_pages()` / `iter_batches()` generators could leak pinned frames if callers `break` early.
  3. **Standalone `StorageEngine` Registry:** `test_m1_persistence_across_restart` opens `StorageEngine` directly without `CatalogManager`, requiring the engine itself to remember created table schemas and layouts (`ROW` vs. `COLUMN`).
* **Resolution:**
  1. `RowTableHeap.update()` validates size limits upfront, unpins the full page when `SlottedPage.update_tuple()` returns `False`, inserts the relocated tuple via `_insert_payload()` first, and tombstones the old slot only after the new `RID` succeeds (maintaining `<= 1` pinned frame).
  2. `ColumnTableHeap` caps `chunk_rows` to the tightest fixed-width capacity (`1020` for 8-byte scalars), computes minimum cross-column fit before writing `insert()`/`insert_batch()`, rolls back partially allocated column pages via `bpm.delete_page(pid)` on error, and wraps `iter_column_pages()` / `iter_batches()` yields in `try/finally` unpin blocks.
  3. `StorageEngine` persists `storage_meta.json` atomically (`storage_meta.json.tmp -> os.fsync -> os.replace`) on `create_table()`, `flush_all()`, and `close()`, enabling automatic reload of `RowTableHeap`, `ColumnTableHeap`, and `ObjectStore`.
* **Lesson / Guardrail Added:** Gate 1 (`pytest -v tests/test_m1_storage.py`) passed `12/12` in `0.55s`, and an edge-case verification confirmed zero pinned frames on early generator breaks, cross-page update relocation, and `200,000`-row `order_items` `insert_batch` (`1.141s`) + vectorized `iter_batches` sum (`0.0130s`).

### [Wave 3 | src/index/bplus_tree.py & lsm_tree.py] Composite Key Suffixes, Bloom Key Scope, and Tombstone Resurrection Guards
* **Symptom / Failing Test:** Secondary retail indexes (`products.price`, `orders.order_date`, `order_items.order_id`) allow duplicate keys, and LSM `SSTable`s must simultaneously reject non-existent keys via `BloomFilter` (`>95%`) and avoid resurrecting deleted keys across compacted levels.
* **Root Cause Analysis:**
  1. Hashing composite `(user_key || >II(rid))` into `SSTable.bloom` would make `bloom.might_contain(encode_key(k))` miss existing keys because the lookup key lacks the `RID` suffix.
  2. In `LSMTreeIndex`, dropping `is_tombstone=1` entries prematurely during `L0 -> L1` compaction when older `SSTable`s still live in `L1+` or deeper levels would resurrect stale values on subsequent reads.
  3. In `BPlusTreeIndex`, holding parent/sibling pages pinned during splits or range scans risks `BufferPoolFullError` on constrained pools (`buffer_pool_size=3`).
* **Resolution:** Standardized `BloomFilter` to hash *only* the fixed-width `user_key` prefix (`f1d937b`, `04f4c9b`), preserved tombstones during compaction whenever any deeper/older SSTable exists, and enforced 2-frame latch-crabbing / sibling unpin discipline in `BPlusTreeIndex`.
* **Lesson / Guardrail Added:** Verified by `test_m2_bplus_50k_inserts_and_invariants`, `test_m2_bplus_sibling_range_scan`, `test_m2_lsm_wal_crash_recovery`, `test_m2_lsm_tombstone_compaction`, and `test_m2_lsm_bloom_filter_effectiveness`.

### [Wave 4 | src/query/catalog.py & simple_executor.py] Atomic DML Rollback, FD Cleanup, and Index Path Containment
* **Symptom / Failing Test:** Hardening review of `CatalogManager` and `SimpleExecutor` identified three edge-case failure modes: partial multi-row `UPDATE`/`DELETE` leaving heap and indexes out of sync if a downstream `heap.update()` or `idx.insert()` failed, OS file-descriptor double-close risk in `CatalogManager.save()`, and global index name collisions or relative-path escapes on reopen.
* **Root Cause Analysis:**
  1. Mutating `RowTableHeap` and multiple active indexes row-by-row during `UPDATE`/`DELETE` lacked rollback compensation if a later row raised `DuplicateKeyError`, `RecordTooLargeError`, or `StorageError`.
  2. Calling `os.close(fd)` inside both the `try` block and `finally` block on `OSError` between `close` and `os.replace` could double-close a recycled FD (`7438506`).
  3. `SimpleExecutor._resolve_index_path()` and `CatalogManager` needed strict containment inside `indexes_dir` plus cross-table global uniqueness enforcement on `index_name` (`c06e030`).
* **Resolution:** Added per-row rollback snapshots across `UPDATE` (including cross-page relocated `RID`s) and `DELETE` (`a057587`, `7438506`), guarded `fd` ownership (`fd_to_close = fd; fd = None; os.close(fd_to_close)`), and enforced `resolved.is_relative_to(indexes_root)` alongside global index-name uniqueness (`c06e030`).
* **Lesson / Guardrail Added:** All DML mutations preserve atomic consistency between `TableHeap` and all registered `BPlusTreeIndex`/`LSMTreeIndex` structures on failure.

### [Wave 4 | bench/retail_loader.py] `runpy` Eager-Import Warning and Bulk Catalog Stat Batching
* **Symptom / Failing Test:** Initial `test_m2_index_vs_fullscan_equivalence` failed (`RED`) with `ModuleNotFoundError: No module named 'bench'`. After implementing `bench/__init__.py` and `bench/retail_loader.py`, invoking `.venv/bin/python -m bench.retail_loader` emitted a `RuntimeWarning` from `runpy` (`'bench.retail_loader' found in sys.modules after import of package 'bench'`), and calling `CatalogManager.update_table_stats()` per inserted row would trigger thousands of `copy.deepcopy(column_stats)` passes.
* **Root Cause Analysis:** Eagerly importing `bench.retail_loader` inside `bench/__init__.py` pre-populated `sys.modules` before `runpy` executed `__main__`; separately, `CatalogManager.update_table_stats()` performs defensive `copy.deepcopy` per call for rollback safety.
* **Resolution:** Converted `bench/__init__.py` to PEP 562 lazy `__getattr__` re-exports (eliminating the `RuntimeWarning` while preserving `from bench import ...`), computed exact per-column min/max rows in one pass (`_compute_min_max_rows`) to update `CatalogManager` in $O(1)$ calls per table before a single `catalog.save()`, and used bounded power-law probabilities (`ranks ** -0.9`) via `np.random.default_rng(seed).choice`.
* **Lesson / Guardrail Added:** Full Gate 2 (`pytest -v tests/test_m1_storage.py tests/test_m2_indexing.py`) passed `23/23` in `3.31s`, and `python -m bench.retail_loader --scale tiny --seed 42` completed cleanly in `0.255s` with zero warnings.
