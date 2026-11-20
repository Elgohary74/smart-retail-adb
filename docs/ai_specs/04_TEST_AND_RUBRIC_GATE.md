# 04 — Adversarial Test Suite & Rubric Gate Specification

## 1. CSAI 302 Rubric Traceability Matrix (30 Marks Total)

Every test in `tests/test_m1_storage.py` and `tests/test_m2_indexing.py` maps directly to a graded rubric criterion in the course brief.

| Milestone | Rubric Criterion | Marks | Mandatory Verification Tests |
| :--- | :--- | :--- | :--- |
| **M1** | Custom persistent storage | 5 | `test_m1_persistence_across_restart`, `test_m1_torn_page_checksum_detection`, `test_m1_fsm_allocation_reuse` |
| **M1** | Buffer pool | 3 | `test_m1_lru_eviction_order`, `test_m1_fifo_eviction_order`, `test_m1_pin_protection_and_pool_full` |
| **M1** | Storage layouts | 3 | `test_m1_slotted_page_crud_and_compaction`, `test_m1_column_page_zero_copy_numpy` |
| **M1** | Testing and measurements | 4 | `test_m1_dirty_writeback_accuracy`, `test_m1_hit_miss_telemetry` + Gate 3 M1 benchmarks |
| **M2** | B+ Tree | 5 | `test_m2_bplus_50k_inserts_and_invariants`, `test_m2_bplus_sibling_range_scan`, `test_m2_bplus_restart_persistence` |
| **M2** | LSM Tree | 4 | `test_m2_lsm_wal_crash_recovery`, `test_m2_lsm_tombstone_compaction`, `test_m2_lsm_bloom_filter_effectiveness` |
| **M2** | Workload-based choice | 2 | `test_m2_workload_advisor_decision_rules` |
| **M2** | Object storage & encryption | 1 | `test_m2_object_store_aes256_gcm_and_tamper_detection` |
| **M2** | Correctness & performance | 3 | `test_m2_sql_executor_and_injection_defense`, `test_m2_index_vs_fullscan_equivalence` + Gate 3 M2 benchmarks |

---

## 2. Gate 1 Specification (`tests/test_m1_storage.py`)

Implement all 10 tests below in `tests/test_m1_storage.py` using pytest `tmp_path` fixtures:

1. **`test_m1_page_header_binary_packing`:**
   * Pack and unpack `PageHeader` with boundary values (`uint32` max, `uint64` max), assert exact 32-byte length, and verify all 8 fields match byte-for-byte.
2. **`test_m1_torn_page_checksum_detection`:**
   * Write a valid page to `DiskManager`, close it, flip 1 byte at offset `4096` directly in the `.db` file on disk, reopen `DiskManager`, and assert `read_page()` raises `PageCorruptedError`.
3. **`test_m1_fsm_allocation_reuse`:**
   * Allocate 10 pages, deallocate pages `3` and `7`, allocate 2 new pages, and assert `DiskManager` reuses `3` and `7` before extending the file.
4. **`test_m1_slotted_page_crud_and_compaction`:**
   * Insert tuples until a page is full (`insert_tuple` returns `None`). Delete alternating even slots (`0, 2, 4, ...`). Verify odd slots remain readable at their original `slot_id`s. Insert a large tuple that only fits after defragmentation, asserting `SlottedPage.compact()` reclaims the fragmented holes without changing existing `slot_id`s.
5. **`test_m1_column_page_zero_copy_numpy`:**
   * Insert `2,500` rows into `ColumnTableHeap` for `order_items` (spanning 3 column pages per column). Fetch the `unit_price` column page, obtain its `np.ndarray` view, verify `arr.flags['OWNDATA'] is False` (true zero-copy view!), and assert `np.sum(qty_arr * price_arr)` matches Python ground-truth math within $10^{-6}$.
6. **`test_m1_lru_eviction_order`:**
   * Initialize `BufferPoolManager(pool_size=3, replacer=LRUReplacer(3))`. Fetch pages `1, 2, 3` and unpin all three. Fetch and unpin page `1` again (making `2` the LRU victim). Fetch page `4` and assert page `2` was evicted while pages `1, 3, 4` remain resident.
7. **`test_m1_fifo_eviction_order`:**
   * Repeat the exact sequence above with `FIFOReplacer(3)`. Assert that re-accessing page `1` does **not** refresh its position, so page `1` (oldest inserted) is evicted when page `4` is fetched.
8. **`test_m1_pin_protection_and_pool_full`:**
   * Initialize `BufferPoolManager(pool_size=2)`. Fetch pages `1` and `2` without unpinning (`pin_count == 1`). Attempt to fetch page `3` and assert `BufferPoolFullError` is raised. Unpin page `1`, retry fetching page `3`, and assert success.
9. **`test_m1_dirty_writeback_accuracy`:**
   * Modify a tuple in buffer pool (`is_dirty=True`), evict that frame by loading other pages, and assert the modified bytes were automatically flushed to disk before eviction.
10. **`test_m1_persistence_across_restart`:**
    * Insert `10,000` `products` and `20,000` `order_items` records, call `storage_engine.close()`, instantiate a brand-new `StorageEngine` pointing to the same directory, and assert 100% of records match the original dataset.

---

## 3. Gate 2 Specification (`tests/test_m2_indexing.py`)

Implement all 11 tests below in `tests/test_m2_indexing.py`:

1. **`test_m2_object_store_aes256_gcm_and_tamper_detection`:**
   * Store 20 binary product images (`10 KB`–`64 KB`) using `AES256GCMHook(key)`.
   * Read raw bytes directly from `objects.dat` on disk and assert that known plaintext substrings (e.g., `b"JFIF"` or `b"IHDR_PRODUCT_SECRET"`) do **not** appear in the raw file.
   * Retrieve all 20 objects via `get_object(oid)` and assert byte-for-byte equality.
   * Corrupt 1 byte inside the ciphertext in `objects.dat` and assert `get_object(oid)` raises `SecurityViolationError` or `PageCorruptedError`.
2. **`test_m2_bplus_50k_inserts_and_invariants`:**
   * Insert `50,000` shuffled `(product_id, RID)` pairs into `BPlusTreeIndex`.
   * Run a structural validator verifying: (a) all leaves sit at identical depth `height`, (b) keys inside every internal and leaf node are strictly sorted, (c) internal separator keys properly bound child subtrees, and (d) all `50,000` keys return their exact `RID`.
3. **`test_m2_bplus_sibling_range_scan`:**
   * Insert `10,000` prices (`float64`, including duplicate prices). Execute `range_scan(100.0, 250.0)` and assert the returned `(price, RID)` list is sorted and matches a ground-truth list comprehension 100%.
4. **`test_m2_bplus_restart_persistence`:**
   * Flush and close `BPlusTreeIndex`, reopen from disk, and verify point lookups and range scans succeed without rebuilding.
5. **`test_m2_lsm_wal_crash_recovery`:**
   * Insert `500` records into `LSMTreeIndex` (below the `1 MB` MemTable flush threshold so data resides only in `MemTable` + `lsm.wal`). Simulate a crash by discarding the in-memory Python object without calling `close()`, reopen `LSMTreeIndex` on the same directory, and assert all `500` records recover from `lsm.wal`.
6. **`test_m2_lsm_tombstone_compaction`:**
   * Insert `5,000` keys across multiple MemTable flushes (creating `L0` SSTables). Update `1,000` keys with new `RID`s and delete `1,000` keys. Trigger compaction into `L1`, and assert: (a) deleted keys return `[]` (never resurrecting older SSTable values), (b) updated keys return only their newest `RID`, and (c) remaining `3,000` keys match ground truth.
7. **`test_m2_lsm_bloom_filter_effectiveness`:**
   * Populate `LSMTreeIndex` with `10,000` even keys (`0, 2, 4, ..., 19998`). Query `5,000` odd keys (`1, 3, 5, ..., 19999`) and assert that Bloom filter rejection rate exceeds `95%` (avoiding disk block reads on non-existent keys).
8. **`test_m2_workload_advisor_decision_rules`:**
   * Feed 100 operations $\rightarrow$ assert `FULL_SCAN`.
   * Feed 500 writes + 100 point reads ($\alpha = 0.83, \beta = 0.0$) $\rightarrow$ assert `LSM_TREE`.
   * Feed 100 writes + 500 range reads ($\alpha = 0.17, \beta = 0.83$) $\rightarrow$ assert `BPLUS_TREE`.
9. **`test_m2_sql_executor_and_injection_defense`:**
   * Execute `CREATE INDEX`, `INSERT`, `SELECT ... WHERE = ?`, `SELECT ... WHERE BETWEEN ? AND ?`, `UPDATE`, and `DELETE` via `SimpleExecutor` with bound parameters.
   * Pass an unparameterized query containing embedded quotes or stacked commands (`"SELECT * FROM products WHERE category = 'Books'; DROP TABLE products;"`) and assert `SecurityViolationError` is raised.
10. **`test_m2_index_vs_fullscan_equivalence`:**
    * Run 200 randomized point and range queries across `BPlusTreeIndex`, `LSMTreeIndex`, and `FullTableScan` on the `tiny` retail dataset and assert identical `RID` result sets across all three access paths.