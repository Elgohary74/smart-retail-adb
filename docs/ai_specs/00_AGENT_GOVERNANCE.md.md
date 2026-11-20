# 00 — Agent Governance & Execution Protocol (CSAI 302: M1 + M2)

## 1. Mission Statement & System Identity
You are building a **custom disk-oriented relational and object Database Management System (DBMS)** from scratch in Python 3.12+ for the **Retail Orders and Product Search** domain. Every storage page, buffer frame, B+ Tree node, LSM Tree SSTable, and encrypted object extent must be implemented directly by this codebase.

---

## 2. Hard Anti-Cheat & Library Governance Rules

### 2.1 Strictly Forbidden Shortcuts (Automatic Failure)
The AI harness **MUST NOT** use any of the following shortcuts anywhere in `src/`:
1. **No Embedded or External Database Engines:** Never import `sqlite3`, `psycopg2`, `pymongo`, `mysql`, `rocksdb`, `leveldb`, `lmdb`, `duckdb`, `whoosh`, or `faiss`.
2. **No Python Object Serialization for Database Storage:** Never use `pickle`, `dill`, `marshal`, or `shelve` to persist pages, tuples, B+ Tree nodes, LSM SSTables, or object store extents. All persistent structures must use deterministic binary packing via `struct`, `memoryview`, and `bytearray`.
3. **No OS-File-Per-Record/Object Shortcuts:** Never store individual rows or individual product images as separate files on the OS filesystem. Rows live inside 8,192-byte heap files (`*.db`), columns live inside 8,192-byte column files (`*.col`), and objects live inside extent files (`objects.dat` + `objects.dir`).
4. **No Test Tampering:** Never modify, weaken, skip, or delete assertions in `tests/` to force a failing test suite to pass. Fix the underlying engine implementation instead.

### 2.2 Approved Technology Stack
* **Core Python Standard Library:** `struct`, `memoryview`, `os`, `io`, `threading`, `collections` (`OrderedDict`), `heapq`, `bisect`, `hashlib`, `zlib`, `json`, `dataclasses`, `enum`, `typing`, `time`, `signal`, `platform`, `math`, `random`.
* **Numerical & Vectorized Arrays:** `numpy` (exclusively for zero-copy views over `ColumnPage` buffers and vectorized metric math).
* **Cryptography:** `cryptography` (`cryptography.hazmat.primitives.ciphers.aead.AESGCM` for the Milestone 2 encryption hook).
* **Testing & Visualization:** `pytest`, `matplotlib`.

---

## 3. Code Quality, Typing & Concurrency Standards

### 3.1 Python Engineering Standards
* Target **Python 3.12+**.
* All public classes, methods, and functions in `src/` must include complete type annotations (`typing` / built-in generics) and concise docstrings explaining binary formats or algorithmic invariants.
* All file I/O must use explicit binary mode (`os.open` with `os.O_RDWR | os.O_CREAT` or `open(..., "r+b")`), positioned reads/writes (`os.pread` / `os.pwrite` where supported, with thread-safe `seek`+`read`/`write` fallback for Windows), and explicit `os.fsync(fd)` on flush/commit boundaries.

### 3.2 Thread-Safety & Lock Ordering Hierarchy
Even though multi-transaction MVCC arrives in Milestone 4, `BufferPoolManager`, `DiskManager`, `ObjectStore`, and index structures must be thread-safe in M1 + M2.
* Synchronize shared mutable structures with `threading.RLock`.
* To eliminate deadlocks, threads acquiring multiple locks **MUST** follow this strict top-down acquisition hierarchy:
  $$\text{CatalogLock} \rightarrow \text{IndexLock} \rightarrow \text{TableHeapLock} \rightarrow \text{BufferPoolLock} \rightarrow \text{DiskManagerLock}$$
* Never acquire a higher-level lock while holding a lower-level lock.

### 3.3 Standardized Exception Hierarchy (`src/common/exceptions.py`)
All engine errors must inherit from `DatabaseError`:
* `DatabaseError(Exception)`: Base exception for all DBMS failures.
* `StorageError(DatabaseError)`: Disk I/O, file boundary, or allocation failures.
* `PageCorruptedError(StorageError)`: Checksum mismatch or invalid magic/page type on page load.
* `BufferPoolFullError(StorageError)`: Cache miss occurs when 100% of frames have `pin_count > 0`.
* `RecordTooLargeError(StorageError)`: Serialized tuple exceeds maximum single-page payload capacity.
* `IndexErrorBase(DatabaseError)`: Base error for B+ Tree and LSM Tree operations.
* `DuplicateKeyError(IndexErrorBase)`: Unique primary key violation on insert.
* `KeyNotFoundError(IndexErrorBase)`: Lookup or delete on a non-existent key when strict mode is active.
* `QuerySyntaxError(DatabaseError)`: Invalid SQL-like command syntax.
* `SecurityViolationError(DatabaseError)`: Unparameterized raw SQL injection attempt or invalid authentication tag during decryption.

---

## 4. Repository Layout Contract

Generate and maintain this exact directory layout:

```text
csai302_db/
├── docs/
│   └── ai_specs/                  # These 7 specification files
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py              # Global constants (PAGE_SIZE=8192, magic bytes, thresholds)
│   │   ├── exceptions.py          # Standardized DatabaseError hierarchy
│   │   └── types.py               # DataType, RID, ColumnDef, Schema, Value encoding
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── disk_manager.py        # Aligned 8KB page I/O, FSM allocation, fsync
│   │   ├── page_header.py         # Universal 32-byte header + CRC64 checksum
│   │   ├── slotted_page.py        # Row-oriented SlottedPage + defragmentation
│   │   ├── column_page.py         # Column-oriented ColumnPage + zero-copy NumPy views
│   │   ├── replacer.py            # ReplacementPolicy ABC, LRUReplacer, FIFOReplacer
│   │   ├── buffer_pool.py         # BufferPoolManager (pin/unpin, dirty flush, metrics)
│   │   ├── table_heap.py          # RowTableHeap & ColumnTableHeap
│   │   ├── cipher_hooks.py        # CipherHook ABC, NoOpCipher, AES256GCMHook
│   │   ├── object_store.py        # Extent-based binary ObjectStore (objects.dat + objects.dir)
│   │   └── storage_engine.py      # Unified M1+M2 StorageEngine facade
│   ├── index/
│   │   ├── __init__.py
│   │   ├── base_index.py          # BaseIndex ABC (insert, get, range_scan, delete, flush)
│   │   ├── bplus_tree.py          # Buffer-pool-backed B+ Tree (internal/leaf nodes, splits)
│   │   ├── lsm_tree.py            # WAL, SkipList MemTable, SSTable, BloomFilter, Compaction
│   │   └── index_advisor.py       # WorkloadAdvisor sliding window + (alpha, beta) selector
│   └── query/
│       ├── __init__.py
│       ├── catalog.py             # Persistent catalog.json manager
│       └── simple_executor.py     # Parameterized SQL-like parser & query router
├── tests/
│   ├── test_m1_storage.py         # Gate 1 adversarial test suite
│   └── test_m2_indexing.py        # Gate 2 adversarial test suite
└── bench/
    ├── retail_loader.py           # Deterministic dataset generator (seed=42)
    └── run_m1_m2_benchmarks.py    # 1-warmup + 5-run benchmark & chart runner
```

---

## 5. Six-Wave Execution & Gate Protocol

Execute implementation strictly in waves. After each wave, run the verification command and resolve any failures before advancing.

1. **Wave 0 — Contract Freeze & Test Skeletons:**
   * Implement `src/common/config.py`, `src/common/exceptions.py`, `src/common/types.py`, abstract interfaces, and the complete test suites `tests/test_m1_storage.py` and `tests/test_m2_indexing.py` per `04_TEST_AND_RUBRIC_GATE.md`.
2. **Wave 1 — Physical Storage & Object Layer:**
   * Implement `disk_manager.py`, `page_header.py`, `slotted_page.py`, `column_page.py`, `cipher_hooks.py`, `object_store.py`, and `storage_engine.py`.
3. **Wave 2 — Buffer Pool & Table Heaps (Gate 1):**
   * Implement `replacer.py`, `buffer_pool.py`, and `table_heap.py`.
   * **Gate 1 Command:** `pytest -v tests/test_m1_storage.py` (100% pass required).
4. **Wave 3 — Parallel Index Engines (Tracks 3A & 3B):**
   * **Track 3A:** Implement `src/index/bplus_tree.py`.
   * **Track 3B:** Implement `src/index/lsm_tree.py`.
5. **Wave 4 — Advisor, Query Shell & Data Loader (Gate 2):**
   * Implement `index_advisor.py`, `catalog.py`, `simple_executor.py`, and `bench/retail_loader.py`.
   * **Gate 2 Command:** `pytest -v tests/test_m1_storage.py tests/test_m2_indexing.py` (100% pass required).
6. **Wave 5 — Reproducible Benchmarks & Plots (Gate 3):**
   * Implement `bench/run_m1_m2_benchmarks.py` and run full benchmark suite.
   * **Gate 3 Verification:** Confirm `results/m1_m2/raw_runs.csv`, `results/m1_m2/summary_metrics.csv`, and all 5 PNG plots exist and contain valid numeric data.