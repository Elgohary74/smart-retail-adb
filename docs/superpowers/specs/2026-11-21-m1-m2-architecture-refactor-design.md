# Smart Retail DB — M1 & M2 Architecture Refactor Design

**Date:** 2026-11-21
**Status:** Proposed (Recommended Direction Selected)
**Scope:** Milestones 1 (Storage Engine Foundation) and 2 (Advanced Indexing System), plus M1/M2 evaluation harness and foundational readiness for M3–M5 per `docs/CSAI302_Project_doc.pdf`.

---

## 1. Intent & Why

The current M1 and M2 implementation passes its 13 unit tests, but an audit against `docs/CSAI302_Project_doc.pdf` uncovered:
1. **Durability & correctness bugs:** unclean exits corrupt LSM-vs-table state (`IndexError` on reopen), `LSMTree.compact()` unlinks runs before writing the compacted output, re-creating an LSM index silently drops unflushed memtable records, low-cardinality LSM keys never flush during inserts, oversized records (>4 KB) leave orphan pages, `None`/mixed keys crash both indexes, `bool`/`int` collide (`True == 1`), and there is no `UPDATE`/`DELETE`.
2. **Rubric & architectural mismatches:** `BPlusTree` is 100% in-memory (not connected to M1 pages/buffer pool, rebuilt via full scan on startup); `LSMTree` writes raw plaintext JSONL files bypassing `LocalObjectStore` and encryption hooks while keeping all runs in RAM (making LSM ~16.7x slower on writes and ~1.9x slower on reads than `BPlusTree`, contradicting `choose_index_type`); `column` layout is JSON-per-page with full-row dict materialization (no projection or I/O benefit); `SmartRetailDB` creates one isolated `BufferPool` per table; range queries are disconnected from the query/engine path.
3. **Missing benchmark evidence:** `benchmark_storage.py` lacks row-vs-column, lookup/scan, and `page`-vs-`objects` comparisons; `benchmark_indexes.py` lacks indexed-vs-full-scan, read-heavy/write-heavy mixed workloads, range queries, and index size/build-cost metrics; neither follows the 5-run warm-up + median/p95 ms protocol on page 5 of the spec.
4. **M3–M5 blockers:** JSON-in-4KB pages block M3 vectorized execution and M5 fixed-dim vectors (>4 KB overflow), and table-local unpinned `BufferPool`s without `pageLSN` block M4 WAL/ARIES recovery and MVCC.

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Correct, crash-safe M1/M2 storage & indexing** with `INSERT`, `SELECT` (projection + equality/range), `UPDATE`, and `DELETE`.
- **Single unified `BufferPool`** managing `(file_id, page_id)` across all tables and index files, with `FIFO`/`LRU` eviction, `pin`/`unpin`, dirty tracking, and `pageLSN` header reservation for M4.
- **Typed schemas (`INT`, `FLOAT`, `BOOL`, `STR`, `BYTES`, `VECTOR(D)`) + binary encoding** supporting both:
  - **Row layout:** slotted pages with stable `(page_id, slot_id)` RIDs and overflow page chains for >4 KB records/vectors.
  - **Column layout:** dedicated per-column page streams with projection pushdown, late materialization, vectorized column batches, and deleted-row bitmap.
- **Page-backed persistent `BPlusTree`** and **disk-backed SSTable `LSMTree`** (Bloom filters + sparse fence pointers + tombstones + memtable WAL + atomic compaction), both using the unified `PageBackend` (`PageFile` / `LocalObjectStore`) + `TransformingBackend` + `BufferPool`.
- **Upgraded `TransformingBackend`** supporting `page_id`-aware AEAD encryption (e.g., AES-256-GCM with reserved 28-byte per-page overhead inside 4,096-byte physical pages) alongside zero-overhead identity/test transforms.
- **Reproducible M1/M2 benchmark suite** (warm-up + 5 repeats, median/mean/p95 ms, throughput, page I/O, index size) demonstrating real trade-offs (FIFO vs. LRU, row vs. column, page vs. objects, B+ Tree vs. LSM vs. full scan).

### Non-Goals (Deferred to M3–M5)
- Full SQL parser, joins, cost-based optimizer, ORM, or TPC-H execution (M3).
- Full ARIES redo/undo, lock manager, or MVCC snapshot isolation (M4).
- Sharding, Raft consensus, 2PC, or top-k vector similarity search (M5).

### Anti-Goals
- Do not outsource storage, buffering, or indexing to SQLite/DuckDB/RocksDB/etc.
- Do not break the core package's zero-external-runtime-dependency constraint (core `smartdb` stays pure Python >=3.10 stdlib; optional `cryptography` and `matplotlib` are used only when explicitly requested by user/benchmarks).

## 3. Approaches Considered

- **Option A (Selected) — Clean Foundational Refactor for M1–M5:** Replace JSON-in-page storage and in-memory B+Tree / JSONL-in-RAM LSM with a unified `(file_id, page_id)` `BufferPool` (pin/unpin, LSN header), typed binary slotted row pages (+ overflow), true per-column page streams with projection & vectorized batches, page-backed `BPlusTree`, disk-backed SSTable `LSMTree` over `PageBackend`, and `page_id`-aware `TransformingBackend`.
  - *Trade-off:* Higher upfront refactoring effort across M1/M2 modules (~10–14 focused tasks), **but** eliminates every M1/M2 rubric gap and prevents throwing away the storage/index layers in M3, M4, and M5.
- **Option B (Rejected) — Incremental Patches on JSON Pages:** Keep JSON-per-page layouts, in-memory `BPlusTree`, and JSONL `LSMTree`, patching only `compact()`, `len(memtable)`, and benchmark scripts.
  - *Why rejected:* Still leaves `BPlusTree` disconnected from M1 storage, `column` layout without real I/O projection savings, LSM slower than B+ Tree on writes, and forces a full rewrite before M3 vectorized execution, M4 ARIES/MVCC, and M5 vectors.

## 4. Architecture & Component Design

### 4.1 Typed Schema & Value Validation (`src/smartdb/schema.py`)
- `DataType` enum: `INT` (signed 64-bit), `FLOAT` (IEEE-754 float64), `BOOL` (uint8), `STR` (UTF-8), `BYTES` (raw bytes), `VECTOR` (fixed `dim` float32 array), `ANY` (legacy fallback when `create_table` is passed plain `list[str]`, encoded with a 1-byte type tag so `bool` and `int` never collide).
- `ColumnSpec(name: str, dtype: DataType, nullable: bool = True, dim: int | None = None)` and `TableSchema(fields: list[ColumnSpec])`.
- `TableSchema.validate_record(record)` checks exact field names, rejects `bool` for `INT`/`FLOAT`, checks `None` against `nullable`, checks `len(vec) == dim` for `VECTOR`, and validates JSON/binary compatibility before any table or index write occurs.
- `encode_key(dtype, value) -> bytes` / comparable key representation:
  - Encodes keys into order-preserving bytes (or canonical `(is_null, type_rank, value)` tuples where `bool` is distinct from `int`, `None` sorts before non-null values without raising `TypeError`, and floats/ints compare deterministically).

### 4.2 Page Backend, Local Object Store & Encryption (`src/smartdb/storage/`)
- **`PageBackend` Protocol (`page_file.py`)**:
  - Properties: `physical_page_size: int`, `usable_page_size: int`, `page_count: int`.
  - Methods: `read_page(page_id: int) -> bytes`, `write_page(page_id: int, data: bytes) -> None`, `allocate_page() -> int`, `delete_after(page_id: int)` / `close()`.
  - `PageFile`: maintains contiguous `physical_page_size`-byte pages (default `4096`).
- **`LocalObjectStore` (`object_store.py`)**:
  - Stores one `physical_page_size`-byte file `page_{page_id:012d}.obj` per page.
  - Tracks `_page_count` in memory (initialized once in `__init__` from existing files, incremented in `allocate_page()`), removing the $O(N)$ directory `glob()` on every access.
  - Supports deleting an entire object directory when an index or temporary SSTable run is dropped.
- **`TransformingBackend` (`encryption.py`)**:
  - Hook protocol: `encrypt(page_id: int, plaintext: bytes) -> bytes` and `decrypt(page_id: int, ciphertext: bytes) -> bytes`, with `overhead_bytes: int = 0` (must satisfy `0 <= overhead_bytes < backend.physical_page_size`).
  - `usable_page_size = backend.physical_page_size - overhead_bytes`.
  - Enforces `len(plaintext) == usable_page_size` and `len(ciphertext) == backend.physical_page_size`.
  - Note on empty/uninitialized pages: `BufferPool.new_page()` initializes a valid page header in RAM before writing through `TransformingBackend`, so `PageFile`/`LocalObjectStore` never serves unencrypted all-zero plaintext as ciphertext when read back through `decrypt`.
  - Optional helper `make_aes_gcm_hooks(key: bytes)` using `cryptography` (28-byte overhead: 12B random nonce + 16B GCM tag, AAD = `page_id` 8B big-endian) when `cryptography` is available, plus stdlib XOR/CTR helpers for zero-dependency unit tests.
- **`BackendFactory` / `StorageManager` (`storage/manager.py`)**:
  - Centralizes opening/closing page backends (`page` or `objects`, optionally wrapped in `TransformingBackend`), mapping logical names (e.g., `"table:Products:row"`, `"table:Products:col:price"`, `"index:Products:product_id:bplus"`, `"index:Products:category:lsm:run_0"`) to stable integer `file_id`s inside the single `BufferPool`.

### 4.3 Shared Buffer Pool (`src/smartdb/storage/buffer_pool.py`)
- Single DB-wide `BufferPool(capacity: int = 64, policy: str = "LRU", wal_flushed_lsn_fn=None)` managing pages keyed by `(file_id: int, page_id: int)`.
  - Preserves backward-friendly direct single-backend mode (`file_id = 0`) for isolated storage unit tests/benchmarks.
- Frame object `BufferFrame`:
  - `key: tuple[int, int]`, `data: bytearray` (length = `backend.usable_page_size`), `pin_count: int`, `dirty: bool`, `page_lsn: int = 0`, `rec_lsn: int = 0`.
- Operations:
  - `fetch_page(file_id, page_id) -> BufferFrame` (increments `pin_count`; counts `hits`/`misses`/`disk_reads`).
  - `new_page(file_id) -> tuple[int, BufferFrame]` (allocates `page_id` from backend metadata/extent without redundant zero-write + immediate read miss, installs pinned dirty zeroed frame in pool).
  - `unpin_page(file_id, page_id, is_dirty: bool = False, lsn: int | None = None)`.
  - Context manager `pin_page(file_id, page_id, dirty_on_exit: bool = False)` for leak-free unpinning.
  - `flush_file(file_id)`, `flush_all()`, `unregister_file(file_id, remove_frames=True)` (drops frames without writing if an SSTable/index is deleted).
  - Eviction (`FIFO` or `LRU`): scans ordered frame map for the oldest/LRU frame with `pin_count == 0`; if dirty, checks `wal_flushed_lsn_fn` (M4 hook) then writes back via backend and increments `disk_writes` and `evictions`. Raises `RuntimeError("All buffer frames are pinned")` if pool is full of pinned pages.

### 4.4 Common Page Header, Row Slotted Pages, Columnar Pages & Overflow (`src/smartdb/storage/`)
- **Common Page Header (`20 bytes`, `struct "<BBHIQi"`):**
  - `page_type: uint8` (`1=ROW_SLOTTED`, `2=OVERFLOW`, `3=COL_FIXED`, `4=COL_VAR`, `5=BITMAP`, `6=BPLUS_META`, `7=BPLUS_INTERNAL`, `8=BPLUS_LEAF`, `9=LSM_DATA`, `10=LSM_META`)
  - `flags: uint8`
  - `item_count: uint16`
  - `free_offset: uint32` (low-water or slot/data cursor)
  - `lsn: uint64` (reserved `0` until M4 WAL/ARIES)
  - `next_page_id: int32` (`-1` if none)
  Wait: slotted page also needs `free_end: uint16` and `prev_page_id: int32` for B+Tree leaf links or `next_page_id` + `prev_page_id`: using a **24-byte common header** (`"<BBHHIqi"` = `page_type(1) + flags(1) + item_count(2) + free_start(2) + free_end(2) + lsn(8) + next_page(4) + prev_page(4)` = 24 bytes!) gives every page type all fields with zero packing hacks.
- **Row Layout (`RowTable` + `SlottedPage`):**
  - Slot directory starts at byte 24: `slot_id` entry is `(offset: uint16, length: uint16)` (`length == 0` marks a deleted/tombstone slot; `slot_id`s never shift on delete).
  - Records serialized in compact binary (`schema.serialize_row`) placed from `free_end` downwards; free space is compacted within the page on demand when fragmented.
  - Records exceeding `usable_page_size - 28` store a 12-byte inline stub `(MAGIC_OVERFLOW, total_bytes: uint32, first_overflow_page_id: int32)` and chain `OVERFLOW` pages via `next_page_id`. Overflow pages are allocated only **after** verifying serialization succeeds, preventing orphan pages.
  - `Locator` (`RID`) = `(page_id, slot_id)`.
  - `update(rid, new_record)`: overwrites in-place if new binary payload fits in slot's old bytes (or page has free space after intra-page compact); otherwise frees old slot (or writes a forwarding stub? Keeping index update via `SmartRetailDB` returning `(old_rid, new_rid)` is cleaner and avoids multi-hop forwarding chains until M4!). Specifically: `RowTable.update(rid, record) -> tuple[RID, bool]` attempts in-place/same-page update (preserving `rid`!), and only relocates to a new `rid` (deleting old slot) if the same page has insufficient space—returning the effective `new_rid` so `SmartRetailDB` updates indexes if `new_rid != old_rid`!
- **Columnar Layout (`ColumnTable`):**
  - Each table stores:
    1. A `BITMAP` page stream (`file_id` `"deleted"`) tracking 1 bit per `rowid` (`0=live`, `1=deleted`), plus a lightweight metadata header page recording `row_count` and per-column page directory.
    2. One `PageBackend` stream per column (`col_<name>`), packed as binary values with null-bitmap:
       - **Fixed-width columns (`INT`, `FLOAT`, `BOOL`, `VECTOR(D)`):** fixed `K` slots per page (`COL_FIXED`), enabling $O(1)$ random lookup `(page_idx = rowid // K, slot = rowid % K)` and zero-copy/fast `struct.unpack` batch reads! (If `VECTOR(D)` exceeds 1 page per item, e.g. `D > 1000`, uses overflow pages referenced by `(page_id, offset)` in `COL_VAR`!)
       - **Variable-width columns (`STR`, `BYTES`, `ANY`, oversized `VECTOR`):** slotted `COL_VAR` pages storing `(first_rowid, row_count)` in page metadata + a persisted per-column directory (`[(first_rowid, page_id), ...]`) binary-searched in $O(\log P)$ for random `rowid` lookup!
  - **Projection pushdown & vectorized batching:**
    - `scan_batches(fields: list[str] | None, batch_size: int = 1024)` reads **only** the requested columns' page streams + deleted bitmap, yielding `dict[str, list[Any]]` (columnar batches ready for M3 vectorized operators!).
    - `scan(fields=..., predicate=...)` evaluates predicates on predicate column pages first, then fetches only matching `rowid`s from output column pages (**late materialization**).
  - `Locator` = `(0, rowid)` (or `(rowid >> 16, rowid & 0xFFFF)` if keeping 16-bit friendly pairs, wait: `(0, rowid)` works cleanly as `tuple[int, int]`!).
  - `update(rid, record)`: marks old `rowid` deleted in bitmap and appends updated row at new `rowid` (standard columnar append+tombstone update), returning `new_rid` so indexes update.

### 4.5 Persistent Page-Backed B+ Tree (`src/smartdb/indexes/bplus_tree.py`)
- Backed by a `PageBackend` registered in the shared `BufferPool` (`file_id`).
- **Page 0 (`BPLUS_META`):** stores `root_page_id`, `height`, `order` / max keys, `key_dtype`, `distinct_keys`, `total_entries`, `first_leaf_page_id`.
- **Internal pages (`BPLUS_INTERNAL`):** sorted encoded keys + `child_page_id`s (`uint32`).
- **Leaf pages (`BPLUS_LEAF`):** sorted encoded keys + RIDs (`(page_id: uint32, slot_id: uint32)`). Duplicate RIDs for high-frequency keys spill cleanly into linked `OVERFLOW` RID-bucket pages (`bucket_page_id`) when inline RID list exceeds threshold (so a low-cardinality column like `category` with 10,000 duplicate RIDs never overflows a single 4 KB leaf page!).
- **Leaf doubly/singly linked list:** `next_page_id` / `prev_page_id` in the 24-byte page header enables $O(\log_B N + K)$ `range_search`.
- **API:** `insert(key, rid)`, `delete(key, rid=None) -> int`, `search(key) -> list[RID]`, `range_search(low=None, high=None, include_low=True, include_high=True) -> list[RID]`, `page_count`, `disk_bytes()`, `close()`.
- **Startup cost:** Opens existing page 0 in $O(1)$ without scanning the table.

### 4.6 Disk-Backed SSTable LSM Tree (`src/smartdb/indexes/lsm_tree.py`)
- Uses `StorageManager` (`PageFile` or `LocalObjectStore`, wrapped with `TransformingBackend` if encryption is active) + shared `BufferPool` for SSTable run pages!
- **Memtable:**
  - Tracks `total_entries: int` (including duplicates and tombstones!) and flushes when `total_entries >= flush_threshold` (fixing the low-cardinality infinite-RAM bug).
  - Entries stored per canonical key as ordered `list[tuple[RID, bool]]` (`is_deleted=False` for insert, `True` for tombstone) — or latest-op sequence per `(key, rid)` pair.
  - **Memtable Write-Ahead Log (`memtable.wal`):** appends small binary/JSONL frames (or encrypted records if `TransformingBackend` is active, using the configured transform/storage abstraction) on `insert`/`delete`, replayed on reopen so un-flushed memtable entries survive crashes! Truncated safely after each SSTable flush.
- **Immutable SSTable Runs (Level-ordered `run_{id}`):**
  - Stored in `PageBackend` pages (`LSM_DATA` + `LSM_META` pages), written sequentially in sorted key order when memtable flushes.
  - Each run keeps in RAM **only**:
    1. **Bloom filter** (`bytearray` bitset, configurable bits/key e.g. 10 bits/key, 7 hashes via `blake2b`/`hashlib`) — skips reading disk pages on negative lookups!
    2. **Sparse fence-pointer array** `[(first_key, page_id), ...]` + `min_key`, `max_key` — `search(key)` binary-searches fence pointers and reads **at most 1 `LSM_DATA` page** via `BufferPool`!
  - **Read resolution (newest -> oldest: Memtable, then Run $N$, ..., Run $0$):**
    - `search(key)`: aggregates `(rid, op_seq, is_deleted)` for `key` across memtable and runs where `min_key <= key <= max_key` and `bloom.might_contain(key)`, returning only RIDs whose newest state is not deleted!
    - `range_search(low, high, ...)`: binary-searches each run's fence pointers for `low`, iterates pages sequentially until `key > high`, merges across memtable + runs by `(key, rid)` keeping newest state, dropping tombstones!
- **Atomic Compaction:**
  - Triggered when run count exceeds threshold (e.g., `>= 4` or `>= 8` runs, configurable).
  - Writes merged non-tombstone entries into a **new** SSTable run (`run_{next_id}`), atomically updates `manifest.json` via `.tmp` + `os.replace`, unregisters old runs from `BufferPool`, and **only then** deletes old run files/objects.

### 4.7 Index Manager, Query Parser & Engine (`manager.py`, `query.py`, `engine.py`, `cli.py`)
- **`choose_index_type(reads: int, writes: int, range_reads: int = 0) -> str`:**
  - If `reads < 0 or writes < 0 or range_reads < 0`: raise `ValueError`.
  - If `writes == 0`: `"BPLUS"`.
  - If `reads == 0 and range_reads == 0`: `"LSM"`.
  - `return "LSM" if writes > reads * 2 and range_reads == 0 else "BPLUS"` (documented rule: write-heavy point workloads pick LSM; read-heavy or range-heavy workloads pick B+ Tree).
- **Restricted Safe SQL Subset (`query.py`)**:
  - `SELECT (* | col1, col2, ...) FROM table [WHERE field (= | > | >= | < | <=) literal | WHERE field BETWEEN low AND high]`
  - `UPDATE table SET field = literal [, ...] [WHERE ...]`
  - `DELETE FROM table [WHERE ...]`
  - `CREATE TABLE name (col1 TYPE, ...) [LAYOUT ROW|COLUMN]` (also accepts legacy untyped `(col1, col2)` using `DataType.ANY`).
  - `INSERT INTO table VALUES (...)` (fixes single-value `(1)` parsing via regex/tokenizer instead of bare `ast.literal_eval` tuple check).
  - Safe literal parsing: accepts `int`, `float`, `bool`, `str`, `None`, rejects `complex`, `Ellipsis`, containers, and SQL injection patterns (`OR 1=1`, `; DROP ...`).
- **`SmartRetailDB` (`engine.py`)**:
  - Owns one `StorageManager` + one shared `BufferPool` + `IndexManager` + atomic `catalog.json` / `indexes.json` (plus per-table workload read/write counters persisted in catalog).
  - `create_index` clears any prior index artifacts on `(table, field)` before creating/populating, only scans the table when creating a new index (never silently skips dirty memtable or loads stale runs).
  - `execute` pushes projections (`fields`) to `RowTable`/`ColumnTable` and uses `IndexManager.search` or `IndexManager.range_search` whenever a matching index exists, verifying predicate + live status cleanly.
  - `explain` reports exact plan (`INDEX POINT LOOKUP`, `INDEX RANGE SCAN`, `COLUMN SCAN [cols=...]`, `ROW TABLE SCAN`).

### 4.8 Retail Workload, Benchmarks & Repo Hygiene
- **`workloads/generate_retail_data.py`**: Add `OrderItems` (`order_item_id, order_id, product_id, quantity, unit_price`) and `Inventory` (`product_id, warehouse_id, quantity_on_hand, last_updated`) alongside `Products`, `Customers`, `Orders` with typed schemas.
- **`benchmarks/common.py` + `benchmark_storage.py` + `benchmark_indexes.py`**:
  - Warm-up + 5 measured runs by default; records OS, Python version, CPU count, seed, dataset size; outputs CSV + terminal summary with `mean_ms`, `median_ms`, `p95_ms`, `stdev_ms`, `ops_per_sec`, `disk_reads`, `disk_writes`, `hit_rate`, and `index_bytes`; optional `--plots` flag outputs PNG charts using `matplotlib`.
  - M1 benchmarks: FIFO vs. LRU buffer pool trace; Row vs. Column (bulk insert, narrow 2-column projection scan, full-row point lookup); `page` vs. `objects` (and optional AEAD encryption overhead).
  - M2 benchmarks: Indexed point/range lookup (`B+ Tree`, `LSM`) vs. full table scan; Write-Heavy (80W/20R) vs. Read-Heavy (90R/10W) vs. Range workloads comparing `B+ Tree` and `LSM`; index build time & disk footprint.
- **Packaging & Cleanup**:
  - Remove empty stale directories (`src/common`, `src/index`, `src/query`, `src/storage`, `bench`), keep `pyproject.toml` editable install working cleanly, document `pip install -e .` and `PYTHONPATH=src` in `README.md`, and update `docs/architecture.md` and `docs/milestone-checklist.md`.

## 5. Settled Design Decisions (Resolved Pre-Plan)

1. **Crash-Consistency Boundary (Pre-M4):**
   - `RowTable`, `ColumnTable`, and `BPlusTree` remain write-back cached in `BufferPool` (requiring `flush()`/`close()` for full durability; full WAL + ARIES belongs to M4).
   - `SmartRetailDB` records `clean_shutdown: bool` in `catalog.json` (set `False` on open, `True` on `close()`).
   - `RowTable.try_get_record(rid)` / `ColumnTable.try_get_record(rid)` return `None` instead of raising `IndexError` for out-of-bounds/unwritten/tombstoned RIDs. On reopen after `clean_shutdown == False`, `SmartRetailDB` runs a startup reconciliation pass that trims or rebuilds any index whose entries reference un-persisted table pages or lag behind persisted table rows, guaranteeing zero `IndexError` crashes after unclean exits.
2. **Clean API Refactor (No Legacy Shims for Old JSON Format):**
   - Replace `storage/layouts.py` and old `StorageTable` cleanly; update `tests/`, `benchmarks/`, `workloads/`, and `cli.py` to match the new contracts.
   - `BufferPool` supports multi-file `(file_id, page_id)` keys **and** accepts an optional single `backend` positional/keyword argument (`file_id=0` default) so standalone storage tests remain concise.
   - `BPlusTree` accepts `storage_manager` + `file_id` (DB mode) **or** standalone `BPlusTree(order=...)` (auto-provisions a temporary `PageFile` + `BufferPool` with `max_keys_per_node = order`, still using real pages).
   - `range_search(...)` returns `list[tuple[Any, RID]]` (matching existing tests/callers).
   - `TransformingBackend` uses `(page_id: int, data: bytes) -> bytes` with `overhead_bytes: int = 0`.
3. **`LSMTree` `memtable.wal` Encryption & `fsync` Policy:**
   - `memtable.wal` uses length-prefixed framed records (`uint32 len + payload`) with buffered OS writes (`file.flush()` per mutation; `os.fsync()` on SSTable flush/compaction, not per insert) so LSM write benchmarks preserve sequential batch speed.
   - `StorageManager` / `TransformingBackend` optionally accept `encrypt_record(bytes) -> bytes` and `decrypt_record(bytes) -> bytes` (automatically provided alongside `make_aes_gcm_hooks(key)`); when set, `memtable.wal` encrypts every WAL frame.
4. **Ordering, Coercion, and `NULL` Semantics:**
   - `INT` and `FLOAT` share a unified numeric ordering (`1 == 1.0`, `1 < 1.5 < 2`) encoded losslessly via bucket + `math.floor` `int64` + fractional `float64` (never losing 64-bit integer precision). `BOOL` has a separate rank and never equals `INT`.
   - `NaN` and `Inf` are rejected by `TableSchema.validate_record`.
   - `None` (`NULL`) sorts first in index keys (`is_null=0`), **and** `SmartRetailDB` enforces SQL three-valued logic by filtering `None` out of `<`, `<=`, `>`, `>=`, `BETWEEN`, and `=` queries (supporting `IS NULL` / `IS NOT NULL` and `= None` only for explicit null matching).
   - `query.py` recognizes SQL `NULL`, `TRUE`, `FALSE` (case-insensitive) as well as Python literals, and coerces `int` literals when querying `FLOAT` columns.
5. **Unified `SmartRetailDB.execute`, `VECTOR` Literals, and `"auto"` Telemetry:**
   - Unify `CREATE TABLE`, `INSERT`, `CREATE INDEX`, `SELECT`, `UPDATE`, and `DELETE` parsing inside `query.py` and `SmartRetailDB.execute(sql)` (with `cli.py` delegating directly to `execute`). `SELECT` returns `list[dict]`; `INSERT` returns `RID` (via `db.insert`) / `int` count `1` (via `execute`); `UPDATE`/`DELETE` return affected `int` count; `CREATE TABLE`/`CREATE INDEX` return status string. `SmartRetailDB` also exposes `select()`, `insert()`, `update()`, `delete()`.
   - `INSERT ... VALUES` permits numeric list/tuple literals specifically for `VECTOR(D)` columns while rejecting containers for scalar columns.
   - `SmartRetailDB` tracks and persists per-table `(reads, writes, range_reads)` counters in `catalog.json`. Calling `create_index(..., "auto")` without explicit `reads`/`writes` reads these observed counters (falling back to `(100, 10, 0)` only when total observed ops == 0).

## 6. What Is Left to the Builder (Implementation Plan Details)
- Exact `struct` packing byte offsets within each page format (`ROW_SLOTTED`, `COL_FIXED`, `COL_VAR`, `BPLUS_*`, `LSM_*`), as long as they share the 24-byte common page header and round-trip deterministically.
- Exact Bloom filter hash function pairing (e.g., `blake2b` split into two 64-bit hashes for double hashing) and default `LSMTree` compaction run threshold (recommended: 4 or 8 runs).
- Exact matplotlib chart styling in benchmark scripts.
