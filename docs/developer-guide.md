# Developer Onboarding & Codebase Guide

Welcome to Smart Retail DB! This guide explains how the codebase is organized, **exactly where to start reading**, the invariants you must preserve when making changes, and where upcoming milestones (M3–M5) plug in.

---

## 1. Where to Start (Recommended Reading Order)

If you are opening the codebase for the first time, read the modules **bottom-up** in this order (~20 minutes total):

1. **Data Types & Binary Records** — [src/smartdb/core/schema.py](../src/smartdb/core/schema.py)
   - Understand `DataType`, `ColumnSpec`, `TableSchema`, and `RID = tuple[int, int]`.
   - Skim [key_codec.py](../src/smartdb/core/key_codec.py) (order-preserving index keys) and [row_codec.py](../src/smartdb/core/row_codec.py) (binary row encoding with null bitmaps).
2. **Physical Pages & Buffer Pool** — [src/smartdb/storage/backends/page_file.py](../src/smartdb/storage/backends/page_file.py) & [src/smartdb/storage/buffer_pool.py](../src/smartdb/storage/buffer_pool.py)
   - See the `PageBackend` protocol implemented by `PageFile`, `LocalObjectStore`, and `TransformingBackend`.
   - Read `BufferPool.fetch_page`, `new_page`, `pin_page`, and `_evict_if_needed` to see how `(file_id, page_id)` frames work.
3. **Table Storage Engines** — [src/smartdb/storage/tables/row.py](../src/smartdb/storage/tables/row.py) & [src/smartdb/storage/tables/column.py](../src/smartdb/storage/tables/column.py)
   - Compare `RowTable` (slotted pages + overflow chains in `storage/pages/`) with `ColumnTable` (per-column streams + deleted bitmap + `scan_batches`).
4. **Indexes** — [src/smartdb/indexes/manager.py](../src/smartdb/indexes/manager.py), [bplus/tree.py](../src/smartdb/indexes/bplus/tree.py), & [lsm/tree.py](../src/smartdb/indexes/lsm/tree.py)
   - Notice both `BPlusTree` and `LSMTree` store their pages through `StorageManager` + `BufferPool` — no external files or in-memory-only indexes.
5. **Query Parser & Engine Coordinator** — [src/smartdb/query/parser.py](../src/smartdb/query/parser.py) & [src/smartdb/engine/db.py](../src/smartdb/engine/db.py)
   - Trace `SmartRetailDB.insert()`, `select()`, `update()`, `delete()`, `create_index()`, and `execute()`.
6. **Tests & Benchmarks** — [tests/test_engine.py](../tests/test_engine.py) & [benchmarks/benchmark_indexes.py](../benchmarks/benchmark_indexes.py)
   - `tests/test_engine.py` is the best executable spec of end-to-end behavior, including crash reconciliation and security checks.

---

## 2. Package Map & Ownership

```text
src/smartdb/
├── cli.py                -> REPL entry point (`python -m smartdb.cli`)
├── core/                 -> Constants, .env settings, schemas, key & row serialization
│   ├── config.py            Public: PAGE_SIZE, defaults, Settings, load_settings
│   ├── schema.py            Public: DataType, ColumnSpec, TableSchema, RID
│   ├── key_codec.py         Internal/Core: encode_key, decode_key
│   └── row_codec.py         Internal/Core: serialize_row, deserialize_row
├── storage/              -> M1 + M2 storage engine
│   ├── buffer_pool.py       Shared (file_id, page_id) frame cache (LRU/FIFO)
│   ├── backends/            PageFile, LocalObjectStore, TransformingBackend, StorageManager
│   ├── pages/               24-byte PageHeader, SlottedPage, OVERFLOW chains
│   └── tables/              RowTable, ColumnTable, column_codec (ColumnStream)
├── indexes/              -> M2 indexing layer
│   ├── manager.py           IndexManager, WorkloadProfile & choose_index_type
│   ├── bplus/               Page-backed BPlusTree, node codecs, RID overflow buckets
│   └── lsm/                 LSMTree, BloomFilter, SSTable writer/reader, LSMWal
├── query/                -> SQL parsing & predicate evaluation (M2 -> M3)
│   ├── parser.py            parse_sql, parse_select, AST statement dataclasses
│   └── predicates.py        matches_predicate (SQL three-valued NULL logic)
└── engine/               -> Database coordinator (M1-M6)
    ├── db.py                SmartRetailDB
    └── catalog.py           Atomic catalog.json & indexes.json persistence
```

---

## 3. Critical Invariants (Do Not Break These)

1. **Always unpin every fetched page:** Use `with pool.pin_page(pid, file_id=fid, dirty=...) as frame:` instead of manual `fetch_page`/`unpin_page` pairs whenever possible. Leaking a single pin eventually causes `RuntimeError("All buffer frames are pinned")`.
2. **Validate before allocating or writing:** Call `schema.validate_record()` / `schema.validate_value()` before calling `pool.new_page()` or modifying indexes, so invalid input never leaves orphan pages or out-of-sync indexes.
3. **Never confuse `bool` and `int`:** In Python, `isinstance(True, int)` is `True` and `1 == True`. Every type check and comparison must check `isinstance(x, bool)` explicitly.
4. **Respect SQL three-valued `NULL` logic:** `NULL` (`None`) sorts first inside indexes (`b"\x00"`), so range queries (`<`, `<=`, `>`, `>=`, `BETWEEN`) must keep filtering out `None` rows.
5. **Respect `usable_page_size`:** Never hardcode `4096` inside page layouts or codecs — encryption hooks (`TransformingBackend`) reserve `overhead_bytes` (e.g., `28` bytes for AES-GCM), reducing `usable_page_size` to `4068`.
6. **Atomic metadata writes:** Always write JSON manifests/catalogs to a `.tmp` file and swap with `os.replace(tmp, target)`.

---

## 4. Common Developer Workflows

### Run Tests
```bash
# Full suite (39 tests)
PYTHONPATH=src python -m unittest discover -s tests -v

# Single test module while iterating
PYTHONPATH=src python -m unittest tests.test_bplus_tree -v
```

### Debug Query Plans & Buffer Activity
```bash
PYTHONPATH=src python -m smartdb.cli --db /tmp/dev_db
```
Inside the prompt, prefix any `SELECT` with `:explain ` to inspect whether the engine picks `INDEX POINT LOOKUP`, `INDEX RANGE SCAN`, `ROW TABLE SCAN`, or `COLUMN TABLE SCAN`, and run `:stats` to inspect buffer hits, misses, evictions, and disk I/O.

### Run Benchmarks
```bash
PYTHONPATH=src python benchmarks/benchmark_storage.py --repeats 5
PYTHONPATH=src python benchmarks/benchmark_indexes.py --repeats 5
```

---

## 5. Where to Add Future Milestones (M3–M5)

- **M3 (Query Processing Engine):**
  - Add vectorized filter/project/aggregate operators consuming `ColumnTable.scan_batches()`, join algorithms (`HashJoin`, `SortMergeJoin`, `NestedLoopJoin`), and a cost-based optimizer under `src/smartdb/query/` (reading `catalog.table_stats` and `IndexManager`).
- **M4 (Transactions & Recovery):**
  - Add `TransactionManager`, `LockManager` (wait-for graph deadlock detection), and ARIES `WALManager` under `src/smartdb/engine/` or `src/smartdb/tx/`. Pass `wal_flushed_lsn_fn` into `BufferPool` and populate `PageHeader.lsn` / `BufferFrame.page_lsn` — the buffer pool guard is already built.
- **M5 (Distributed Layer & Exact Top-K Vector Search):**
  - `DataType.VECTOR(D)` and `OVERFLOW` page chains already store fixed-dimensional `float32` vectors. Add exact cosine/Euclidean top-$k$ search over `ColumnTable`/`RowTable` batches and add sharding/Raft/2PC coordination around `SmartRetailDB` nodes.
