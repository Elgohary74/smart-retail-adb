# Smart Retail DB — Milestones 1 and 2

A custom educational DBMS implemented in Python using the course's retail-orders domain (`Products`, `Customers`, `Orders`, `OrderItems`, `Inventory`). Core storage, buffering, and indexing are implemented from scratch with zero external database engines.

## What's implemented

### M1 — Storage engine foundation
- Persistent 4,096-byte pages (`PageFile` and `LocalObjectStore`) with a 24-byte `PageHeader` reserving `lsn` for M4 WAL/ARIES.
- Single shared database-wide `BufferPool` keyed by `(file_id, page_id)` with `FIFO`/`LRU` eviction, `pin`/`unpin`, dirty write-back, and hit/miss/I/O counters.
- Typed schemas (`INT`, `FLOAT`, `BOOL`, `STR`, `BYTES`, `VECTOR(D)`, `ANY`) with binary `RowTable` (slotted pages + overflow chains) and `ColumnTable` (per-column page streams, projection pushdown, late materialization, vectorized batches).
- Reproducible 5-run M1 benchmarks (FIFO vs. LRU, Row vs. Column projection, `page` vs. `objects` vs. AES-GCM).

### M2 — Advanced indexing
- Persistent page-backed `BPlusTree` (`BPLUS_META`, `BPLUS_INTERNAL`, `BPLUS_LEAF`, overflow RID buckets) supporting equality, range, and delete.
- Disk-backed SSTable `LSMTree` over `StorageManager`/`BufferPool` with Bloom filters, sparse fence pointers, entry-count memtable flush, buffered `memtable.wal`, tombstones, and atomic compaction.
- Workload-based `auto` index selection (`choose_index_type`) using explicit or observed table telemetry.
- `TransformingBackend` supporting `(page_id, data) -> bytes` hooks and `make_aes_gcm_hooks(key)` (28-byte AES-256-GCM per-page envelope).
- Safe SQL subset (`CREATE TABLE`, `INSERT`, `SELECT` projection + `=`/`<`/`<=`/`>`/`>=`/`BETWEEN`/`IS NULL`/`IS NOT NULL`, `UPDATE`, `DELETE`, `CREATE INDEX`).

## Quick start

```bash
cd smart-retail-adb
python3 -m venv .venv
source .venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

*(On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` and `$env:PYTHONPATH="src"`.)*

## Run the CLI

```bash
PYTHONPATH=src python -m smartdb.cli --db ./retail_data
```

Example session:

```sql
CREATE TABLE Products (product_id INT, name STR, category STR, price FLOAT, stock INT) LAYOUT ROW;
INSERT INTO Products VALUES (1, 'Keyboard', 'Accessories', 29.99, 20);
INSERT INTO Products VALUES (2, 'Mouse', 'Accessories', 15.50, 35);
CREATE INDEX ON Products (price) USING BPLUS;
SELECT product_id, name FROM Products WHERE price BETWEEN 10.0 AND 30.0;
:explain SELECT product_id, name FROM Products WHERE price BETWEEN 10.0 AND 30.0
:stats
```

## Generate demo dataset & run benchmarks

```bash
PYTHONPATH=src python workloads/generate_retail_data.py --db ./retail_data --products 10000 --customers 1000 --orders 5000 --seed 42
PYTHONPATH=src python benchmarks/benchmark_storage.py --pages 128 --accesses 10000 --records 2000 --repeats 5
PYTHONPATH=src python benchmarks/benchmark_indexes.py --records 2000 --repeats 5
```

Pass `--plots` to save PNG charts alongside the CSVs in `benchmark_results/` (requires `matplotlib`).
