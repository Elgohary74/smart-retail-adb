# Smart Retail DB — Milestones 1 and 2

A small educational DBMS implemented in Python using the course's retail-orders example. The core data path is custom page storage and custom indexes; it does **not** use SQLite, PostgreSQL, MySQL, or another database engine.

## What's implemented

### M1 — Storage engine foundation
- Persistent fixed-size 4,096-byte pages.
- Configurable FIFO or LRU buffer pool with dirty-page write-back and counters for hits, misses, reads, writes, evictions, and hit rate.
- Row-oriented and column-oriented encodings for records in pages.
- Repeatable benchmark comparing FIFO and LRU with the same deterministic trace.

### M2 — Advanced indexing
- B+ Tree index with leaf links, duplicate-key buckets, point lookup, and range lookup.
- LSM Tree index with an in-memory memtable, sorted JSONL runs on disk, point/range lookup, and compaction.
- `auto` index selection: write-heavy workloads choose LSM; otherwise choose B+ Tree. This is a documented heuristic, not a learned optimizer.
- A local object-store backend (one page per object file) behind the same page interface.
- Optional encryption/decryption transform hooks. They are **hooks only**; the default does not encrypt. Supply paired, same-length transforms from an approved cryptographic implementation if encryption is required.
- Restricted SQL-like `SELECT * FROM table [WHERE field = literal]` with indexed equality lookup when an index exists. The grammar rejects expressions such as `OR 1=1`; it is intentionally not a full SQL parser.

## Quick start (Windows PowerShell)

```powershell
cd smart-retail-db
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
```

No third-party runtime dependencies are needed. The test suite uses Python's standard `unittest` framework.

## Run the CLI

```powershell
smartdb --db .\retail_data
```

Try these commands in the prompt:

```sql
CREATE TABLE Products (product_id, name, category, price, stock) LAYOUT ROW;
INSERT INTO Products VALUES (1, 'Keyboard', 'Accessories', 29.99, 20);
INSERT INTO Products VALUES (2, 'Mouse', 'Accessories', 15.50, 35);
SELECT * FROM Products;
SELECT * FROM Products WHERE product_id = 1;
CREATE INDEX ON Products (product_id) USING AUTO READS 100 WRITES 10;
SELECT * FROM Products WHERE product_id = 1;
```

Then use `:explain SELECT * FROM Products WHERE product_id = 1` to see whether the engine picks a table scan or index lookup. `:stats` prints buffer-pool statistics.

Use `--backend objects` to store pages as separate local object files:

```powershell
smartdb --db .\retail_objects --backend objects
```

The backend is a local adapter to demonstrate the object-storage abstraction; it is not an S3 client.

## Generate a demo dataset

```powershell
python .\workloads\generate_retail_data.py --db .\retail_data --products 10000 --customers 1000 --orders 5000 --seed 42
```

To use object-backed pages, add `--backend objects`. The data is synthetic and generated with a fixed seed. Adjust sizes to your computer. The generator is intended for a first benchmark/demo, not a replacement for the TPC-H subset required in M3.

## Benchmarks

Run each benchmark multiple times and report the median; a single run is not sufficient for a final report.

```powershell
python .\benchmarks\benchmark_storage.py --pages 128 --accesses 10000 --capacity 16 --seed 42
python .\benchmarks\benchmark_indexes.py --records 10000 --seed 42
```

CSV results are written to `benchmark_results/`. Keep your Python version, OS, CPU/RAM, seed, page count, buffer capacity, and command line with your final measurements. Timings vary by machine; do not copy example numbers from another system.

## Project structure

```text
src/smartdb/
  engine.py                 high-level database API
  cli.py                    interactive command line
  query.py                  restricted safe SELECT parser
  storage/page_file.py      persistent fixed-size pages
  storage/buffer_pool.py    FIFO/LRU cache and metrics
  storage/layouts.py        row/column page encoding
  storage/table.py          page-backed records and locators
  storage/object_store.py   local object-store adapter
  storage/encryption.py     optional same-length transform hooks
  indexes/bplus_tree.py     B+ Tree
  indexes/lsm_tree.py       LSM memtable, disk runs, compaction
  indexes/manager.py        auto index choice and common interface
tests/                      storage, index, and integration tests
benchmarks/                 M1 and M2 benchmark scripts
workloads/                  reproducible synthetic data generator
```

