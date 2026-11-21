# Smart Retail DB (CSAI 302 — Milestones 1 & 2)

A simple, custom educational Database Management System (DBMS) written in pure Python for the **CSAI 302 (Advanced Database Systems)** course project using the Retail Orders domain.

All data storage, buffer management, page layouts, and indexes are implemented from scratch—this project does **not** wrap SQLite, PostgreSQL, MySQL, or another database engine.

---

## Implemented Features

### M1 — Storage Engine Foundation
* **Persistent Fixed-Size Pages (`4,096` bytes):** Contiguous binary page files (`PageFile`) behind a common `PageBackend` protocol.
* **Shared DB-Wide Buffer Pool (`BufferPool`):** One pool per database instance keyed by `(stream_id, page_id)` with configurable `FIFO` or `LRU` eviction, `pin_page`/`unpin_page`, dirty-page write-back, and `BufferStats` (`hits`, `misses`, `hit_rate`, `disk_reads`, `disk_writes`, `evictions`).
* **Row & Column Layout Managers:**
  * `RowLayout`: Slotted JSON pages with `None` tombstones and stable `(page_id, slot_id)` locators.
  * `ColumnLayout`: Separate page stream per column (`<col>.pages` + `_tombstones.pages`) with logical `(0, row_id)` locators, projection-only reads (`scan(columns=...)`), and columnar batch iteration (`iter_batches`).
* **Typed Schemas & CRUD:** Optional column types (`INT`, `FLOAT`, `STR`, `BOOL`, `NOT NULL`) and `insert`, `get_record`, `update`, `delete`, and `scan`.

### M2 — Advanced Indexing
* **B+ Tree (`BPlusTree`):** Leaf sibling links for range scans and duplicate-key locator buckets (rebuilt from storage on open).
* **LSM Tree (`LSMTree`):** In-memory memtable flushing to sorted JSONL runs on disk with compaction.
* **Workload-Based Auto Selection:** `auto` chooses `LSM` when `writes > 2 * max(1, reads)` and `BPLUS` otherwise.
* **Object-Store Backend (`LocalObjectStore`):** One local object file per page behind the `PageBackend` interface.
* **Encryption Hooks (`TransformingBackend`):** Optional paired same-length `encrypt`/`decrypt` hooks (identity pass-through by default; supply an approved cipher externally if needed).
* **Safe Restricted Query Parser:** Supports `SELECT * FROM table [WHERE field = literal]` parsed via `ast.literal_eval` (rejects expressions such as `OR 1=1`).

---

## Setup

**Requirements:** Python `>= 3.10` (no third-party runtime dependencies).

### Option A: Editable install (recommended)

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### Option B: Run directly without installing
Prefix commands with `PYTHONPATH=src` (Linux/macOS) or `$env:PYTHONPATH="src"` (PowerShell), e.g.:
```bash
PYTHONPATH=src python3 -m smartdb.cli --db ./my_retail_db
```

---

## How to Use

### 1. Interactive CLI

Start the shell (using default `LRU` policy, `16` buffer frames, and `page` backend):
```bash
smartdb --db ./my_retail_db --buffer-frames 16 --policy LRU --backend page
```

Example session using typed schemas, columnar layout, indexing, and plan inspection:
```sql
CREATE TABLE Products (product_id INT NOT NULL, name STR NOT NULL, category STR, price FLOAT, stock INT) LAYOUT COLUMN;
INSERT INTO Products VALUES (1, 'Keyboard', 'Accessories', 29.99, 20);
INSERT INTO Products VALUES (2, 'Mouse', 'Accessories', 15.50, 35);
INSERT INTO Products VALUES (3, 'Monitor', 'Electronics', 199.99, 8);

SELECT * FROM Products;
:explain SELECT * FROM Products WHERE product_id = 1

CREATE INDEX ON Products (product_id) USING AUTO READS 100 WRITES 10;
:explain SELECT * FROM Products WHERE product_id = 1
SELECT * FROM Products WHERE product_id = 1;

:stats
:quit
```

* Use `--backend objects` to store pages as individual `.obj` files (`LocalObjectStore`).
* `:explain <SELECT ...>` prints whether the query uses `TABLE SCAN` or `INDEX LOOKUP`.
* `:stats` prints the shared `BufferPool` metrics (`hits`, `misses`, `hit_rate`, `disk_reads`, `disk_writes`, `evictions`).

### 2. Python API Example

```python
from smartdb.engine import SmartRetailDB

with SmartRetailDB("./demo_db", buffer_frames=16, replacement_policy="LRU") as db:
    # Create a columnar table with typed columns
    db.create_table(
        "Products",
        [
            ("product_id", "INT", False),
            ("name", "STR", False),
            ("category", "STR", False),
            ("price", "FLOAT", False),
            ("stock", "INT", False),
        ],
        layout="column",
    )

    loc1 = db.insert("Products", {"product_id": 1, "name": "Keyboard", "category": "Accessories", "price": 29.99, "stock": 20})
    loc2 = db.insert("Products", {"product_id": 2, "name": "Mouse", "category": "Accessories", "price": 15.50, "stock": 35})

    # Update and delete records safely (locators of other rows remain valid)
    db.update("Products", loc2, {"product_id": 2, "name": "Pro Mouse", "category": "Accessories", "price": 24.99, "stock": 30})
    db.delete("Products", loc1)

    # Create an index (auto-selects B+ Tree for read-heavy workload)
    chosen = db.create_index("Products", "product_id", index_type="auto", reads=100, writes=10)
    print("Chosen index:", chosen)

    # Indexed lookup via SQL-like interface
    print(db.execute("SELECT * FROM Products WHERE product_id = 2"))

    # Column projection scan (reads only product_id and price column streams)
    print(db.tables["Products"].scan(columns=["product_id", "price"]))

    # Shared DB-wide buffer pool stats
    print(db.buffer_stats())
```

### 3. Generate the Retail Orders Dataset

Creates all 5 retail tables (`Products`, `Customers`, `Inventory`, `Orders`, `OrderItems`) with typed schemas and consistent order totals:

```bash
python3 workloads/generate_retail_data.py --db ./my_retail_db --products 1000 --customers 200 --orders 500 --seed 42
```

Optional flags:
* `--layout {row,column}` (default: `row`)
* `--backend {page,objects}` (default: `page`)

*(Note: running against an existing `--db` appends new rows when schemas match. Use a fresh directory for a clean dataset.)*

---

## Testing

Run the full `unittest` suite (25 tests):

```bash
# After `pip install -e .`
python3 -m unittest discover -s tests -v

# Or without installing
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Test coverage by file:
* [tests/test_storage.py](tests/test_storage.py) — `PageFile`, `LocalObjectStore`, `TransformingBackend`, `TableSchema`, `BufferPool` (`FIFO` vs `LRU`, multi-stream sharing, pinning, dirty write-back), `RowLayout`, `ColumnLayout` (projection I/O savings), and `StorageTable` reopen persistence.
* [tests/test_indexes.py](tests/test_indexes.py) — `BPlusTree` splits/point/range lookup, `LSMTree` memtable flush/reload, and `auto` index selection.
* [tests/test_engine.py](tests/test_engine.py) — `SmartRetailDB` end-to-end persistence, shared buffer pool, `update`/`delete` with indexes, injection rejection, and CLI schema parsing.
* [tests/test_workload.py](tests/test_workload.py) — 5-table retail dataset generator integrity and incremental append.

---

## Benchmarks

### M1 Storage Benchmark
Runs a warm-up pass followed by 5 measured repeats (`--repeats 5`), reporting `mean_ms`, `median_ms`, `p95_ms`, `stdev_ms`, `ops_per_sec`, `hit_rate`, `disk_reads`, `disk_writes`, and `evictions`:

```bash
PYTHONPATH=src python3 benchmarks/benchmark_storage.py \
  --pages 128 --accesses 10000 --records 2000 --insert-records 500 --capacity 16 --repeats 5 --seed 42
```

Suites measured and saved to `benchmark_results/m1_storage.csv`:
1. **`buffer_policy`:** `FIFO` vs `LRU` on a mixed 80% hot / 20% cold page trace.
2. **`projection_scan`:** `row` vs `column` scanning 2 projected columns (`product_id`, `price`) out of 5.
3. **`point_lookup`:** `row` vs `column` random `get_record()` lookups.
4. **`backend_insert`:** `page_plain` vs `objects_plain` vs `page_xor_hook`.

### M2 Index Benchmark
Compares `BPlusTree` vs `LSMTree` write and point-lookup throughput across repeated runs and saves results to `benchmark_results/m2_indexes.csv`:

```bash
PYTHONPATH=src python3 benchmarks/benchmark_indexes.py --records 10000 --repeats 3 --seed 42
```

---

## Project Structure

```text
src/smartdb/
  engine.py                 SmartRetailDB coordinator, shared BufferPool, catalog, SQL execution
  cli.py                    Interactive shell (CREATE TABLE, INSERT, CREATE INDEX, SELECT, :explain, :stats)
  query.py                  Restricted safe SELECT parser
  storage/
    page_file.py            PageBackend protocol and contiguous PageFile (4,096 B pages)
    object_store.py         LocalObjectStore adapter (one .obj file per page)
    encryption.py           Optional same-length transform hooks (TransformingBackend)
    buffer_pool.py          Shared multi-stream BufferPool (FIFO/LRU, pin/unpin, BufferStats)
    schema.py               Column and TableSchema validation (INT, FLOAT, STR, BOOL, ANY)
    layouts.py              RowLayout and ColumnLayout managers with tombstones and projection
    table.py                StorageTable CRUD, scan, and batch facade
  indexes/
    bplus_tree.py           B+ Tree with leaf links and duplicate-key buckets
    lsm_tree.py             LSM memtable, sorted JSONL runs, and compaction
    manager.py              Auto index selection heuristic and unified IndexManager
tests/                      Unit and integration tests (storage, indexes, engine, workloads)
benchmarks/                 M1 storage and M2 index benchmark scripts
workloads/                  Synthetic 5-table retail dataset generator
docs/                       Architecture documentation, milestone checklist, and course brief
```
