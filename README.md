# Smart Retail DB

A custom educational relational and vector-capable database management system implemented from scratch in Python (CSAI 302 — Advanced Database Systems). It models a **Retail Orders** domain (`Products`, `Customers`, `Orders`, `OrderItems`, `Inventory`) without wrapping SQLite, PostgreSQL, MySQL, DuckDB, or any external storage/index engine.

---

## Implemented Milestones

### Milestone 1 — Storage Engine Foundation
- **Persistent Page Backends:** Fixed `4,096`-byte physical pages via contiguous files ([page_file.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/storage/backends/page_file.py)) or one-object-per-page directories ([object_store.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/storage/backends/object_store.py)) behind a shared `PageBackend` protocol and `StorageManager`.
- **Shared Database-Wide Buffer Pool:** Single `BufferPool` ([buffer_pool.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/storage/buffer_pool.py)) caching `(file_id, page_id)` frames across all tables and indexes with configurable `LRU`/`FIFO` eviction, `pin_page`/`unpin_page`, dirty write-back, WAL-LSN flush guard hook, and hit/miss/I/O counters.
- **Typed Schemas & Binary Layouts:** [schema.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/core/schema.py) supports `INT`, `FLOAT`, `BOOL`, `STR`, `BYTES`, `VECTOR(D)`, and `ANY` with strict `bool` vs. `int` separation and lossless order-preserving key encoding.
  - **`RowTable`** ([row.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/storage/tables/row.py)): Binary slotted pages (`(page_id, slot_id)` RIDs) with stable slots, in-place/relocating updates, tombstones, and linked `OVERFLOW` pages for records >4 KB (including large embeddings).
  - **`ColumnTable`** ([column.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/storage/tables/column.py)): Dedicated per-column page streams + deleted-row bitmap supporting column projection pushdown, late materialization, and vectorized `scan_batches`.

### Milestone 2 — Advanced Indexing System
- **Page-Backed B+ Tree** ([tree.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/indexes/bplus/tree.py)): Persistent `BPLUS_META`, `BPLUS_INTERNAL`, and doubly-linked `BPLUS_LEAF` pages inside `StorageManager` + `BufferPool`, supporting equality, inclusive/exclusive range scans, deletes, and overflow RID buckets for high-duplicate keys.
- **Disk-Backed SSTable LSM Tree** ([tree.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/indexes/lsm/tree.py)): Entry-count memtable + framed `memtable.wal`, immutable `LSM_DATA`/`LSM_META` SSTable runs stored in `StorageManager` + `BufferPool`, `blake2b` Bloom filters, sparse fence pointers, tombstone sequence resolution, and atomic manifest compaction.
- **Workload-Driven Index Selection** ([manager.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/indexes/manager.py)): `choose_index_type(reads, writes, range_reads)` uses observed table telemetry (or explicit counts) to pick `LSM` for write-heavy point workloads and `BPLUS` for read-heavy or range workloads.
- **Page-Aware Encryption Hooks** ([encryption.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/storage/backends/encryption.py)): `TransformingBackend` wraps any backend with `(page_id, bytes) -> bytes` hooks and configurable `overhead_bytes`, plus `make_aes_gcm_hooks(key)` for authenticated AES-256-GCM encryption on pages and WAL records.
- **Safe Restricted SQL/DML Engine** ([parser.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/query/parser.py), [db.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/src/smartdb/engine/db.py)): Parses literals via `ast.literal_eval` (rejecting SQL injection, expressions, and `NaN`/`Inf`) and enforces SQL three-valued `NULL` logic.

---

## Prerequisites & Setup

- **Python:** `>= 3.10`
- **Core Runtime Dependencies:** None (pure Python standard library).
- **Optional Extras:** `cryptography` (for `make_aes_gcm_hooks`) and `matplotlib` (for `--plots` benchmark charts).

### Linux / macOS

```bash
cd smart-retail-adb
python3 -m venv .venv
source .venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Windows (PowerShell)

```powershell
cd smart-retail-adb
py -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

*(If `setuptools` is installed in your environment, `pip install -e .` also registers the `smartdb` console entry point.)*

---

## Interactive CLI & Supported SQL

Start the interactive shell:

```bash
PYTHONPATH=src python -m smartdb.cli --db ./retail_data --buffer-frames 64 --policy LRU --backend page
```

Use `--backend objects` to store every table, column, B+ Tree, and LSM SSTable page as an individual object file via `LocalObjectStore`.

### Supported Commands

| Feature | Syntax Example |
| :--- | :--- |
| Typed / Untyped Table | `CREATE TABLE Products (product_id INT, name STR, category STR, price FLOAT, stock INT) LAYOUT ROW;` |
| Columnar Table | `CREATE TABLE Sales (id INT, category STR, price FLOAT) LAYOUT COLUMN;` |
| Vector Column | `CREATE TABLE Embeddings (id INT, vec VECTOR(3)) LAYOUT ROW;` |
| Insert | `INSERT INTO Products VALUES (1, 'Keyboard', 'Accessories', 29.99, 20);` |
| Projection & Filter | `SELECT product_id, name FROM Products WHERE price BETWEEN 10.0 AND 50.0;` |
| Comparison & Nulls | `SELECT * FROM Products WHERE price >= 20;` / `WHERE price IS NOT NULL;` |
| Update & Delete | `UPDATE Products SET stock = 19 WHERE product_id = 1;` / `DELETE FROM Products WHERE product_id = 1;` |
| Index Creation | `CREATE INDEX ON Products (price) USING BPLUS;` or `USING AUTO READS 100 WRITES 10;` |
| Shell Meta-Commands | `:explain <SELECT ...>`, `:stats`, `:tables`, `:help`, `:quit` |

---

## Python API Quick Example

```python
from smartdb.core import ColumnSpec, DataType
from smartdb.engine import SmartRetailDB

with SmartRetailDB("./retail_demo", buffer_frames=64, replacement_policy="LRU") as db:
    db.create_table(
        "Products",
        [
            ColumnSpec("product_id", DataType.INT, nullable=False),
            ColumnSpec("name", DataType.STR, nullable=False),
            ColumnSpec("price", DataType.FLOAT),
        ],
        layout="column",
    )
    db.create_index("Products", "price", index_type="BPLUS")
    db.insert("Products", {"product_id": 1, "name": "Keyboard", "price": 29.99})
    db.insert("Products", {"product_id": 2, "name": "Mouse", "price": 15.50})

    rows = db.execute("SELECT product_id, name FROM Products WHERE price BETWEEN 10.0 AND 25.0")
    print(rows)          # [{'product_id': 2, 'name': 'Mouse'}]
    print(db.buffer_stats())
```

---

## Retail Dataset & Reproducible Benchmarks

Generate a deterministic synthetic Retail Orders dataset (`Products`, `Customers`, `Orders`, `OrderItems`, `Inventory`):

```bash
PYTHONPATH=src python workloads/generate_retail_data.py --db ./retail_data --products 10000 --customers 1000 --orders 5000 --seed 42
```

Run the M1 and M2 benchmarks (each defaults to **1 warm-up + 5 measured repeats**, reporting environment metadata, wall-clock `mean_ms`/`median_ms`/`p95_ms`/`stdev_ms`, `ops_per_sec`, page I/O, and index byte size to `benchmark_results/`):

```bash
PYTHONPATH=src python benchmarks/benchmark_storage.py --pages 128 --accesses 10000 --records 2000 --repeats 5 --plots
PYTHONPATH=src python benchmarks/benchmark_indexes.py --records 2000 --repeats 5 --plots
```

| Script | Scenarios Measured | Output CSV |
| :--- | :--- | :--- |
| [benchmark_storage.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/benchmarks/benchmark_storage.py) | FIFO vs. LRU trace; Row vs. Column 2-column projection scan; `page` vs. `objects` vs. AES-GCM insert throughput | `benchmark_results/m1_storage.csv` |
| [benchmark_indexes.py](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/benchmarks/benchmark_indexes.py) | Full scan vs. B+ Tree vs. LSM point lookup; Write-heavy (`80W/20R`) vs. Read-heavy (`90R/10W`) vs. Range scan; index disk footprint | `benchmark_results/m2_indexes.csv` |

---

## Repository Structure

```text
src/smartdb/
  cli.py                    # Interactive REPL entry point
  core/
    schema.py               # Typed columns, TableSchema, and record validation
    key_codec.py            # Lossless order-preserving key encoding/decoding
    row_codec.py            # Binary value & row serialization with projection skip
  engine/
    catalog.py              # Atomic catalog.json & indexes.json persistence
    db.py                   # SmartRetailDB coordinator, DML/SELECT execution, reconciliation
  query/
    parser.py               # Restricted safe SQL/DML/DDL parser
    predicates.py           # SQL three-valued predicate evaluation
  storage/
    buffer_pool.py          # Shared multi-file BufferPool (LRU/FIFO, pin/unpin, LSN guard)
    backends/
      page_file.py          # PageBackend protocol & contiguous PageFile (4,096 B)
      object_store.py       # LocalObjectStore (one .obj file per page, cached count)
      encryption.py         # TransformingBackend hooks & optional AES-256-GCM helper
      manager.py            # StorageManager mapping logical files <-> file_id
    pages/
      slotted.py            # 24-byte PageHeader, PageType enum, SlottedPage
      overflow.py           # Linked OVERFLOW page chain reader/writer
    tables/
      row.py                # TableStorage protocol & slotted-page RowTable
      column_codec.py       # Columnar bitmap and fixed/variable page codecs
      column.py             # Per-column ColumnTable (projection pushdown, vectorized batches)
  indexes/
    manager.py              # IndexManager & choose_index_type heuristic
    bplus/
      pages.py              # Binary B+ Tree meta/internal/leaf page codecs
      buckets.py            # Linked OVERFLOW RID-bucket pages for duplicate keys
      tree.py               # Page-backed B+ Tree with leaf links
    lsm/
      bloom.py              # Compact blake2b Bloom filter
      sstable.py            # Immutable SSTable run page writer/reader
      wal.py                # Buffered framed Write-Ahead Log for memtable durability
      tree.py               # Disk-backed SSTable LSM Tree (compaction & tombstones)
tests/                      # 36 unit & integration tests (8 test modules)
benchmarks/                 # Reproducible M1/M2 benchmark harness & plotting
workloads/                  # Deterministic Retail Orders data generator
docs/
  architecture.md           # Component interfaces, binary page formats, and M3-M5 integration design
  milestone-checklist.md    # M1/M2 verification checklist
```

See [docs/architecture.md](file:///run/media/radwan/data/University/Year%204/Semester%201/Advanced%20DB/smart-retail-adb/docs/architecture.md) for binary page layouts, design rationale, and how Milestones 3–5 integrate on top of this engine.
