# `csai302_db` — Custom Disk-Oriented Relational & Object DBMS

A custom, disk-oriented relational and object Database Management System built from first principles in Python 3.12+ for the **CSAI 302** course (**Milestones 1 & 2**), modeled around the **Retail Orders & Product Search** domain (`customers`, `products`, `inventory`, `orders`, `order_items`, and encrypted product images).

All persistence, buffer management, indexing, and query execution are implemented directly in custom binary formats (`*.db`, `*.col`, `*.bpt`, `*.sst`, `*.wal`, `objects.dat`, `objects.dir`) without SQLite, external DB engines, `pickle`, or OS-file-per-record shortcuts.

---

## 1. Architecture Overview

* **Milestone 1 (Storage Engine & Buffer Pool):**
  * Universal `8,192`-byte checksummed pages (`32`-byte `<IBBHQIIQ` header + 64-bit BLAKE2b integrity digest) with Page 0 Free Space Map (FSM) bitmaps and `os.pread`/`os.pwrite` + `os.fsync` persistence.
  * Dual storage layouts sharing a unified `RID(page_id, slot_id)` abstraction:
    * `SlottedPage` (row-oriented) with `16`-byte MVCC headers (`xmin`, `xmax`) and `RID`-preserving `compact()`.
    * `ColumnPage` (column-oriented, `1,020` rows per 8-byte chunk) exposing zero-copy `numpy.ndarray` views.
  * Thread-safe, pin-aware `BufferPoolManager` supporting pluggable `LRUReplacer` and `FIFOReplacer` policies, dirty write-back, and hit/miss telemetry.
* **Milestone 2 (Object Store, Indexing & Query Shell):**
  * Extent-based binary `ObjectStore` (`objects.dat` + `objects.dir`) with `NoOpCipher` and `AES256GCMHook` (`aad = struct.pack("<Q", oid)`).
  * Buffer-pool-backed `BPlusTreeIndex` (`*.bpt`, `fsm_offset=64`) with order-preserving key codecs, duplicate-key composite encoding, leaf sibling range scans, and latch-crabbing pin discipline (`<= 2` pinned frames).
  * Three-tier `LSMTreeIndex` with CRC32-protected `lsm.wal`, `1 MB` `MemTable`, immutable `SSTable`s with SHA-256 Bloom filters and sparse fence indexes, and size-tiered compaction.
  * Sliding-window $(\alpha, \beta)$ `WorkloadAdvisor` and parameterized, injection-safe `SimpleExecutor` backed by an atomic `CatalogManager`.

---

## 2. Environment & Installation

Requires **Python 3.12+**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy cryptography pytest matplotlib
```

Quick dependency verification:

```bash
.venv/bin/python -c "import numpy, cryptography, pytest, matplotlib; print('OK')"
```

---

## 3. Repository Map

| Path | Role |
| :--- | :--- |
| [`src/`](src/) | Core DBMS package (`common/`, `storage/`, `index/`, `query/`) |
| [`tests/`](tests/) | Gate 1 ([`test_m1_storage.py`](tests/test_m1_storage.py)) and Gate 2 ([`test_m2_indexing.py`](tests/test_m2_indexing.py)) adversarial suites |
| [`bench/`](bench/) | Deterministic Retail data loader (`retail_loader.py`) and benchmark suite (`run_m1_m2_benchmarks.py`) |
| [`docs/`](docs/) | Architecture spec ([`architecture.md`](docs/architecture.md)), [`DEV_JOURNAL.md`](docs/DEV_JOURNAL.md), ADRs ([`ADR_M1_M2.md`](docs/decisions/ADR_M1_M2.md)), M1–M2 report ([`M1_M2_TECHNICAL_NARRATIVE.md`](docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md)), and M3–M6 SDK ([`M3_M6_HANDOFF_CONTRACT.md`](docs/handoff/M3_M6_HANDOFF_CONTRACT.md)) |
| [`results/m1_m2/`](results/m1_m2/) | Gate 3 CSV metrics ([`summary_metrics.csv`](results/m1_m2/summary_metrics.csv)), [`env_metadata.json`](results/m1_m2/env_metadata.json), and 300-DPI PNG [`plots/`](results/m1_m2/plots/) |
| `data/` | Runtime database files (gitignored) |

---

## 4. Reproducible Execution Commands

### Gate 1 — M1 Storage Engine & Buffer Pool
```bash
.venv/bin/pytest -v tests/test_m1_storage.py
```

### Gate 2 — M2 Indexing, Object Store & Query Shell (Full M1 + M2 Suite)
```bash
.venv/bin/pytest -v tests/test_m1_storage.py tests/test_m2_indexing.py
```

### Generate Retail Dataset (`seed=42`)
```bash
# Standard scale (10k customers, 5k products, 50k orders, 200k order_items, 500 encrypted PNGs + 7 indexes)
.venv/bin/python -m bench.retail_loader --scale standard --seed 42 --data-dir data

# Fast tiny scale (500 customers, 250 products, 2k orders, 8k order_items, 25 encrypted PNGs)
.venv/bin/python -m bench.retail_loader --scale tiny --seed 42 --data-dir /tmp/csai302_tiny

# Skip index generation (tables + encrypted ObjectStore + catalog.json only)
.venv/bin/python -m bench.retail_loader --scale tiny --seed 42 --data-dir /tmp/csai302_tiny --no-indexes
```

### Parameterized Query Shell (`SimpleExecutor`) Usage
```python
import hashlib
from src.query import CatalogManager, SimpleExecutor
from src.storage import AES256GCMHook, StorageEngine

key = hashlib.sha256(b"csai302-retail-master-key").digest()
engine = StorageEngine("data", buffer_pool_size=64, cipher=AES256GCMHook(key))
engine.open()
catalog = CatalogManager("data/catalog.json")
executor = SimpleExecutor(engine, catalog)

try:
    # Point lookup via idx_customers_pk (BPlusTreeIndex)
    customer = executor.execute("SELECT * FROM customers WHERE customer_id = ?", (1,))
    # Range scan via idx_products_price (BPlusTreeIndex)
    mid_tier = executor.execute("SELECT * FROM products WHERE price BETWEEN ? AND ?", (50.0, 200.0))
    # WorkloadAdvisor-guided index DDL (USING BTREE | LSM | AUTO)
    # executor.execute("CREATE INDEX idx_orders_cust ON orders (customer_id) USING AUTO")
finally:
    executor.close()
    engine.close()
```

### Gate 3 — Full M1 + M2 Benchmarks & Plots
```bash
.venv/bin/python -m bench.run_m1_m2_benchmarks --runs 5 --seed 42 --out-dir results/m1_m2
```

---

## 5. Gate 3 Benchmark Summary & Documentation

Full methodology, environment metadata ([`env_metadata.json`](results/m1_m2/env_metadata.json)), complete tables ([`summary_metrics.csv`](results/m1_m2/summary_metrics.csv)), and analysis live in [`docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md`](docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md).

| Experiment | Key Finding (`runs=5`, `seed=42`) | Plot |
| :--- | :--- | :--- |
| **1A** Buffer Pool (`LRU` vs. `FIFO`) | At 16 frames under Zipfian ($\theta=0.9$), `LRU` achieves **70.72%** hit ratio vs. **66.34%** `FIFO`; sequential scans thrash at **0.0%** ($\le 64$ frames) and jump to **98.7%** at `128` frames (`65`-page working set). | [`fig1_buffer_pool_hit_ratio.png`](results/m1_m2/plots/fig1_buffer_pool_hit_ratio.png) |
| **1B** Row vs. Column (`200k` `order_items`) | `ColumnTableHeap` runs OLAP `SUM(qty * unit_price)` in **10.70 ms** vs. **1,525.42 ms** (`~142.6x` faster, `394` vs. `1,283` pages); `RowTableHeap` wins `5,000` OLTP point lookups (**127.31 ms** vs. **409.30 ms**, `~3.2x` faster). | [`fig2_row_vs_column_latency.png`](results/m1_m2/plots/fig2_row_vs_column_latency.png) |
| **2A** Read-Heavy `90/10` (`50k` products) | `LSM_TREE` (**23,225 ops/s**, `0.059 ms` P95) and `BPLUS_TREE` (**19,453 ops/s**, `0.072 ms` P95) outperform `FULL_SCAN` (**4.17 ops/s**) by `~4,668x–5,574x`. | [`fig3_index_read_vs_write_throughput.png`](results/m1_m2/plots/fig3_index_read_vs_write_throughput.png) |
| **2B** Write-Heavy `10/90` (`50k` inventory) | `LSM_TREE` achieves **13,108 ops/s** & **1.27 MB** vs. `BPLUS_TREE` **8,961 ops/s** & **2.30 MB** (`~3,332x` over `FULL_SCAN` at **3.93 ops/s**). | [`fig4_index_p95_latency_and_size.png`](results/m1_m2/plots/fig4_index_p95_latency_and_size.png) |
| **2C** `WorkloadAdvisor` Trace (`4,150` ops) | Deterministic transitions: `FULL_SCAN` ($\alpha=0.5, \beta=0.0, N=150$) $\to$ `LSM_TREE` ($\alpha=0.903, \beta=0.0, N=1000$) $\to$ `BPLUS_TREE` ($\alpha=0.049, \beta=0.902, N=1000$). | — |
| **2D** `ObjectStore` (`500` images, `34.01` MB) | `AES256GCMHook` sustains **1,008.1 MB/s** reads vs. **1,213.0 MB/s** `NoOpCipher` with **+0.107%** framing overhead (`34.04 MB` total). | [`fig5_encryption_throughput.png`](results/m1_m2/plots/fig5_encryption_throughput.png) |

**Engineering & Handoff Docs:**
* [`docs/decisions/ADR_M1_M2.md`](docs/decisions/ADR_M1_M2.md) — Architecture Decision Records (`ADR-001` through `ADR-007`)
* [`docs/DEV_JOURNAL.md`](docs/DEV_JOURNAL.md) — Chronological debugging, bottlenecks, and failed-attempts log
* [`docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md`](docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md) — Report-ready M1–M2 design, experimental setup, and analysis
* [`docs/handoff/M3_M6_HANDOFF_CONTRACT.md`](docs/handoff/M3_M6_HANDOFF_CONTRACT.md) — Verified Python API signatures, M3–M5 recipes, and invariant checklist

---

## 6. Rubric Traceability Summary (30 Marks)

| Milestone | Rubric Criterion | Marks | Target Modules | Gate Verification |
| :--- | :--- | :---: | :--- | :--- |
| **M1** | Custom persistent storage | 5 | `src/storage/disk_manager.py`, `page_header.py`, `storage_engine.py` | `test_m1_persistence_across_restart`, `test_m1_torn_page_checksum_detection`, `test_m1_fsm_allocation_reuse` |
| **M1** | Buffer pool | 3 | `src/storage/buffer_pool.py`, `replacer.py` | `test_m1_lru_eviction_order`, `test_m1_fifo_eviction_order`, `test_m1_pin_protection_and_pool_full` |
| **M1** | Storage layouts | 3 | `src/storage/slotted_page.py`, `column_page.py`, `table_heap.py` | `test_m1_slotted_page_crud_and_compaction`, `test_m1_column_page_zero_copy_numpy` |
| **M1** | Testing and measurements | 4 | `tests/test_m1_storage.py`, `bench/run_m1_m2_benchmarks.py` | `test_m1_dirty_writeback_accuracy`, `test_m1_hit_miss_telemetry` + Gate 3 M1 benchmarks |
| **M2** | B+ Tree | 5 | `src/index/bplus_tree.py` | `test_m2_bplus_50k_inserts_and_invariants`, `test_m2_bplus_sibling_range_scan`, `test_m2_bplus_restart_persistence` |
| **M2** | LSM Tree | 4 | `src/index/lsm_tree.py` | `test_m2_lsm_wal_crash_recovery`, `test_m2_lsm_tombstone_compaction`, `test_m2_lsm_bloom_filter_effectiveness` |
| **M2** | Workload-based choice | 2 | `src/index/index_advisor.py` | `test_m2_workload_advisor_decision_rules` |
| **M2** | Object storage & encryption | 1 | `src/storage/object_store.py`, `cipher_hooks.py` | `test_m2_object_store_aes256_gcm_and_tamper_detection` |
| **M2** | Correctness & performance | 3 | `src/query/catalog.py`, `simple_executor.py`, `bench/` | `test_m2_sql_executor_and_injection_defense`, `test_m2_index_vs_fullscan_equivalence` + Gate 3 M2 benchmarks |
