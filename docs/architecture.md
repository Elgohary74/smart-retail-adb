# Architecture — Smart Retail DB (M1 & M2)

This document explains the internal design, data formats, key design trade-offs, and extension points of the custom DBMS across **M1 (Storage Engine)** and **M2 (Advanced Indexing)**.

---

## 1. End-to-End Component Architecture

```text
+-------------------------------------------------------------------------+
|                     CLI (cli.py)  /  Python API                         |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       SmartRetailDB (engine.py)                         |
|  - Persists schemas in catalog.json & index definitions in indexes.json |
|  - Owns ONE shared BufferPool across all tables                         |
|  - Coordinates StorageTable (data) + IndexManager (M2) + query.py       |
+-------------------+---------------------------------+-------------------+
                    |                                 |
                    v                                 v
+---------------------------------------+ +-------------------------------+
|         StorageTable (table.py)       | |   IndexManager (indexes/)     |
|  - Validates records via TableSchema  | |  - BPlusTree (in-memory,      |
|  - Delegates to RowLayout or          | |    rebuilt on open)           |
|    ColumnLayout                       | |  - LSMTree (memtable +        |
+-------------------+-------------------+ |    sorted JSONL disk runs)    |
                    |                     +-------------------------------+
                    v
+-------------------------------------------------------------------------+
|                  Layout Manager (storage/layouts.py)                    |
|  - RowLayout:    1 stream  (<Table>.pages), locator = (page_id, slot_id)|
|  - ColumnLayout: N+1 streams (<table>/<col>.pages + _tombstones.pages), |
|                  locator = (0, row_id), projection-only reads           |
+------------------------------------+------------------------------------+
                                     | read_page / write_page / pin / unpin
                                     v
+-------------------------------------------------------------------------+
|               Shared BufferPool (storage/buffer_pool.py)                |
|  - OrderedDict keyed by (stream_id, page_id) with fixed capacity        |
|  - Eviction: LRU or FIFO (skips pinned frames; writes back dirty pages) |
|  - Metrics: hits, misses, hit_rate, disk_reads, disk_writes, evictions  |
+------------------------------------+------------------------------------+
                                     | PageBackend Protocol
                                     v
+-------------------------------------------------------------------------+
|            TransformingBackend (storage/encryption.py, optional)        |
+-------------------+---------------------------------+-------------------+
                    | backend_kind="page"             | backend_kind="objects"
                    v                                 v
+---------------------------------------+ +-------------------------------+
|        PageFile (page_file.py)        | | LocalObjectStore (object_store)|
|  - Contiguous binary file             | | - Directory of 4 KB .obj files |
|  - Page i at byte offset i * page_size| | - Cached _page_count           |
+---------------------------------------+ +-------------------------------+
```

---

## 2. Storage Manager (`src/smartdb/storage/`)

The Storage Manager is responsible only for **fixed-size pages, caching, and persistence**—it does not inspect record fields or JSON contents.

### 2.1 `PageBackend` & Physical Backends
All page backends implement the `PageBackend` protocol (`page_file.py`): `page_size`, `page_count`, `allocate_page()`, `read_page(page_id)`, `write_page(page_id, data)`, and `close()`.

* **`PageFile` (`page_file.py`):** Stores pages contiguously in a single binary file (`<db>/tables/<Table>.pages` or `<Table>/<col>.pages`). Page `i` lives at byte offset `i * page_size` (default `PAGE_SIZE = 4096`).
* **`LocalObjectStore` (`object_store.py`):** Stores each page as an independent file `page_000000000000.obj`. `_page_count` is computed once on initialization and incremented on `allocate_page()`, avoiding repeated directory scans during inserts.
* **`TransformingBackend` (`encryption.py`):** Wraps any `PageBackend` to apply paired same-length `encrypt`/`decrypt` hooks of length `backend.page_size`. Default is identity pass-through; production cryptography must be supplied externally.

### 2.2 Shared DB-Wide `BufferPool` (`buffer_pool.py`)
* **Single Shared Pool:** `SmartRetailDB` creates one `BufferPool` shared across all tables and column streams, keyed by `(stream_id: str, page_id: int)`.
  * *Why:* A single shared capacity makes `RowLayout` vs `ColumnLayout` benchmarks fair (a 5-column table gets the same memory budget as a 1-column/row table) and prepares directly for M3 multi-table joins.
  * *Standalone use:* `BufferPool(backend)` still works with a default stream (`"default"`).
* **Replacement Policies:** Uses an `OrderedDict` of cached frames:
  * `LRU`: moves a frame to the end on every hit (`read_page` or `write_page`).
  * `FIFO`: preserves insertion order regardless of hits.
* **Pinning & Write Validation:**
  * `pin_page` / `unpin_page` track per-frame pin counts so active frames are skipped during eviction (raises `RuntimeError` if every frame in the pool is pinned; ready for M4 transactions).
  * `write_page` validates `0 <= page_id < backend.page_count` and `len(data) == backend.page_size` up front, marking the frame `_dirty` for write-back on eviction or `flush()`/`close()`.

---

## 3. Typed Schema & Catalog (`schema.py` & `catalog.json`)

* `Column(name, dtype, nullable)` supports `INT` (excluding `bool`), `FLOAT` (accepts `int`/`float`), `STR`, `BOOL`, and `ANY`.
* Calling `create_table("T", ["id", "name"])` without types defaults to `ANY` (`nullable=True`) for backward compatibility.
* Schemas are persisted in `catalog.json` under `tables.<name>.columns` and validated by `TableSchema.validate_record()` on every `insert` and `update`.

---

## 4. Layout Manager (`layouts.py` & `table.py`)

The Layout Manager translates Python records into 4 KB pages and handles record addressing, tombstones, and column projection.

### 4.1 `RowLayout`
* **On-disk stream:** `<db>/tables/<Table>.pages` (or `.pages.objects/`).
* **Page binary format (`4,096` bytes):**
  ```text
  |<-- 4 bytes -->|<--------- payload_len bytes -------------->|<-- padding -->|
  +---------------+--------------------------------------------+---------------+
  |  payload_len  | {"layout":"row","records":[{...},None,...]}|  0x00 bytes   |
  |  (uint32 >I)  |            (UTF-8 JSON)                    |               |
  +---------------+--------------------------------------------+---------------+
  ```
* **Locator:** `(page_id, slot_id)`.
* **Updates & Deletes:**
  * `delete` sets `records[slot_id] = None` (tombstone) instead of removing the list element, so later `slot_id`s on the page never shift and existing M2 index locators remain valid.
  * `update` overwrites the slot in place if the page stays within `page_size`; if it overflows, the old slot is tombstoned and the updated record is appended to the tail page, returning the new locator.

### 4.2 `ColumnLayout`
* **On-disk streams:** `<db>/tables/<Table>/<col>.pages` for each column, plus `<db>/tables/<Table>/_tombstones.pages`.
* **Page binary format (`4,096` bytes):**
  ```text
  |<-- 4 bytes -->|<-- 4 bytes -->|<---- payload_len bytes ---->|<-- padding -->|
  +---------------+---------------+-----------------------------+---------------+
  |  payload_len  |  item_count   |  [val0, val1, val2, ...]    |  0x00 bytes   |
  |  (uint32 >I)  |  (uint32 >I)  |       (UTF-8 JSON)          |               |
  +---------------+---------------+-----------------------------+---------------+
  ```
* **Locator:** `(0, row_id)` — stays a 2-int tuple so `BPlusTree` and `LSMTree` store it identically to row locators without index changes.
* **Fast `row_id` Lookup:** On open, `ColumnLayout` reads the 8-byte header of each column's pages to build cumulative `item_count` offsets (`_page_starts[col]`). `bisect_right` maps `row_id -> (col_page_id, local_idx)` in `O(log pages)`.
* **Projection-Only I/O:** `scan(columns=[...])` and `iter_batches(columns=[...])` read **only** the page streams for the requested columns (plus `_deleted` tombstones), skipping unused columns completely.
* **Updates & Deletes:** `delete` appends `row_id` to `_tombstones.pages`. `update` writes changed column pages in place if all fit; if any column overflows its page, the updated row is inserted as a new `row_id` and the old `row_id` is tombstoned.

### 4.3 Row vs. Column Trade-Off Summary

| Operation | `RowLayout` | `ColumnLayout` |
|---|---|---|
| `insert(record)` | Fast (1 stream write on tail page) | Slower (`K` column stream writes) |
| `get_record(locator)` | Fast (1 page read) | Slower (`K` column page reads) |
| `scan(columns=[c1, c2])` | Reads and parses all `K` columns | Reads and parses **only** `c1`, `c2` |
| Batch output (`iter_batches`) | Re-packs rows into column batches | Streams column pages directly |

---

## 5. M2 Indexing & Query Execution

* **`BPlusTree` (`indexes/bplus_tree.py`):** In-memory B+ Tree with linked leaves and duplicate-key buckets; rebuilt from `StorageTable.iter_records()` when the DB opens.
* **`LSMTree` (`indexes/lsm_tree.py`):** Buffers writes in `memtable` and flushes sorted `run_*.jsonl` files to `<db>/index_data/<table>/<field>/`; compacts runs when `len(runs) >= 8`.
* **Index Selection (`indexes/manager.py`):** `auto` chooses `LSM` when `writes > 2 * max(1, reads)`; otherwise chooses `BPLUS`.
* **Safe Query Execution (`query.py` & `engine.py`):** `parse_select` accepts only `SELECT * FROM table [WHERE field = literal]` using `ast.literal_eval`. On indexed lookups, `SmartRetailDB.execute()` deduplicates locators, skips tombstoned/deleted records cleanly via `_safe_get_record()`, and re-verifies `record.get(field) == value`.

---

## 6. Intentional Simplifications & M3–M5 Readiness

To keep the educational DBMS easy to understand and maintain:
* **JSON payloads inside 4 KB pages:** Avoids manual byte-offset packing; single records larger than ~`4,092` bytes raise `OverflowError`.
* **Append-only page allocation:** Tombstoned slots are not reclaimed automatically; full page compaction can be added later if needed.
* **M3 & M4 hooks already in place:** `iter_batches(columns=...)` provides columnar batches for M3 vectorized operators, `TableSchema` provides column types for expressions/joins, and `BufferPool.pin_page`/`unpin_page` provides frame pinning for M4 transactions and WAL.
