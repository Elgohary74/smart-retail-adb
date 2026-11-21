# Architecture Decisions — M1/M2

## 1. Storage Manager vs. Layout Manager Separation
- **Storage Manager (`src/smartdb/storage/`):**
  - `PageBackend` (`page_file.py`): Fixed-size page interface (`read_page`, `write_page`, `allocate_page`, `page_count`, `page_size`, `close`). Default `PAGE_SIZE = 4096`.
  - `PageFile`: Stores pages contiguously in one binary file (`page_id * page_size`).
  - `LocalObjectStore` (`object_store.py`): Stores each page in a separate local object file (`page_000000000000.obj`) with cached `_page_count`.
  - `TransformingBackend` (`encryption.py`): Optional same-length encrypt/decrypt hook wrapper respecting `backend.page_size`. Actual cryptography must be supplied externally.
  - `BufferPool` (`buffer_pool.py`): **One shared pool per `SmartRetailDB`**, keyed by `(stream_id, page_id)`. Supports `FIFO` and `LRU` eviction, `pin_page`/`unpin_page`, dirty-page write-back, `write_page` bounds validation, and `BufferStats` (`hits`, `misses`, `hit_rate`, `disk_reads`, `disk_writes`, `evictions`).

## 2. Typed Schema (`schema.py`)
- `TableSchema` and `Column(name, dtype, nullable)` support `INT`, `FLOAT`, `STR`, `BOOL`, and `ANY` (untyped fallback when only column names are supplied).
- Schemas are persisted in `catalog.json` under `columns` and validated on every `insert` and `update`.

## 3. Layout Manager (`layouts.py` & `table.py`)
- **Row Layout (`RowLayout`):**
  - Stores full table records in one stream (`<table>.pages` or `.pages.objects/`).
  - Page format: `[4B payload_len (>I) | UTF-8 JSON {"layout":"row","records":[...]} | 0x00 padding]`.
  - Record locator: `(page_id, slot_id)`.
  - Deletes set the slot to `None` (tombstone) so later `slot_id`s on the page never shift. In-place updates preserve `(page_id, slot_id)`; oversized updates tombstone the old slot and append to the tail page.
  - Trade-off: 1 page read per point lookup (`get_record`), but scanning 1–2 columns still reads and parses entire rows.

- **Column Layout (`ColumnLayout`):**
  - Stores each column in its own stream (`tables/<table>/<col>.pages`) plus a tombstone stream (`_tombstones.pages`).
  - Page format: `[4B payload_len (>I) | 4B item_count (>I) | UTF-8 JSON array | 0x00 padding]`.
  - Record locator: `(0, row_id)` — a 2-int tuple compatible with M2 indexes; `row_id` maps to `(col_page_id, local_idx)` via cumulative `item_count` offsets (`bisect_right`).
  - `scan(columns=[...])` and `iter_batches(columns=[...])` read **only** the requested column streams, skipping unused columns.
  - Trade-off: faster projection scans and analytical reads; slower full-row inserts and point lookups (`K` column page reads).

## 4. M2 Indexing & Query Interface
- **B+ Tree (`indexes/bplus_tree.py`):** Keys and `(a, b)` locators held in leaf nodes with sibling `next` links for range scans; rebuilt from storage on open.
- **LSM Tree (`indexes/lsm_tree.py`):** Writes buffer in an in-memory memtable and flush to sorted JSONL runs on disk; compaction merges runs.
- **Auto-selection heuristic (`indexes/manager.py`):** Chooses `LSM` if `writes > 2 * max(1, reads)`; otherwise `BPLUS`.
- **Restricted query parser (`query.py`):** Supports `SELECT * FROM table [WHERE field = literal]` using `ast.literal_eval` (no arbitrary SQL/Python execution).
- **Stale locator safety:** `SmartRetailDB.execute()` skips deleted/tombstoned locators safely via `_safe_get_record` and deduplicates repeated locators.
