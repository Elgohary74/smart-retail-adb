# 06 — Forward-Compatibility Hooks for Milestones 3 to 6

## 1. Purpose
To prevent rewrite bottlenecks during Weeks 9–14, Milestones 1 and 2 embed zero-overhead structural hooks that Milestones 3, 4, 5, and 6 plug into directly without altering binary disk formats.

---

## 2. Hook 1: Milestone 3 — Vectorized Query Processing & Cost-Based Optimizer

### 2.1 Zero-Copy Batch Vectorization (`BATCH_SIZE = 1024`)
* `ColumnPage.as_numpy_array(dtype)` exposes contiguous `int64`, `int32`, and `float64` memory buffers directly as NumPy arrays without copying bytes.
* Milestone 3 vectorized operators (`VectorFilter`, `VectorProject`, `HashJoin`, `HashAggregate`) will consume `dict[str, np.ndarray]` batches of up to `1,020` rows (1 full `ColumnPage` chunk) and apply boolean NumPy masks (`mask = (prices >= 50.0) & (prices <= 200.0)`).

### 2.2 Catalog Statistics Hook (`CatalogManager.get_table_stats`)
* Ensure `catalog.json` tracks and persists `row_count`, `page_count`, and per-column `min_val` / `max_val` on insert/load.
* Milestone 3's `CostBasedOptimizer` reads these exact counters to estimate scan I/O cost ($C_{\text{scan}} = \text{page\_count} \times C_{\text{seq\_page}}$) versus B+ Tree index lookup cost ($C_{\text{idx}} = (\text{height} + \text{selectivity} \times \text{leaf\_pages}) \times C_{\text{rand\_page}}$).

---

## 3. Hook 2: Milestone 4 — MVCC Transactions & ARIES Crash Recovery

### 3.1 Reserved 16-Byte MVCC Tuple Header (`SlottedPage`)
* Every row tuple stored in `SlottedPage` begins with `<QQ` (`xmin: uint64`, `xmax: uint64`):
  * In M1–M2, tuples are written with `xmin = 1` (committed bootstrap transaction) and `xmax = 0` (active/not deleted).
  * Expose helper methods `SlottedPage.get_tuple_mvcc_header(slot_id) -> tuple[int, int]` and `SlottedPage.set_tuple_xmax(slot_id, xmax: int) -> None` so Milestone 4's Snapshot Isolation visibility check (`xmin <= snapshot_xmax and (xmax == 0 or xmax > snapshot_xmax)`) operates in place without changing tuple byte offsets.

### 3.2 Reserved `page_lsn` Header Field (`PageHeader`)
* Bytes `8..15` of every 8,192-byte page store `page_lsn: uint64` (initialized to `0` in M1–M2).
* Expose `PageHeader.set_page_lsn(buffer, lsn: int)` and `PageHeader.get_page_lsn(buffer) -> int` so Milestone 4's ARIES Redo phase can skip reapplying log records where `page_lsn >= record.lsn`, and `BufferPoolManager` can enforce the Write-Ahead Logging (WAL) protocol (`flushed_lsn >= page_lsn` before dirty page eviction).

---

## 4. Hook 3: Milestone 5 — Distributed Sharding & Exact Vector Search Baseline

### 4.1 Immutable `RID` & Deterministic Shard Key Routing
* Because `SlottedPage.compact()` never alters `slot_id` inside the slot directory, `RID(page_id, slot_id)` remains stable across page defragmentations.
* `customers` and `orders` include `customer_id` (`INT64`) and `region` (`VARCHAR(16)`), allowing Milestone 5's `ShardRouter` to partition records across 3 gRPC nodes via `hash(shard_key) % 3` while keeping `customers` and `orders` co-located on `customer_id`.

### 4.2 Vector Column Alignment for Product Embeddings
* `ColumnPage` supports arbitrary fixed-width byte strides (`stride = 384 * 4 = 1,536` bytes per vector, fitting `5` 384-dimensional `float32` product embeddings per 8,192-byte page).
* Milestone 5's exact top-$k$ cosine similarity module (`np.dot(matrix, query_vec) / norms`) and Bonus Milestone 7's HNSW/IVF index will store product embeddings in `products.desc_vec.col` using the exact same `ColumnPage` and `BufferPoolManager` built in Milestone 1.