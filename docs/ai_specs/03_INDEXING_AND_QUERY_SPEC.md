# 03 — B+ Tree, LSM Tree, Workload Advisor & Query Specification

## 1. Unified Index Interface (`src/index/base_index.py`)

Both `BPlusTreeIndex` and `LSMTreeIndex` must inherit from `BaseIndex` and expose identical operational semantics:

```python
class BaseIndex(ABC):
    @abstractmethod
    def insert(self, key: Any, rid: RID) -> None: ...
    @abstractmethod
    def get(self, key: Any) -> list[RID]: ...
    @abstractmethod
    def range_scan(self, low_key: Any, high_key: Any, inclusive: bool = True) -> list[tuple[Any, RID]]: ...
    @abstractmethod
    def delete(self, key: Any, rid: RID | None = None) -> bool: ...
    @abstractmethod
    def flush(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def get_stats(self) -> dict[str, Any]: ...
```
* **Duplicate Key Support:** Non-unique secondary indexes (e.g., `products.price`, `orders.order_date`, `order_items.order_id`) store composite keys `(key, rid.page_id, rid.slot_id)` internally so every entry is physically unique and deterministic. `get(key)` returns all matching `RID`s in ascending order.

---

## 2. Buffer-Pool-Backed B+ Tree Index (`src/index/bplus_tree.py`)

The B+ Tree persists every node inside an 8,192-byte page managed by its own `DiskManager` and `BufferPoolManager` (`*.bpt` file).

### 2.1 Index Metadata Page (`page_id = 0`)
* Bytes `0..31`: Standard `PageHeader` (`page_type = 0`).
* Bytes `32..55` (`<IBBHQ`):
  * `root_page_id: uint32` (`0xFFFFFFFF` when empty)
  * `key_dtype: uint8` (`1=INT32, 2=INT64, 3=UINT64, 4=FLOAT64, 5=VARCHAR`)
  * `key_size: uint16` (Fixed byte width of serialized key, e.g., `8` for `INT64`/`FLOAT64`, `max_len` for `VARCHAR`)
  * `height: uint16` (`0` when empty, `1` when root is a leaf)
  * `entry_count: uint64` (Total indexed key-RID pairs)
* **Critical Persistence Rule:** Whenever `root_page_id` or `height` changes during a split, immediately update `page_id = 0` via the buffer pool so process restarts always find the current root.

### 2.2 Fixed-Size Key Encoding (`encode_key` / `decode_key`)
* `INT32` / `INT64`: Map signed integers to big-endian unsigned order-preserving bytes by XORing the sign bit (`val ^ (1 << 63)` packed as `>Q` for `INT64`), allowing lexicographical byte comparison `bytes_a < bytes_b` to preserve exact numeric order!
* `FLOAT64`: IEEE-754 order-preserving 8-byte big-endian mapping (flip sign bit if positive; flip all 64 bits if negative).
* `VARCHAR(max_len)`: UTF-8 bytes right-padded with `\x00` to `max_len`.
* Composite uniqueness suffix: Append `rid.page_id` (`>I`) + `rid.slot_id` (`>I`) = 8 bytes to form a fixed-length `effective_key_size = key_size + 8` byte key. This eliminates duplicate-key overflow edge cases completely!

### 2.3 Internal & Leaf Node Binary Layouts
* **Internal Node (`page_type = 3`):**
  * Header (32 B): `slot_count` = number of separator keys $k$ (there are $k + 1$ child pointers).
  * Body starts at offset `32`:
    `[child_0: uint32 (4B)]` followed by $k$ records of `[sep_key_i: effective_key_size B][child_{i+1}: uint32 (4B)]`.
  * Maximum keys per internal node: $M_{\text{int}} = \lfloor \frac{8192 - 36}{\text{effective\_key\_size} + 4} \rfloor$.
* **Leaf Node (`page_type = 4`):**
  * Header (32 B): `slot_count` = number of entries $k$; `next_page_id` = right sibling leaf page ID (`0xFFFFFFFF` if rightmost).
  * Body starts at offset `32`:
    $k$ records of `[entry_key_i: effective_key_size B][rid_page_id_i: uint32 (4B)][rid_slot_id_i: uint32 (4B)]`.
  * Maximum entries per leaf node: $M_{\text{leaf}} = \lfloor \frac{8192 - 32}{\text{effective\_key\_size} + 8} \rfloor$.

### 2.4 B+ Tree Algorithms
1. **Point Lookup (`get(key)`):**
   * Construct `low_composite = encode_key(key) + b"\x00" * 8` and `high_composite = encode_key(key) + b"\xff" * 8`.
   * Traverse from `root_page_id` down to the target leaf using binary search (`bisect_left`), unpinning each parent page immediately after reading the child pointer (crabbing/latch-release discipline so at most 2 pages are pinned simultaneously!).
   * Scan leaf entries while `entry_key <= high_composite`, following `next_page_id` if entries spill onto the right sibling leaf.
2. **Range Scan (`range_scan(low_key, high_key, inclusive=True)`):**
   * Descend to the leaf containing `low_key`, then traverse sequentially along `next_page_id` pointers until `decoded_key > high_key` (or `>=` when `inclusive=False`). Unpin each leaf page as soon as traversal moves to `next_page_id`.
3. **Insertion & Split-Upward (`insert(key, rid)`):**
   * Record the traversal stack of `(page_id, child_index)` from root to leaf.
   * Insert `(composite_key, rid)` into the target leaf in sorted order.
   * If `slot_count > M_leaf`: allocate a new leaf `right_page_id`, move the upper half $\lfloor (M_{\text{leaf}} + 1) / 2 \rfloor$ entries to `right_page_id`, set `right.next_page_id = left.next_page_id`, set `left.next_page_id = right_page_id`, and push `(right.first_key, right_page_id)` up to the parent internal node.
   * Propagate internal node splits upward recursively; if the root splits, allocate a new root page (`flags |= IS_ROOT`), increment `height`, and update `page_id = 0`.

---

## 3. Write-Optimized LSM Tree Index (`src/index/lsm_tree.py`)

`LSMTreeIndex` stores data in `data/indexes/<index_name>/` using a Write-Ahead Log, an in-memory sorted `MemTable`, and immutable tiered `SSTables`.

### 3.1 Durability WAL (`lsm.wal`) & `MemTable`
* **WAL Record Format (`<IBH` + key + `<II`):**
  `[crc32: uint32][op_type: uint8 (1=PUT, 2=DELETE)][key_len: uint16][key_bytes][rid_page: uint32][rid_slot: uint32]`.
  * Every `insert` or `delete` appends to `lsm.wal` and updates the in-memory sorted `MemTable` (backed by a Python `dict` + sorted key structure or SkipList; byte size tracked via `memtable_bytes`).
  * On startup, `LSMTreeIndex` replays `lsm.wal` (ignoring any incomplete trailing record with invalid CRC) to restore uncommitted `MemTable` entries.
* **Flush Threshold:** When `memtable_bytes >= 1_048_576` (`1 MB` default for fast benchmarking; configurable via `memtable_size_bytes`), freeze the `MemTable`, flush it to a new Level-0 `SSTable` (`sst_0_<seq:06d>.sst`), update `manifest.json` atomically, and truncate `lsm.wal`.

### 3.2 Immutable `SSTable` Binary Layout
Each `SSTable` file consists of:
1. **Data Blocks (8,192 bytes each):**
   * Packed sorted entries: `[key_len: uint16][key_bytes][is_tombstone: uint8][rid_page: uint32][rid_slot: uint32]`.
   * Zero-padded to `8,192` bytes at block boundaries.
2. **Bloom Filter Section:**
   * Sized for $n$ entries and target false-positive rate $p = 0.01$:
     $$m = \left\lceil -\frac{n \ln p}{(\ln 2)^2} \right\rceil \text{ bits}, \quad k = \max\left(1, \text{round}\left(\frac{m}{n} \ln 2\right)\right) \text{ hash functions}$$
   * Double-hashing using SHA-256 (`h1, h2 = struct.unpack("<QQ", hashlib.sha256(key).digest()[:16])`), setting bit $(h_1 + i \cdot h_2) \bmod m$ for $i \in [0, k-1]$.
3. **Sparse Fence Index Section:**
   * Stores `[num_blocks: uint32]` followed by `[first_key_len: uint16][first_key_bytes][block_offset: uint64]` for every 8 KB data block (loaded into RAM when the `SSTable` opens).
4. **64-Byte Footer at EOF (`<8sQIIQIQQI12s`):**
   * `magic = b"CSAI_SST"`, `bloom_offset`, `bloom_len`, `num_hashes`, `index_offset`, `index_len`, `entry_count`, `crc32`, `padding`.

### 3.3 Lookup, Range Scan & Size-Tiered Compaction
* **Point Lookup (`get(key)`):**
  1. Check active `MemTable` and any immutable flushing `MemTable`.
  2. Search `SSTables` from **newest to oldest** (`L0` descending by sequence number, then `L1`, then `L2`).
  3. For each `SSTable`, check its in-memory **Bloom Filter** first; if negative, skip disk I/O completely!
  4. If Bloom filter is positive, binary-search the in-memory **Fence Index** to locate the single 8 KB data block, read only that block, and scan for matching keys. If the newest record for a composite key is a `TOMBSTONE` (`is_tombstone == 1`), suppress older versions in lower levels.
* **Range Scan (`range_scan(low_key, high_key)`):**
  * Open merge iterators across `MemTable` and all active `SSTables` using `heapq.merge`, deduplicate identical composite keys keeping the newest version, filter out `is_tombstone == 1`, and yield records in sorted key order.
* **Size-Tiered Compaction:**
  * Whenever Level $L_i$ reaches `4` `SSTables` (`compaction_trigger = 4`), merge all `SSTables` in $L_i$ into a single sorted `SSTable` in $L_{i+1}$.
  * Deduplicate identical keys (retaining the newest version). If $L_{i+1}$ is the deepest existing level, drop `is_tombstone == 1` records permanently. Atomically update `manifest.json` and delete old `SSTable` files.

---

## 4. Workload-Based Index Auto-Selector (`src/index/index_advisor.py`)

`WorkloadAdvisor` maintains a sliding window (`collections.deque(maxlen=1000)`) of recent operations `(op_type, table_name, column_name)` where `op_type` $\in$ `{"POINT_READ", "RANGE_READ", "WRITE"}`.

### 4.1 Quantitative Selection Formula
For a candidate `(table_name, column_name)` over the current window:
* Let $R_{\text{eq}}$ = count of `POINT_READ` operations.
* Let $R_{\text{rng}}$ = count of `RANGE_READ` operations.
* Let $U$ = count of `WRITE` (`INSERT`, `UPDATE`, `DELETE`) operations.
* Let $N_{\text{ops}} = R_{\text{eq}} + R_{\text{rng}} + U$, and let $P$ = current page count of `table_name`.
* Compute:
  $$\alpha = \frac{U}{\max(1, N_{\text{ops}})} \quad (\text{Write Intensity Ratio})$$
  $$\beta = \frac{R_{\text{rng}}}{\max(1, R_{\text{eq}} + R_{\text{rng}})} \quad (\text{Range Query Ratio})$$

### 4.2 Deterministic Decision Table
1. **Rule 1 (Small Table / Cold Start):** If $N_{\text{ops}} < 200$ or $P < 4$ $\rightarrow$ Select **`FULL_SCAN`**.
2. **Rule 2 (Write-Heavy Point Workload):** Else if $\alpha > 0.45$ and $\beta < 0.25$ $\rightarrow$ Select **`LSM_TREE`** (eliminates random B+ Tree page splits during heavy inventory/order writes).
3. **Rule 3 (Read-Heavy or Range-Scan Workload):** Else ($\alpha \le 0.45$ or $\beta \ge 0.25$) $\rightarrow$ Select **`BPLUS_TREE`** (leverages $O(\log_B N)$ tree traversal and sequential leaf sibling pointers).

Every call to `advisor.recommend(table_name, column_name)` returns a `Recommendation(strategy, alpha, beta, n_ops, reason)` dataclass and appends an audit entry to `advisor.decision_log`.

---

## 5. Parameterized SQL-Like Query Shell (`src/query/simple_executor.py`)

Provides a clean, SQL-injection-proof interface for Milestone 2 testing and benchmarking:
* **Supported Statements:**
  * `CREATE INDEX <idx_name> ON <table> (<col>) [USING BTREE | LSM | AUTO]`
  * `INSERT INTO <table> VALUES (?, ?, ...)`
  * `SELECT * FROM <table> WHERE <col> = ?`
  * `SELECT * FROM <table> WHERE <col> BETWEEN ? AND ?`
  * `UPDATE <table> SET <col> = ? WHERE <pk_col> = ?`
  * `DELETE FROM <table> WHERE <pk_col> = ?`
* **Security Rule:** All literal values **MUST** pass through the `params: tuple | list` argument of `execute(sql: str, params: Sequence[Any] = ())`. If raw string literals or semicolon-stacked statements (`; DROP ...`) appear in `sql`, raise `SecurityViolationError`.