# System Architecture Specification — Milestones 1 & 2
**Project:** Custom Relational & Object Database Management System (`csai302_db`)  
**Application Domain:** Retail Orders and Product Search  
**Target Runtime:** Python 3.12+ (Custom Binary Engine)

---

## 1. Architectural Overview & System Positioning

`csai302_db` is a custom, disk-oriented Database Management System built from first principles in Python. It implements its own fixed-page disk manager, buffer pool manager, dual row/columnar storage layouts, encrypted binary object store, disk-backed B+ Tree index, tiered LSM Tree index, workload-driven index advisor, and parameterized query executor.

### 1.1 Core Architectural Principles
1. **Zero External Database Engines:** All persistence, caching, indexing, and query execution occur inside team-implemented binary formats (`*.db`, `*.col`, `*.bpt`, `*.sst`, `*.wal`, `objects.dat`).
2. **Universal 8,192-Byte Page Abstraction:** Every relational table page and B+ Tree node conforms to an identical `8,192`-byte binary geometry with a `32`-byte header, allowing a single `BufferPoolManager` implementation to manage heterogeneous storage and index structures uniformly.
3. **Zero-Copy Memory Alignment:** Binary buffers use Python `bytearray` and `memoryview` slices to eliminate redundant copies, while columnar pages expose C-contiguous typed arrays directly to `numpy.ndarray` views via `np.frombuffer`.
4. **Forward-Compatible Hooks (M3–M6):** Page headers reserve `page_lsn` (`8` bytes), row tuples reserve `xmin` and `xmax` (`16` bytes), and record identifiers (`RID`) remain immutable across page defragmentations so Milestone 3 vectorized joins, Milestone 4 MVCC/ARIES recovery, and Milestone 5 Raft/2PC sharding attach without altering on-disk layouts.

### 1.2 End-to-End System Architecture Diagram (M1 & M2 Focus + M3–M6 Extension Points)

```text
+-----------------------------------------------------------------------------------+
|                        CLIENT / BENCHMARK / TEST HARNESS                          |
|   tests/test_m1_storage.py  |  tests/test_m2_indexing.py  |  bench/benchmarks.py  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     QUERY & CATALOG LAYER (src/query/) [M2]                       |
|  +-------------------------------------+   +-----------------------------------+  |
|  | SimpleExecutor                      |   | CatalogManager (catalog.json)     |  |
|  | - Parameterized SQL-like Parser     |   | - Table Schemas & Layout Types    |  |
|  | - Injection Guard (SecurityError)   |   | - Row/Page Stats & Index Registry |  |
|  +------------------+------------------+   +-----------------+-----------------+  |
+---------------------|----------------------------------------|--------------------+
                      |                                        |
                      v                                        v
+-----------------------------------------------------------------------------------+
|                 ACCESS METHODS & WORKLOAD INTELLIGENCE (src/index/) [M2]          |
|  +-----------------------------------------------------------------------------+  |
|  | WorkloadAdvisor: Sliding Window (W=1000) -> (alpha, beta) Index Selector    |  |
|  +--------------------+-----------------------------------+--------------------+  |
|                       |                                   |                       |
|                       v                                   v                       |
|  +-------------------------------------+   +-----------------------------------+  |
|  | BPlusTreeIndex (*.bpt)              |   | LSMTreeIndex (data/indexes/<id>/) |  |
|  | - Read & Range-Scan Optimized       |   | - Write-Optimized Append Engine   |  |
|  | - Buffer-Pool-Backed 8KB Nodes      |   | - WAL + SkipList MemTable (1MB)   |  |
|  | - Leaf Sibling Pointers (next_page) |   | - Immutable SSTables + Bloom + L* |  |
|  +--------------------+----------------+   +-----------------+-----------------+  |
+-----------------------|--------------------------------------|--------------------+
                        |                                      |
                        v                                      |
+--------------------------------------------------------+     |
|           BUFFER POOL LAYER (src/storage/) [M1]        |     |
|  +--------------------------------------------------+  |     |
|  | BufferPoolManager                                |  |     |
|  | - Frame Pool (N x 8,192B bytearray)              |  |     |
|  | - Page Table (page_id -> frame_id)               |  |     |
|  | - Pin Reference Counters & Dirty Flags           |  |     |
|  | - Pluggable ReplacementPolicy (LRU / FIFO)       |  |     |
|  +-------------------------+------------------------+  |     |
+----------------------------|---------------------------+     |
                             |                                 |
         +-------------------+-------------------+             |
         v                                       v             |
+---------------------------------+   +-----------------------+|
| RELATIONAL STORAGE [M1]         |   | OBJECT STORAGE [M2]   ||
| +-----------------------------+ |   | +-------------------+ ||
| | RowTableHeap (SlottedPage)  | |   | | ObjectStore       | ||
| | - customers, products,      | |   | | - objects.dat     | ||
| |   inventory, orders         | |   | | - objects.dir     | ||
| +-----------------------------+ |   | +---------+---------+ ||
| +-----------------------------+ |   |           |           ||
| | ColumnTableHeap (ColumnPage)| |   |           v           ||
| | - order_items (*.col)       | |   | +-------------------+ ||
| | - Zero-Copy NumPy Views     | |   | | CipherHook        | ||
| +-----------------------------+ |   | | - NoOpCipher      | ||
+----------------+----------------+   | | - AES256GCMHook   | ||
                 |                    | +---------+---------+ ||
                 +--------------------+-----------+-----------++
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                   PERSISTENCE & DISK LAYER (src/storage/) [M1 + M2]               |
|  +-----------------------------------------------------------------------------+  |
|  | Unified StorageEngine & DiskManager                                         |  |
|  | - Page 0 Free Space Map (FSM) Bitmap  |  CRC64 Torn-Page Verification       |  |
|  | - Aligned 8KB OS I/O (pread / pwrite / fsync)                               |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
|  [M3 Hook: Vectorized Filter/Join]  [M4 Hook: MVCC xmin/xmax & ARIES WAL LSN]     |
|  [M5 Hook: 3-Node Raft/2PC Shards & 384-d Exact Cosine Vector Search]             |
+-----------------------------------------------------------------------------------+
```

---

## 2. Complete Repository & File Structure

### 2.1 Annotated Directory Tree

```text
csai302_db/
├── docs/
│   ├── architecture.md                    # Complete M1+M2 system architecture & diagrams
│   └── ai_specs/
│       ├── 00_AGENT_GOVERNANCE.md         # Anti-cheat rules, lock hierarchy, wave gates
│       ├── 01_RETAIL_SCHEMA_AND_DATA.md   # Retail tables, types, catalog.json, data scale
│       ├── 02_STORAGE_AND_OBJECT_SPEC.md  # 8KB pages, Slotted/Column pages, BPM, ObjectStore
│       ├── 03_INDEXING_AND_QUERY_SPEC.md  # B+ Tree, LSM Tree, WorkloadAdvisor, SQL shell
│       ├── 04_TEST_AND_RUBRIC_GATE.md     # Adversarial test specs mapped to 30 rubric marks
│       ├── 05_BENCHMARK_PROTOCOL.md       # 1-warmup + 5-run protocol & 5 plot definitions
│       └── 06_M3_TO_M6_HOOKS.md           # Reserved headers & hooks for Milestones 3-6
│
├── src/
│   ├── __init__.py
│   ├── common/                            # Layer 0: Shared primitives, types & exceptions
│   │   ├── __init__.py
│   │   ├── config.py                      # PAGE_SIZE=8192, HEADER_SIZE=32, sentinels, defaults
│   │   ├── exceptions.py                  # DatabaseError hierarchy (PageCorrupted, PoolFull, etc.)
│   │   └── types.py                       # DataType enum, RID, ColumnDef, Schema, pack/unpack
│   │
│   ├── storage/                           # Layer 1: M1 Storage Engine + M2 Encrypted ObjectStore
│   │   ├── __init__.py
│   │   ├── disk_manager.py                # Binary file I/O, FSM page 0 bitmap, checksums, fsync
│   │   ├── page_header.py                 # 32-byte universal header (<IBBHQIIQ) & PageType enum
│   │   ├── slotted_page.py                # Row-oriented SlottedPage + MVCC header + compact()
│   │   ├── column_page.py                 # Column-oriented ColumnPage + zero-copy NumPy arrays
│   │   ├── replacer.py                    # ReplacementPolicy ABC, LRUReplacer, FIFOReplacer
│   │   ├── buffer_pool.py                 # BufferPoolManager (pin/unpin, eviction, hit/miss stats)
│   │   ├── table_heap.py                  # RowTableHeap & ColumnTableHeap multi-page managers
│   │   ├── cipher_hooks.py                # CipherHook ABC, CipherPayload, NoOpCipher, AES256GCMHook
│   │   ├── object_store.py                # Extent-based binary ObjectStore (objects.dat + .dir)
│   │   └── storage_engine.py              # Unified M1+M2 facade coordinating heaps & ObjectStore
│   │
│   ├── index/                             # Layer 2: M2 Access Methods & Workload Intelligence
│   │   ├── __init__.py
│   │   ├── base_index.py                  # BaseIndex ABC (insert, get, range_scan, delete, flush)
│   │   ├── bplus_tree.py                  # Disk-backed B+ Tree (metadata p0, splits, leaf chain)
│   │   ├── lsm_tree.py                    # WAL, SkipList MemTable, SSTable, BloomFilter, Compaction
│   │   └── index_advisor.py               # WorkloadAdvisor sliding-window (alpha, beta) selector
│   │
│   └── query/                             # Layer 3: M2 Metadata Catalog & Safe Query Shell
│       ├── __init__.py
│       ├── catalog.py                     # Persistent atomic catalog.json manager & table stats
│       └── simple_executor.py             # Parameterized SQL-like parser & index/scan router
│
├── tests/                                 # Automated Adversarial Gate Suites
│   ├── __init__.py
│   ├── test_m1_storage.py                 # Gate 1: 10 tests covering M1 storage, BPM & restart
│   └── test_m2_indexing.py                # Gate 2: 10 tests covering M2 B+Tree, LSM, Advisor, AES
│
├── bench/                                 # Reproducible Workload & Benchmark Suite
│   ├── __init__.py
│   ├── retail_loader.py                   # Deterministic Retail dataset generator (seed=42)
│   └── run_m1_m2_benchmarks.py            # 1-warmup + 5-run benchmark harness & plot generator
│
├── data/                                  # Runtime Database Storage Directory (gitignored)
│   ├── catalog.json                       # Persistent schema & index catalog
│   ├── tables/                            # *.db (row heaps) and *.col (columnar files)
│   ├── indexes/                           # *.bpt (B+ Tree files) and lsm_<name>/ (WAL + SSTables)
│   └── objects/                           # objects.dat (encrypted blobs) + objects.dir
│
└── results/                               # Reproducible Benchmark Artifacts
    └── m1_m2/
        ├── env_metadata.json              # CPU, RAM, OS, Python/NumPy versions, seed
        ├── raw_runs.csv                   # All individual trial timings & counters
        ├── summary_metrics.csv            # Mean, median, std_dev, p95 latency, throughput
        └── plots/                         # 5 high-resolution (300 DPI) PNG comparison figures
```

### 2.2 Module Dependency DAG (Zero Circular Imports)

```text
[src/common/*]  (config, exceptions, types)
      ▲
      ├─────────────────────────┬─────────────────────────┐
      │                         │                         │
[src/storage/*]           [src/index/*]             [src/query/*]
(disk_manager,            (base_index,              (catalog,
 page_header,              bplus_tree,               simple_executor)
 slotted_page,             lsm_tree,                      ▲
 column_page,              index_advisor)                 │
 replacer,                      ▲                         │
 buffer_pool,                   └─────────────────────────┤
 table_heap,                                              │
 cipher_hooks,                                            │
 object_store,                                            │
 storage_engine) ─────────────────────────────────────────┘
```

### 2.3 Module & Class Responsibility Matrix

| File Path | Primary Classes | Key Responsibilities | Lock Domain |
| :--- | :--- | :--- | :--- |
| `src/common/types.py` | `DataType`, `RID`, `ColumnDef`, `Schema` | Binary row serialization/deserialization, fixed-width key encoding, and 8-byte `RID` packing. | Stateless |
| `src/storage/disk_manager.py` | `DiskManager` | Manages single binary file, Page 0 FSM allocation bitmap, `8,192` B aligned I/O, and 64-bit page checksums. | `DiskManagerLock` |
| `src/storage/page_header.py` | `PageType`, `PageFlags`, `PageHeader` | Packs/unpacks the universal `32`-byte page header and validates page integrity hashes. | Stateless |
| `src/storage/slotted_page.py` | `SlottedPage` | Manages variable-length row tuples, `4`-byte slot entries, `16`-byte MVCC headers, and in-page `compact()`. | Caller Frame Pin |
| `src/storage/column_page.py` | `ColumnPage` | Stores contiguous typed column arrays (`1,020` rows/page for 8 B types) and exposes zero-copy `np.ndarray` views. | Caller Frame Pin |
| `src/storage/replacer.py` | `ReplacementPolicy`, `LRUReplacer`, `FIFOReplacer` | $O(1)$ eviction victim selection over unpinned frames using `OrderedDict`. | `BufferPoolLock` |
| `src/storage/buffer_pool.py` | `FrameHeader`, `BufferPoolManager` | Manages memory frames, pin counts, dirty write-back, and cache hit/miss telemetry. | `BufferPoolLock` |
| `src/storage/table_heap.py` | `RowTableHeap`, `ColumnTableHeap` | Multi-page CRUD, full table scans, and aligned multi-column batch ingestion. | `TableHeapLock` |
| `src/storage/cipher_hooks.py` | `CipherHook`, `NoOpCipher`, `AES256GCMHook` | Pluggable encryption interface with authenticated `oid` binding (`AES-256-GCM`). | Stateless |
| `src/storage/object_store.py` | `ObjectStore`, `ObjectMeta` | Appends encrypted binary blobs to `objects.dat` and persists `oid` offsets in `objects.dir`. | `ObjectStoreLock` |
| `src/storage/storage_engine.py` | `StorageEngine` | Unified facade managing lifecycle (`open`, `flush_all`, `close`) across tables and object store. | Coordinates sub-locks |
| `src/index/bplus_tree.py` | `BPlusTreeIndex` | Buffer-pool-backed B+ Tree supporting equality lookups, leaf-linked range scans, and recursive node splits. | `IndexLock` |
| `src/index/lsm_tree.py` | `LSMTreeIndex`, `MemTable`, `SSTable`, `BloomFilter` | Write-ahead logged `MemTable`, immutable tiered `SSTable`s, Bloom filters, and size-tiered compaction. | `IndexLock` |
| `src/index/index_advisor.py` | `WorkloadAdvisor`, `Recommendation` | Sliding-window telemetry collector computing $(\alpha, \beta)$ ratios to select `FULL_SCAN`, `BPLUS_TREE`, or `LSM_TREE`. | `IndexLock` |
| `src/query/catalog.py` | `CatalogManager` | Atomically persists table schemas, row/page statistics, and index metadata in `data/catalog.json`. | `CatalogLock` |
| `src/query/simple_executor.py` | `SimpleExecutor` | Parses parameterized SQL-like queries, blocks SQL injection, and routes execution through indexes or scans. | `CatalogLock` |

---

## 3. Milestone 1: Storage Engine & Buffer Pool Architecture

### 3.1 Physical File Organization & Free Space Map (FSM)
Each relational table heap (`*.db`), column file (`*.col`), and B+ Tree file (`*.bpt`) begins with **Page 0** acting as a persistent **Free Space Map (FSM)**:
* Bytes `0..31`: `PageHeader` (`page_id = 0`, `page_type = 0`).
* Bytes `32..8191` (`8,160` bytes = `65,280` bits): Bit $i$ indicates whether physical `page_id = i` is currently allocated (`1`) or free (`0`). Bit `0` is permanently `1`.
* `DiskManager.allocate_page()` scans for the lowest zero bit $i \ge 1$, sets it to `1`, writes `8,192` zeroed bytes at offset $i \times 8,192$ if extending the file, syncs Page 0, and returns `i`.

### 3.2 Universal 32-Byte Binary Page Header (`<IBBHQIIQ`)

```text
Byte Offset:  0       4   5   6       8              16             20             24             32
             +-------+---+---+-------+--------------+--------------+--------------+--------------+
             |page_id|typ|flg|slots  |   page_lsn   |free_space_ptr| next_page_id |   checksum   |
             | (4B)  |1B |1B | (2B)  |     (8B)     |     (4B)     |     (4B)     |     (8B)     |
             +-------+---+---+-------+--------------+--------------+--------------+--------------+
```

* **`checksum` (`bytes 24..31`):** Computed as a 64-bit BLAKE2b digest (`hashlib.blake2b(header[0:24] + body[32:8192], digest_size=8)`) on every `DiskManager.write_page()` and verified on every `DiskManager.read_page()`. Any single-bit corruption on disk immediately triggers `PageCorruptedError`.

### 3.3 Dual Storage Layouts: `SlottedPage` (Row) vs. `ColumnPage` (Column)

```text
[A] SLOTTED ROW PAGE (page_type = 1) — Used by customers, products, inventory, orders
0              32                     32 + 4*N                free_space_ptr                   8192
+--------------+----------------------+-----------------------+--------------------------------+
| PageHeader   | Slot 0 | Slot 1 |... |   Contiguous Free     | Tuple N-1 | ... | Tuple 0      |
| (32 Bytes)   | (4B ea: off, len)    |   Space               | [16B MVCC Header + User Data]  |
+--------------+----------------------+-----------------------+--------------------------------+
               |---- Grows Downward -->                       <-- Grows Upward ----------------|

[B] COLUMNAR CHUNK PAGE (page_type = 2) — Used by order_items (*.col files)
0              32                                      32 + N * item_size                      8192
+--------------+---------------------------------------+---------------------------------------+
| PageHeader   | val[0] | val[1] | val[2] | ... val[N-1] |          Unused Trailing Bytes        |
| (32 Bytes)   | <--- Zero-Copy np.frombuffer() View --->|                                       |
+--------------+---------------------------------------+---------------------------------------+
  Capacity: N = 1,020 elements for 8-byte types (INT64, FLOAT64); N = 2,040 for 4-byte (INT32)
```

* **Immutable `RID` Invariant in `SlottedPage`:** Deleting a tuple sets its slot entry to `(offset=0, length=0)` without shifting subsequent slot indices. When `SlottedPage.compact()` defragments live tuples toward offset `8192`, it updates only `tuple_offset` inside each slot entry, preserving `(page_id, slot_id)` for all external B+ Tree and LSM Tree indexes.
* **Aligned Multi-Column Chunks in `ColumnTableHeap`:** `order_items` stores 5 separate `.col` files (`order_id.col`, `item_seq.col`, `product_id.col`, `qty.col`, `unit_price.col`). Every column page holds up to `CHUNK_ROWS = 1,020` values (matching the 8-byte capacity of `INT64`/`FLOAT64`), ensuring logical row $r$ resides at identical `(page_idx = r // 1020 + 1, slot_id = r % 1020)` coordinates across all 5 files.

### 3.4 Buffer Pool Pinning & Eviction State Machine

```text
                    new_page() / fetch_page() [MISS]
  +------------+ --------------------------------------> +-------------------+
  | FREE FRAME |                                         |  PINNED IN POOL   |
  +------------+ <-------------------------------------- +-------------------+
                    flush & reset on table drop            |   ^           |
                                                           |   |           |
                                           fetch_page()    |   |           | unpin_page()
                                           [HIT: pin++]    |   |           | [pin-- > 0]
                                                           +---+           |
                                                                           | unpin_page()
                                                                           | [pin == 0]
                                                                           v
  +------------+  victim() selected & is_dirty == True   +-------------------+
  | DISK WRITE | <-------------------------------------- | UNPINNED IN       |
  | (pwrite)   | --------------------------------------> | REPLACER          |
  +------------+  reassigned to new page_id (pin = 1)    | (LRU or FIFO)     |
                                                         +-------------------+
```

* **Pin Exhaustion Safety:** If `fetch_page()` encounters a cache miss and `len(free_list) == 0` while `replacer.size() == 0` (meaning 100% of frames have `pin_count > 0`), `BufferPoolManager` immediately raises `BufferPoolFullError` instead of corrupting active memory.

---

## 4. Milestone 2: Encrypted Object Store & Advanced Indexing Architecture

### 4.1 Encrypted Object Store (`objects.dat` + `objects.dir`)

Product images (`10 KB` to `250 KB`) bypass fixed `8 KB` relational pages and are persisted inside an append-oriented binary extent store managed by `StorageEngine.object_store`:

```text
objects.dat (Binary Extent Stream):
+--------+-----------+---------+------------+-----------+----------+-------------------+--------+
| Magic  |    OID    | raw_len | cipher_len |   Nonce   | GCM Tag  |    Ciphertext     | CRC32  |
| "OBJ1" |  (8 B)    |  (4 B)  |   (4 B)    |  (12 B)   |  (16 B)  |  (cipher_len B)   | (4 B)  |
+--------+-----------+---------+------------+-----------+----------+-------------------+--------+
<-------------------- 48-Byte Extent Header -------------------->                       <Trailer>

objects.dir (Persistent OID Directory Table):
+--------+-------------+-------------------------------------------------------------------------+
| Magic  | Entry Count | Entry 0 (24B: oid, offset, cipher_len, raw_len) | Entry 1 (24B) | ...   |
| "ODIR" |   (4 B)     |                                                                         |
+--------+-------------+-------------------------------------------------------------------------+
```

* **Authenticated Encryption Binding:** `AES256GCMHook` generates a fresh `96`-bit random `nonce` (`os.urandom(12)`) per object write and passes `aad = struct.pack("<Q", oid)` into `AESGCM.encrypt()`. This prevents both ciphertext inspection on disk and cross-object swapping attacks.

### 4.2 Disk-Backed B+ Tree Index (`*.bpt`)

Every B+ Tree node lives inside an `8,192`-byte buffer-pool page, giving high fanout ($M_{\text{leaf}} \approx 340$ for 8-byte keys + 8-byte `RID`s) and keeping tree height $\le 3$ for millions of records.

```text
                        [Page 0: Metadata]
                        root_page_id = 5, height = 3
                                    |
                                    v
                     +-----------------------------+
                     | Internal Node (Page 5, t=3) |
                     | [P1 | key=500 | P2 | key=1200 | P3]
                     +----+----------+---------+---+
                          |          |         |
            +-------------+          |         +-------------+
            v                        v                       v
+-----------------------+  +-----------------------+  +-----------------------+
| Leaf Node (Page 1,t=4)|  | Leaf Node (Page 2,t=4)|  | Leaf Node (Page 3,t=4)|
| (10,RID) (45,RID) ... |->| (500,RID) (720,RID)...|->| (1200,RID) ...        |
| next_page_id = 2      |  | next_page_id = 3      |  | next_page_id = 0xFFFF |
+-----------------------+  +-----------------------+  +-----------------------+
```

* **Order-Preserving Key Serialization:** Signed integers (`INT32`, `INT64`) flip their sign bit and pack big-endian (`>Q`), and `FLOAT64` values apply IEEE-754 sign-bit normalization so raw byte comparison (`key_a < key_b`) matches exact numeric ordering. Appending the 8-byte `RID` (`>II`) makes every composite key unique, handling duplicate secondary keys (such as `products.price`) cleanly.
* **Latch-Crabbing Pin Discipline:** Top-down tree traversal unpins each parent page immediately after pinning its child page, guaranteeing at most `2` pages are pinned simultaneously during lookups and range scans.

### 4.3 Three-Tier Write-Optimized LSM Tree Index (`data/indexes/<name>/`)

```text
  insert(key, rid) / delete(key)
         │
         ├──► [1] Append to lsm.wal (CRC32 + OpType + Key + RID) [Crash Durability]
         │
         └──► [2] Insert into Active MemTable (Sorted SkipList / Buffer, Cap = 1 MB)
                       │
                       │ (When memtable_bytes >= 1 MB: Flush sequentially)
                       v
              ┌─────────────────────────────────────────────────────────┐
              │ Level L0:  [SST_001]   [SST_002]   [SST_003]   [SST_004]│ (Overlapping Key Ranges)
              └───────────────────────────┬─────────────────────────────┘
                                          │ Size-Tiered Compaction (Trigger = 4 SSTables)
                                          │ heapq.merge + Deduplicate + Purge Bottom Tombstones
                                          v
              ┌─────────────────────────────────────────────────────────┐
              │ Level L1:  [SST_005 (Merged & Deduplicated)]            │
              └─────────────────────────────────────────────────────────┘

  Single SSTable File Layout:
  +-------------------+-------------------+--------------+--------------------+--------------+
  | Data Block 0 (8KB)| Data Block 1 (8KB)| Bloom Filter | Sparse Fence Index | 64B Footer   |
  | (Sorted Entries)  | (Sorted Entries)  | (10 bits/key)| (Block First Keys) | "CSAI_SST"   |
  +-------------------+-------------------+--------------+--------------------+--------------+
```

### 4.4 Quantitative WorkloadAdvisor Decision Boundaries

`WorkloadAdvisor` maintains a sliding window of the last $W = 1,000$ operations per `(table, column)` and computes write intensity $\alpha = \frac{U}{N_{\text{ops}}}$ and range-read ratio $\beta = \frac{R_{\text{rng}}}{R_{\text{eq}} + R_{\text{rng}}}$:

```text
  beta (Range Ratio)
  1.0 +-------------------------------------------------------+
      |                                                       |
      |                    BPLUS_TREE                         |
 0.25 + - - - - - - - - - - - - - - +-------------------------+
      |                             |                         |
      |        BPLUS_TREE           |        LSM_TREE         |
      |   (Read-Heavy / Balanced)   |  (Write-Heavy Point)    |
  0.0 +-----------------------------+-------------------------+----> alpha (Write Ratio)
     0.0                           0.45                      1.0
     * Override: If N_ops < 200 or table_pages < 4 -> FULL_SCAN
```

---

## 5. End-to-End Execution Data Flows

### 5.1 Flow A: Parameterized Indexed Range Query (`BPlusTreeIndex`)
1. Client calls `executor.execute("SELECT * FROM products WHERE price BETWEEN ? AND ?", (50.0, 150.0))`.
2. `SimpleExecutor` verifies parameter placeholders, rejects unparameterized string literals, and records `RANGE_READ` in `WorkloadAdvisor`.
3. `SimpleExecutor` queries `idx_products_price.range_scan(50.0, 150.0)`.
4. `BPlusTreeIndex` traverses from `root_page_id` down to the first leaf containing `50.0` (unpinning internal pages along the path), then follows `next_page_id` sibling pointers across leaves until `price > 150.0`, collecting matching `RID(page_id, slot_id)` pairs.
5. `RowTableHeap.get(rid)` pins each target data page in `BufferPoolManager`, extracts the tuple via `SlottedPage.get_tuple(slot_id)`, deserializes columns via `Schema`, and unpins the data page.

### 5.2 Flow B: High-Frequency Inventory Update (`LSMTreeIndex`)
1. Client calls `executor.execute("UPDATE inventory SET stock_qty = ? WHERE product_id = ?", (42, 1001))`.
2. `RowTableHeap` updates the tuple in `SlottedPage` (marking the buffer pool frame dirty) and returns its `RID`.
3. `LSMTreeIndex.insert(1001, rid)` appends the mutation record to `lsm.wal` and inserts `(1001, rid)` into the in-memory `MemTable` in $O(\log M)$ time without performing random disk seeks.
4. When `MemTable` reaches `1 MB`, it flushes sequentially to a new `L0` `SSTable` with an embedded Bloom filter and fence index, updates `manifest.json` atomically, and truncates `lsm.wal`.

### 5.3 Flow C: Zero-Copy Columnar Aggregation (`ColumnTableHeap`)
1. Analytical benchmark executes `SUM(qty * unit_price)` over `200,000` `order_items` rows.
2. `ColumnTableHeap` iterates through the `197` data pages of `order_items.qty.col` and `order_items.unit_price.col` (`1,020` rows per page).
3. For each page pair, `ColumnPage.as_numpy_array()` wraps the raw `memoryview` slice (`bytes 32..8192`) in a zero-copy `np.ndarray` (`arr.flags['OWNDATA'] == False`) and computes `float(np.dot(qty_arr, price_arr))`, unpinning both pages immediately afterward.

---

## 6. Concurrency, Durability & Crash-Safety Invariants

### 6.1 Strict Top-Down Lock Acquisition Order
All multi-component operations acquire `threading.RLock` instances strictly in the following order to guarantee deadlock freedom:

$$\text{CatalogLock} \rightarrow \text{IndexLock} \rightarrow \text{TableHeapLock} \rightarrow \text{BufferPoolLock} \rightarrow \text{DiskManagerLock}$$

### 6.2 Crash & Restart Durability Guarantees
* **Relational & B+ Tree Pages:** `StorageEngine.flush_all()` flushes all dirty frames in `BufferPoolManager`, stamps 64-bit BLAKE2b checksums on every page, and issues `os.fsync(fd)` on every open `DiskManager`.
* **LSM Tree Unflushed Writes:** Every `MemTable` mutation writes a CRC32-protected record to `lsm.wal` before updating memory. On restart after an abrupt crash, `LSMTreeIndex` replays valid `lsm.wal` records while discarding any torn trailing record whose CRC32 check fails.
* **Atomic Metadata Updates:** `catalog.json` and LSM `manifest.json` are never overwritten in place; updates write to `<file>.tmp`, call `os.fsync()`, and atomically swap via `os.replace()`.