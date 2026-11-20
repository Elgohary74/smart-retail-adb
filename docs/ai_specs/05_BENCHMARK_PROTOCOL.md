# 05 — Reproducible Benchmark & Visualization Protocol (Wave 5)

## 1. Mandatory Measurement Protocol (CSAI 302 Standard)

Every experiment in `bench/run_m1_m2_benchmarks.py` must strictly follow the course assessment protocol:
1. **Environment & Configuration Capture:** Before running benchmarks, write `results/m1_m2/env_metadata.json` containing:
   * OS name/version (`platform.platform()`), CPU architecture/cores (`os.cpu_count()`), Python version (`platform.python_version()`), NumPy version, `PAGE_SIZE = 8192`, `random_seed = 42`, and dataset row counts.
2. **Warm-Up + 5 Measured Trials:**
   * For every configuration point, execute **1 untimed warm-up run** (discarded to stabilize OS page cache and JIT/import overhead) followed by **5 independent timed runs** (`runs = 5`).
3. **Required Statistical Aggregations:**
   * Record every raw run in `results/m1_m2/raw_runs.csv`.
   * Compute **mean**, **median**, **std_dev**, **p95 latency (ms)** (`np.percentile(latencies, 95)`), and **throughput (ops/sec)** in `results/m1_m2/summary_metrics.csv`.
4. **Correctness Assertions Inside Benchmarks:**
   * Every benchmark function must assert result correctness (e.g., exact checksum/sum match or exact matching `RID` count) before recording timing so broken fast paths are never plotted.

---

## 2. Controlled Benchmark Suite (6 Experiments)

### Experiment 1A: Buffer Pool Replacement Policy (`LRU` vs. `FIFO`)
* **Dataset:** `10,000` `products` rows (~`96` pages of `8,192` bytes).
* **Independent Variables:**
  * Policy: `LRUReplacer` vs. `FIFOReplacer`.
  * Pool Size (`frames`): `[16, 32, 64, 128]`.
  * Access Pattern (`5,000` page fetches per run):
    1. *Zipfian Skew ($\theta = 0.9$)* — hot products requested repeatedly.
    2. *Sequential Scan* — full table sweep repeated 3 times.
* **Metrics:** Cache hit ratio ($\frac{\text{hits}}{\text{hits} + \text{misses}}$), disk reads, disk writes, and wall-clock time (ms).

### Experiment 1B: Storage Layout (`SlottedPage` Row vs. `ColumnPage` Column)
* **Dataset:** `200,000` `order_items` records stored in both a `RowTableHeap` and a `ColumnTableHeap`.
* **Workloads:**
  1. *Analytical Column Aggregation:* Compute `SUM(qty * unit_price)` across all `200,000` records.
  2. *Full-Tuple Point Lookup:* Fetch all 5 columns for `5,000` random `RID`s.
* **Metrics:** Wall-clock execution time (ms), total pages read from buffer pool, and throughput (rows/sec).

### Experiment 2A: Index Read-Heavy Workload (`90% Reads / 10% Writes`)
* **Dataset:** `50,000` `products` records (`product_id` keys).
* **Access Methods:** `FullTableScan` vs. `BPlusTreeIndex` vs. `LSMTreeIndex` (`1 MB` MemTable).
* **Workload:** `10,000` operations (`7,000` point lookups, `2,000` narrow range scans of width 20, `1,000` new product inserts).
* **Metrics:** Mean & p95 operation latency (ms), total throughput (ops/sec), index build time (ms), and on-disk index size (MB).

### Experiment 2B: Index Write-Heavy Workload (`10% Reads / 90% Writes`)
* **Dataset:** `50,000` `inventory` records (`product_id` keys).
* **Access Methods:** `FullTableScan` vs. `BPlusTreeIndex` vs. `LSMTreeIndex`.
* **Workload:** `20,000` operations (`18,000` stock updates/upserts, `2,000` point stock checks).
* **Metrics:** Mean & p95 write latency (ms), write throughput (ops/sec), and compaction/flush overhead.

### Experiment 2C: WorkloadAdvisor Dynamic Adaptation Trace
* **Workload:** Feed a 3-phase workload trace (`Phase 1: 150 ops cold start` $\rightarrow$ `Phase 2: 2,000 ops 90% inventory writes` $\rightarrow$ `Phase 3: 2,000 ops 85% price range scans`) into `WorkloadAdvisor`.
* **Metrics:** Log $(\alpha, \beta)$ trajectory and selected strategy transitions (`FULL_SCAN` $\rightarrow$ `LSM_TREE` $\rightarrow$ `BPLUS_TREE`).

### Experiment 2D: Object Store Encryption Overhead (`NoOpCipher` vs. `AES256GCMHook`)
* **Dataset:** `500` synthetic product images (`16 KB` to `128 KB` each, ~`35 MB` total).
* **Metrics:** `put_object` write throughput (MB/s), `get_object` read throughput (MB/s), and storage space overhead (%).

---

## 3. Required Output Charts (`results/m1_m2/plots/`)

Generate 5 high-resolution (`300 DPI`) labeled `matplotlib` figures with legends, axis units, and error bars ($\pm 1\text{ std}$):
1. `fig1_buffer_pool_hit_ratio.png`: Cache Hit Ratio (%) vs. Buffer Pool Size (`16..128` frames) comparing `LRU` and `FIFO` under Zipfian and Sequential workloads.
2. `fig2_row_vs_column_latency.png`: Grouped bar chart comparing `SlottedPage (Row)` vs. `ColumnPage (Column)` latency (ms) on OLAP aggregation vs. OLTP point lookups.
3. `fig3_index_read_vs_write_throughput.png`: Side-by-side throughput (ops/sec) for `Full Scan`, `B+ Tree`, and `LSM Tree` under Read-Heavy (90/10) and Write-Heavy (10/90) workloads.
4. `fig4_index_p95_latency_and_size.png`: Dual-panel chart showing p95 latency (ms) and disk footprint (MB) across `B+ Tree` vs. `LSM Tree`.
5. `fig5_encryption_throughput.png`: Bar chart comparing `NoOpCipher` vs. `AES256GCMHook` read/write throughput (MB/s) on product images.