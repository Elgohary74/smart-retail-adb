# M1/M2 checklist

## M1
- [x] Custom persistent page file (`PageFile`) and 24-byte `PageHeader`
- [x] Shared DB-wide `BufferPool` with `FIFO`/`LRU`, `pin`/`unpin`, and `lsn` hook
- [x] Typed binary `RowTable` (slotted pages + overflow) and `ColumnTable` (projection pushdown + batches)
- [x] Reproducible 5-run benchmarks: FIFO vs. LRU, Row vs. Column, `page` vs. `objects` vs. AES-GCM
- [x] Automated unit tests across schema, backends, buffer pool, row table, and column table

## M2
- [x] `LocalObjectStore` backend with cached page count
- [x] Page-aware `TransformingBackend` encryption hooks + `make_aes_gcm_hooks`
- [x] Persistent page-backed `BPlusTree` (equality, range, delete, duplicate RID buckets)
- [x] Disk-backed SSTable `LSMTree` (Bloom filters, fence pointers, WAL, tombstones, atomic compaction)
- [x] Workload telemetry & `choose_index_type` heuristic
- [x] Safe SQL/DML subset (`SELECT` projection/range/`BETWEEN`/`IS NULL`, `INSERT`, `UPDATE`, `DELETE`)
- [x] Reproducible 5-run M2 benchmarks: indexed vs. full scan, read-heavy vs. write-heavy, range, index size

## Before formal submission
- [ ] Run benchmarks on the final target machine with `--plots` and archive CSVs/charts in the report
- [ ] Record OS, Python version, CPU/RAM, seed, commands, and median/p95 latencies
