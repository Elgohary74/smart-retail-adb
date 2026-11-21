# M1/M2 checklist

## M1
- [x] Custom persistent page file
- [x] Buffer pool with configurable FIFO/LRU
- [x] Row and column page encodings
- [x] Counters for cache hits/misses and disk activity
- [x] Reproducible FIFO-vs-LRU benchmark script
- [x] Automated storage and layout tests

## M2
- [x] Common page backend interface and local object-per-page backend
- [x] Optional encryption/decryption hooks (not encryption by default)
- [x] B+ Tree point/range lookup
- [x] LSM memtable, sorted disk runs, point/range lookup, compaction
- [x] Workload heuristic for selecting B+ Tree vs LSM
- [x] Simple indexed equality query path
- [x] B+ Tree/LSM unit tests and an index benchmark script

## Before formal submission
- [ ] Run tests and benchmarks on the target machine
- [ ] Record OS, Python version, machine specs, seed, commands, and repeated-run median/p95 where applicable
- [ ] Include actual output tables and charts; never invent benchmark measurements
