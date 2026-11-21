# Architecture decisions — M1/M2

1. **Page size:** fixed 4,096 bytes, with a four-byte payload-length header in data pages.
2. **Persistent storage:** `PageFile` stores pages contiguously in one binary file. `LocalObjectStore` exposes the same interface, but stores each page in a separate local object file.
3. **Buffer pool:** FIFO and LRU use an ordered frame map; dirty pages are written on eviction or explicit flush. Metrics are exposed through `BufferPool.snapshot()`.
4. **Record layouts:** row format stores a list of records; column format stores one list per field. Both are encoded as JSON payloads inside a fixed-size page.
5. **Record locators:** `(page_id, row_offset)` allows indexes to locate a row without scanning the table.
6. **B+ Tree:** keys and locators are held in leaf nodes; leaf links enable range scans. The implementation is rebuildable from storage.
7. **LSM Tree:** writes go to a memtable. At a configurable threshold, data is sorted and written to immutable JSON Lines runs. Compaction merges runs.
8. **Auto-selection heuristic:** choose LSM if `writes > 2 * max(1, reads)`; otherwise choose B+ Tree. Benchmark against representative workload traces and tune the rule if results support it.
9. **Restricted query parser:** supports only `SELECT * FROM table` and equality predicates with scalar literals. Values are parsed with `ast.literal_eval`, not executed as SQL or Python code.
10. **Security boundary:** encryption is exposed through same-length transform hooks; actual encryption must be supplied externally. The hooks are not cryptography by themselves.
