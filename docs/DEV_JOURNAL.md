# Development Journal — `csai302_db` (Milestones 1 & 2)

Chronological engineering log documenting legacy audits, specification conflict resolutions, bugs, bottlenecks, and debugging guardrails across Waves 0–5 (per `docs/docify.md`).

---

### [Wave 0 | Legacy Audit] Replacing 4 KB JSON Prototype with Spec-Compliant 8 KB Binary Engine
* **Symptom / Failing Test:** Pre-refactor audit of `src/smartdb/` revealed fundamental architectural divergence from `docs/ai_specs/00_AGENT_GOVERNANCE.md` through `06_M3_TO_M6_HOOKS.md`.
* **Root Cause Analysis:**
  * `src/smartdb/storage/page_file.py` used `PAGE_SIZE = 4096` with raw file append (`page_count`) rather than `8,192`-byte pages, universal `32`-byte headers, BLAKE2b checksums, or a Page-0 Free Space Map.
  * `src/smartdb/storage/layouts.py` packed rows and columns as UTF-8 JSON (`json.dumps` behind a 4-byte length prefix) instead of binary `SlottedPage` tuples with `16`-byte MVCC headers or zero-copy NumPy `ColumnPage` buffers.
  * `src/smartdb/indexes/lsm_tree.py` persisted text `run_*.jsonl` files without a CRC32 WAL, binary `SSTable` blocks, or Bloom filters; `src/smartdb/storage/encryption.py` only offered identity lambda transforms rather than `AES-256-GCM` extent storage.
  * `pyproject.toml` targeted `smart-retail-db` on Python `>=3.10` with zero dependencies and legacy `benchmarks/` / `workloads/` scripts.
* **Resolution:** Removed the legacy `src/smartdb/` package, legacy `tests/`, `benchmarks/`, `workloads/`, `retail_benchmark_data/`, and `src/smart_retail_db.egg-info/`. Created a clean Python `3.14` `.venv` with `numpy`, `cryptography`, `pytest`, and `matplotlib`, updated `pyproject.toml` (`csai302-db`, `requires-python = ">=3.12"`), and aligned `docs/architecture.md` and `docs/ai_specs/00_AGENT_GOVERNANCE.md`.
* **Lesson / Guardrail Added:** Wave 0 freezes exact binary contracts, standardized exceptions, and all Gate 1 + Gate 2 adversarial tests before any storage or index module is implemented.

### [Wave 0 | Binary Spec Reconciliation] Resolving B+ Tree Page 0 FSM Collision and SSTable Footer Size Conflict
* **Symptom / Failing Test:** Cross-checking `docs/ai_specs/03_INDEXING_AND_QUERY_SPEC.md` against `02_STORAGE_AND_OBJECT_SPEC.md` prior to coding exposed two struct-packing mismatches in `03_INDEXING_AND_QUERY_SPEC.md`.
* **Root Cause Analysis:**
  1. **B+ Tree Page 0 Metadata vs. FSM Offset:** Section 2.1 specified bytes `32..55` (`24` bytes) with format `<IBBHQ` (`16` bytes: `uint32, uint8, uint8, uint16, uint64`), mis-sizing `key_size` (`uint16`, not `uint8`) and leaving no room for the `unique` flag while colliding with `DEFAULT_FSM_OFFSET = 32`.
  2. **SSTable 64-Byte EOF Footer:** Section 3.2 specified a "64-Byte Footer at EOF" with format `<8sQIIQIQQI12s`, which sums to `68` bytes (`8+8+4+4+8+4+8+8+4+12`), and omitted `bloom_num_bits` from the prose field list.
* **Resolution:**
  1. Standardized B+ Tree Page 0 metadata at bytes `32..55` to `<IBHHQ?6x` (`24` bytes: `root_page_id`, `key_dtype`, `key_size`, `height`, `entry_count`, `unique`, 6-byte padding) and set `BPLUS_FSM_OFFSET = 64` so the FSM occupies bytes `64..8191` (`65,024` pages).
  2. Standardized the SSTable EOF footer to `<8sQIIQIQQI8s` (exact `64` bytes: `magic`, `bloom_offset`, `bloom_len`, `num_hashes`, `index_offset`, `index_len`, `entry_count`, `bloom_num_bits`, `crc32`, `8B` padding).
  Both clarifications are documented in `docs/architecture.md`.
* **Lesson / Guardrail Added:** Every `struct.Struct(...).size` in `src/` must match its documented byte span in `docs/architecture.md` and class ASCII diagrams, verified by Gate 1/2 binary packing tests.
