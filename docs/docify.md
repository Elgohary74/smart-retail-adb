# `docify.md` — Dual-Audience Documentation & Future-Handoff Protocol

**Target System:** `csai302_db` (Custom Relational & Object DBMS — Retail Orders & Product Search)  
**Target Consumers:**
1. **Human Stakeholders:** Team members, peer reviewers, and CSAI 302 instructors evaluating code clarity, design rationale, reproducibility, and the Week 15 Formal Technical Report.
2. **Future AI Harness Waves (M3–M6):** Autonomous agents implementing Query Processing (M3), MVCC & ARIES Recovery (M4), Distributed Raft/2PC & Vector Search (M5), and Full Integration (M6) without breaking M1–M2 invariants.

---

## 1. Documentation Strategy & Cognitive Bias Controls

### 1.1 Parallel Evaluation of AI Documentation Approaches

| Documentation Strategy | Mechanism | Failure Mode | Verdict |
| :--- | :--- | :--- | :--- |
| **Approach 1: Post-Hoc Summary Dump** | Ask the AI to generate a README and report after all code and benchmarks finish. | **High.** Suffers from hindsight bias; omits failed attempts, subtle concurrency invariants, and exact binary offsets. | **Reject** |
| **Approach 2: Line-by-Line Narration** | Comment every line of Python code explaining basic syntax (`# increment counter`). | **High.** Bloats token context for future AI waves, obscures complex algorithms, and drifts out of sync immediately. | **Reject** |
| **Approach 3: Dual-Audience Living Docs-as-Code** | Embed ASCII binary diagrams and invariant contracts in docstrings, and update structured ADRs, dev journals, and M3–M6 handoff specs at each Wave Gate. | Requires strict templates and gate verification at the end of Waves 2, 4, and 5. | **Adopt** |

### 1.2 Mitigations Against AI Documentation Biases
* **Anti-Hindsight & Anti-Sanitization Bias:** AI agents naturally conceal bugs they fixed during test iterations, presenting code as if it worked on the first try. However, the CSAI 302 Formal Technical Report explicitly requires documenting *"bottlenecks, trade-offs, bugs and debugging decisions, failed attempts and lessons learned."* Whenever a Wave Gate test fails and is subsequently fixed, the AI **must** log the root cause and fix in `docs/DEV_JOURNAL.md`.
* **Anti-Curse-of-Knowledge Bias:** Future AI waves will not have the full chat transcript from M1–M2. Every public interface, reserved byte offset, and lock-ordering invariant must be explicitly codified in `docs/handoff/M3_M6_HANDOFF_CONTRACT.md` so future waves can consume M1–M2 as a black-box SDK.
* **Anti-Drift Rule:** Documentation is treated as a build artifact. No Wave Gate (`Gate 1`, `Gate 2`, `Gate 3`) is considered passed until its corresponding documentation deliverables are updated to match the exact committed code.

---

## 2. In-Code Documentation Standards (Python 3.12+)

### 2.1 Module-Level Header Template
Every `.py` file under `src/`, `tests/`, and `bench/` must start with a structured module docstring specifying its architectural layer, responsibilities, lock domain, and key invariants:

```python
"""
Module: src/storage/slotted_page.py
Layer: Storage Engine (Milestone 1 — Row-Oriented Page Layout)
Lock Domain: Caller must hold an active buffer pool pin on the underlying frame.

Purpose:
    Implements an 8,192-byte slotted page for variable-length and fixed-length
    relational tuples (customers, products, inventory, orders).

Key Invariants:
    1. Slot IDs are immutable once assigned; deletions zero out (offset, length)
       without shifting subsequent slot indices so external RIDs remain valid.
    2. Every tuple begins with a 16-byte MVCC header (xmin: uint64, xmax: uint64)
       reserved for Milestone 4 Snapshot Isolation.
"""
```

### 2.2 Mandatory ASCII Binary Layout Diagrams
Any class that packs or unpacks raw bytes (`PageHeader`, `SlottedPage`, `ColumnPage`, `ObjectStore`, `BPlusTreeIndex`, `LSMTreeIndex`) **must** include an exact byte-offset ASCII diagram inside its class docstring:

```python
class SlottedPage:
    """
    8,192-byte row-oriented page using a forward-growing slot directory and
    backward-growing tuple heap.

    Binary Layout:
    +------------------+----------------------------------+--------------------+
    | PageHeader (32B) | Slot 0 (4B) | Slot 1 (4B) | ...  |   Free Space ...   |
    +------------------+----------------------------------+--------------------+
    0                 32                               32+4N              free_ptr
                                                          +--------------------+
                                                          | ... Packed Tuples  |
                                                          +--------------------+
                                                        free_ptr              8192
    """
```

### 2.3 Public Method Docstring Standard
All public methods and functions must use Google-style docstrings containing **only** high-signal engineering metadata:
* **Summary:** One-sentence imperative description.
* **Args / Returns / Raises:** Explicit types and exact `DatabaseError` subclasses raised (`PageCorruptedError`, `BufferPoolFullError`, `SecurityViolationError`, etc.).
* **Complexity & Concurrency:** Time/space complexity (e.g., $O(\log_B N)$ disk page reads, max `2` pinned frames simultaneously) and thread-safety notes.

### 2.4 Inline Comment Rules ("Explain *Why*, Never *What*")
* **Forbidden:** `# Increment pin count by 1` or `# Pack header using struct`.
* **Required:** Explain non-obvious edge cases, hardware/OS behaviors, and cross-milestone hooks:
  * `# Unpin parent before descending to child to guarantee <= 2 pinned frames per thread.`
  * `# Flip IEEE-754 sign bit so raw big-endian byte comparison preserves negative float ordering.`
  * `# Write xmin=1, xmax=0 now so M4 Snapshot Isolation works without rewriting M1 heap files.`

---

## 3. Repository-Level Living Documentation Artifacts

In addition to `docs/architecture.md` and `docs/ai_specs/*`, the AI harness must generate and maintain **five human-and-agent documentation files** inside the repository:

```text
csai302_db/
├── README.md                                  # Quickstart, architecture map, CLI/bench commands
└── docs/
    ├── architecture.md                        # System architecture & binary formats (already created)
    ├── docify.md                              # This documentation protocol
    ├── DEV_JOURNAL.md                         # Chronological log of bugs, bottlenecks & debugging fixes
    ├── decisions/
    │   └── ADR_M1_M2.md                       # Architecture Decision Records (trade-offs & rationale)
    ├── milestones/
    │   └── M1_M2_TECHNICAL_NARRATIVE.md       # Formal report-ready narrative, results & analysis
    └── handoff/
        └── M3_M6_HANDOFF_CONTRACT.md          # Machine/human SDK specification for M3–M6 waves
```

### 3.1 Artifact 1: `README.md` (Project Front Door)
Must provide an immediate, reproducible entry point for teammates and instructors:
1. **System Overview:** Concise summary of `csai302_db` and the Retail Orders & Product Search workload.
2. **Environment & Installation:** Python 3.12+ setup, virtual environment creation, and `pip install numpy cryptography pytest matplotlib`.
3. **Repository Map:** Clean directory table linking `src/`, `tests/`, `bench/`, `docs/`, and `results/`.
4. **Reproducible Execution Commands:** Copy-pasteable commands to:
   * Run Gate 1 (`pytest -v tests/test_m1_storage.py`)
   * Run Gate 2 (`pytest -v tests/test_m2_indexing.py`)
   * Generate Retail dataset (`python -m bench.retail_loader --scale standard --seed 42`)
   * Execute full M1+M2 benchmarks (`python -m bench.run_m1_m2_benchmarks`)
5. **Rubric Traceability Summary:** Quick table mapping implemented modules to the 30 points of M1 and M2.

### 3.2 Artifact 2: `docs/decisions/ADR_M1_M2.md` (Architecture Decision Records)
Document every major engineering choice using a standardized 4-part ADR template so human reviewers and future AI agents understand *why* alternatives were rejected:
* **ADR-001:** Fixed 8,192-Byte Pages (`struct` + `memoryview`) vs. 4 KB Pages or Python Object Serialization.
* **ADR-002:** Unified 64-Bit `RID(page_id, slot_id)` Across Slotted Row and Columnar Layouts.
* **ADR-003:** $O(1)$ `OrderedDict` Pin-Aware `BufferPoolManager` with Pluggable `LRU` and `FIFO` Policies.
* **ADR-004:** Extent-Based Encrypted `ObjectStore` (`AES-256-GCM` with `oid` AAD Binding) vs. OS-File-Per-Image.
* **ADR-005:** Buffer-Pool-Backed B+ Tree with Order-Preserving Key Encoding vs. In-Memory Tree.
* **ADR-006:** Three-Tier LSM Tree (`WAL` + `MemTable` + `SSTable` + Bloom Filters + Size-Tiered Compaction).
* **ADR-007:** Quantitative Sliding-Window $(\alpha, \beta)$ `WorkloadAdvisor` Index Selection Thresholds.

Each ADR must follow this exact structure:
```markdown
## ADR-00X: <Decision Title>
* **Status:** Accepted (Waves 1–4)
* **Context:** <Problem statement and CSAI 302 requirement>
* **Alternatives Considered:** <Option A, Option B, Option C with pros/cons>
* **Decision & Rationale:** <Chosen approach and quantitative justification>
* **Consequences & M3–M6 Impact:** <How this constrains or enables future milestones>
```

### 3.3 Artifact 3: `docs/DEV_JOURNAL.md` (Debugging, Bottlenecks & Failed Attempts Log)
Directly feeds the *"Discussion: bottlenecks, trade-offs, bugs and debugging decisions, failed attempts and lessons learned"* section of the CSAI 302 Final Report. Whenever the AI encounters a test failure, performance bottleneck, or design adjustment during Waves 0–5, it appends an entry:

```markdown
### [Wave X | Module] <Short Issue Title>
* **Symptom / Failing Test:** <Exact pytest failure or benchmark anomaly>
* **Root Cause Analysis:** <Technical explanation of the bug or bottleneck>
* **Failed Attempt (if any):** <What was tried first and why it failed>
* **Resolution:** <Exact algorithmic or structural fix applied>
* **Lesson / Guardrail Added:** <Test assertion or invariant added to prevent recurrence>
```

### 3.4 Artifact 4: `docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md` (Report-Ready Chapter)
Structured to map 1-to-1 onto the required CSAI 302 Final Technical Report sections for Milestones 1 and 2:
1. **Design & Implementation Narrative (M1 & M2):** Data structures, algorithms, storage formats, and integration mechanics.
2. **Experimental Setup & Reproducibility:** Hardware specs (read from `results/m1_m2/env_metadata.json`), dataset scale, random seed (`42`), and warm-up + 5-run methodology.
3. **Measured Results & Quantitative Analysis:** Embedded Markdown tables summarizing `results/m1_m2/summary_metrics.csv` alongside references to `fig1` through `fig5` in `results/m1_m2/plots/`.
4. **Hypothesis vs. Observation Discussion:** Explicit comparison of theoretical expectations vs. empirical measurements (e.g., where `LSMTreeIndex` outperforms `BPlusTreeIndex` on 90% writes, where `ColumnPage` accelerates OLAP aggregation via zero-copy NumPy, and `AES-256-GCM` throughput cost).
5. **Known Limitations:** Honest engineering bounds (e.g., single-process file locking, fixed `65,280`-page single-FSM capacity per file, non-concurrent background compaction in M2).

### 3.5 Artifact 5: `docs/handoff/M3_M6_HANDOFF_CONTRACT.md` (Future AI Harness SDK)
Written specifically as the **context-injection prompt manual** for future AI waves building Milestones 3–6. It must contain:
1. **Exact Public API Signatures:** Copy-ready Python signatures for `StorageEngine`, `BufferPoolManager`, `RowTableHeap`, `ColumnTableHeap`, `ObjectStore`, `BPlusTreeIndex`, `LSMTreeIndex`, `CatalogManager`, and `SimpleExecutor`.
2. **Step-by-Step Extension Recipes for Future Waves:**
   * **How to Build M3 (Query Engine) on M1–M2:** How to iterate `ColumnTableHeap` pages into `dict[str, np.ndarray]` batches of `1,020` rows; how to read table/index stats from `CatalogManager`; how to fetch rows via `RID` after index scans.
   * **How to Build M4 (MVCC & ARIES) on M1–M2:** How to read/write `xmin`/`xmax` via `SlottedPage.get_tuple_mvcc_header()` / `set_tuple_xmax()`; how to stamp `page_lsn` via `PageHeader.set_page_lsn()`; how to hook WAL flush checks into `BufferPoolManager.flush_page()`.
   * **How to Build M5 (Distributed & Vector Search) on M1–M2:** How to store `float32[384]` product embeddings in a `ColumnTableHeap` (`5` vectors per `8,192`-byte `ColumnPage`) and route shards deterministically using `customer_id` / `region`.
3. **Do-Not-Break Invariants List:** Strict checklist of binary offsets, lock ordering (`CatalogLock -> IndexLock -> TableHeapLock -> BufferPoolLock -> DiskManagerLock`), and pin-count rules that future waves must never violate.

---

## 4. Wave-by-Wave Documentation Execution Schedule

The AI harness must update documentation incrementally at each Wave Gate rather than waiting until the end:

| Execution Wave | Mandatory Documentation Deliverables | Verification Gate |
| :--- | :--- | :--- |
| **Wave 0** (Contracts) | Initialize `README.md`, `docs/DEV_JOURNAL.md`, and complete module/class docstrings in `src/common/*`. | `src/common/*` passes type and docstring inspection. |
| **Wave 2** (Gate 1: M1) | Complete ASCII binary layout docstrings in `src/storage/*`; write `ADR-001` through `ADR-004` in `docs/decisions/ADR_M1_M2.md`; log any M1 debugging fixes in `docs/DEV_JOURNAL.md`. | `pytest -v tests/test_m1_storage.py` passes + M1 ADRs present. |
| **Wave 4** (Gate 2: M2) | Complete ASCII binary layout docstrings in `src/index/*` and `src/query/*`; write `ADR-005` through `ADR-007`; update `README.md` with full query/CLI usage. | `pytest -v tests/test_m1_storage.py tests/test_m2_indexing.py` passes. |
| **Wave 5** (Gate 3: Final) | Generate `docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md` populated with actual benchmark numbers from `summary_metrics.csv`, and finalize `docs/handoff/M3_M6_HANDOFF_CONTRACT.md`. | All 5 documentation artifacts exist, contain zero placeholders (`TODO`/`TBD`), and match actual CSV metrics. |

---

## 5. Automated Documentation Verification Checklist

Before declaring Milestone 1 + Milestone 2 complete, the AI harness must verify all six quality rules:
1. **Zero Placeholders:** No `TODO`, `TBD`, `FIXME`, or dummy numbers remain in `README.md` or `docs/*.md`.
2. **Metric Consistency:** Every latency, throughput, and hit-ratio figure cited in `docs/milestones/M1_M2_TECHNICAL_NARRATIVE.md` matches `results/m1_m2/summary_metrics.csv` exactly.
3. **Binary Diagram Accuracy:** Every byte offset in docstring ASCII diagrams matches the exact `struct.Struct` format strings in `src/storage/*` and `src/index/*`.
4. **Honest Failure History:** `docs/DEV_JOURNAL.md` records actual bugs/edge cases resolved during Waves 0–5 (or documents why specific adversarial tests required careful handling).
5. **Reproducibility:** Following the commands in `README.md` from a clean directory reproduces Gate 1, Gate 2, and Gate 3 without manual intervention.
6. **M3–M6 Readiness:** `docs/handoff/M3_M6_HANDOFF_CONTRACT.md` documents every public class signature and extension hook required to start Milestone 3 immediately.