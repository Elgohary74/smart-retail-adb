"""
Module: src/common/config.py
Layer: Layer 0 — Common Primitives & Configuration
Lock Domain: Stateless (immutable module-level constants)

Purpose:
    Defines canonical binary page geometry, sentinel IDs, buffer pool and LSM
    defaults, WorkloadAdvisor decision thresholds, and binary format magic tags
    shared across all storage, indexing, and query layers of ``csai302_db``.

Key Invariants:
    1. Every relational table page and B+ Tree node is strictly ``PAGE_SIZE``
       (8,192 bytes) beginning with a ``HEADER_SIZE`` (32-byte) header.
    2. ``DEFAULT_FSM_OFFSET = 32`` applies to heap/column files, while
       ``BPLUS_FSM_OFFSET = 64`` reserves bytes 32..63 on B+ Tree Page 0 for
       persistent root and schema metadata.
"""

PAGE_SIZE: int = 8192
HEADER_SIZE: int = 32

INVALID_PAGE_ID: int = 0xFFFFFFFF
INVALID_SLOT_ID: int = 0xFFFFFFFF

MVCC_HEADER_SIZE: int = 16
SLOT_ENTRY_SIZE: int = 4
CHUNK_ROWS: int = 1020

DEFAULT_FSM_OFFSET: int = 32
BPLUS_FSM_OFFSET: int = 64

DEFAULT_BUFFER_POOL_SIZE: int = 64
DEFAULT_MEMTABLE_BYTES: int = 1_048_576
LSM_COMPACTION_TRIGGER: int = 4
BLOOM_FPR: float = 0.01

ADVISOR_WINDOW_SIZE: int = 1000
ADVISOR_MIN_OPS: int = 200
ADVISOR_MIN_PAGES: int = 4
ADVISOR_ALPHA_THRESHOLD: float = 0.45
ADVISOR_BETA_THRESHOLD: float = 0.25

MAGIC_OBJECT_RECORD: bytes = b"OBJ1"
MAGIC_OBJECT_DIR: bytes = b"ODIR"
MAGIC_SSTABLE: bytes = b"CSAI_SST"
