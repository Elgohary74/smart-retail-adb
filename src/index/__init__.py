"""
Module: src/index/__init__.py
Layer: Layer 2 — Access Methods & Workload Intelligence
Lock Domain: IndexLock

Purpose:
    Package marker for Milestone 2 index engines (``BPlusTreeIndex``,
    ``LSMTreeIndex``) and workload-driven selection (``WorkloadAdvisor``).

Key Invariants:
    1. All concrete indexes inherit from ``BaseIndex`` and acquire ``IndexLock``
       before lower storage locks.
"""
