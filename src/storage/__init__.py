"""
Module: src/storage/__init__.py
Layer: Layer 1 — Physical Storage, Buffer Pool & Object Store
Lock Domain: Submodule-specific (DiskManagerLock, BufferPoolLock, TableHeapLock, ObjectStoreLock)

Purpose:
    Package marker for Milestone 1 storage components and the Milestone 2
    encrypted extent-based object store.

Key Invariants:
    1. All persistent structures use 8,192-byte checksummed pages or binary
       extent streams without Python object serialization.
"""
