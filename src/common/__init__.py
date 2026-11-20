"""
Module: src/common/__init__.py
Layer: Layer 0 — Common Primitives
Lock Domain: Stateless

Purpose:
    Exports shared configuration constants, exception classes, relational
    schema definitions, ``RID``, and order-preserving binary key codecs.

Key Invariants:
    1. Layer 0 has zero dependencies on ``src.storage``, ``src.index``, or
       ``src.query``, guaranteeing an acyclic dependency DAG.
"""

from src.common import config, exceptions, types

__all__ = ["config", "exceptions", "types"]
