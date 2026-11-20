"""
Module: src/query/__init__.py
Layer: Layer 3 — Metadata Catalog & Query Layer
Lock Domain: CatalogLock

Purpose:
    Package marker exporting ``CatalogManager`` (atomic persistent JSON
    metadata and Milestone 3 CBO table statistics) and ``SimpleExecutor``
    (parameterized, SQL-injection-proof query shell and index router).

Key Invariants:
    1. Query execution always enforces parameterized ``?`` placeholders and
       strict top-down lock acquisition (``CatalogLock -> IndexLock ->
       TableHeapLock -> BufferPoolLock -> DiskManagerLock``).
"""

from src.query.catalog import CatalogManager
from src.query.simple_executor import SimpleExecutor

__all__ = ["CatalogManager", "SimpleExecutor"]
