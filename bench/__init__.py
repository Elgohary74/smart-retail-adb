"""
Module: bench/__init__.py
Layer: Benchmark & Deterministic Dataset Tooling
Lock Domain: Stateless (lazy re-exports of schemas, scales, generator, and loader APIs)

Purpose:
    Exposes the deterministic Retail Orders & Product Search data generator and
    dataset loader for Gate 2 equivalence tests and Gate 3 benchmark suites
    without eagerly importing ``bench.retail_loader`` during ``python -m``.

Key Invariants:
    1. Re-exports the exact five retail table schemas, ``SCALES``,
       ``RetailDataGenerator``, and ``load_retail_dataset`` contracts defined in
       ``docs/ai_specs/01_RETAIL_SCHEMA_AND_DATA.md``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bench.retail_loader import (
        CUSTOMERS_SCHEMA,
        INVENTORY_SCHEMA,
        ORDER_ITEMS_SCHEMA,
        ORDERS_SCHEMA,
        PRODUCTS_SCHEMA,
        SCALES,
        RetailDataGenerator,
        load_retail_dataset,
    )

__all__ = [
    "CUSTOMERS_SCHEMA",
    "INVENTORY_SCHEMA",
    "ORDER_ITEMS_SCHEMA",
    "ORDERS_SCHEMA",
    "PRODUCTS_SCHEMA",
    "RetailDataGenerator",
    "SCALES",
    "load_retail_dataset",
]

_RETAIL_LOADER_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Lazily resolve ``bench.retail_loader`` exports to avoid ``runpy`` warnings."""
    if name in _RETAIL_LOADER_EXPORTS:
        module = import_module("bench.retail_loader")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
