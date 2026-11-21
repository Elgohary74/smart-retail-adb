"""Atomic JSON catalog persistence for tables, workload stats, and indexes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import atomic_write_json


class CatalogStore:
    """Loads and atomically persists catalog.json and indexes.json."""

    def __init__(self, directory: Path):
        self.catalog_path = directory / "catalog.json"
        self.index_catalog_path = directory / "indexes.json"
        self.schemas: dict[str, dict[str, Any]] = {}
        self.table_stats: dict[str, dict[str, int]] = {}
        self.clean_shutdown = True
        self._load_catalog()

    def _load_catalog(self) -> None:
        if not self.catalog_path.exists():
            return
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.schemas = data.get("tables", {})
        self.clean_shutdown = bool(data.get("clean_shutdown", True))
        for name, meta in self.schemas.items():
            self.table_stats[name] = dict(meta.get("stats", {"reads": 0, "writes": 0, "range_reads": 0}))

    def save_catalog(self, clean_shutdown: bool = False) -> None:
        for name, stats in self.table_stats.items():
            if name in self.schemas:
                self.schemas[name]["stats"] = stats
        atomic_write_json(self.catalog_path, {"clean_shutdown": clean_shutdown, "tables": self.schemas})

    def load_indexes(self) -> list[dict[str, str]]:
        if not self.index_catalog_path.exists():
            return []
        return json.loads(self.index_catalog_path.read_text(encoding="utf-8")).get("indexes", [])

    def save_indexes(self, entries: list[tuple[str, str, str]]) -> None:
        items = [{"table": t, "field": f, "type": kind} for t, f, kind in entries]
        atomic_write_json(self.index_catalog_path, {"indexes": items})
