"""
Module: bench/retail_loader.py
Layer: Benchmark & Deterministic Dataset Tooling
Lock Domain: Single-writer loader lifecycle (StorageEngine -> CatalogManager -> Index)

Purpose:
    Implements the deterministic ``RetailDataGenerator`` (using ``random.Random``
    and ``numpy.random.default_rng`` with Zipfian skew ``theta = 0.9``) and the
    ``load_retail_dataset`` pipeline that populates all 5 retail tables,
    AES-256-GCM encrypted PNG-prefixed ``product_images`` in ``ObjectStore``,
    atomic ``CatalogManager`` statistics, and all 7 B+ Tree / LSM Tree indexes.

Key Invariants:
    1. Given identical ``seed`` and ``scale``, ``RetailDataGenerator.generate()``
       produces byte-for-byte identical relational rows and plaintext image
       payloads across runs and platforms.
    2. Bounded Zipfian sampling uses normalized power-law weights
       ``p[i] proportional to (i + 1) ** (-0.9)`` over exact ``1..N`` IDs.
    3. All 5 tables, ``ObjectStore`` blobs, ``catalog.json``, and 7 indexes are
       flushed and closed cleanly before ``load_retail_dataset`` returns.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from src.common.exceptions import StorageError
from src.common.types import RID, ColumnDef, DataType, LayoutType, Schema
from src.index.bplus_tree import BPlusTreeIndex
from src.index.lsm_tree import LSMTreeIndex
from src.query.catalog import CatalogManager
from src.storage.cipher_hooks import AES256GCMHook
from src.storage.storage_engine import StorageEngine
from src.storage.table_heap import ColumnTableHeap

PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"
ZIPF_THETA: float = 0.9
REGIONS: tuple[str, ...] = ("EGYPT", "MENA", "EUROPE", "NA", "APAC")
CATEGORIES: tuple[str, ...] = ("Electronics", "Apparel", "Home", "Books", "Sports", "Beauty")

# 2026-01-01T00:00:00Z in Unix milliseconds and exact 365-day span.
_YEAR_2026_START_MS: int = 1_767_225_600_000
_YEAR_2026_SPAN_MS: int = 365 * 86_400_000
_DEFAULT_MASTER_KEY_SEED: bytes = b"csai302-retail-master-key"
_MIN_IMAGE_BYTES: int = 16 * 1024
_MAX_IMAGE_BYTES: int = 128 * 1024

CUSTOMERS_SCHEMA: Schema = Schema(
    table_name="customers",
    columns=(
        ColumnDef("customer_id", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("name", DataType.VARCHAR, max_len=64, nullable=False),
        ColumnDef("region", DataType.VARCHAR, max_len=16, nullable=False),
        ColumnDef("tier", DataType.INT32, max_len=4, nullable=False),
    ),
    primary_key="customer_id",
)

PRODUCTS_SCHEMA: Schema = Schema(
    table_name="products",
    columns=(
        ColumnDef("product_id", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("category", DataType.VARCHAR, max_len=32, nullable=False),
        ColumnDef("price", DataType.FLOAT64, max_len=8, nullable=False),
        ColumnDef("image_oid", DataType.UINT64, max_len=8, nullable=False),
    ),
    primary_key="product_id",
)

INVENTORY_SCHEMA: Schema = Schema(
    table_name="inventory",
    columns=(
        ColumnDef("product_id", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("stock_qty", DataType.INT32, max_len=4, nullable=False),
        ColumnDef("warehouse_id", DataType.INT32, max_len=4, nullable=False),
        ColumnDef("updated_ts", DataType.INT64, max_len=8, nullable=False),
    ),
    primary_key="product_id",
)

ORDERS_SCHEMA: Schema = Schema(
    table_name="orders",
    columns=(
        ColumnDef("order_id", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("customer_id", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("order_date", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("total_amt", DataType.FLOAT64, max_len=8, nullable=False),
    ),
    primary_key="order_id",
)

ORDER_ITEMS_SCHEMA: Schema = Schema(
    table_name="order_items",
    columns=(
        ColumnDef("order_id", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("item_seq", DataType.INT32, max_len=4, nullable=False),
        ColumnDef("product_id", DataType.INT64, max_len=8, nullable=False),
        ColumnDef("qty", DataType.INT32, max_len=4, nullable=False),
        ColumnDef("unit_price", DataType.FLOAT64, max_len=8, nullable=False),
    ),
    primary_key=None,
)

SCALES: dict[str, dict[str, int]] = {
    "standard": {
        "customers": 10_000,
        "products": 5_000,
        "inventory": 5_000,
        "orders": 50_000,
        "order_items": 200_000,
        "product_images": 500,
    },
    "tiny": {
        "customers": 500,
        "products": 250,
        "inventory": 250,
        "orders": 2_000,
        "order_items": 8_000,
        "product_images": 25,
    },
}


def _validate_seed(seed: int) -> int:
    """Validate that ``seed`` is a non-boolean integer."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    return seed


def _validate_scale(scale: str) -> dict[str, int]:
    """Validate ``scale`` against ``SCALES`` and return its count dictionary."""
    if not isinstance(scale, str) or scale not in SCALES:
        raise ValueError(
            f"Unsupported dataset scale {scale!r}; expected one of {sorted(SCALES)}"
        )
    return SCALES[scale]


def _default_encryption_key() -> bytes:
    """Derive the deterministic 32-byte AES-256-GCM key used when none is supplied."""
    return hashlib.sha256(_DEFAULT_MASTER_KEY_SEED).digest()


def _sample_zipfian_ids(
    np_rng: np.random.Generator,
    domain_size: int,
    count: int,
    theta: float = ZIPF_THETA,
) -> np.ndarray:
    """Sample ``count`` IDs in ``1..domain_size`` with bounded Zipfian skew ``theta``.

    Args:
        np_rng: Deterministic NumPy random generator.
        domain_size: Upper bound ``N`` of the ``1..N`` ID domain.
        count: Number of IDs to sample with replacement.
        theta: Power-law skew exponent (default ``0.9``).

    Returns:
        1-D ``int64`` NumPy array of sampled ``1..domain_size`` IDs.
    """
    ids = np.arange(1, domain_size + 1, dtype=np.int64)
    weights = np.power(ids.astype(np.float64), -theta)
    probs = weights / weights.sum()
    return np_rng.choice(ids, size=count, replace=True, p=probs)


def _compute_min_max_rows(
    schema: Schema, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Compute per-column minimum and maximum rows in a single pass for catalog stats."""
    if not rows:
        return None
    first = rows[0]
    min_row = {col.name: first[col.name] for col in schema.columns}
    max_row = {col.name: first[col.name] for col in schema.columns}
    for row in rows[1:]:
        for col in schema.columns:
            cname = col.name
            val = row[cname]
            if val < min_row[cname]:
                min_row[cname] = val
            if val > max_row[cname]:
                max_row[cname] = val
    return min_row, max_row


class RetailDataGenerator:
    """Deterministic synthetic generator for the Retail Orders & Product Search domain.

    Produces reproducible rows for ``customers``, ``products``, ``inventory``,
    ``orders``, and ``order_items`` alongside PNG-prefixed ``product_images``
    binary payloads using ``random.Random(seed)`` and
    ``numpy.random.default_rng(seed)``.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the generator with a deterministic PRNG seed.

        Args:
            seed: Deterministic seed governing both Python and NumPy RNG streams.

        Raises:
            TypeError: If ``seed`` is not an integer.
        """
        self.seed: int = _validate_seed(seed)
        self.py_rng: random.Random = random.Random(self.seed)
        self.np_rng: np.random.Generator = np.random.default_rng(self.seed)

    def generate(self, scale: str = "standard") -> dict[str, Any]:
        """Generate the complete synthetic retail dataset for ``scale``.

        Args:
            scale: Dataset scale key (``"standard"`` or ``"tiny"``).

        Returns:
            Dictionary containing ``"customers"``, ``"products"``,
            ``"inventory"``, ``"orders"``, ``"order_items"`` row lists and
            ``"product_images"`` ``(oid, bytes)`` pairs.

        Raises:
            ValueError: If ``scale`` is not in ``SCALES``.

        Complexity:
            $O(N_{\\text{rows}} + B_{\\text{images}})$ time and memory using
            freshly seeded ``Random(seed)`` and ``default_rng(seed)`` streams so
            repeated calls remain 100% reproducible.
        """
        counts = _validate_scale(scale)
        py_rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)
        self.py_rng = py_rng
        self.np_rng = np_rng

        num_customers = counts["customers"]
        num_products = counts["products"]
        num_inventory = counts["inventory"]
        num_orders = counts["orders"]
        num_order_items = counts["order_items"]
        num_images = counts["product_images"]

        customers: list[dict[str, Any]] = [
            {
                "customer_id": cid,
                "name": f"Customer_{cid:05d}",
                "region": py_rng.choice(REGIONS),
                "tier": py_rng.randint(1, 3),
            }
            for cid in range(1, num_customers + 1)
        ]

        products: list[dict[str, Any]] = []
        product_prices: list[float] = []
        for pid in range(1, num_products + 1):
            price = round(py_rng.uniform(5.0, 2500.0), 2)
            product_prices.append(price)
            products.append(
                {
                    "product_id": pid,
                    "category": py_rng.choice(CATEGORIES),
                    "price": price,
                    "image_oid": pid if pid <= num_images else 0,
                }
            )

        inventory: list[dict[str, Any]] = [
            {
                "product_id": pid,
                "stock_qty": py_rng.randint(50, 500),
                "warehouse_id": py_rng.randint(1, 5),
                "updated_ts": _YEAR_2026_START_MS
                + py_rng.randrange(0, _YEAR_2026_SPAN_MS),
            }
            for pid in range(1, num_inventory + 1)
        ]

        order_customer_ids = _sample_zipfian_ids(
            np_rng, num_customers, num_orders, theta=ZIPF_THETA
        )
        order_dates = np_rng.integers(
            _YEAR_2026_START_MS,
            _YEAR_2026_START_MS + _YEAR_2026_SPAN_MS,
            size=num_orders,
            dtype=np.int64,
        )

        # Guarantee at least 1 item per order while hitting exact num_order_items total.
        if num_order_items >= num_orders:
            extra_items = num_order_items - num_orders
            item_counts = np.ones(num_orders, dtype=np.int64) + np_rng.multinomial(
                extra_items, np.full(num_orders, 1.0 / num_orders)
            )
        else:
            item_counts = np_rng.multinomial(
                num_order_items, np.full(num_orders, 1.0 / num_orders)
            )

        item_product_ids = _sample_zipfian_ids(
            np_rng, num_products, num_order_items, theta=ZIPF_THETA
        )
        item_qtys = np_rng.integers(1, 6, size=num_order_items, dtype=np.int32)

        orders: list[dict[str, Any]] = []
        order_items: list[dict[str, Any]] = []
        item_cursor = 0
        for order_idx in range(num_orders):
            order_id = order_idx + 1
            n_items = int(item_counts[order_idx])
            order_total = 0.0
            for seq in range(1, n_items + 1):
                pid = int(item_product_ids[item_cursor])
                qty = int(item_qtys[item_cursor])
                unit_price = product_prices[pid - 1]
                order_total += qty * unit_price
                order_items.append(
                    {
                        "order_id": order_id,
                        "item_seq": seq,
                        "product_id": pid,
                        "qty": qty,
                        "unit_price": unit_price,
                    }
                )
                item_cursor += 1

            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": int(order_customer_ids[order_idx]),
                    "order_date": int(order_dates[order_idx]),
                    "total_amt": round(order_total, 2),
                }
            )

        image_sizes = np_rng.integers(
            _MIN_IMAGE_BYTES,
            _MAX_IMAGE_BYTES + 1,
            size=num_images,
            dtype=np.int64,
        )
        magic_len = len(PNG_MAGIC)
        product_images: list[tuple[int, bytes]] = [
            (
                oid,
                PNG_MAGIC + np_rng.bytes(int(image_sizes[oid - 1]) - magic_len),
            )
            for oid in range(1, num_images + 1)
        ]

        return {
            "customers": customers,
            "products": products,
            "inventory": inventory,
            "orders": orders,
            "order_items": order_items,
            "product_images": product_images,
        }


def _build_bplus_index(
    idx_path: Path,
    schema: Schema,
    col_name: str,
    unique: bool,
    entries: Sequence[tuple[Any, RID]],
) -> None:
    """Create, populate, flush, and close a ``BPlusTreeIndex`` cleanly."""
    col_def = schema.get_column(col_name)
    idx = BPlusTreeIndex(
        idx_path,
        key_dtype=col_def.dtype,
        key_size=col_def.key_size,
        unique=unique,
    )
    try:
        for key_val, rid in entries:
            idx.insert(key_val, rid)
        idx.flush()
    finally:
        idx.close()


def _build_lsm_index(
    idx_path: Path,
    schema: Schema,
    col_name: str,
    unique: bool,
    entries: Sequence[tuple[Any, RID]],
) -> None:
    """Create, populate, flush, and close an ``LSMTreeIndex`` cleanly."""
    col_def = schema.get_column(col_name)
    idx = LSMTreeIndex(
        idx_path,
        key_dtype=col_def.dtype,
        key_size=col_def.key_size,
        unique=unique,
    )
    try:
        for key_val, rid in entries:
            idx.insert(key_val, rid)
        idx.flush()
    finally:
        idx.close()


def load_retail_dataset(
    data_dir: str | Path,
    scale: str = "standard",
    seed: int = 42,
    build_indexes: bool = True,
    encryption_key: bytes | None = None,
) -> dict[str, Any]:
    """Generate and persist the full retail database, encrypted objects, catalog, and indexes.

    Args:
        data_dir: Target database directory (created if missing).
        scale: Dataset scale (``"standard"`` or ``"tiny"``).
        seed: Deterministic seed forwarded to ``RetailDataGenerator``.
        build_indexes: When ``True``, build and register all 7 retail B+ Tree
            and LSM Tree indexes under ``<data_dir>/indexes/``.
        encryption_key: Optional 32-byte AES-256-GCM key; defaults to a
            deterministic SHA-256 digest when omitted.

    Returns:
        Summary dictionary containing ``scale``, ``seed``, ``data_dir``,
        ``counts``, ``indexes``, and ``elapsed_seconds``.

    Raises:
        TypeError: If ``seed`` or ``encryption_key`` has an invalid type.
        ValueError: If ``scale`` is unknown or ``encryption_key`` is not 32 bytes.
        StorageError: If any table already exists in ``data_dir`` or disk I/O fails.
    """
    t0 = time.perf_counter()
    _validate_scale(scale)
    _validate_seed(seed)

    key = (
        bytes(encryption_key)
        if encryption_key is not None
        else _default_encryption_key()
    )
    cipher = AES256GCMHook(key)

    dataset = RetailDataGenerator(seed=seed).generate(scale=scale)
    db_root = Path(data_dir)
    tables_dir = db_root / "tables"
    indexes_dir = db_root / "indexes"

    table_specs: tuple[tuple[Schema, LayoutType, Path], ...] = (
        (CUSTOMERS_SCHEMA, LayoutType.ROW, tables_dir / "customers.db"),
        (PRODUCTS_SCHEMA, LayoutType.ROW, tables_dir / "products.db"),
        (INVENTORY_SCHEMA, LayoutType.ROW, tables_dir / "inventory.db"),
        (ORDERS_SCHEMA, LayoutType.ROW, tables_dir / "orders.db"),
        (ORDER_ITEMS_SCHEMA, LayoutType.COLUMN, tables_dir / "order_items"),
    )

    catalog = CatalogManager(db_root / "catalog.json")
    rids_by_table: dict[str, list[RID]] = {}
    built_indexes: list[str] = []

    engine = StorageEngine(db_root, buffer_pool_size=64, cipher=cipher)
    engine.open()
    try:
        for schema, _, _ in table_specs:
            if engine.has_table(schema.table_name) or catalog.has_table(schema.table_name):
                raise StorageError(
                    f"Table {schema.table_name!r} already exists in {db_root}; "
                    "use a fresh data directory"
                )

        for schema, layout, table_path in table_specs:
            heap = engine.create_table(schema, layout)
            catalog.register_table(schema, layout, file_path=table_path)
            rows: list[dict[str, Any]] = dataset[schema.table_name]

            if isinstance(heap, ColumnTableHeap):
                rids = heap.insert_batch(rows)
            else:
                rids = [heap.insert(row) for row in rows]
            rids_by_table[schema.table_name] = rids

            bounds = _compute_min_max_rows(schema, rows)
            page_cnt = heap.page_count()
            if bounds is not None:
                min_row, max_row = bounds
                catalog.update_table_stats(
                    schema.table_name,
                    row_delta=len(rows),
                    page_count=page_cnt,
                    row_values=min_row,
                    persist=False,
                )
                catalog.update_table_stats(
                    schema.table_name,
                    row_delta=0,
                    page_count=page_cnt,
                    row_values=max_row,
                    persist=False,
                )
            else:
                catalog.update_table_stats(
                    schema.table_name,
                    row_delta=0,
                    page_count=page_cnt,
                    persist=False,
                )

        obj_store = engine.object_store
        for oid, payload in dataset["product_images"]:
            obj_store.put_object(oid, payload)

        engine.flush_all()
        catalog.save()
    finally:
        engine.close()

    if build_indexes:
        indexes_dir.mkdir(parents=True, exist_ok=True)
        index_specs: tuple[tuple[Schema, str, str, str, bool], ...] = (
            (CUSTOMERS_SCHEMA, "idx_customers_pk", "customer_id", "BPLUS_TREE", True),
            (PRODUCTS_SCHEMA, "idx_products_pk", "product_id", "BPLUS_TREE", True),
            (PRODUCTS_SCHEMA, "idx_products_price", "price", "BPLUS_TREE", False),
            (INVENTORY_SCHEMA, "idx_inventory_pk", "product_id", "LSM_TREE", True),
            (ORDERS_SCHEMA, "idx_orders_pk", "order_id", "BPLUS_TREE", True),
            (ORDERS_SCHEMA, "idx_orders_date", "order_date", "BPLUS_TREE", False),
            (ORDER_ITEMS_SCHEMA, "idx_order_items_order_id", "order_id", "LSM_TREE", False),
        )

        for schema, idx_name, col_name, idx_type, unique in index_specs:
            rows = dataset[schema.table_name]
            rids = rids_by_table[schema.table_name]
            entries = [(row[col_name], rid) for row, rid in zip(rows, rids, strict=True)]

            if idx_type == "BPLUS_TREE":
                idx_path = (indexes_dir / f"{idx_name}.bpt").resolve()
                _build_bplus_index(idx_path, schema, col_name, unique, entries)
            else:
                idx_path = (indexes_dir / idx_name).resolve()
                _build_lsm_index(idx_path, schema, col_name, unique, entries)

            catalog.register_index(
                schema.table_name,
                idx_name,
                col_name,
                idx_type,
                unique,
                idx_path,
            )
            built_indexes.append(idx_name)

    elapsed = time.perf_counter() - t0
    return {
        "scale": scale,
        "seed": seed,
        "data_dir": str(db_root),
        "counts": {k: len(v) for k, v in dataset.items()},
        "indexes": built_indexes,
        "elapsed_seconds": round(elapsed, 4),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``python -m bench.retail_loader``."""
    parser = argparse.ArgumentParser(
        description="Generate and load deterministic CSAI 302 Retail dataset."
    )
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES.keys()),
        default="standard",
        help="Dataset scale preset (default: standard).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic PRNG seed (default: 42).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Output database directory (default: data).",
    )
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        help="Skip building B+ Tree and LSM Tree secondary/PK indexes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for loading the synthetic retail dataset."""
    args = _parse_args(argv)
    summary = load_retail_dataset(
        data_dir=args.data_dir,
        scale=args.scale,
        seed=args.seed,
        build_indexes=not args.no_indexes,
    )
    print(
        f"Loaded retail dataset scale={summary['scale']} seed={summary['seed']} "
        f"into {summary['data_dir']} in {summary['elapsed_seconds']:.3f}s "
        f"(counts={summary['counts']}, indexes={len(summary['indexes'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATEGORIES",
    "CUSTOMERS_SCHEMA",
    "INVENTORY_SCHEMA",
    "ORDER_ITEMS_SCHEMA",
    "ORDERS_SCHEMA",
    "PNG_MAGIC",
    "PRODUCTS_SCHEMA",
    "REGIONS",
    "RetailDataGenerator",
    "SCALES",
    "ZIPF_THETA",
    "load_retail_dataset",
    "main",
]
