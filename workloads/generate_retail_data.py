"""Generate deterministic synthetic retail rows in the custom SmartDB engine."""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from smartdb.engine import SmartRetailDB
from smartdb.schema import ColumnSpec, DataType

PRODUCT_SCHEMA = [
    ColumnSpec("product_id", DataType.INT, nullable=False),
    ColumnSpec("name", DataType.STR, nullable=False),
    ColumnSpec("category", DataType.STR, nullable=False),
    ColumnSpec("price", DataType.FLOAT, nullable=False),
    ColumnSpec("stock", DataType.INT, nullable=False),
]
CUSTOMER_SCHEMA = [
    ColumnSpec("customer_id", DataType.INT, nullable=False),
    ColumnSpec("name", DataType.STR, nullable=False),
    ColumnSpec("email", DataType.STR, nullable=False),
    ColumnSpec("region", DataType.STR, nullable=False),
]
ORDER_SCHEMA = [
    ColumnSpec("order_id", DataType.INT, nullable=False),
    ColumnSpec("customer_id", DataType.INT, nullable=False),
    ColumnSpec("order_date", DataType.STR, nullable=False),
    ColumnSpec("total_amount", DataType.FLOAT, nullable=False),
]
ORDER_ITEM_SCHEMA = [
    ColumnSpec("order_item_id", DataType.INT, nullable=False),
    ColumnSpec("order_id", DataType.INT, nullable=False),
    ColumnSpec("product_id", DataType.INT, nullable=False),
    ColumnSpec("quantity", DataType.INT, nullable=False),
    ColumnSpec("unit_price", DataType.FLOAT, nullable=False),
]
INVENTORY_SCHEMA = [
    ColumnSpec("product_id", DataType.INT, nullable=False),
    ColumnSpec("warehouse_id", DataType.INT, nullable=False),
    ColumnSpec("quantity_on_hand", DataType.INT, nullable=False),
    ColumnSpec("last_updated", DataType.STR, nullable=False),
]


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _existing_rows(db: SmartRetailDB, table_name: str, expected_cols: list[ColumnSpec]) -> list[dict[str, Any]]:
    if table_name not in db.tables:
        return []
    table = db.tables[table_name]
    if table.schema.field_names != [c.name for c in expected_cols]:
        raise ValueError(f"{table_name} schema mismatch; use a separate --db directory.")
    return [record for _, record in table.iter_records()]


def _next_id(rows: list[dict[str, Any]], field: str) -> int:
    return max((int(row[field]) for row in rows), default=0) + 1


def generate(
    directory: Path,
    product_count: int,
    customer_count: int,
    order_count: int,
    seed: int,
    backend: str = "page",
    layout: str = "row",
) -> None:
    rng = random.Random(seed)
    categories = ["Accessories", "Electronics", "Home", "Office", "Clothing"]
    regions = ["North", "South", "East", "West", "Central"]

    with SmartRetailDB(directory, backend_kind=backend) as db:
        for name, cols in (
            ("Products", PRODUCT_SCHEMA),
            ("Customers", CUSTOMER_SCHEMA),
            ("Orders", ORDER_SCHEMA),
            ("OrderItems", ORDER_ITEM_SCHEMA),
            ("Inventory", INVENTORY_SCHEMA),
        ):
            if name not in db.tables:
                db.create_table(name, cols, layout=layout)

        products = _existing_rows(db, "Products", PRODUCT_SCHEMA)
        customers = _existing_rows(db, "Customers", CUSTOMER_SCHEMA)
        orders = _existing_rows(db, "Orders", ORDER_SCHEMA)
        order_items = _existing_rows(db, "OrderItems", ORDER_ITEM_SCHEMA)

        p_start = _next_id(products, "product_id")
        c_start = _next_id(customers, "customer_id")
        o_start = _next_id(orders, "order_id")
        oi_start = _next_id(order_items, "order_item_id")

        new_products = []
        new_inventory = []
        for pid in range(p_start, p_start + product_count):
            stock = rng.randint(0, 250)
            new_products.append(
                {
                    "product_id": pid,
                    "name": f"Product-{pid:06d}",
                    "category": rng.choice(categories),
                    "price": round(rng.uniform(2.0, 500.0), 2),
                    "stock": stock,
                }
            )
            new_inventory.append(
                {
                    "product_id": pid,
                    "warehouse_id": (pid % 3) + 1,
                    "quantity_on_hand": stock,
                    "last_updated": "2026-01-01",
                }
            )

        new_customers = [
            {
                "customer_id": cid,
                "name": f"Customer-{cid:06d}",
                "email": f"customer{cid:06d}@example.test",
                "region": rng.choice(regions),
            }
            for cid in range(c_start, c_start + customer_count)
        ]

        prices = {r["product_id"]: r["price"] for r in products}
        prices.update({r["product_id"]: r["price"] for r in new_products})
        product_ids = list(prices)
        customer_ids = [r["customer_id"] for r in customers] + [r["customer_id"] for r in new_customers]
        if order_count and (not product_ids or not customer_ids):
            raise ValueError("Cannot generate orders without at least one product and one customer")

        new_orders = []
        new_items = []
        first_day = date(2026, 1, 1)
        next_oi = oi_start
        for oid in range(o_start, o_start + order_count):
            item_count = rng.randint(1, 3)
            total = 0.0
            for _ in range(item_count):
                pid = rng.choice(product_ids)
                qty = rng.randint(1, 5)
                unit_price = prices[pid]
                total += unit_price * qty
                new_items.append(
                    {
                        "order_item_id": next_oi,
                        "order_id": oid,
                        "product_id": pid,
                        "quantity": qty,
                        "unit_price": unit_price,
                    }
                )
                next_oi += 1
            new_orders.append(
                {
                    "order_id": oid,
                    "customer_id": rng.choice(customer_ids),
                    "order_date": (first_day + timedelta(days=rng.randrange(365))).isoformat(),
                    "total_amount": round(total, 2),
                }
            )

        for table_name, batch in (
            ("Products", new_products),
            ("Inventory", new_inventory),
            ("Customers", new_customers),
            ("Orders", new_orders),
            ("OrderItems", new_items),
        ):
            for rec in batch:
                db.insert(table_name, rec)

    print(
        f"Added {len(new_products)} products, {len(new_customers)} customers, "
        f"{len(new_orders)} orders, {len(new_items)} order items to {directory}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("./retail_data"))
    parser.add_argument("--products", type=_non_negative_int, default=10_000)
    parser.add_argument("--customers", type=_non_negative_int, default=1_000)
    parser.add_argument("--orders", type=_non_negative_int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["page", "objects"], default="page")
    parser.add_argument("--layout", choices=["row", "column"], default="row")
    args = parser.parse_args()
    generate(args.db, args.products, args.customers, args.orders, args.seed, args.backend, args.layout)


if __name__ == "__main__":
    main()
