"""Generate deterministic synthetic retail rows in the custom SmartDB engine."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import random
from typing import Any

from smartdb.engine import SmartRetailDB

PRODUCT_SCHEMA = [
    ("product_id", "INT", False),
    ("name", "STR", False),
    ("category", "STR", False),
    ("price", "FLOAT", False),
    ("stock", "INT", False),
]
CUSTOMER_SCHEMA = [
    ("customer_id", "INT", False),
    ("name", "STR", False),
    ("email", "STR", False),
    ("region", "STR", False),
]
ORDER_SCHEMA = [
    ("order_id", "INT", False),
    ("customer_id", "INT", False),
    ("order_date", "STR", False),
    ("total_amount", "FLOAT", False),
]
ORDER_ITEM_SCHEMA = [
    ("order_item_id", "INT", False),
    ("order_id", "INT", False),
    ("product_id", "INT", False),
    ("quantity", "INT", False),
    ("unit_price", "FLOAT", False),
]
INVENTORY_SCHEMA = [
    ("product_id", "INT", False),
    ("warehouse_id", "INT", False),
    ("quantity_on_hand", "INT", False),
    ("last_updated", "STR", False),
]

TABLE_SCHEMAS = (
    ("Products", PRODUCT_SCHEMA),
    ("Customers", CUSTOMER_SCHEMA),
    ("Orders", ORDER_SCHEMA),
    ("OrderItems", ORDER_ITEM_SCHEMA),
    ("Inventory", INVENTORY_SCHEMA),
)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _existing_rows(
    db: SmartRetailDB, table_name: str, expected_fields: list[str], id_field: str | None = None
) -> list[dict[str, Any]]:
    if table_name not in db.tables:
        return []
    table = db.tables[table_name]
    if table.fields != expected_fields:
        raise ValueError(
            f"{table_name} has fields {table.fields}; expected {expected_fields}. "
            "Use a different --db directory or make the schemas match."
        )
    rows = [record for _, record in table.iter_records()]
    if id_field is not None:
        if any(not isinstance(row[id_field], int) or isinstance(row[id_field], bool) for row in rows):
            raise ValueError(f"{table_name}.{id_field} values must be integers")
    return rows


def _next_id(rows: list[dict[str, Any]], field: str) -> int:
    return max((row[field] for row in rows), default=0) + 1


def _build_products(rng: random.Random, start_id: int, count: int) -> list[dict[str, Any]]:
    categories = ["Accessories", "Electronics", "Home", "Office", "Clothing"]
    return [
        {
            "product_id": product_id,
            "name": f"Product-{product_id:06d}",
            "category": rng.choice(categories),
            "price": round(rng.uniform(2.0, 500.0), 2),
            "stock": rng.randint(0, 250),
        }
        for product_id in range(start_id, start_id + count)
    ]


def _build_customers(rng: random.Random, start_id: int, count: int) -> list[dict[str, Any]]:
    regions = ["North", "South", "East", "West", "Central"]
    return [
        {
            "customer_id": customer_id,
            "name": f"Customer-{customer_id:06d}",
            "email": f"customer{customer_id:06d}@example.test",
            "region": rng.choice(regions),
        }
        for customer_id in range(start_id, start_id + count)
    ]


def _build_inventory(rng: random.Random, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_day = date(2026, 1, 1)
    return [
        {
            "product_id": product["product_id"],
            "warehouse_id": rng.randint(1, 3),
            "quantity_on_hand": product["stock"],
            "last_updated": (first_day + timedelta(days=rng.randrange(365))).isoformat(),
        }
        for product in products
    ]


def _build_orders_and_items(
    rng: random.Random,
    order_start: int,
    item_start: int,
    order_count: int,
    customer_ids: list[int],
    product_prices: dict[int, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if order_count and (not product_prices or not customer_ids):
        raise ValueError("Cannot generate orders without at least one product and one customer")

    first_day = date(2026, 1, 1)
    product_ids = list(product_prices)
    orders: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    next_item_id = item_start

    for order_id in range(order_start, order_start + order_count):
        line_count = rng.randint(1, 3)
        total = 0.0
        for _ in range(line_count):
            product_id = rng.choice(product_ids)
            quantity = rng.randint(1, 5)
            unit_price = float(product_prices[product_id])
            total += quantity * unit_price
            items.append(
                {
                    "order_item_id": next_item_id,
                    "order_id": order_id,
                    "product_id": product_id,
                    "quantity": quantity,
                    "unit_price": unit_price,
                }
            )
            next_item_id += 1
        orders.append(
            {
                "order_id": order_id,
                "customer_id": rng.choice(customer_ids),
                "order_date": (first_day + timedelta(days=rng.randrange(365))).isoformat(),
                "total_amount": round(total, 2),
            }
        )
    return orders, items


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
    with SmartRetailDB(directory, backend_kind=backend) as db:
        products = _existing_rows(db, "Products", [c[0] for c in PRODUCT_SCHEMA], "product_id")
        customers = _existing_rows(db, "Customers", [c[0] for c in CUSTOMER_SCHEMA], "customer_id")
        orders = _existing_rows(db, "Orders", [c[0] for c in ORDER_SCHEMA], "order_id")
        order_items = _existing_rows(db, "OrderItems", [c[0] for c in ORDER_ITEM_SCHEMA], "order_item_id")
        _existing_rows(db, "Inventory", [c[0] for c in INVENTORY_SCHEMA], "product_id")

        new_products = _build_products(rng, _next_id(products, "product_id"), product_count)
        new_customers = _build_customers(rng, _next_id(customers, "customer_id"), customer_count)
        new_inventory = _build_inventory(rng, new_products)

        product_prices = {row["product_id"]: row["price"] for row in products}
        product_prices.update({row["product_id"]: row["price"] for row in new_products})
        customer_ids = [row["customer_id"] for row in customers] + [
            row["customer_id"] for row in new_customers
        ]
        new_orders, new_items = _build_orders_and_items(
            rng,
            _next_id(orders, "order_id"),
            _next_id(order_items, "order_item_id"),
            order_count,
            customer_ids,
            product_prices,
        )

        for table_name, schema in TABLE_SCHEMAS:
            if table_name not in db.tables:
                db.create_table(table_name, schema, layout=layout)

        for record in new_products:
            db.insert("Products", record)
        for record in new_customers:
            db.insert("Customers", record)
        for record in new_inventory:
            db.insert("Inventory", record)
        for record in new_orders:
            db.insert("Orders", record)
        for record in new_items:
            db.insert("OrderItems", record)

    print(
        f"Added {len(new_products)} products, {len(new_customers)} customers, "
        f"{len(new_inventory)} inventory rows, {len(new_orders)} orders, "
        f"and {len(new_items)} order items to {directory}"
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
    generate(
        args.db,
        args.products,
        args.customers,
        args.orders,
        args.seed,
        args.backend,
        args.layout,
    )


if __name__ == "__main__":
    main()
