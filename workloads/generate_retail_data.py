"""Generate deterministic synthetic retail rows in the custom SmartDB engine."""
from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from smartdb.engine import SmartRetailDB


PRODUCT_FIELDS = ["product_id", "name", "category", "price", "stock"]
CUSTOMER_FIELDS = ["customer_id", "name", "email", "region"]
ORDER_FIELDS = [
    "order_id",
    "customer_id",
    "product_id",
    "quantity",
    "order_date",
    "total_amount",
]


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _existing_rows(
    db: SmartRetailDB, table_name: str, fields: list[str], id_field: str
) -> list[dict[str, Any]]:
    if table_name not in db.tables:
        return []
    table = db.tables[table_name]
    if table.fields != fields:
        raise ValueError(
            f"{table_name} has fields {table.fields}; expected {fields}. "
            "Use a different --db directory or make the schemas match."
        )
    rows = [record for _, record in table.iter_records()]
    if any(not isinstance(row[id_field], int) or isinstance(row[id_field], bool) for row in rows):
        raise ValueError(f"{table_name}.{id_field} values must be integers")
    return rows


def _first_id(rows: list[dict[str, Any]], field: str) -> int:
    return max((row[field] for row in rows), default=0) + 1


def generate(
    directory: Path,
    product_count: int,
    customer_count: int,
    order_count: int,
    seed: int,
    backend: str = "page",
) -> None:
    rng = random.Random(seed)
    categories = ["Accessories", "Electronics", "Home", "Office", "Clothing"]
    regions = ["North", "South", "East", "West", "Central"]

    with SmartRetailDB(directory, backend_kind=backend) as db:
        products = _existing_rows(db, "Products", PRODUCT_FIELDS, "product_id")
        customers = _existing_rows(db, "Customers", CUSTOMER_FIELDS, "customer_id")
        orders = _existing_rows(db, "Orders", ORDER_FIELDS, "order_id")

        product_start = _first_id(products, "product_id")
        customer_start = _first_id(customers, "customer_id")
        order_start = _first_id(orders, "order_id")

        new_products = []
        for product_id in range(product_start, product_start + product_count):
            new_products.append(
                {
                    "product_id": product_id,
                    "name": f"Product-{product_id:06d}",
                    "category": rng.choice(categories),
                    "price": round(rng.uniform(2.0, 500.0), 2),
                    "stock": rng.randint(0, 250),
                }
            )

        new_customers = []
        for customer_id in range(customer_start, customer_start + customer_count):
            new_customers.append(
                {
                    "customer_id": customer_id,
                    "name": f"Customer-{customer_id:06d}",
                    "email": f"customer{customer_id:06d}@example.test",
                    "region": rng.choice(regions),
                }
            )

        product_prices = {
            row["product_id"]: row["price"] for row in products
        }
        product_prices.update(
            {row["product_id"]: row["price"] for row in new_products}
        )
        if any(not isinstance(price, (int, float)) for price in product_prices.values()):
            raise ValueError("Products.price values must be numeric to generate orders")
        product_ids = list(product_prices)
        customer_ids = [row["customer_id"] for row in customers] + [
            row["customer_id"] for row in new_customers
        ]
        if order_count and (not product_prices or not customer_ids):
            raise ValueError(
                "Cannot generate orders without at least one product and one customer"
            )

        new_orders = []
        first_day = date(2026, 1, 1)
        for order_id in range(order_start, order_start + order_count):
            product_id = rng.choice(product_ids)
            quantity = rng.randint(1, 5)
            new_orders.append(
                {
                    "order_id": order_id,
                    "customer_id": rng.choice(customer_ids),
                    "product_id": product_id,
                    "quantity": quantity,
                    "order_date": (first_day + timedelta(days=rng.randrange(365))).isoformat(),
                    "total_amount": round(product_prices[product_id] * quantity, 2),
                }
            )

        for table_name, fields in (
            ("Products", PRODUCT_FIELDS),
            ("Customers", CUSTOMER_FIELDS),
            ("Orders", ORDER_FIELDS),
        ):
            if table_name not in db.tables:
                db.create_table(table_name, fields, layout="row")

        for record in new_products:
            db.insert("Products", record)
        for record in new_customers:
            db.insert("Customers", record)
        for record in new_orders:
            db.insert("Orders", record)

    print(
        f"Added {len(new_products)} products, {len(new_customers)} customers, "
        f"and {len(new_orders)} orders to {directory}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("./retail_data"))
    parser.add_argument("--products", type=_non_negative_int, default=10_000)
    parser.add_argument("--customers", type=_non_negative_int, default=1_000)
    parser.add_argument("--orders", type=_non_negative_int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["page", "objects"], default="page")
    args = parser.parse_args()
    generate(
        args.db,
        args.products,
        args.customers,
        args.orders,
        args.seed,
        args.backend,
    )


if __name__ == "__main__":
    main()
