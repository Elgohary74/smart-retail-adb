from collections import defaultdict
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartdb.engine import SmartRetailDB
from workloads.generate_retail_data import generate


class WorkloadGeneratorTests(unittest.TestCase):
    def test_generate_creates_all_five_retail_tables_with_consistent_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_dir = Path(tmp) / "retail"
            generate(db_dir, product_count=5, customer_count=3, order_count=4, seed=42, layout="column")

            with SmartRetailDB(db_dir) as db:
                self.assertEqual(
                    set(db.tables),
                    {"Products", "Customers", "Orders", "OrderItems", "Inventory"},
                )
                for table in db.tables.values():
                    self.assertEqual(table.layout, "column")

                products = {r["product_id"]: r for r in db.execute("SELECT * FROM Products")}
                customers = {r["customer_id"]: r for r in db.execute("SELECT * FROM Customers")}
                orders = {r["order_id"]: r for r in db.execute("SELECT * FROM Orders")}
                items = db.execute("SELECT * FROM OrderItems")
                inventory = db.execute("SELECT * FROM Inventory")

                self.assertEqual(len(products), 5)
                self.assertEqual(len(customers), 3)
                self.assertEqual(len(orders), 4)
                self.assertEqual(len(inventory), 5)
                self.assertGreaterEqual(len(items), 4)

                item_totals: dict[int, float] = defaultdict(float)
                for item in items:
                    self.assertIn(item["order_id"], orders)
                    self.assertIn(item["product_id"], products)
                    self.assertEqual(item["unit_price"], products[item["product_id"]]["price"])
                    item_totals[item["order_id"]] += item["quantity"] * item["unit_price"]

                for order_id, order in orders.items():
                    self.assertIn(order["customer_id"], customers)
                    self.assertAlmostEqual(order["total_amount"], round(item_totals[order_id], 2), places=2)

            # Incremental append preserves existing rows and increments IDs cleanly.
            generate(db_dir, product_count=2, customer_count=1, order_count=2, seed=43, layout="column")
            with SmartRetailDB(db_dir) as db:
                self.assertEqual(len(db.execute("SELECT * FROM Products")), 7)
                self.assertEqual(len(db.execute("SELECT * FROM Customers")), 4)
                self.assertEqual(len(db.execute("SELECT * FROM Orders")), 6)
                self.assertEqual(len(db.execute("SELECT * FROM Inventory")), 7)


if __name__ == "__main__":
    unittest.main()
