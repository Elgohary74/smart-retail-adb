import tempfile
import unittest
from pathlib import Path
from smartdb.cli import _parse_create_fields
from smartdb.engine import SmartRetailDB


class EngineTests(unittest.TestCase):
    def test_create_insert_index_query_and_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "db"
            db = SmartRetailDB(root)
            db.create_table("Products", ["product_id", "name", "category", "price", "stock"], "row")
            db.insert("Products", {"product_id": 1, "name": "Keyboard", "category": "Accessories", "price": 29.99, "stock": 10})
            db.insert("Products", {"product_id": 2, "name": "Mouse", "category": "Accessories", "price": 15.5, "stock": 20})
            self.assertEqual(len(db.execute("SELECT * FROM Products WHERE product_id = 1")), 1)
            self.assertIn("TABLE SCAN", db.explain("SELECT * FROM Products WHERE product_id = 1"))
            kind = db.create_index("Products", "product_id", "auto", reads=100, writes=5)
            self.assertEqual(kind, "BPLUS")
            self.assertIn("INDEX LOOKUP", db.explain("SELECT * FROM Products WHERE product_id = 1"))
            self.assertEqual(db.execute("SELECT * FROM Products WHERE product_id = 2")[0]["name"], "Mouse")
            db.close()

            reopened = SmartRetailDB(root)
            self.assertEqual(len(reopened.execute("SELECT * FROM Products")), 2)
            self.assertEqual(reopened.execute("SELECT * FROM Products WHERE product_id = 1")[0]["name"], "Keyboard")
            reopened.close()

    def test_lsm_index_survives_reopen_and_new_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "db"
            db = SmartRetailDB(root)
            db.create_table("Products", ["product_id", "name"], "column")
            db.insert("Products", {"product_id": 1, "name": "Book"})
            db.create_index("Products", "product_id", "LSM")
            db.close()
            db = SmartRetailDB(root)
            self.assertEqual(db.execute("SELECT * FROM Products WHERE product_id = 1")[0]["name"], "Book")
            db.insert("Products", {"product_id": 2, "name": "Pen"})
            self.assertEqual(db.execute("SELECT * FROM Products WHERE product_id = 2")[0]["name"], "Pen")
            db.close()

    def test_object_backend_survives_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "objects_db"
            db = SmartRetailDB(root, backend_kind="objects")
            db.create_table("Products", ["product_id", "name"], "row")
            db.insert("Products", {"product_id": 7, "name": "Monitor"})
            db.close()
            reopened = SmartRetailDB(root)
            self.assertEqual(reopened.execute("SELECT * FROM Products WHERE product_id = 7")[0]["name"], "Monitor")
            reopened.close()

    def test_rejects_injection_like_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = SmartRetailDB(Path(tmp) / "db")
            db.create_table("Products", ["product_id", "name"])
            with self.assertRaises(ValueError):
                db.execute("SELECT * FROM Products WHERE product_id = 1 OR 1=1")
            db.close()

    def test_shared_buffer_pool_typed_schema_and_update_delete_with_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "db"
            db = SmartRetailDB(root, buffer_frames=8)
            db.create_table(
                "Products",
                [("product_id", "INT", False), ("name", "STR", False), ("price", "FLOAT", True)],
                layout="column",
            )
            db.create_table("Customers", [("customer_id", "INT", False), ("name", "STR", False)], layout="row")

            loc1 = db.insert("Products", {"product_id": 1, "name": "Keyboard", "price": 29.99})
            loc2 = db.insert("Products", {"product_id": 2, "name": "Mouse", "price": 15.5})
            db.insert("Customers", {"customer_id": 10, "name": "Alice"})
            db.create_index("Products", "product_id", "BPLUS")

            with self.assertRaises(TypeError):
                db.insert("Products", {"product_id": "bad", "name": "Bad", "price": 1.0})

            new_loc2 = db.update("Products", loc2, {"product_id": 2, "name": "Pro Mouse", "price": 39.0})
            self.assertEqual(db.execute("SELECT * FROM Products WHERE product_id = 2")[0]["name"], "Pro Mouse")

            db.delete("Products", loc1)
            self.assertEqual(db.execute("SELECT * FROM Products WHERE product_id = 1"), [])
            self.assertEqual(len(db.execute("SELECT * FROM Products")), 1)
            self.assertEqual(db.tables["Products"].get_record(new_loc2)["price"], 39.0)

            stats = db.buffer_stats()
            self.assertEqual(stats["capacity"], 8)
            self.assertIn("hits", stats)
            db.close()

            reopened = SmartRetailDB(root, buffer_frames=8)
            self.assertEqual(reopened.execute("SELECT * FROM Products WHERE product_id = 2")[0]["name"], "Pro Mouse")
            with self.assertRaises(TypeError):
                reopened.insert("Products", {"product_id": "bad", "name": "Bad", "price": 1.0})
            reopened.close()

    def test_cli_parse_create_fields_typed_and_untyped(self):
        self.assertEqual(_parse_create_fields("id, name, price"), ["id", "name", "price"])
        self.assertEqual(
            _parse_create_fields("id INT NOT NULL, name STR, price FLOAT"),
            [("id", "INT", False), ("name", "STR", True), ("price", "FLOAT", True)],
        )


if __name__ == "__main__":
    unittest.main()
