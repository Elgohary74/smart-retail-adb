import tempfile
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
