import os
import tempfile
import unittest
from pathlib import Path

from smartdb.engine import SmartRetailDB
from smartdb.indexes.manager import choose_index_type
from smartdb.query import parse_sql
from smartdb.schema import ColumnSpec, DataType
from smartdb.storage.encryption import make_aes_gcm_hooks


class EngineTests(unittest.TestCase):
    def test_choose_index_type_boundaries(self):
        self.assertEqual(choose_index_type(reads=0, writes=1), "LSM")
        self.assertEqual(choose_index_type(reads=10, writes=0), "BPLUS")
        self.assertEqual(choose_index_type(reads=10, writes=100, range_reads=5), "BPLUS")
        self.assertEqual(choose_index_type(reads=10, writes=100, range_reads=0), "LSM")

    def test_full_sql_dml_projection_range_and_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "db"
            db = SmartRetailDB(root)
            db.execute("CREATE TABLE Products (product_id INT, name STR, price FLOAT, vec VECTOR(3)) LAYOUT ROW;")
            db.execute("INSERT INTO Products VALUES (1, 'Keyboard', 29.99, [1.0, 0.0, 0.0]);")
            db.execute("INSERT INTO Products VALUES (2, 'Mouse', 15.50, [0.0, 1.0, 0.0]);")
            db.execute("INSERT INTO Products VALUES (3, 'Monitor', 199.0, [0.0, 0.0, 1.0]);")

            # Single-column table insert without trailing comma
            db.execute("CREATE TABLE Ids (id INT);")
            db.execute("INSERT INTO Ids VALUES (7);")
            self.assertEqual(db.execute("SELECT * FROM Ids"), [{"id": 7}])

            kind = db.create_index("Products", "price", "BPLUS")
            self.assertEqual(kind, "BPLUS")
            self.assertIn("INDEX RANGE SCAN", db.explain("SELECT product_id, name FROM Products WHERE price BETWEEN 15.0 AND 30.0"))
            rows = db.execute("SELECT product_id, name FROM Products WHERE price BETWEEN 15.0 AND 30.0")
            self.assertEqual(sorted(r["name"] for r in rows), ["Keyboard", "Mouse"])

            updated = db.execute("UPDATE Products SET price = 25.0 WHERE product_id = 2;")
            self.assertEqual(updated, 1)
            self.assertEqual(db.execute("SELECT price FROM Products WHERE product_id = 2"), [{"price": 25.0}])

            deleted = db.execute("DELETE FROM Products WHERE product_id = 1;")
            self.assertEqual(deleted, 1)
            self.assertEqual(len(db.execute("SELECT * FROM Products")), 2)
            db.close()

            reopened = SmartRetailDB(root)
            self.assertEqual(len(reopened.execute("SELECT * FROM Products WHERE price > 20")), 2)
            reopened.close()

    def test_bool_int_query_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = SmartRetailDB(tmp)
            db.create_table("T", ["id", "name"])
            db.create_index("T", "id", "BPLUS")
            db.insert("T", {"id": 1, "name": "one"})
            self.assertEqual(db.execute("SELECT * FROM T WHERE id = TRUE"), [])
            self.assertEqual(db.execute("SELECT * FROM T WHERE id = 1"), [{"id": 1, "name": "one"}])
            db.close()

    def test_null_excluded_from_range_and_inequality_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = SmartRetailDB(tmp)
            db.create_table("P", [ColumnSpec("id", DataType.INT), ColumnSpec("price", DataType.FLOAT)])
            db.create_index("P", "price", "BPLUS")
            db.insert("P", {"id": 1, "price": None})
            db.insert("P", {"id": 2, "price": 5.0})
            db.insert("P", {"id": 3, "price": 20.0})

            self.assertEqual([r["id"] for r in db.execute("SELECT id FROM P WHERE price < 10")], [2])
            self.assertEqual([r["id"] for r in db.execute("SELECT id FROM P WHERE price <= 10")], [2])
            self.assertEqual([r["id"] for r in db.execute("SELECT id FROM P WHERE price > 0")], [2, 3])
            self.assertEqual([r["id"] for r in db.execute("SELECT id FROM P WHERE price BETWEEN 0 AND 10")], [2])
            self.assertEqual([r["id"] for r in db.execute("SELECT id FROM P WHERE price IS NULL")], [1])
            self.assertEqual([r["id"] for r in db.execute("SELECT id FROM P WHERE price IS NOT NULL")], [2, 3])
            db.close()

    def test_unclean_shutdown_reconciles_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "crash_db"
            db = SmartRetailDB(root, buffer_frames=16)
            db.create_table("T", [ColumnSpec("id", DataType.INT), ColumnSpec("val", DataType.STR)])
            db.create_index("T", "id", "LSM")
            for i in range(150):
                db.insert("T", {"id": i, "val": f"v{i}"})
            # Close raw backend handles WITHOUT calling db.close()
            idx = db.index_manager.indexes[("T", "id")][1]
            idx._wal_file.close()
            db.storage_manager.close_all()

            reopened = SmartRetailDB(root)
            # Reconciled cleanly: no IndexError, index matches durable table state
            all_rows = reopened.execute("SELECT * FROM T")
            for row in all_rows:
                got = reopened.execute(f"SELECT * FROM T WHERE id = {row['id']}")
                self.assertEqual(got, [row])
            reopened.close()

    def test_failed_insert_is_atomic_across_table_and_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = SmartRetailDB(tmp)
            db.create_table("T", [ColumnSpec("id", DataType.INT, nullable=False), ColumnSpec("v", DataType.FLOAT)])
            db.create_index("T", "id", "BPLUS")
            with self.assertRaises(TypeError):
                db.insert("T", {"id": True, "v": 1.0})
            self.assertEqual(db.execute("SELECT * FROM T"), [])
            self.assertEqual(db.index_manager.search("T", "id", True), [])
            db.close()

    def test_auto_telemetry_and_index_switching(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = SmartRetailDB(tmp)
            db.create_table("T", [ColumnSpec("id", DataType.INT), ColumnSpec("cat", DataType.STR)])
            for i in range(20):
                db.insert("T", {"id": i, "cat": f"C{i % 2}"})
            # 20 writes, 0 reads -> USING AUTO picks LSM
            kind = db.create_index("T", "id", "auto")
            self.assertEqual(kind, "LSM")
            self.assertEqual(len(db.execute("SELECT * FROM T WHERE id = 5")), 1)
            # Switch to BPLUS and back to LSM without stale runs
            self.assertEqual(db.create_index("T", "id", "BPLUS"), "BPLUS")
            db.insert("T", {"id": 99, "cat": "C1"})
            self.assertEqual(db.create_index("T", "id", "LSM"), "LSM")
            self.assertEqual(len(db.execute("SELECT * FROM T WHERE id = 99")), 1)
            db.close()

    def test_object_backend_with_aes_gcm_and_injection_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sec_obj"
            enc, dec, ovh, rec_enc, rec_dec = make_aes_gcm_hooks(os.urandom(32))
            db = SmartRetailDB(
                root,
                backend_kind="objects",
                encrypt_page=enc,
                decrypt_page=dec,
                crypto_overhead=ovh,
                encrypt_record=rec_enc,
                decrypt_record=rec_dec,
            )
            db.create_table("Products", [ColumnSpec("id", DataType.INT), ColumnSpec("name", DataType.STR)], "column")
            db.create_index("Products", "id", "LSM")
            db.insert("Products", {"id": 7, "name": "Monitor"})
            with self.assertRaises(ValueError):
                parse_sql("SELECT * FROM Products WHERE id = 1 OR 1=1")
            with self.assertRaises(ValueError):
                parse_sql("SELECT * FROM Products WHERE id = 1+2j")
            db.close()

            reopened = SmartRetailDB(
                root,
                backend_kind="objects",
                encrypt_page=enc,
                decrypt_page=dec,
                crypto_overhead=ovh,
                encrypt_record=rec_enc,
                decrypt_record=rec_dec,
            )
            self.assertEqual(reopened.execute("SELECT name FROM Products WHERE id = 7"), [{"name": "Monitor"}])
            reopened.close()


if __name__ == "__main__":
    unittest.main()
