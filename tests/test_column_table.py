import tempfile
import unittest

from smartdb.schema import ColumnSpec, DataType, TableSchema
from smartdb.storage.buffer_pool import BufferPool
from smartdb.storage.column_table import ColumnTable
from smartdb.storage.manager import StorageManager


class ColumnTableTests(unittest.TestCase):
    def test_column_table_round_trip_reopen_update_and_delete(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("name", DataType.STR),
            ColumnSpec("price", DataType.FLOAT),
            ColumnSpec("active", DataType.BOOL),
            ColumnSpec("vec", DataType.VECTOR, dim=3),
            ColumnSpec("extra", DataType.ANY),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=16, storage_manager=mgr)
            tbl = ColumnTable(mgr, pool, "Items", schema)
            r0 = tbl.insert({"id": 1, "name": "A", "price": 10.5, "active": True, "vec": [1.0, 2.0, 3.0], "extra": None})
            r1 = tbl.insert({"id": 2, "name": "B", "price": None, "active": False, "vec": [0.0, 0.5, 1.5], "extra": "ok"})
            r0_new = tbl.update(r0, {"price": 12.0})
            self.assertNotEqual(r0_new, r0)
            self.assertIsNone(tbl.try_get_record(r0))
            self.assertTrue(tbl.delete(r1))
            self.assertIsNone(tbl.try_get_record(r1))
            self.assertEqual(tbl.get_record(r0_new)["price"], 12.0)
            pool.close()
            mgr.close_all()

            mgr2 = StorageManager(tmp)
            pool2 = BufferPool(capacity=16, storage_manager=mgr2)
            tbl2 = ColumnTable(mgr2, pool2, "Items", schema)
            rows = tbl2.scan()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], 1)
            self.assertEqual(rows[0]["price"], 12.0)
            self.assertIsNone(tbl2.try_get_record((0, 999999)))
            pool2.close()
            mgr2.close_all()

    def test_projection_pushdown_avoids_reading_unprojected_columns(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("price", DataType.FLOAT),
            ColumnSpec("big_notes", DataType.STR),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=32, storage_manager=mgr)
            tbl = ColumnTable(mgr, pool, "Big", schema)
            for i in range(1500):
                tbl.insert({"id": i, "price": float(i), "big_notes": ("N" * 200) + str(i)})
            pool.close()
            mgr.close_all()

            # Reopen with fresh cold BufferPool
            mgr2 = StorageManager(tmp)
            pool2 = BufferPool(capacity=16, storage_manager=mgr2)
            tbl2 = ColumnTable(mgr2, pool2, "Big", schema)
            before_reads = pool2.stats.disk_reads
            batches = list(tbl2.scan_batches(fields=["price"], batch_size=500))
            after_reads = pool2.stats.disk_reads

            self.assertEqual(sum(len(b["price"]) for b in batches), 1500)
            self.assertEqual(batches[0]["price"][:3], [0.0, 1.0, 2.0])
            # big_notes takes ~80+ pages; projecting only price reads <= 10 pages total
            self.assertLess(after_reads - before_reads, 15)
            pool2.close()
            mgr2.close_all()


if __name__ == "__main__":
    unittest.main()
