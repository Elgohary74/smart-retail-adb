import tempfile
import unittest
from pathlib import Path

from smartdb.core import ColumnSpec, DataType, TableSchema
from smartdb.storage import BufferPool, RowTable, StorageManager
from smartdb.storage.pages import PageType, SlottedPage


class RowTableTests(unittest.TestCase):
    def test_slotted_page_insert_delete_keeps_stable_slot_ids_and_reclaims(self):
        buf = bytearray(4096)
        SlottedPage.init(buf, PageType.ROW_SLOTTED)
        s0 = SlottedPage.insert_payload(buf, b"alpha" * 20)
        s1 = SlottedPage.insert_payload(buf, b"bravo" * 20)
        s2 = SlottedPage.insert_payload(buf, b"charlie" * 20)
        self.assertEqual((s0, s1, s2), (0, 1, 2))
        self.assertTrue(SlottedPage.delete_slot(buf, s1))
        self.assertIsNone(SlottedPage.get_payload(buf, s1))
        self.assertEqual(SlottedPage.get_payload(buf, s2), b"charlie" * 20)
        # Re-insert large payload after compacting deleted space
        SlottedPage.compact(buf)
        self.assertEqual(SlottedPage.get_payload(buf, s2), b"charlie" * 20)
        self.assertEqual(list(SlottedPage.live_slots(buf)), [(0, b"alpha" * 20), (2, b"charlie" * 20)])

    def test_row_table_insert_get_reopen_and_delete(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("name", DataType.STR),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=8, storage_manager=mgr)
            tbl = RowTable(mgr, pool, "Products", schema)
            r0 = tbl.insert({"id": 1, "name": "Keyboard"})
            r1 = tbl.insert({"id": 2, "name": "Mouse"})
            self.assertTrue(tbl.delete(r0))
            self.assertFalse(tbl.delete(r0))
            self.assertIsNone(tbl.try_get_record(r0))
            self.assertEqual(tbl.get_record(r1), {"id": 2, "name": "Mouse"})
            pool.close()
            mgr.close_all()

            mgr2 = StorageManager(tmp)
            pool2 = BufferPool(capacity=8, storage_manager=mgr2)
            tbl2 = RowTable(mgr2, pool2, "Products", schema)
            self.assertEqual(tbl2.scan(), [{"id": 2, "name": "Mouse"}])
            pool2.close()
            mgr2.close_all()

    def test_row_table_in_place_and_relocating_update(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("note", DataType.STR),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=8, storage_manager=mgr)
            tbl = RowTable(mgr, pool, "Notes", schema)
            r0 = tbl.insert({"id": 1, "note": "short"})
            r0_updated = tbl.update(r0, {"note": "shrt"})
            self.assertEqual(r0_updated, r0)
            self.assertEqual(tbl.get_record(r0), {"id": 1, "note": "shrt"})

            # Fill page then grow r0 -> forces relocation to a new RID
            for i in range(2, 300):
                tbl.insert({"id": i, "note": "x" * 20})
            r0_reloc = tbl.update(r0, {"note": "Y" * 3000})
            self.assertNotEqual(r0_reloc, r0)
            self.assertIsNone(tbl.try_get_record(r0))
            self.assertEqual(tbl.get_record(r0_reloc)["note"], "Y" * 3000)
            pool.close()
            mgr.close_all()

    def test_oversized_record_and_1536d_vector_overflow_chain(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("desc", DataType.STR),
            ColumnSpec("vec", DataType.VECTOR, dim=1536),
        ])
        vec = [float(i % 7) * 0.25 for i in range(1536)]
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=8, storage_manager=mgr)
            tbl = RowTable(mgr, pool, "Docs", schema)
            rid = tbl.insert({"id": 1, "desc": "A" * 5000, "vec": vec})
            got = tbl.get_record(rid)
            self.assertEqual(got["id"], 1)
            self.assertEqual(got["desc"], "A" * 5000)
            self.assertEqual(got["vec"], vec)
            self.assertTrue(tbl.delete(rid))
            self.assertIsNone(tbl.try_get_record(rid))
            pool.close()
            mgr.close_all()

    def test_failed_insert_allocates_no_orphan_pages(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("price", DataType.FLOAT),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=8, storage_manager=mgr)
            tbl = RowTable(mgr, pool, "Products", schema)
            self.assertEqual(tbl.page_count, 0)
            with self.assertRaises(ValueError):
                tbl.insert({"id": 1, "price": float("nan")})
            self.assertEqual(tbl.page_count, 0)
            pool.close()
            mgr.close_all()

    def test_try_get_record_returns_none_on_out_of_bounds(self):
        schema = TableSchema([ColumnSpec("id", DataType.INT)])
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=8, storage_manager=mgr)
            tbl = RowTable(mgr, pool, "Empty", schema)
            self.assertIsNone(tbl.try_get_record((0, 0)))
            self.assertIsNone(tbl.try_get_record((99, 5)))
            pool.close()
            mgr.close_all()


if __name__ == "__main__":
    unittest.main()
