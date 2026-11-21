import tempfile
import unittest

from smartdb.core import DataType
from smartdb.indexes import BPlusTree
from smartdb.storage import BufferPool, StorageManager


class BPlusTreeTests(unittest.TestCase):
    def test_standalone_bplus_insert_search_and_range_splits(self):
        tree = BPlusTree(order=4)
        for i in range(500):
            tree.insert(i, (i, 0))
        tree.insert(30, (999, 1))
        self.assertEqual(tree.search(30), [(30, 0), (999, 1)])
        self.assertEqual(tree.search(501), [])
        self.assertEqual(
            tree.range_search(10, 12, include_low=True, include_high=True),
            [(10, (10, 0)), (11, (11, 0)), (12, (12, 0))],
        )
        self.assertEqual(
            tree.range_search(10, 12, include_low=False, include_high=False),
            [(11, (11, 0))],
        )
        tree.close()

    def test_persistent_reopen_without_table_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=16, storage_manager=mgr)
            tree = BPlusTree(storage_manager=mgr, pool=pool, name="T.id", key_dtype=DataType.INT)
            for i in range(400):
                tree.insert(i * 2, (0, i))
            self.assertGreater(tree.page_count, 1)
            self.assertGreater(tree.disk_bytes(), 4096)
            pool.close()
            mgr.close_all()

            mgr2 = StorageManager(tmp)
            pool2 = BufferPool(capacity=16, storage_manager=mgr2)
            tree2 = BPlusTree(storage_manager=mgr2, pool=pool2, name="T.id", key_dtype=DataType.INT)
            self.assertEqual(tree2.search(40), [(0, 20)])
            self.assertEqual(tree2.range_search(10, 14), [(10, (0, 5)), (12, (0, 6)), (14, (0, 7))])
            pool2.close()
            mgr2.close_all()

    def test_bplus_high_duplicate_rid_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=16, storage_manager=mgr)
            tree = BPlusTree(storage_manager=mgr, pool=pool, name="T.cat", key_dtype=DataType.STR)
            for i in range(1500):
                tree.insert("Electronics", (i // 50, i % 50))
            tree.insert("Books", (99, 0))
            rids = tree.search("Electronics")
            self.assertEqual(len(rids), 1500)
            self.assertEqual(rids[0], (0, 0))
            self.assertEqual(rids[-1], (29, 49))
            self.assertEqual(tree.search("Books"), [(99, 0)])
            pool.close()
            mgr.close_all()

    def test_delete_specific_and_all_plus_null_and_bool_keys(self):
        tree = BPlusTree(order=5, key_dtype=DataType.ANY)
        tree.insert(None, (0, 0))
        tree.insert(1, (0, 1))
        tree.insert(True, (0, 2))
        tree.insert(1, (0, 3))
        self.assertEqual(tree.search(None), [(0, 0)])
        self.assertEqual(tree.search(1), [(0, 1), (0, 3)])
        self.assertEqual(tree.search(True), [(0, 2)])

        self.assertEqual(tree.delete(1, (0, 1)), 1)
        self.assertEqual(tree.search(1), [(0, 3)])
        self.assertEqual(tree.delete(1, None), 1)
        self.assertEqual(tree.search(1), [])
        self.assertEqual(tree.search(True), [(0, 2)])
        tree.close()

    def test_bucket_reclamation_and_tight_buffer_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=2, storage_manager=mgr)
            with BPlusTree(order=4, storage_manager=mgr, pool=pool, name="T.dup", key_dtype=DataType.STR) as tree:
                for i in range(600):
                    tree.insert("dup", (i // 100, i % 100))
                for i in range(600):
                    self.assertEqual(tree.delete("dup", (i // 100, i % 100)), 1)
                self.assertEqual(len(tree), 0)
                self.assertEqual(tree.search("dup"), [])
                tree.insert("dup", (9, 9))
                self.assertEqual(tree.search("dup"), [(9, 9)])
            pool.close()
            mgr.close_all()


if __name__ == "__main__":
    unittest.main()
