import tempfile
import unittest
from pathlib import Path
from smartdb.indexes.bplus_tree import BPlusTree
from smartdb.indexes.lsm_tree import LSMTree
from smartdb.indexes.manager import choose_index_type


class IndexTests(unittest.TestCase):
    def test_bplus_insert_search_and_range_after_splits(self):
        tree = BPlusTree(order=4)
        for i in range(100):
            tree.insert(i, (i, 0))
        tree.insert(30, (999, 0))
        self.assertEqual(tree.search(30), [(30, 0), (999, 0)])
        self.assertEqual(len(tree.range_search(10, 12)), 3)
        self.assertEqual(tree.search(101), [])

    def test_lsm_flush_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "lsm"
            idx = LSMTree(directory, flush_threshold=2)
            idx.insert(1, (0, 0))
            idx.insert(2, (0, 1))
            idx.insert(1, (0, 2))
            idx.close()
            reopened = LSMTree(directory, flush_threshold=2)
            self.assertEqual(set(reopened.search(1)), {(0, 0), (0, 2)})
            self.assertEqual(reopened.search(2), [(0, 1)])
            reopened.close()

    def test_bplus_remove_locators_and_keys(self):
        tree = BPlusTree(order=4)
        for i in range(50):
            tree.insert(i, (i, 0))
        tree.insert(10, (999, 0))
        self.assertTrue(tree.remove(10, (10, 0)))
        self.assertEqual(tree.search(10), [(999, 0)])
        self.assertTrue(tree.remove(10, (999, 0)))
        self.assertEqual(tree.search(10), [])
        self.assertFalse(tree.remove(10, (999, 0)))
        self.assertEqual([key for key, _ in tree.range_search(9, 11)], [9, 11])
        # Empty leaf still accepts inserts and routes correctly.
        tree.insert(10, (7, 0))
        self.assertEqual(tree.search(10), [(7, 0)])
        self.assertEqual(len(tree), 50)

    def test_lsm_remove_filters_and_persists_tombstones(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "lsm"
            idx = LSMTree(directory, flush_threshold=2)
            idx.insert(1, (0, 0))
            idx.insert(1, (0, 1))
            idx.insert(2, (0, 2))
            idx.insert(3, (0, 3))
            self.assertTrue(idx.remove(1, (0, 0)))
            self.assertFalse(idx.remove(1, (0, 0)))
            self.assertEqual(idx.search(1), [(0, 1)])
            self.assertEqual(idx.range_search(1, 2), [(1, (0, 1)), (2, (0, 2))])
            idx.close()
            reopened = LSMTree(directory, flush_threshold=2)
            self.assertEqual(reopened.search(1), [(0, 1)])
            self.assertEqual(reopened.range_search(1, 3), [(1, (0, 1)), (2, (0, 2)), (3, (0, 3))])
            reopened.close()

    def test_lsm_compaction_drops_tombstoned_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "lsm"
            idx = LSMTree(directory, flush_threshold=1)
            idx.insert(1, (0, 0))
            idx.insert(2, (0, 1))
            idx.remove(1, (0, 0))
            self.assertEqual(len(idx.runs), 2)
            idx.compact()
            self.assertEqual(idx.search(1), [])
            self.assertEqual(idx.search(2), [(0, 1)])
            self.assertFalse(idx.tombstone_path.exists())
            self.assertEqual(len(idx.runs), 1)
            idx.close()

    def test_auto_selection(self):
        self.assertEqual(choose_index_type(reads=100, writes=10), "BPLUS")
        self.assertEqual(choose_index_type(reads=10, writes=100), "LSM")


if __name__ == "__main__":
    unittest.main()
