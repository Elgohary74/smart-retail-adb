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

    def test_auto_selection(self):
        self.assertEqual(choose_index_type(reads=100, writes=10), "BPLUS")
        self.assertEqual(choose_index_type(reads=10, writes=100), "LSM")


if __name__ == "__main__":
    unittest.main()
