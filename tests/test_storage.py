import tempfile
import unittest
from pathlib import Path

from smartdb.storage.page_file import PAGE_SIZE, PageFile
from smartdb.storage.buffer_pool import BufferPool
from smartdb.storage.layouts import encode_records, decode_records
from smartdb.storage.object_store import LocalObjectStore
from smartdb.storage.encryption import TransformingBackend


class PageFileTests(unittest.TestCase):
    def test_allocate_write_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.pages"
            pf = PageFile(path)
            page_id = pf.allocate_page()
            self.assertEqual(page_id, 0)
            expected = b"A" * PAGE_SIZE
            pf.write_page(page_id, expected)
            pf.close()
            with PageFile(path) as reopened:
                self.assertEqual(reopened.read_page(0), expected)
                self.assertEqual(reopened.page_count, 1)

    def test_invalid_page_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            pf = PageFile(Path(tmp) / "db.pages")
            with self.assertRaises(IndexError):
                pf.read_page(0)
            pf.allocate_page()
            with self.assertRaises(ValueError):
                pf.write_page(0, b"too short")
            with self.assertRaises(ValueError):
                pf.read_page(-1)
            pf.close()

    def test_buffer_pool_hit_and_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = PageFile(Path(tmp) / "db.pages")
            for _ in range(3):
                backend.allocate_page()
            pool = BufferPool(backend, capacity=2, policy="LRU")
            pool.read_page(0)
            pool.read_page(0)
            pool.read_page(1)
            pool.read_page(2)
            self.assertEqual(pool.stats.hits, 1)
            self.assertEqual(pool.stats.misses, 3)
            self.assertEqual(pool.stats.evictions, 1)
            pool.close()

    def test_row_and_column_layout_round_trip(self):
        records = [{"product_id": 1, "name": "Book"}, {"product_id": 2, "name": "Pen"}]
        for layout in ("row", "column"):
            self.assertEqual(decode_records(encode_records(records, layout)), records)

    def test_transform_hooks_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = PageFile(Path(tmp) / "wrapped.pages")
            xor = lambda data: bytes(value ^ 0xAA for value in data)
            wrapped = TransformingBackend(raw, encrypt=xor, decrypt=xor)
            page_id = wrapped.allocate_page()
            content = b"Q" * PAGE_SIZE
            wrapped.write_page(page_id, content)
            self.assertEqual(wrapped.read_page(page_id), content)
            wrapped.close()

    def test_object_store_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp) / "objects")
            page_id = store.allocate_page()
            content = b"Z" * PAGE_SIZE
            store.write_page(page_id, content)
            self.assertEqual(store.read_page(page_id), content)
            store.close()


if __name__ == "__main__":
    unittest.main()
