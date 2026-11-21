import tempfile
import unittest
from pathlib import Path

from smartdb.storage.page_file import PAGE_SIZE, PageFile
from smartdb.storage.buffer_pool import BufferPool
from smartdb.storage.layouts import encode_records, decode_records
from smartdb.storage.object_store import LocalObjectStore
from smartdb.storage.encryption import TransformingBackend
from smartdb.storage.schema import Column, TableSchema


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

    def test_object_store_custom_page_size_and_cached_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / "objects_small"
            store = LocalObjectStore(store_dir, page_size=512)
            p0 = store.allocate_page()
            p1 = store.allocate_page()
            self.assertEqual((p0, p1), (0, 1))
            self.assertEqual(store.page_count, 2)
            store.write_page(p1, b"X" * 512)
            store.close()

            reopened = LocalObjectStore(store_dir, page_size=512)
            self.assertEqual(reopened.page_count, 2)
            self.assertEqual(reopened.read_page(1), b"X" * 512)
            reopened.close()

    def test_transform_backend_respects_custom_page_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = PageFile(Path(tmp) / "small.pages", page_size=512)
            xor = lambda data: bytes(value ^ 0x55 for value in data)
            wrapped = TransformingBackend(raw, encrypt=xor, decrypt=xor)
            self.assertEqual(wrapped.page_size, 512)
            page_id = wrapped.allocate_page()
            content = b"M" * 512
            wrapped.write_page(page_id, content)
            self.assertEqual(wrapped.read_page(page_id), content)
            wrapped.close()

    def test_table_schema_validation_and_serialization(self):
        schema = TableSchema.from_spec([
            ("product_id", "INT", False),
            ("name", "STR", False),
            ("price", "FLOAT", True),
            ("active", "BOOL", True),
        ])
        normalized = schema.validate_record({
            "product_id": 1,
            "name": "Mouse",
            "price": 15,
            "active": True,
        })
        self.assertEqual(normalized["price"], 15.0)
        self.assertIsInstance(normalized["price"], float)

        with self.assertRaises(TypeError):
            schema.validate_record({
                "product_id": True,
                "name": "Mouse",
                "price": 15.0,
                "active": True,
            })
        with self.assertRaises(ValueError):
            schema.validate_record({
                "product_id": 1,
                "name": None,
                "price": 15.0,
                "active": True,
            })

        restored = TableSchema.from_dict(schema.to_dict())
        self.assertEqual(restored.field_names, ["product_id", "name", "price", "active"])
        self.assertEqual([c.dtype for c in restored.columns], ["INT", "STR", "FLOAT", "BOOL"])

        untyped = TableSchema.from_spec(["a", "b"])
        self.assertEqual(untyped.validate_record({"a": 1, "b": "x"}), {"a": 1, "b": "x"})

    def test_buffer_pool_fifo_vs_lru_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            for policy, expected_hits_after_recheck in (("FIFO", 1), ("LRU", 2)):
                backend = PageFile(Path(tmp) / f"{policy}.pages")
                for i in range(3):
                    pid = backend.allocate_page()
                    backend.write_page(pid, bytes([65 + i]) * PAGE_SIZE)
                pool = BufferPool(backend, capacity=2, policy=policy)
                pool.read_page(0)
                pool.read_page(1)
                pool.read_page(0)  # Hit: refreshes LRU, does not refresh FIFO
                pool.read_page(2)  # Evicts page 1 (LRU) or page 0 (FIFO)
                self.assertEqual(pool.read_page(0), b"A" * PAGE_SIZE)
                self.assertEqual(pool.stats.hits, expected_hits_after_recheck)
                pool.close()

    def test_buffer_pool_multi_stream_and_unregister(self):
        with tempfile.TemporaryDirectory() as tmp:
            b1 = PageFile(Path(tmp) / "t1.pages", page_size=512)
            b2 = PageFile(Path(tmp) / "t2.pages", page_size=512)
            b1.allocate_page()
            b2.allocate_page()
            b2.allocate_page()
            pool = BufferPool(capacity=2, policy="LRU")
            pool.register_backend("t1", b1)
            pool.register_backend("t2", b2)

            pool.write_page(0, b"1" * 512, stream_id="t1")
            pool.read_page(0, stream_id="t2")
            pool.read_page(1, stream_id="t2")  # Evicts ("t1", 0) and writes it back
            self.assertEqual(pool.stats.evictions, 1)
            self.assertEqual(pool.stats.disk_writes, 1)

            pool.write_page(0, b"2" * 512, stream_id="t2")
            pool.unregister_backend("t2")
            self.assertEqual(b2.read_page(0), b"2" * 512)
            b2.close()
            pool.close()
            with PageFile(Path(tmp) / "t1.pages", page_size=512) as reopened:
                self.assertEqual(reopened.read_page(0), b"1" * 512)

    def test_buffer_pool_validates_write_and_pinning(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = PageFile(Path(tmp) / "pin.pages", page_size=512)
            for _ in range(3):
                backend.allocate_page()
            pool = BufferPool(backend, capacity=2, policy="LRU")

            with self.assertRaises(IndexError):
                pool.write_page(99, b"A" * 512)
            with self.assertRaises(ValueError):
                pool.write_page(0, b"short")

            pool.pin_page(0)
            pool.read_page(1)
            pool.read_page(2)  # Must evict unpinned page 1 instead of pinned page 0
            self.assertIn(("default", 0), pool._frames)

            pool.pin_page(2)
            with self.assertRaises(RuntimeError):
                pool.read_page(1)  # All 2 frames are pinned

            pool.unpin_page(0)
            pool.unpin_page(2)
            pool.read_page(1)
            pool.close()


if __name__ == "__main__":
    unittest.main()
