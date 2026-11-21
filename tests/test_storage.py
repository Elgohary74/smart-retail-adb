import tempfile
import unittest
from pathlib import Path

from smartdb.storage.page_file import PAGE_SIZE, PageFile
from smartdb.storage.buffer_pool import BufferPool
from smartdb.storage.layouts import (
    ColumnLayout,
    RowLayout,
    decode_records,
    encode_records,
)
from smartdb.storage.object_store import LocalObjectStore
from smartdb.storage.encryption import TransformingBackend
from smartdb.storage.schema import Column, TableSchema
from smartdb.storage.table import StorageTable


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

    def test_row_layout_crud_projection_and_stable_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = TableSchema.from_spec([("id", "INT"), ("note", "STR")])
            backend = PageFile(Path(tmp) / "rows.pages", page_size=256)
            pool = BufferPool(capacity=4, policy="LRU")
            pool.register_backend("t.data", backend)
            layout = RowLayout(pool, "t", schema)

            locs = [layout.insert({"id": i, "note": f"item-{i:03d}"}) for i in range(15)]
            self.assertGreater(backend.page_count, 1)

            # Deleting slot 0 on page 0 must not shift slot 1 on page 0.
            loc_after = locs[1]
            layout.delete(locs[0])
            with self.assertRaises(KeyError):
                layout.get_record(locs[0])
            self.assertEqual(layout.get_record(loc_after)["id"], 1)

            # In-place update keeps locator; oversized update relocates cleanly.
            same_loc = layout.update(loc_after, {"id": 1, "note": "short"})
            self.assertEqual(same_loc, loc_after)
            self.assertEqual(layout.get_record(same_loc)["note"], "short")

            moved_loc = layout.update(loc_after, {"id": 1, "note": "X" * 180})
            self.assertNotEqual(moved_loc, loc_after)
            with self.assertRaises(KeyError):
                layout.get_record(loc_after)
            self.assertEqual(layout.get_record(moved_loc)["note"], "X" * 180)

            projected = [row for _, row in layout.iter_records(columns=["id"])]
            self.assertEqual(len(projected), 14)
            self.assertEqual(set(projected[0].keys()), {"id"})

            batches = list(layout.iter_batches(columns=["id"], batch_size=5))
            self.assertEqual(sum(len(b["id"]) for b in batches), 14)
            pool.close()

    def test_column_layout_crud_and_projection_io_savings(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = TableSchema.from_spec([
                ("id", "INT"),
                ("price", "FLOAT"),
                ("desc_a", "STR"),
                ("desc_b", "STR"),
                ("desc_c", "STR"),
            ])
            pool = BufferPool(capacity=2, policy="LRU")
            for stream in ("t.col.id", "t.col.price", "t.col.desc_a", "t.col.desc_b", "t.col.desc_c", "t._tombstones"):
                pool.register_backend(stream, PageFile(Path(tmp) / f"{stream}.pages", page_size=512))
            layout = ColumnLayout(pool, "t", schema)

            locs = [
                layout.insert({
                    "id": i,
                    "price": float(i) + 0.5,
                    "desc_a": f"alpha-{i}-" + ("a" * 30),
                    "desc_b": f"beta-{i}-" + ("b" * 30),
                    "desc_c": f"gamma-{i}-" + ("c" * 30),
                })
                for i in range(60)
            ]
            layout.delete(locs[5])
            with self.assertRaises(KeyError):
                layout.get_record(locs[5])
            self.assertEqual(layout.get_record(locs[6])["id"], 6)

            same_loc = layout.update(locs[6], {
                "id": 6,
                "price": 99.5,
                "desc_a": "small-a",
                "desc_b": "small-b",
                "desc_c": "small-c",
            })
            self.assertEqual(same_loc, locs[6])
            self.assertEqual(layout.get_record(locs[6])["price"], 99.5)

            moved_loc = layout.update(locs[7], {
                "id": 7,
                "price": 12.5,
                "desc_a": "Z" * 350,
                "desc_b": "small-b",
                "desc_c": "small-c",
            })
            self.assertNotEqual(moved_loc, locs[7])
            with self.assertRaises(KeyError):
                layout.get_record(locs[7])
            self.assertEqual(layout.get_record(moved_loc)["desc_a"], "Z" * 350)

            pool.flush()
            pool.close()

            # Reopen with cold cache and compare projection disk_reads vs full-table scan.
            proj_pool = BufferPool(capacity=2, policy="LRU")
            for stream in ("t.col.id", "t.col.price", "t.col.desc_a", "t.col.desc_b", "t.col.desc_c", "t._tombstones"):
                proj_pool.register_backend(stream, PageFile(Path(tmp) / f"{stream}.pages", page_size=512))
            reopened = ColumnLayout(proj_pool, "t", schema)
            reads_before_proj = proj_pool.stats.disk_reads
            proj_rows = [row for _, row in reopened.iter_records(columns=["id", "price"])]
            proj_reads = proj_pool.stats.disk_reads - reads_before_proj
            self.assertEqual(len(proj_rows), 59)
            self.assertEqual(set(proj_rows[0].keys()), {"id", "price"})

            reads_before_full = proj_pool.stats.disk_reads
            full_rows = [row for _, row in reopened.iter_records()]
            full_reads = proj_pool.stats.disk_reads - reads_before_full
            self.assertEqual(len(full_rows), 59)
            self.assertLess(proj_reads, full_reads)
            proj_pool.close()

    def test_storage_table_persists_row_and_column_across_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = [("id", "INT", False), ("name", "STR", False), ("price", "FLOAT", True)]
            for layout in ("row", "column"):
                for backend_kind in ("page", "objects"):
                    table_path = Path(tmp) / f"{layout}_{backend_kind}" / "Products.pages"
                    table = StorageTable(
                        table_path,
                        fields=schema,
                        layout=layout,
                        buffer_frames=4,
                        replacement_policy="LRU",
                        backend_kind=backend_kind,
                    )
                    loc0 = table.insert({"id": 1, "name": "Keyboard", "price": 29.99})
                    loc1 = table.insert({"id": 2, "name": "Mouse", "price": 15.50})
                    table.update(loc1, {"id": 2, "name": "Gaming Mouse", "price": 25.00})
                    table.delete(loc0)
                    table.close()

                    reopened = StorageTable(
                        table_path,
                        fields=schema,
                        layout=layout,
                        buffer_frames=4,
                        replacement_policy="LRU",
                        backend_kind=backend_kind,
                    )
                    rows = reopened.scan(columns=["id", "name"])
                    self.assertEqual(rows, [{"id": 2, "name": "Gaming Mouse"}])
                    self.assertEqual(reopened.scan(field="id", value=2), [{"id": 2, "name": "Gaming Mouse", "price": 25.0}])
                    reopened.close()


if __name__ == "__main__":
    unittest.main()
