import tempfile
import unittest
from pathlib import Path

from smartdb.storage import PAGE_SIZE, BufferPool, PageFile


class BufferPoolTests(unittest.TestCase):
    def test_fifo_vs_lru_eviction_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for policy, expected_evicted in (("FIFO", 0), ("LRU", 1)):
                backend = PageFile(Path(tmp) / f"{policy}.pages")
                for pid in range(3):
                    backend.allocate_page()
                    backend.write_page(pid, bytes([pid]) * PAGE_SIZE)
                pool = BufferPool(backend, capacity=2, policy=policy)
                with pool.pin_page(0):
                    pass
                with pool.pin_page(1):
                    pass
                with pool.pin_page(0):
                    pass
                with pool.pin_page(2):
                    pass
                self.assertEqual(pool.stats.evictions, 1)
                self.assertNotIn((0, expected_evicted), pool._frames)
                pool.close()

    def test_pin_blocks_eviction_and_raises_when_all_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = PageFile(Path(tmp) / "pin.pages")
            for _ in range(3):
                backend.allocate_page()
            pool = BufferPool(backend, capacity=2, policy="LRU")
            f0 = pool.fetch_page(0)
            f1 = pool.fetch_page(1)
            with self.assertRaises(RuntimeError):
                pool.fetch_page(2)
            pool.unpin_page(0)
            f2 = pool.fetch_page(2)
            self.assertEqual(pool.stats.evictions, 1)
            pool.unpin_page(1)
            pool.unpin_page(2)
            self.assertEqual((f0.pin_count, f1.pin_count, f2.pin_count), (0, 0, 0))
            pool.close()

    def test_new_page_avoids_disk_read_miss_and_flushes_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.pages"
            backend = PageFile(path)
            pool = BufferPool(backend, capacity=2, policy="LRU")
            pid, frame = pool.new_page()
            frame.data[:4] = b"TEST"
            pool.unpin_page(pid, is_dirty=True)
            self.assertEqual(pool.stats.disk_reads, 0)
            pool.close()

            reopened = PageFile(path)
            self.assertEqual(reopened.read_page(pid)[:4], b"TEST")
            reopened.close()

    def test_multi_file_shared_pool_and_wal_lsn_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            b0 = PageFile(Path(tmp) / "f0.pages")
            b1 = PageFile(Path(tmp) / "f1.pages")
            b0.allocate_page()
            b1.allocate_page()
            flushed_lsn = 5
            pool = BufferPool(capacity=1, policy="LRU", wal_flushed_lsn_fn=lambda: flushed_lsn)
            f0 = pool.register_backend(b0)
            f1 = pool.register_backend(b1)

            with pool.pin_page(0, file_id=f0, dirty=True) as frame:
                frame.page_lsn = 10  # exceeds flushed_lsn=5

            with self.assertRaises(RuntimeError):
                pool.fetch_page(0, file_id=f1)

            flushed_lsn = 10
            with pool.pin_page(0, file_id=f1) as frame:
                self.assertEqual(len(frame.data), PAGE_SIZE)
            pool.close()


if __name__ == "__main__":
    unittest.main()
