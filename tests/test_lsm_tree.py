import os
import tempfile
import unittest
from pathlib import Path

from smartdb.core import DataType
from smartdb.indexes import BloomFilter, LSMTree
from smartdb.storage import BufferPool, StorageManager, make_aes_gcm_hooks


class LSMTreeTests(unittest.TestCase):
    def test_lsm_low_cardinality_flushes_by_entry_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            lsm = LSMTree(tmp, flush_threshold=100, compaction_threshold=10, key_dtype=DataType.STR)
            for i in range(300):
                lsm.insert(f"C{i % 3}", (0, i))
            self.assertGreaterEqual(lsm.flushes, 3)
            self.assertGreaterEqual(lsm.run_count, 3)
            self.assertEqual(len(lsm.search("C0")), 100)
            lsm.close()

    def test_lsm_memtable_wal_recovers_without_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            lsm = LSMTree(tmp, flush_threshold=128, key_dtype=DataType.INT)
            for i in range(50):
                lsm.insert(i, (0, i))
            # Simulate unclean exit: close only raw WAL handle without flushing memtable
            lsm._wal_file.close()

            reopened = LSMTree(tmp, flush_threshold=128, key_dtype=DataType.INT)
            self.assertEqual(reopened.search(25), [(0, 25)])
            self.assertEqual(len(reopened.range_search(10, 19)), 10)
            reopened.close()

    def test_lsm_atomic_compaction_and_tombstones(self):
        with tempfile.TemporaryDirectory() as tmp:
            lsm = LSMTree(tmp, flush_threshold=20, compaction_threshold=4, key_dtype=DataType.INT)
            for i in range(200):
                lsm.insert(i, (0, i))
            self.assertGreaterEqual(lsm.compactions, 1)
            self.assertLess(lsm.run_count, 4)

            lsm.delete(50, (0, 50))
            lsm.insert(50, (1, 50))
            lsm.delete(60, None)
            self.assertEqual(lsm.search(50), [(1, 50)])
            self.assertEqual(lsm.search(60), [])

            lsm.compact()
            self.assertEqual(lsm.search(50), [(1, 50)])
            self.assertEqual(lsm.search(60), [])
            rng = lsm.range_search(49, 61, include_low=True, include_high=False)
            keys = [k for k, _ in rng]
            self.assertIn(49, keys)
            self.assertIn(50, keys)
            self.assertNotIn(60, keys)
            self.assertNotIn(61, keys)
            lsm.close()

    def test_bloom_filter_skips_absent_run_reads(self):
        bf = BloomFilter(num_bits=1024, num_hashes=7)
        for i in range(100):
            bf.add(f"k{i}".encode())
        for i in range(100):
            self.assertTrue(bf.might_contain(f"k{i}".encode()))
        restored = BloomFilter.from_bytes(bf.to_bytes())
        self.assertTrue(restored.might_contain(b"k42"))

        with tempfile.TemporaryDirectory() as tmp:
            mgr = StorageManager(tmp)
            pool = BufferPool(capacity=16, storage_manager=mgr)
            lsm = LSMTree(tmp, flush_threshold=50, storage_manager=mgr, pool=pool, name_prefix="T_id", key_dtype=DataType.INT)
            for i in range(100):
                lsm.insert(i, (0, i))
            lsm.flush()
            reads_before = pool.stats.disk_reads
            self.assertEqual(lsm.search(999999), [])
            self.assertEqual(pool.stats.disk_reads, reads_before)
            lsm.close()
            pool.close()
            mgr.close_all()

    def test_lsm_with_object_store_and_aes_gcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc, dec, ovh, rec_enc, rec_dec = make_aes_gcm_hooks(os.urandom(32))
            mgr = StorageManager(
                tmp,
                backend_kind="objects",
                encrypt_page=enc,
                decrypt_page=dec,
                crypto_overhead=ovh,
                encrypt_record=rec_enc,
                decrypt_record=rec_dec,
            )
            pool = BufferPool(capacity=16, storage_manager=mgr)
            lsm = LSMTree(tmp, flush_threshold=10, storage_manager=mgr, pool=pool, name_prefix="sec", key_dtype=DataType.STR)
            for i in range(25):
                lsm.insert(f"item-{i:02d}", (0, i))
            lsm.close()
            pool.close()
            mgr.close_all()

            mgr2 = StorageManager(
                tmp,
                backend_kind="objects",
                encrypt_page=enc,
                decrypt_page=dec,
                crypto_overhead=ovh,
                encrypt_record=rec_enc,
                decrypt_record=rec_dec,
            )
            pool2 = BufferPool(capacity=16, storage_manager=mgr2)
            lsm2 = LSMTree(tmp, flush_threshold=10, storage_manager=mgr2, pool=pool2, name_prefix="sec", key_dtype=DataType.STR)
            self.assertEqual(lsm2.search("item-12"), [(0, 12)])
            self.assertGreater(lsm2.disk_bytes(), 0)
            lsm2.close()
            pool2.close()
            mgr2.close_all()


if __name__ == "__main__":
    unittest.main()
