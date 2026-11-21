import os
import tempfile
import unittest
from pathlib import Path

from smartdb.storage.encryption import TransformingBackend, make_aes_gcm_hooks
from smartdb.storage.manager import StorageManager
from smartdb.storage.object_store import LocalObjectStore
from smartdb.storage.page_file import PAGE_SIZE, PageFile


class StorageBackendTests(unittest.TestCase):
    def test_page_file_custom_page_size_with_transforming_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = PageFile(Path(tmp) / "small.pages", page_size=512)
            xor = lambda page_id, data: bytes(b ^ ((page_id + 0xAA) & 0xFF) for b in data)
            wrapped = TransformingBackend(raw, encrypt=xor, decrypt=xor)
            self.assertEqual(wrapped.physical_page_size, 512)
            self.assertEqual(wrapped.usable_page_size, 512)
            pid = wrapped.allocate_page()
            payload = b"X" * 512
            wrapped.write_page(pid, payload)
            self.assertEqual(wrapped.read_page(pid), payload)
            wrapped.close()

    def test_local_object_store_cached_count_reopen_and_destroy(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_dir = Path(tmp) / "objs"
            store = LocalObjectStore(obj_dir)
            self.assertEqual(store.page_count, 0)
            p0 = store.allocate_page()
            p1 = store.allocate_page()
            self.assertEqual((p0, p1), (0, 1))
            self.assertEqual(store.page_count, 2)
            store.write_page(p1, b"B" * PAGE_SIZE)
            store.close()

            reopened = LocalObjectStore(obj_dir)
            self.assertEqual(reopened.page_count, 2)
            self.assertEqual(reopened.read_page(p1), b"B" * PAGE_SIZE)
            reopened.destroy()
            self.assertFalse(obj_dir.exists())

    def test_aes_gcm_hooks_with_overhead_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc, dec, overhead, rec_enc, rec_dec = make_aes_gcm_hooks(os.urandom(32))
            self.assertEqual(overhead, 28)
            raw = PageFile(Path(tmp) / "sec.pages")
            wrapped = TransformingBackend(raw, encrypt=enc, decrypt=dec, overhead_bytes=overhead)
            self.assertEqual(wrapped.physical_page_size, 4096)
            self.assertEqual(wrapped.usable_page_size, 4068)

            pid = wrapped.allocate_page()
            msg = b"S" * wrapped.usable_page_size
            wrapped.write_page(pid, msg)
            self.assertEqual(wrapped.read_page(pid), msg)

            # Record transform for WAL
            ct_rec = rec_enc(b"wal-entry")
            self.assertEqual(rec_dec(ct_rec), b"wal-entry")

            # Tamper physical byte on disk -> decryption must fail
            raw_bytes = bytearray(raw.read_page(pid))
            raw_bytes[-1] ^= 0x01
            raw.write_page(pid, bytes(raw_bytes))
            with self.assertRaises(Exception):
                wrapped.read_page(pid)
            wrapped.close()

    def test_storage_manager_registers_and_drops_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("page", "objects"):
                mgr = StorageManager(Path(tmp) / kind, backend_kind=kind)
                fid, backend = mgr.open_file("table:P:row", "tables/P.pages")
                pid = backend.allocate_page()
                backend.write_page(pid, b"Z" * mgr.usable_page_size)
                self.assertEqual(mgr.get_backend(fid).read_page(pid), b"Z" * mgr.usable_page_size)
                mgr.drop_file("table:P:row")
                with self.assertRaises(KeyError):
                    mgr.get_backend(fid)
                mgr.close_all()


if __name__ == "__main__":
    unittest.main()
