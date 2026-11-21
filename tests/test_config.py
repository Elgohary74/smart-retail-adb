import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from smartdb.core.config import (
    BLOOM_BITS_PER_KEY,
    BPLUS_INLINE_RID_CAP,
    LSM_FLUSH_THRESHOLD,
    PAGE_SIZE,
    load_env,
    load_settings,
)
from smartdb.engine import SmartRetailDB


class ConfigTests(unittest.TestCase):
    def test_constants_and_env_parsing(self):
        self.assertEqual(PAGE_SIZE, 4096)
        self.assertEqual(BPLUS_INLINE_RID_CAP, 8)
        self.assertEqual(LSM_FLUSH_THRESHOLD, 128)
        self.assertEqual(BLOOM_BITS_PER_KEY, 10)

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# Comment line\n"
                "\n"
                "SMARTDB_BUFFER_FRAMES=32\n"
                "SMARTDB_POLICY='FIFO'\n"
                'SMARTDB_BACKEND="objects"\n',
                encoding="utf-8",
            )
            parsed = load_env(env_file)
            self.assertEqual(parsed["SMARTDB_BUFFER_FRAMES"], "32")
            self.assertEqual(parsed["SMARTDB_POLICY"], "FIFO")
            self.assertEqual(parsed["SMARTDB_BACKEND"], "objects")

            with mock.patch.dict(os.environ, {"SMARTDB_POLICY": "LRU"}, clear=False):
                settings = load_settings(env_file)
                self.assertEqual(settings.buffer_frames, 32)
                self.assertEqual(settings.replacement_policy, "LRU")
                self.assertEqual(settings.backend_kind, "objects")

    def test_invalid_settings_raise_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("SMARTDB_BUFFER_FRAMES=0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(env_file)

    def test_engine_honors_env_defaults_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"SMARTDB_BUFFER_FRAMES": "24", "SMARTDB_POLICY": "FIFO"}, clear=False):
                db1 = SmartRetailDB(Path(tmp) / "db1")
                self.assertEqual(db1.buffer_frames, 24)
                self.assertEqual(db1.replacement_policy, "FIFO")
                db1.close()

                db2 = SmartRetailDB(Path(tmp) / "db2", buffer_frames=8, replacement_policy="LRU")
                self.assertEqual(db2.buffer_frames, 8)
                self.assertEqual(db2.replacement_policy, "LRU")
                db2.close()


if __name__ == "__main__":
    unittest.main()
