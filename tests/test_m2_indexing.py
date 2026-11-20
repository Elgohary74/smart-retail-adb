"""
Module: tests/test_m2_indexing.py
Layer: Gate 2 Adversarial Verification Suite (M2 Object Store, Indexing & Query)
Lock Domain: Test harness (imports wave-specific modules lazily inside tests)

Purpose:
    Verifies all 11 Milestone 2 Gate 2 requirements covering AES-256-GCM
    ObjectStore encryption and cross-OID swap detection, B+ Tree 50k inserts and
    structural invariants (including 3-frame latch-crabbing), float64 range
    scans, restart persistence, delete behavior, LSM WAL crash/torn-tail
    recovery, multi-level tombstone compaction, Bloom filter effectiveness,
    WorkloadAdvisor rules, injection-safe SimpleExecutor, and index-vs-fullscan
    equivalence on the deterministic retail dataset.

Key Invariants:
    1. Top-level imports contain only stdlib, ``pytest``, ``numpy``, and
       ``src.common.*`` so collection succeeds cleanly across Waves 0-4.
"""

from __future__ import annotations

import random
import struct
import zlib

import pytest

from src.common.config import MAGIC_OBJECT_DIR, MAGIC_OBJECT_RECORD
from src.common.exceptions import (
    DuplicateKeyError,
    PageCorruptedError,
    SecurityViolationError,
)
from src.common.types import RID, ColumnDef, DataType, LayoutType, Schema


def test_m2_object_store_aes256_gcm_and_tamper_detection(tmp_path) -> None:
    """Verify AES-256-GCM encryption on disk, ciphertext tamper, and cross-OID swap detection."""
    from src.storage.cipher_hooks import AES256GCMHook
    from src.storage.object_store import ObjectStore

    key = bytes(range(32))
    obj_dir = tmp_path / "objects"
    store = ObjectStore(obj_dir, cipher=AES256GCMHook(key))

    rng = random.Random(42)
    payloads: dict[int, bytes] = {}
    for oid in range(1, 21):
        size = rng.randint(10_240, 65_536)
        marker = f"JFIF_PRODUCT_SECRET_{oid:04d}_IHDR_PRODUCT_SECRET".encode("ascii")
        body = marker + rng.randbytes(size - len(marker))
        payloads[oid] = body
        store.put_object(oid, body)

    store.flush()
    store.close()

    raw_dat = (obj_dir / "objects.dat").read_bytes()
    assert b"JFIF_PRODUCT_SECRET" not in raw_dat
    assert b"IHDR_PRODUCT_SECRET" not in raw_dat

    store2 = ObjectStore(obj_dir, cipher=AES256GCMHook(key))
    for oid, expected in payloads.items():
        assert store2.exists(oid)
        assert store2.get_object(oid) == expected
    store2.close()

    # Flip 1 byte inside ciphertext of oid=1 and verify tamper detection
    dir_bytes = (obj_dir / "objects.dir").read_bytes()
    magic, entry_count = struct.unpack_from("<4sI", dir_bytes, 0)
    assert magic == MAGIC_OBJECT_DIR
    assert entry_count == 20
    dir_entries: dict[int, int] = {}
    for i in range(entry_count):
        e_oid, e_offset, _, _ = struct.unpack_from("<QQII", dir_bytes, 8 + i * 24)
        dir_entries[e_oid] = e_offset
    first_offset = dir_entries[1]

    with open(obj_dir / "objects.dat", "r+b") as f:
        tamper_pos = first_offset + 48 + 64
        f.seek(tamper_pos)
        orig = f.read(1)
        f.seek(tamper_pos)
        f.write(bytes([orig[0] ^ 0xFF]))
        f.flush()

    store3 = ObjectStore(obj_dir, cipher=AES256GCMHook(key))
    try:
        with pytest.raises((SecurityViolationError, PageCorruptedError)):
            store3.get_object(1)
    finally:
        store3.close()

    # Cross-OID swap attack: copy oid=1 nonce/tag/ciphertext into oid=2 and recompute CRC32
    swap_dir = tmp_path / "objects_swap"
    swap_store = ObjectStore(swap_dir, cipher=AES256GCMHook(key))
    same_len_a = b"SECRET_IMAGE_ONE_" + b"A" * 16_367
    same_len_b = b"SECRET_IMAGE_TWO_" + b"B" * 16_367
    swap_store.put_object(1, same_len_a)
    swap_store.put_object(2, same_len_b)
    swap_store.flush()
    swap_store.close()

    swap_dir_bytes = (swap_dir / "objects.dir").read_bytes()
    _, swap_count = struct.unpack_from("<4sI", swap_dir_bytes, 0)
    offsets: dict[int, tuple[int, int]] = {}
    for i in range(swap_count):
        e_oid, e_off, e_clen, _ = struct.unpack_from("<QQII", swap_dir_bytes, 8 + i * 24)
        offsets[e_oid] = (e_off, e_clen)

    off1, clen1 = offsets[1]
    off2, clen2 = offsets[2]
    assert clen1 == clen2

    with open(swap_dir / "objects.dat", "r+b") as f:
        f.seek(off1)
        rec1 = f.read(48 + clen1 + 4)
        f.seek(off2)
        rec2 = f.read(48 + clen2 + 4)
        assert rec1[:4] == MAGIC_OBJECT_RECORD
        assert rec2[:4] == MAGIC_OBJECT_RECORD

        # Preserve rec2's header prefix (magic, oid=2, raw_len, cipher_len) and swap nonce+tag+ciphertext
        forged_body = rec2[:20] + rec1[20 : 48 + clen1]
        forged_crc = struct.pack("<I", zlib.crc32(forged_body) & 0xFFFFFFFF)
        f.seek(off2)
        f.write(forged_body + forged_crc)
        f.flush()

    swap_store2 = ObjectStore(swap_dir, cipher=AES256GCMHook(key))
    try:
        with pytest.raises(SecurityViolationError):
            swap_store2.get_object(2)
    finally:
        swap_store2.close()


def test_m2_bplus_50k_inserts_and_invariants(tmp_path) -> None:
    """Verify 50k shuffled B+ Tree inserts, tree invariants, and 3-frame latch-crabbing."""
    from src.index.bplus_tree import BPlusTreeIndex

    idx_path = tmp_path / "pk_50k.bpt"
    idx = BPlusTreeIndex(
        idx_path,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        buffer_pool_size=64,
    )
    keys = list(range(1, 50_001))
    random.Random(42).shuffle(keys)

    expected: dict[int, RID] = {}
    for k in keys:
        rid = RID(page_id=(k // 100) + 1, slot_id=k % 100)
        expected[k] = rid
        idx.insert(k, rid)

    assert idx.validate_invariants() is True

    for k, rid in expected.items():
        assert idx.get(k) == [rid]

    with pytest.raises(DuplicateKeyError):
        idx.insert(1, RID(9999, 1))

    idx.flush()
    idx.close()

    # Reopen with a 3-frame buffer pool to enforce strict <=2 simultaneous frame pinning
    idx_tiny = BPlusTreeIndex(
        idx_path,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        buffer_pool_size=3,
    )
    try:
        assert idx_tiny.get(1) == [expected[1]]
        assert idx_tiny.get(25_000) == [expected[25_000]]
        assert idx_tiny.get(50_000) == [expected[50_000]]
        scan = idx_tiny.range_scan(100, 300, inclusive=True)
        assert scan == [(k, expected[k]) for k in range(100, 301)]
    finally:
        idx_tiny.close()


def test_m2_bplus_sibling_range_scan(tmp_path) -> None:
    """Verify B+ Tree FLOAT64 duplicate-key range scans across leaf sibling pointers."""
    from src.index.bplus_tree import BPlusTreeIndex

    idx = BPlusTreeIndex(
        tmp_path / "price.bpt",
        key_dtype=DataType.FLOAT64,
        key_size=8,
        unique=False,
        buffer_pool_size=64,
    )
    try:
        rng = random.Random(42)
        pairs: list[tuple[float, RID]] = []
        for i in range(10_000):
            # Discrete step ensures duplicate prices as well as values on bounds
            price = round(rng.uniform(-50.0, 500.0), 1)
            rid = RID(page_id=(i // 80) + 1, slot_id=i % 80)
            pairs.append((price, rid))
            idx.insert(price, rid)

        res_incl = idx.range_scan(100.0, 250.0, inclusive=True)
        expected_incl = sorted([(p, r) for p, r in pairs if 100.0 <= p <= 250.0])
        assert res_incl == expected_incl

        res_excl = idx.range_scan(100.0, 250.0, inclusive=False)
        expected_excl = sorted([(p, r) for p, r in pairs if 100.0 < p < 250.0])
        assert res_excl == expected_excl
    finally:
        idx.close()


def test_m2_bplus_restart_persistence(tmp_path) -> None:
    """Verify B+ Tree persists metadata page 0, internal nodes, and leaves across restart."""
    from src.index.bplus_tree import BPlusTreeIndex

    idx_path = tmp_path / "persist.bpt"
    idx = BPlusTreeIndex(
        idx_path,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        buffer_pool_size=32,
    )
    expected = {k: RID(page_id=k // 50 + 1, slot_id=k % 50) for k in range(1, 5001)}
    for k, rid in expected.items():
        idx.insert(k, rid)
    idx.flush()
    idx.close()

    idx2 = BPlusTreeIndex(
        idx_path,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        buffer_pool_size=32,
    )
    try:
        assert idx2.validate_invariants() is True
        for k in (1, 1234, 2500, 4999, 5000):
            assert idx2.get(k) == [expected[k]]
        assert idx2.range_scan(500, 1500, inclusive=True) == [
            (k, expected[k]) for k in range(500, 1501)
        ]
    finally:
        idx2.close()


def test_m2_lsm_wal_crash_recovery(tmp_path) -> None:
    """Verify LSM WAL crash recovery ignores torn trailing bytes and restores MemTable."""
    from src.index.lsm_tree import LSMTreeIndex

    lsm_dir = tmp_path / "lsm_crash"
    lsm = LSMTreeIndex(
        lsm_dir,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=1_048_576,
    )
    expected: dict[int, RID] = {}
    for k in range(1, 501):
        rid = RID(page_id=k // 40 + 1, slot_id=k % 40)
        expected[k] = rid
        lsm.insert(k, rid)

    assert lsm.get_stats()["sstable_count"] == 0
    # Simulate abrupt process crash without calling flush() or close()
    del lsm

    # Append partial/corrupt trailing bytes simulating a torn WAL write at crash time
    wal_path = lsm_dir / "lsm.wal"
    with open(wal_path, "ab") as f:
        f.write(b"\xDE\xAD\xBE\xEF\x01\x00\x20\x00\x11\x22")
        f.flush()

    lsm2 = LSMTreeIndex(
        lsm_dir,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=1_048_576,
    )
    for k, rid in expected.items():
        assert lsm2.get(k) == [rid]

    # Truncated torn tail must allow clean subsequent appends across another restart
    extra_rid = RID(page_id=77, slot_id=7)
    lsm2.insert(501, extra_rid)
    lsm2.close()

    lsm3 = LSMTreeIndex(
        lsm_dir,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=1_048_576,
    )
    try:
        assert lsm3.get(501) == [extra_rid]
        assert lsm3.get(250) == [expected[250]]
    finally:
        lsm3.close()


def test_m2_lsm_tombstone_compaction(tmp_path) -> None:
    """Verify LSM updates and deletes never resurrect stale SSTable values after compaction."""
    from src.index.lsm_tree import LSMTreeIndex

    lsm_dir = tmp_path / "lsm_compact"
    lsm = LSMTreeIndex(
        lsm_dir,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=16_384,
        compaction_trigger=4,
    )
    ground_truth: dict[int, RID] = {}
    for k in range(5000):
        rid = RID(page_id=k // 64 + 1, slot_id=k % 64)
        ground_truth[k] = rid
        lsm.insert(k, rid)

    for k in range(1000):
        new_rid = RID(page_id=9000 + k // 64, slot_id=k % 64)
        ground_truth[k] = new_rid
        lsm.insert(k, new_rid)

    for k in range(1000, 2000):
        assert lsm.delete(k) is True
        del ground_truth[k]

    lsm.flush()
    lsm.compact()

    for k in range(1000, 2000):
        assert lsm.get(k) == []
    for k in range(1000):
        assert lsm.get(k) == [ground_truth[k]]
    for k in range(2000, 5000):
        assert lsm.get(k) == [ground_truth[k]]

    lsm.close()

    lsm2 = LSMTreeIndex(
        lsm_dir,
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=16_384,
        compaction_trigger=4,
    )
    try:
        for k in range(1000, 2000):
            assert lsm2.get(k) == []
        for k in (0, 999, 2000, 4999):
            assert lsm2.get(k) == [ground_truth[k]]
    finally:
        lsm2.close()


def test_m2_lsm_bloom_filter_effectiveness(tmp_path) -> None:
    """Verify SSTable Bloom filters reject >95% of missing key lookups without disk reads."""
    from src.index.lsm_tree import LSMTreeIndex

    lsm = LSMTreeIndex(
        tmp_path / "lsm_bloom",
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=1_048_576,
    )
    try:
        for i in range(10_000):
            k = i * 2
            lsm.insert(k, RID(page_id=i // 100 + 1, slot_id=i % 100))

        # Force MemTable to SSTable so all point lookups exercise SSTable Bloom filters
        lsm.flush()
        assert lsm.get_stats()["sstable_count"] >= 1

        for i in range(5_000):
            missing_k = i * 2 + 1
            assert lsm.get(missing_k) == []

        stats = lsm.get_stats()
        assert stats["bloom_checks"] >= 5_000
        assert stats["bloom_rejection_rate"] > 0.95
    finally:
        lsm.close()


def test_m2_workload_advisor_decision_rules() -> None:
    """Verify WorkloadAdvisor enforces FULL_SCAN, LSM_TREE, and BPLUS_TREE boundaries."""
    from src.index.index_advisor import IndexStrategy, WorkloadAdvisor

    advisor = WorkloadAdvisor()

    # Rule 1a: cold start (< 200 ops) -> FULL_SCAN
    for _ in range(100):
        advisor.record("POINT_READ", "products", "price")
    rec_cold = advisor.recommend("products", "price", table_pages=50)
    assert rec_cold.strategy == IndexStrategy.FULL_SCAN
    assert rec_cold.n_ops == 100

    # Rule 1b: small table (< 4 pages) -> FULL_SCAN even with many ops
    for _ in range(250):
        advisor.record("POINT_READ", "tiny_tbl", "id")
    rec_small = advisor.recommend("tiny_tbl", "id", table_pages=2)
    assert rec_small.strategy == IndexStrategy.FULL_SCAN

    # Rule 2: write-heavy point workload (alpha > 0.45, beta < 0.25) -> LSM_TREE
    for _ in range(500):
        advisor.record("WRITE", "inventory", "product_id")
    for _ in range(100):
        advisor.record("POINT_READ", "inventory", "product_id")
    rec_lsm = advisor.recommend("inventory", "product_id", table_pages=50)
    assert rec_lsm.strategy == IndexStrategy.LSM_TREE
    assert rec_lsm.alpha > 0.45
    assert rec_lsm.beta < 0.25

    # Rule 3: read-heavy or range-scan workload (alpha <= 0.45 or beta >= 0.25) -> BPLUS_TREE
    for _ in range(100):
        advisor.record("WRITE", "orders", "order_date")
    for _ in range(500):
        advisor.record("RANGE_READ", "orders", "order_date")
    rec_bpt = advisor.recommend("orders", "order_date", table_pages=50)
    assert rec_bpt.strategy == IndexStrategy.BPLUS_TREE
    assert rec_bpt.alpha <= 0.45
    assert rec_bpt.beta >= 0.25

    assert len(advisor.decision_log) >= 4


def test_m2_sql_executor_and_injection_defense(tmp_path) -> None:
    """Verify parameterized SimpleExecutor DDL/DML and SQL injection blocking."""
    from src.index.index_advisor import WorkloadAdvisor
    from src.query.catalog import CatalogManager
    from src.query.simple_executor import SimpleExecutor
    from src.storage.storage_engine import StorageEngine

    db_dir = tmp_path / "exec_db"
    products_schema = Schema(
        table_name="products",
        columns=(
            ColumnDef("product_id", DataType.INT64),
            ColumnDef("category", DataType.VARCHAR, max_len=32),
            ColumnDef("price", DataType.FLOAT64),
            ColumnDef("image_oid", DataType.UINT64),
        ),
        primary_key="product_id",
    )

    engine = StorageEngine(db_dir, buffer_pool_size=32)
    engine.open()
    engine.create_table(products_schema, LayoutType.ROW)
    catalog = CatalogManager(db_dir / "catalog.json")
    catalog.register_table(
        products_schema,
        LayoutType.ROW,
        file_path=db_dir / "tables" / "products.db",
    )
    executor = SimpleExecutor(
        engine,
        catalog,
        advisor=WorkloadAdvisor(),
        indexes_dir=db_dir / "indexes",
    )
    try:
        executor.execute("CREATE INDEX idx_products_pk ON products (product_id) USING BTREE")
        executor.execute("CREATE INDEX idx_products_price ON products (price) USING BTREE")

        sample_rows = [
            (1, "Books", 15.0, 101),
            (2, "Electronics", 299.0, 102),
            (3, "Home", 45.0, 103),
            (4, "Apparel", 65.0, 104),
            (5, "Books", 25.0, 105),
        ]
        for row in sample_rows:
            executor.execute("INSERT INTO products VALUES (?, ?, ?, ?)", row)

        res_pk = executor.execute("SELECT * FROM products WHERE product_id = ?", (3,))
        assert len(res_pk) == 1
        assert res_pk[0]["category"] == "Home"
        assert res_pk[0]["price"] == 45.0

        res_rng = executor.execute(
            "SELECT * FROM products WHERE price BETWEEN ? AND ?",
            (20.0, 70.0),
        )
        assert sorted(r["product_id"] for r in res_rng) == [3, 4, 5]

        executor.execute("UPDATE products SET price = ? WHERE product_id = ?", (99.5, 3))
        res_upd = executor.execute("SELECT * FROM products WHERE product_id = ?", (3,))
        assert len(res_upd) == 1
        assert res_upd[0]["price"] == 99.5
        res_old_rng = executor.execute(
            "SELECT * FROM products WHERE price BETWEEN ? AND ?",
            (40.0, 50.0),
        )
        assert res_old_rng == []
        res_new_rng = executor.execute(
            "SELECT * FROM products WHERE price BETWEEN ? AND ?",
            (99.0, 100.0),
        )
        assert [r["product_id"] for r in res_new_rng] == [3]

        executor.execute("DELETE FROM products WHERE product_id = ?", (3,))
        assert executor.execute("SELECT * FROM products WHERE product_id = ?", (3,)) == []

        # SQL injection and unparameterized literal defense
        malicious_queries: list[tuple[str, tuple[object, ...]]] = [
            ("SELECT * FROM products WHERE category = 'Books'; DROP TABLE products;", ()),
            ("SELECT * FROM products WHERE category = 'Books'", ()),
            ("SELECT * FROM products WHERE product_id = 1 OR 1=1", ()),
            ("SELECT * FROM products WHERE product_id = ? -- drop table", (1,)),
        ]
        for bad_sql, bad_params in malicious_queries:
            with pytest.raises(SecurityViolationError):
                executor.execute(bad_sql, bad_params)
    finally:
        executor.close()
        engine.close()


def test_m2_index_vs_fullscan_equivalence(tmp_path) -> None:
    """Verify B+ Tree, LSM Tree, and FullTableScan yield identical RIDs on 200 queries."""
    from bench.retail_loader import load_retail_dataset
    from src.index.bplus_tree import BPlusTreeIndex
    from src.index.lsm_tree import LSMTreeIndex
    from src.storage.storage_engine import StorageEngine

    db_dir = tmp_path / "retail_tiny"
    load_retail_dataset(db_dir, scale="tiny", seed=42, build_indexes=True)

    engine = StorageEngine(db_dir, buffer_pool_size=64)
    engine.open()
    bpt = BPlusTreeIndex(
        tmp_path / "eq_price.bpt",
        key_dtype=DataType.FLOAT64,
        key_size=8,
        unique=False,
        buffer_pool_size=64,
    )
    lsm = LSMTreeIndex(
        tmp_path / "eq_price_lsm",
        key_dtype=DataType.FLOAT64,
        key_size=8,
        unique=False,
        memtable_size_bytes=8_192,
    )
    try:
        rows = engine.get_table("products").full_scan()
        assert len(rows) == 250
        for rid, row in rows:
            bpt.insert(row["price"], rid)
            lsm.insert(row["price"], rid)
        bpt.flush()
        lsm.flush()

        prices = sorted({row["price"] for _, row in rows})
        rng = random.Random(42)

        # 100 randomized point queries (mix of existing and non-existing prices)
        for q_idx in range(100):
            target = rng.choice(prices) if q_idx < 80 else round(rng.uniform(1.0, 3000.0), 4)
            fs_rids = sorted(rid for rid, row in rows if row["price"] == target)
            bpt_rids = bpt.get(target)
            lsm_rids = lsm.get(target)
            assert bpt_rids == fs_rids
            assert lsm_rids == fs_rids

        # 100 randomized range queries
        for _ in range(100):
            a = rng.uniform(0.0, 2600.0)
            b = rng.uniform(0.0, 2600.0)
            low, high = (a, b) if a <= b else (b, a)
            fs_rids = sorted(rid for rid, row in rows if low <= row["price"] <= high)
            bpt_rids = sorted(rid for _, rid in bpt.range_scan(low, high, inclusive=True))
            lsm_rids = sorted(rid for _, rid in lsm.range_scan(low, high, inclusive=True))
            assert bpt_rids == fs_rids
            assert lsm_rids == fs_rids
    finally:
        bpt.close()
        lsm.close()
        engine.close()


def test_m2_bplus_delete_and_lsm_range_scan(tmp_path) -> None:
    """Verify B+ Tree unique/non-unique delete and LSM multi-tier range scans."""
    from src.index.bplus_tree import BPlusTreeIndex
    from src.index.lsm_tree import LSMTreeIndex

    # Part 1: B+ Tree unique and non-unique delete semantics
    bpt_uniq = BPlusTreeIndex(
        tmp_path / "bpt_del_uniq.bpt",
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        buffer_pool_size=32,
    )
    try:
        expected_u = {k: RID(k // 50 + 1, k % 50) for k in range(2000)}
        for k, rid in expected_u.items():
            bpt_uniq.insert(k, rid)

        for k in range(0, 2000, 2):
            assert bpt_uniq.delete(k) is True
            del expected_u[k]
        assert bpt_uniq.delete(999_999) is False

        assert bpt_uniq.validate_invariants() is True
        for k in range(0, 2000, 2):
            assert bpt_uniq.get(k) == []
        for k, rid in expected_u.items():
            assert bpt_uniq.get(k) == [rid]

        assert bpt_uniq.range_scan(100, 110, inclusive=True) == [
            (k, expected_u[k]) for k in range(101, 110, 2)
        ]
    finally:
        bpt_uniq.close()

    bpt_dup = BPlusTreeIndex(
        tmp_path / "bpt_del_dup.bpt",
        key_dtype=DataType.INT64,
        key_size=8,
        unique=False,
        buffer_pool_size=32,
    )
    try:
        r1, r2, r3 = RID(1, 1), RID(1, 2), RID(1, 3)
        bpt_dup.insert(42, r1)
        bpt_dup.insert(42, r2)
        bpt_dup.insert(42, r3)
        assert bpt_dup.delete(42, r2) is True
        assert bpt_dup.get(42) == [r1, r3]
        assert bpt_dup.delete(42) is True
        assert bpt_dup.get(42) == []
    finally:
        bpt_dup.close()

    # Part 2: LSM multi-tier range_scan merging SSTables, MemTable, overwrites, and tombstones
    lsm = LSMTreeIndex(
        tmp_path / "lsm_range",
        key_dtype=DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=8_192,
        compaction_trigger=4,
    )
    try:
        truth: dict[int, RID] = {}
        for k in range(3000):
            rid = RID(page_id=k // 64 + 1, slot_id=k % 64)
            truth[k] = rid
            lsm.insert(k, rid)

        # Overwrite and delete keys across active MemTable while SSTables exist on disk
        for k in range(500, 700):
            new_rid = RID(page_id=8000, slot_id=k)
            truth[k] = new_rid
            lsm.insert(k, new_rid)
        for k in range(600, 650):
            assert lsm.delete(k) is True
            del truth[k]

        got_incl = lsm.range_scan(450, 750, inclusive=True)
        expected_incl = sorted((k, truth[k]) for k in range(450, 751) if k in truth)
        assert got_incl == expected_incl

        got_excl = lsm.range_scan(450, 750, inclusive=False)
        expected_excl = sorted((k, truth[k]) for k in range(451, 750) if k in truth)
        assert got_excl == expected_excl
    finally:
        lsm.close()
