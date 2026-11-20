"""
Module: tests/test_m1_storage.py
Layer: Gate 1 Adversarial Verification Suite (Common Codecs & M1 Storage Engine)
Lock Domain: Test harness (imports wave-specific storage modules lazily inside tests)

Purpose:
    Verifies Layer 0 order-preserving key codecs and all 11 Milestone 1 Gate 1
    requirements covering PageHeader binary packing, BLAKE2b torn-page
    detection, FSM allocation reuse, SlottedPage CRUD/compaction, zero-copy
    NumPy ColumnPage execution, LRU/FIFO eviction, pin protection, dirty
    write-back, hit/miss telemetry, and restart persistence.

Key Invariants:
    1. Top-level imports contain only stdlib, ``pytest``, ``numpy``, and
       ``src.common.*`` so focused runs never fail collection on unimplemented
       storage modules in early waves.
    2. Schema definitions in restart tests are standalone without importing
       ``bench.retail_loader``.
"""

from __future__ import annotations

import math
import random
import struct

import numpy as np
import pytest

from src.common.config import (
    CHUNK_ROWS,
    HEADER_SIZE,
    INVALID_PAGE_ID,
    PAGE_SIZE,
)
from src.common.exceptions import (
    BufferPoolFullError,
    PageCorruptedError,
    RecordTooLargeError,
)
from src.common.types import (
    RID,
    ColumnDef,
    DataType,
    LayoutType,
    Schema,
    decode_key,
    encode_key,
    make_composite_key,
    split_composite_key,
)


def test_common_key_encoding_total_order() -> None:
    """Verify order-preserving key encoding/decoding across all DataTypes."""
    # FLOAT64: negative values, -0.0 normalization, infinities, and NaN rejection
    floats = [
        -math.inf,
        -1e12,
        -1000.5,
        -0.5,
        -1e-12,
        0.0,
        1e-12,
        0.5,
        1000.5,
        1e12,
        math.inf,
    ]
    encoded_floats = [encode_key(v, DataType.FLOAT64, 8) for v in floats]
    assert all(len(b) == 8 for b in encoded_floats)
    for i in range(len(encoded_floats) - 1):
        assert encoded_floats[i] < encoded_floats[i + 1]
    for orig, raw in zip(floats, encoded_floats):
        assert decode_key(raw, DataType.FLOAT64) == orig

    neg_zero = encode_key(-0.0, DataType.FLOAT64, 8)
    pos_zero = encode_key(+0.0, DataType.FLOAT64, 8)
    assert neg_zero == pos_zero
    decoded_zero = decode_key(neg_zero, DataType.FLOAT64)
    assert decoded_zero == 0.0
    assert math.copysign(1.0, decoded_zero) > 0

    with pytest.raises(ValueError):
        encode_key(float("nan"), DataType.FLOAT64, 8)

    # INT32: full signed 32-bit range ordering
    int32_vals = [-(1 << 31), -100_000, -1, 0, 1, 100_000, (1 << 31) - 1]
    enc_i32 = [encode_key(v, DataType.INT32, 4) for v in int32_vals]
    assert all(len(b) == 4 for b in enc_i32)
    for i in range(len(enc_i32) - 1):
        assert enc_i32[i] < enc_i32[i + 1]
    assert [decode_key(b, DataType.INT32) for b in enc_i32] == int32_vals

    # INT64: full signed 64-bit range ordering
    int64_vals = [-(1 << 63), -10**15, -1, 0, 1, 10**15, (1 << 63) - 1]
    enc_i64 = [encode_key(v, DataType.INT64, 8) for v in int64_vals]
    assert all(len(b) == 8 for b in enc_i64)
    for i in range(len(enc_i64) - 1):
        assert enc_i64[i] < enc_i64[i + 1]
    assert [decode_key(b, DataType.INT64) for b in enc_i64] == int64_vals

    # UINT64: unsigned 64-bit ordering
    uint64_vals = [0, 1, 1 << 32, (1 << 64) - 1]
    enc_u64 = [encode_key(v, DataType.UINT64, 8) for v in uint64_vals]
    for i in range(len(enc_u64) - 1):
        assert enc_u64[i] < enc_u64[i + 1]
    assert [decode_key(b, DataType.UINT64) for b in enc_u64] == uint64_vals

    # VARCHAR: UTF-8 padding, lexicographical order, and length rejection
    strings = ["", "A", "Apparel", "Books", "Electronics", "a", "é"]
    enc_str = [encode_key(s, DataType.VARCHAR, 16) for s in strings]
    assert all(len(b) == 16 for b in enc_str)
    for i in range(len(enc_str) - 1):
        assert enc_str[i] < enc_str[i + 1]
    assert [decode_key(b, DataType.VARCHAR) for b in enc_str] == strings

    with pytest.raises((ValueError, RecordTooLargeError)):
        encode_key("x" * 17, DataType.VARCHAR, 16)

    # Composite key + RID ordering and round-trip
    base = encode_key(42.5, DataType.FLOAT64, 8)
    rids = [RID(1, 2), RID(1, 256), RID(2, 0)]
    comp = [make_composite_key(base, r) for r in rids]
    assert comp[0] < comp[1] < comp[2]
    for r, c in zip(rids, comp):
        raw_k, dec_r = split_composite_key(c, 8)
        assert raw_k == base
        assert dec_r == r

    # Schema.encode_row type enforcement for FLOAT64 (rejects bool and str, accepts numerics)
    price_schema = Schema(
        table_name="price_check",
        columns=(ColumnDef("id", DataType.INT64), ColumnDef("price", DataType.FLOAT64)),
        primary_key="id",
    )
    for valid_price in (10, 12.5, np.float64(99.25)):
        encoded_row = price_schema.encode_row({"id": 1, "price": valid_price})
        assert price_schema.decode_row(encoded_row) == {"id": 1, "price": float(valid_price)}
    for bad_price in (True, False, "9.99"):
        with pytest.raises(TypeError):
            price_schema.encode_row({"id": 1, "price": bad_price})


def test_m1_page_header_binary_packing() -> None:
    """Verify exact 32-byte PageHeader pack/unpack, checksums, and LSN hooks."""
    from src.storage.page_header import PageFlags, PageHeader, PageType

    hdr = PageHeader(
        page_id=0xFFFFFFFF,
        page_type=PageType.BPLUS_LEAF,
        flags=PageFlags.IS_ROOT | PageFlags.ENCRYPTED,
        slot_count=0xFFFF,
        page_lsn=0xFFFFFFFFFFFFFFFF,
        free_space_ptr=0xFFFFFFFF,
        next_page_id=0xFFFFFFFE,
        checksum=0x0123456789ABCDEF,
    )
    packed = hdr.pack()
    assert len(packed) == HEADER_SIZE == 32

    unpacked = PageHeader.unpack(packed)
    assert unpacked.page_id == 0xFFFFFFFF
    assert unpacked.page_type == PageType.BPLUS_LEAF
    assert unpacked.flags == (PageFlags.IS_ROOT | PageFlags.ENCRYPTED)
    assert unpacked.slot_count == 0xFFFF
    assert unpacked.page_lsn == 0xFFFFFFFFFFFFFFFF
    assert unpacked.free_space_ptr == 0xFFFFFFFF
    assert unpacked.next_page_id == 0xFFFFFFFE
    assert unpacked.checksum == 0x0123456789ABCDEF

    buf = bytearray(PAGE_SIZE)
    page_hdr = PageHeader(
        page_id=7,
        page_type=PageType.SLOTTED_ROW,
        flags=PageFlags.NONE,
        slot_count=3,
        page_lsn=0,
        free_space_ptr=7900,
        next_page_id=INVALID_PAGE_ID,
        checksum=0,
    )
    page_hdr.pack_into(buf, 0)
    buf[4000:4010] = b"0123456789"
    PageHeader.stamp_checksum(buf)
    assert PageHeader.verify_checksum(buf) is True

    PageHeader.set_page_lsn(buf, 1234567890123)
    assert PageHeader.get_page_lsn(buf) == 1234567890123
    PageHeader.stamp_checksum(buf)
    assert PageHeader.verify_checksum(buf) is True

    # Single-bit corruption anywhere in payload must fail verification
    buf[4005] ^= 0x01
    assert PageHeader.verify_checksum(buf) is False


def test_m1_torn_page_checksum_detection(tmp_path) -> None:
    """Verify DiskManager detects 1-byte torn page corruption via PageCorruptedError."""
    from src.storage.disk_manager import DiskManager
    from src.storage.page_header import PageHeader, PageType

    db_file = tmp_path / "torn_test.db"
    dm = DiskManager(db_file)
    pid = dm.allocate_page()

    buf = bytearray(PAGE_SIZE)
    PageHeader(
        page_id=pid,
        page_type=PageType.SLOTTED_ROW,
        flags=0,
        slot_count=1,
        page_lsn=0,
        free_space_ptr=PAGE_SIZE - 32,
        next_page_id=INVALID_PAGE_ID,
    ).pack_into(buf, 0)
    buf[4096] = 0x5A
    dm.write_page(pid, buf)
    dm.sync()
    dm.close()

    # Flip 1 byte directly on disk at offset 4096 within the allocated page
    with open(db_file, "r+b") as f:
        corrupt_offset = pid * PAGE_SIZE + 4096
        f.seek(corrupt_offset)
        orig = f.read(1)
        f.seek(corrupt_offset)
        f.write(bytes([orig[0] ^ 0xFF]))
        f.flush()

    dm2 = DiskManager(db_file)
    try:
        with pytest.raises(PageCorruptedError):
            dm2.read_page(pid)
    finally:
        dm2.close()


def test_m1_fsm_allocation_reuse(tmp_path) -> None:
    """Verify DiskManager Page 0 FSM allocates sequentially and reuses freed IDs."""
    from src.storage.disk_manager import DiskManager

    db_file = tmp_path / "fsm_test.db"
    dm = DiskManager(db_file)
    allocated = [dm.allocate_page() for _ in range(10)]
    assert allocated == list(range(1, 11))
    assert dm.num_pages() == 11

    dm.deallocate_page(3)
    dm.deallocate_page(7)
    assert not dm.is_allocated(3)
    assert not dm.is_allocated(7)

    reused_1 = dm.allocate_page()
    reused_2 = dm.allocate_page()
    assert {reused_1, reused_2} == {3, 7}
    assert reused_1 == 3
    assert reused_2 == 7
    assert dm.num_pages() == 11

    next_pid = dm.allocate_page()
    assert next_pid == 11
    assert dm.num_pages() == 12
    dm.sync()
    dm.close()

    dm2 = DiskManager(db_file)
    try:
        assert dm2.is_allocated(3)
        assert dm2.is_allocated(7)
        assert dm2.is_allocated(11)
        assert set(dm2.allocated_pages()) == set(range(1, 12))
    finally:
        dm2.close()


def test_m1_slotted_page_crud_and_compaction() -> None:
    """Verify SlottedPage CRUD, MVCC hooks, RID preservation, and compaction."""
    from src.storage.slotted_page import SlottedPage

    buf = bytearray(PAGE_SIZE)
    sp = SlottedPage.init_page(buf, page_id=1)

    with pytest.raises(RecordTooLargeError):
        sp.insert_tuple(b"Z" * PAGE_SIZE)

    inserted: dict[int, bytes] = {}
    i = 0
    while True:
        payload = f"row-{i:04d}-".encode("ascii") + b"X" * 119
        slot_id = sp.insert_tuple(payload)
        if slot_id is None:
            break
        inserted[slot_id] = payload
        i += 1

    assert len(inserted) >= 40
    # Verify MVCC header defaults (xmin=1, xmax=0) and set_tuple_xmax hook
    assert sp.get_tuple_mvcc_header(1) == (1, 0)
    sp.set_tuple_xmax(1, 99)
    assert sp.get_tuple_mvcc_header(1) == (1, 99)
    sp.set_tuple_xmax(1, 0)

    # Delete alternating even slots without shifting odd slot IDs
    for sid in list(inserted):
        if sid % 2 == 0:
            assert sp.delete_tuple(sid) is True
            assert sp.get_tuple(sid) is None

    for sid, expected in inserted.items():
        if sid % 2 == 1:
            assert sp.get_tuple(sid) == expected

    # Before compaction, contiguous free space cannot hold a large tuple
    large_size = sp.contiguous_free_space() + 500
    assert sp.total_free_space() >= large_size + 20

    # Compaction reclaims fragmented holes while keeping every odd slot_id intact
    sp.compact()
    assert sp.contiguous_free_space() >= large_size + 20
    for sid, expected in inserted.items():
        if sid % 2 == 1:
            assert sp.get_tuple(sid) == expected

    large_payload = b"L" * large_size
    large_slot = sp.insert_tuple(large_payload)
    assert large_slot is not None
    assert sp.get_tuple(large_slot) == large_payload

    # Also verify insert_tuple automatically compacts when fragmented holes suffice
    buf2 = bytearray(PAGE_SIZE)
    sp2 = SlottedPage.init_page(buf2, page_id=2)
    slots2: list[int] = []
    while True:
        sid = sp2.insert_tuple(b"Y" * 128)
        if sid is None:
            break
        slots2.append(sid)
    for sid in slots2[::2]:
        assert sp2.delete_tuple(sid) is True
    assert sp2.contiguous_free_space() < 400
    auto_slot = sp2.insert_tuple(b"M" * 400)
    assert auto_slot is not None
    assert sp2.get_tuple(auto_slot) == b"M" * 400
    for sid in slots2[1::2]:
        assert sp2.get_tuple(sid) == b"Y" * 128


def test_m1_column_page_zero_copy_numpy(tmp_path) -> None:
    """Verify ColumnPage zero-copy NumPy views and multi-page ColumnTableHeap math."""
    from src.storage.buffer_pool import BufferPoolManager
    from src.storage.column_page import ColumnPage
    from src.storage.disk_manager import DiskManager
    from src.storage.table_heap import ColumnTableHeap

    schema = Schema(
        table_name="order_items",
        columns=(
            ColumnDef("order_id", DataType.INT64),
            ColumnDef("item_seq", DataType.INT32),
            ColumnDef("product_id", DataType.INT64),
            ColumnDef("qty", DataType.INT32),
            ColumnDef("unit_price", DataType.FLOAT64),
        ),
        primary_key=None,
    )

    dms: list[DiskManager] = []
    column_bpms: dict[str, BufferPoolManager] = {}
    for col in schema.columns:
        dm = DiskManager(tmp_path / f"order_items.{col.name}.col")
        dms.append(dm)
        column_bpms[col.name] = BufferPoolManager(dm, pool_size=8)

    try:
        heap = ColumnTableHeap(schema, column_bpms, chunk_rows=CHUNK_ROWS)
        rng = random.Random(42)
        rows = []
        expected_sum = 0.0
        for idx in range(2500):
            qty = rng.randint(1, 10)
            unit_price = round(rng.uniform(5.0, 250.0), 2)
            expected_sum += qty * unit_price
            rows.append(
                {
                    "order_id": idx // 4 + 1,
                    "item_seq": idx % 4 + 1,
                    "product_id": (idx % 200) + 1,
                    "qty": qty,
                    "unit_price": unit_price,
                }
            )

        rids = heap.insert_batch(rows[:2495])
        for row in rows[2495:]:
            rids.append(heap.insert(row))
        assert len(rids) == 2500

        # 2,500 rows at CHUNK_ROWS=1020 span 3 data pages (pages 1, 2, 3)
        qty_bpm = column_bpms["qty"]
        price_bpm = column_bpms["unit_price"]
        computed_sum = 0.0
        for page_id in (1, 2, 3):
            qty_buf = qty_bpm.fetch_page(page_id)
            try:
                price_buf = price_bpm.fetch_page(page_id)
                try:
                    qty_cp = ColumnPage(qty_buf, DataType.INT32)
                    price_cp = ColumnPage(price_buf, DataType.FLOAT64)
                    qty_arr = qty_cp.as_numpy_array()
                    price_arr = price_cp.as_numpy_array()
                    assert qty_arr.flags["OWNDATA"] is False
                    assert price_arr.flags["OWNDATA"] is False
                    computed_sum += float(np.sum(qty_arr.astype(np.float64) * price_arr))
                finally:
                    price_bpm.unpin_page(page_id, is_dirty=False)
            finally:
                qty_bpm.unpin_page(page_id, is_dirty=False)

        assert abs(computed_sum - expected_sum) < 1e-6

        # Verify M5 vector stride hook (384 float32 dims = 1536 bytes -> 5 vectors/page)
        vec_buf = bytearray(PAGE_SIZE)
        vec_cp = ColumnPage.init_page(vec_buf, page_id=1, dtype=DataType.FLOAT64, stride=1536)
        assert vec_cp.capacity == 5
    finally:
        for dm in dms:
            dm.close()


def test_m1_lru_eviction_order(tmp_path) -> None:
    """Verify LRUReplacer evicts the least-recently-unpinned frame."""
    from src.storage.buffer_pool import BufferPoolManager
    from src.storage.disk_manager import DiskManager
    from src.storage.page_header import PageHeader, PageType
    from src.storage.replacer import LRUReplacer

    dm = DiskManager(tmp_path / "lru.db")
    try:
        for _ in range(4):
            pid = dm.allocate_page()
            buf = bytearray(PAGE_SIZE)
            PageHeader(page_id=pid, page_type=PageType.SLOTTED_ROW).pack_into(buf, 0)
            dm.write_page(pid, buf)

        bpm = BufferPoolManager(dm, pool_size=3, replacer=LRUReplacer(3))
        for pid in (1, 2, 3):
            bpm.fetch_page(pid)
            bpm.unpin_page(pid, is_dirty=False)

        # Re-accessing page 1 moves it to MRU, leaving page 2 as LRU victim
        bpm.fetch_page(1)
        bpm.unpin_page(1, is_dirty=False)

        bpm.fetch_page(4)
        bpm.unpin_page(4, is_dirty=False)

        assert not bpm.is_resident(2)
        assert bpm.is_resident(1)
        assert bpm.is_resident(3)
        assert bpm.is_resident(4)
    finally:
        dm.close()


def test_m1_fifo_eviction_order(tmp_path) -> None:
    """Verify FIFOReplacer evicts oldest-inserted frame regardless of re-access."""
    from src.storage.buffer_pool import BufferPoolManager
    from src.storage.disk_manager import DiskManager
    from src.storage.page_header import PageHeader, PageType
    from src.storage.replacer import FIFOReplacer

    dm = DiskManager(tmp_path / "fifo.db")
    try:
        for _ in range(4):
            pid = dm.allocate_page()
            buf = bytearray(PAGE_SIZE)
            PageHeader(page_id=pid, page_type=PageType.SLOTTED_ROW).pack_into(buf, 0)
            dm.write_page(pid, buf)

        bpm = BufferPoolManager(dm, pool_size=3, replacer=FIFOReplacer(3))
        for pid in (1, 2, 3):
            bpm.fetch_page(pid)
            bpm.unpin_page(pid, is_dirty=False)

        # Re-accessing page 1 must NOT refresh its FIFO insertion position
        bpm.fetch_page(1)
        bpm.unpin_page(1, is_dirty=False)

        bpm.fetch_page(4)
        bpm.unpin_page(4, is_dirty=False)

        assert not bpm.is_resident(1)
        assert bpm.is_resident(2)
        assert bpm.is_resident(3)
        assert bpm.is_resident(4)
    finally:
        dm.close()


def test_m1_pin_protection_and_pool_full(tmp_path) -> None:
    """Verify BufferPoolFullError is raised when all frames are pinned and no page leaks."""
    from src.storage.buffer_pool import BufferPoolManager
    from src.storage.disk_manager import DiskManager
    from src.storage.page_header import PageHeader, PageType

    dm = DiskManager(tmp_path / "pin_full.db")
    try:
        for _ in range(3):
            pid = dm.allocate_page()
            buf = bytearray(PAGE_SIZE)
            PageHeader(page_id=pid, page_type=PageType.SLOTTED_ROW).pack_into(buf, 0)
            dm.write_page(pid, buf)

        bpm = BufferPoolManager(dm, pool_size=2)
        bpm.fetch_page(1)
        bpm.fetch_page(2)

        pages_before = dm.num_pages()
        with pytest.raises(BufferPoolFullError):
            bpm.fetch_page(3)
        with pytest.raises(BufferPoolFullError):
            bpm.new_page()
        assert dm.num_pages() == pages_before

        assert bpm.unpin_page(1, is_dirty=False) is True
        bpm.fetch_page(3)
        assert bpm.is_resident(3)
        bpm.unpin_page(3, is_dirty=False)
        bpm.unpin_page(2, is_dirty=False)
    finally:
        dm.close()


def test_m1_dirty_writeback_accuracy(tmp_path) -> None:
    """Verify dirty frames are flushed to disk before eviction."""
    from src.storage.buffer_pool import BufferPoolManager
    from src.storage.disk_manager import DiskManager
    from src.storage.page_header import PageHeader, PageType
    from src.storage.replacer import LRUReplacer
    from src.storage.slotted_page import SlottedPage

    dm = DiskManager(tmp_path / "dirty_wb.db")
    try:
        for _ in range(3):
            pid = dm.allocate_page()
            buf = bytearray(PAGE_SIZE)
            PageHeader(page_id=pid, page_type=PageType.SLOTTED_ROW).pack_into(buf, 0)
            dm.write_page(pid, buf)

        bpm = BufferPoolManager(dm, pool_size=2, replacer=LRUReplacer(2))
        buf1 = bpm.fetch_page(1)
        sp1 = SlottedPage.init_page(buf1, page_id=1)
        slot_id = sp1.insert_tuple(b"dirty-writeback-secret")
        assert slot_id == 0
        bpm.unpin_page(1, is_dirty=True)

        # Load pages 2 and 3 to force eviction of dirty page 1
        bpm.fetch_page(2)
        bpm.unpin_page(2, is_dirty=False)
        bpm.fetch_page(3)
        bpm.unpin_page(3, is_dirty=False)

        assert not bpm.is_resident(1)
        assert bpm.eviction_write_count >= 1

        raw_disk_page = dm.read_page(1)
        assert SlottedPage(raw_disk_page).get_tuple(0) == b"dirty-writeback-secret"
    finally:
        dm.close()


def test_m1_hit_miss_telemetry(tmp_path) -> None:
    """Verify BufferPoolManager accurately tracks hit_count, miss_count, and hit_ratio."""
    from src.storage.buffer_pool import BufferPoolManager
    from src.storage.disk_manager import DiskManager
    from src.storage.page_header import PageHeader, PageType
    from src.storage.replacer import LRUReplacer

    dm = DiskManager(tmp_path / "telemetry.db")
    try:
        for _ in range(5):
            pid = dm.allocate_page()
            buf = bytearray(PAGE_SIZE)
            PageHeader(page_id=pid, page_type=PageType.SLOTTED_ROW).pack_into(buf, 0)
            dm.write_page(pid, buf)

        bpm = BufferPoolManager(dm, pool_size=3, replacer=LRUReplacer(3))
        for pid in (1, 2, 3):
            bpm.fetch_page(pid)
            bpm.unpin_page(pid, is_dirty=False)
        assert bpm.miss_count == 3
        assert bpm.hit_count == 0

        for pid in (1, 2, 3, 1):
            bpm.fetch_page(pid)
            bpm.unpin_page(pid, is_dirty=False)
        assert bpm.hit_count == 4
        assert bpm.miss_count == 3

        bpm.fetch_page(4)
        bpm.unpin_page(4, is_dirty=False)
        bpm.fetch_page(2)
        bpm.unpin_page(2, is_dirty=False)
        assert bpm.miss_count == 5
        assert bpm.hit_count == 4

        stats = bpm.get_stats()
        assert stats["hit_count"] == 4
        assert stats["miss_count"] == 5
        assert abs(stats["hit_ratio"] - (4.0 / 9.0)) < 1e-9
    finally:
        dm.close()


def test_m1_persistence_across_restart(tmp_path) -> None:
    """Verify 10k products (row) and 20k order_items (column) survive full restart."""
    from src.storage.storage_engine import StorageEngine

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
    order_items_schema = Schema(
        table_name="order_items",
        columns=(
            ColumnDef("order_id", DataType.INT64),
            ColumnDef("item_seq", DataType.INT32),
            ColumnDef("product_id", DataType.INT64),
            ColumnDef("qty", DataType.INT32),
            ColumnDef("unit_price", DataType.FLOAT64),
        ),
        primary_key=None,
    )

    categories = ("Electronics", "Apparel", "Home", "Books", "Sports", "Beauty")
    expected_products = [
        {
            "product_id": i + 1,
            "category": categories[i % len(categories)],
            "price": round(5.0 + (i % 500) * 1.25, 2),
            "image_oid": (i + 1) if i < 500 else 0,
        }
        for i in range(10_000)
    ]
    expected_items = [
        {
            "order_id": i // 4 + 1,
            "item_seq": i % 4 + 1,
            "product_id": (i % 5000) + 1,
            "qty": (i % 5) + 1,
            "unit_price": round(10.0 + (i % 100) * 0.75, 2),
        }
        for i in range(20_000)
    ]

    db_dir = tmp_path / "restart_db"
    engine = StorageEngine(db_dir, buffer_pool_size=64, policy="LRU")
    engine.open()
    p_heap = engine.create_table(products_schema, LayoutType.ROW)
    oi_heap = engine.create_table(order_items_schema, LayoutType.COLUMN)
    for row in expected_products:
        p_heap.insert(row)
    oi_heap.insert_batch(expected_items)
    engine.flush_all()
    engine.close()

    engine2 = StorageEngine(db_dir, buffer_pool_size=64, policy="LRU")
    engine2.open()
    try:
        restored_products = [row for _, row in engine2.get_table("products").full_scan()]
        restored_items = [row for _, row in engine2.get_table("order_items").full_scan()]
        assert restored_products == expected_products
        assert restored_items == expected_items
    finally:
        engine2.close()
