"""
Module: bench/run_m1_m2_benchmarks.py
Layer: Benchmark & Visualization Suite (Wave 5 — Gate 3)
Lock Domain: Isolated single-runner temporary workspaces (no shared mutable state across trials)

Purpose:
    Executes the six reproducible Milestone 1 and Milestone 2 controlled
    experiments (1A: Buffer Pool LRU vs. FIFO, 1B: Row vs. Column Storage Layout,
    2A: Index Read-Heavy 90/10, 2B: Index Write-Heavy 10/90, 2C: WorkloadAdvisor
    Dynamic Adaptation Trace, 2D: ObjectStore NoOpCipher vs. AES256GCMHook),
    enforces correctness assertions before recording timings, exports
    ``env_metadata.json``, ``raw_runs.csv``, and ``summary_metrics.csv``, and
    renders five 300-DPI matplotlib figures.

Key Invariants:
    1. ``MPLCONFIGDIR`` is set and ``matplotlib.use("Agg")`` is activated before
       importing ``matplotlib.pyplot`` so headless rendering emits zero cache
       warnings.
    2. Every configuration point executes 1 untimed warm-up trial followed by
       ``runs`` (default 5) measured trials using deterministic seeding.
    3. Every benchmark asserts exact functional equivalence (row equality,
       numeric sum match, RID match, advisor strategy sequence, or byte-for-byte
       image decryption) before recording timings.
    4. All temporary benchmark directories under the OS temp folder are removed
       on completion or failure.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import argparse
from collections.abc import Sequence
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bench.retail_loader import (
    CATEGORIES,
    INVENTORY_SCHEMA,
    ORDER_ITEMS_SCHEMA,
    PRODUCTS_SCHEMA,
    RetailDataGenerator,
    _sample_zipfian_ids,
)
from src.common.config import (
    DEFAULT_BUFFER_POOL_SIZE,
    DEFAULT_MEMTABLE_BYTES,
    PAGE_SIZE,
)
from src.common.types import RID, DataType
from src.index.bplus_tree import BPlusTreeIndex
from src.index.index_advisor import IndexStrategy, WorkloadAdvisor
from src.index.lsm_tree import LSMTreeIndex
from src.storage.buffer_pool import BufferPoolManager
from src.storage.cipher_hooks import AES256GCMHook, NoOpCipher
from src.storage.disk_manager import DiskManager
from src.storage.object_store import ObjectStore
from src.storage.page_header import PageHeader
from src.storage.replacer import FIFOReplacer, LRUReplacer
from src.storage.slotted_page import SlottedPage
from src.storage.table_heap import ColumnTableHeap, RowTableHeap

_RAW_CSV_FIELDS: tuple[str, ...] = (
    "experiment",
    "config",
    "access_method",
    "workload",
    "pool_size",
    "run_idx",
    "elapsed_ms",
    "ops_count",
    "throughput_ops_sec",
    "mean_op_ms",
    "median_op_ms",
    "p95_op_ms",
    "hit_ratio",
    "disk_reads",
    "disk_writes",
    "pages_read",
    "build_time_ms",
    "disk_size_mb",
    "throughput_mb_s",
    "storage_overhead_pct",
    "flushes",
    "compactions",
)

_SUMMARY_CSV_FIELDS: tuple[str, ...] = (
    "experiment",
    "config",
    "access_method",
    "workload",
    "pool_size",
    "runs",
    "mean_ms",
    "median_ms",
    "std_dev_ms",
    "p95_ms",
    "mean_op_ms",
    "median_op_ms",
    "p95_op_ms",
    "std_p95_op_ms",
    "throughput_ops_sec",
    "std_throughput_ops_sec",
    "mean_hit_ratio",
    "std_hit_ratio",
    "mean_disk_reads",
    "mean_disk_writes",
    "mean_pages_read",
    "mean_build_time_ms",
    "mean_disk_size_mb",
    "std_disk_size_mb",
    "mean_throughput_mb_s",
    "std_throughput_mb_s",
    "storage_overhead_pct",
    "mean_flushes",
    "mean_compactions",
)


@dataclass
class RawRunRecord:
    """Single measured trial row written to ``raw_runs.csv``."""

    experiment: str
    config: str
    access_method: str
    workload: str
    pool_size: int
    run_idx: int
    elapsed_ms: float
    ops_count: int
    throughput_ops_sec: float
    mean_op_ms: float = 0.0
    median_op_ms: float = 0.0
    p95_op_ms: float = 0.0
    hit_ratio: float = 0.0
    disk_reads: int = 0
    disk_writes: int = 0
    pages_read: int = 0
    build_time_ms: float = 0.0
    disk_size_mb: float = 0.0
    throughput_mb_s: float = 0.0
    storage_overhead_pct: float = 0.0
    flushes: int = 0
    compactions: int = 0


def _dir_size_bytes(path: Path) -> int:
    """Return total byte size of a single file or all files inside a directory."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def _generate_products(count: int, seed: int) -> list[dict[str, Any]]:
    """Generate ``count`` deterministic product rows matching ``PRODUCTS_SCHEMA``."""
    py_rng = random.Random(seed)
    return [
        {
            "product_id": pid,
            "category": py_rng.choice(CATEGORIES),
            "price": round(py_rng.uniform(5.0, 2500.0), 2),
            "image_oid": pid if pid <= 500 else 0,
        }
        for pid in range(1, count + 1)
    ]


def _generate_inventory(count: int, seed: int) -> list[dict[str, Any]]:
    """Generate ``count`` deterministic inventory rows matching ``INVENTORY_SCHEMA``."""
    py_rng = random.Random(seed)
    year_start = 1_767_225_600_000
    year_span = 365 * 86_400_000
    return [
        {
            "product_id": pid,
            "stock_qty": py_rng.randint(50, 500),
            "warehouse_id": py_rng.randint(1, 5),
            "updated_ts": year_start + py_rng.randrange(0, year_span),
        }
        for pid in range(1, count + 1)
    ]


def _compute_latency_stats(latencies_ms: list[float]) -> tuple[float, float, float]:
    """Return ``(mean_ms, median_ms, p95_ms)`` for non-empty per-op latencies."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.percentile(arr, 95)),
    )


def _run_exp_1a(
    tmp_root: Path, runs: int, seed: int
) -> tuple[list[RawRunRecord], dict[str, Any]]:
    """Run Experiment 1A: Buffer Pool ``LRU`` vs. ``FIFO`` on 10,000 products.

    Tests ``pool_size`` in ``[16, 32, 64, 128]`` across ``5,000`` fetches per
    run under both Zipfian (``theta = 0.9``) and Sequential access patterns.

    Args:
        tmp_root: Temporary workspace directory cleaned up by caller.
        runs: Number of measured trials per configuration point.
        seed: Base deterministic PRNG seed.

    Returns:
        Tuple of ``(raw_run_records, metadata_dict)``.
    """
    exp_dir = tmp_root / "exp1a"
    exp_dir.mkdir(parents=True, exist_ok=True)
    db_path = exp_dir / "products_10k.db"

    products = _generate_products(10_000, seed)
    dm_init = DiskManager(db_path)
    bpm_init = BufferPoolManager(dm_init, pool_size=128)
    heap_init = RowTableHeap(PRODUCTS_SCHEMA, bpm_init)
    rids = [heap_init.insert(row) for row in products]
    heap_init.flush()
    allocated_pages = dm_init.allocated_pages()
    heap_init.close()

    num_pages = len(allocated_pages)
    fetches_per_run = 5_000
    pool_sizes = (16, 32, 64, 128)
    policies = (("LRU", LRUReplacer), ("FIFO", FIFOReplacer))
    patterns = ("zipfian", "sequential")

    records: list[RawRunRecord] = []

    for pattern in patterns:
        for policy_name, policy_cls in policies:
            for pool_size in pool_sizes:
                for trial in range(-1, runs):
                    is_warmup = trial < 0
                    trial_seed = seed if is_warmup else seed + trial + 1

                    if pattern == "zipfian":
                        np_rng = np.random.default_rng(trial_seed)
                        sampled_pids = _sample_zipfian_ids(
                            np_rng, len(products), fetches_per_run, theta=0.9
                        )
                        target_rids = [rids[int(pid) - 1] for pid in sampled_pids]
                        expected_pids = [int(pid) for pid in sampled_pids]
                    else:
                        seq_pages = [
                            allocated_pages[i % num_pages]
                            for i in range(fetches_per_run)
                        ]
                        target_rids = []
                        expected_pids = []

                    dm = DiskManager(db_path)
                    bpm = BufferPoolManager(
                        dm, pool_size=pool_size, replacer=policy_cls(pool_size)
                    )
                    heap = RowTableHeap(PRODUCTS_SCHEMA, bpm)
                    try:
                        t0 = time.perf_counter()
                        if pattern == "zipfian":
                            for rid, exp_pid in zip(
                                target_rids, expected_pids, strict=True
                            ):
                                row = heap.get(rid)
                                assert row is not None and row["product_id"] == exp_pid
                        else:
                            for pid_page in seq_pages:
                                buf = bpm.fetch_page(pid_page)
                                try:
                                    hdr = PageHeader.unpack(buf, 0)
                                    sp = SlottedPage(buf)
                                    assert hdr.page_id == pid_page and sp.slot_count > 0
                                finally:
                                    bpm.unpin_page(pid_page, is_dirty=False)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        stats = bpm.get_stats()
                        assert stats["pinned_frames"] == 0
                        assert stats["hit_count"] + stats["miss_count"] == fetches_per_run
                    finally:
                        heap.close()

                    if not is_warmup:
                        records.append(
                            RawRunRecord(
                                experiment="1A",
                                config=f"{policy_name}_{pattern}_pool{pool_size}",
                                access_method=policy_name,
                                workload=pattern,
                                pool_size=pool_size,
                                run_idx=trial,
                                elapsed_ms=elapsed_ms,
                                ops_count=fetches_per_run,
                                throughput_ops_sec=fetches_per_run
                                / max(elapsed_ms / 1000.0, 1e-9),
                                mean_op_ms=elapsed_ms / fetches_per_run,
                                median_op_ms=elapsed_ms / fetches_per_run,
                                p95_op_ms=elapsed_ms / fetches_per_run,
                                hit_ratio=float(stats["hit_ratio"]),
                                disk_reads=int(stats["disk_reads"]),
                                disk_writes=int(stats["disk_writes"]),
                                pages_read=int(stats["disk_reads"]),
                            )
                        )

    return records, {"products_rows": len(products), "products_pages": num_pages}


def _open_column_heap(
    schema: Any, col_dir: Path, prefix: str, pool_size: int = DEFAULT_BUFFER_POOL_SIZE
) -> ColumnTableHeap:
    """Open a ``ColumnTableHeap`` over ``col_dir/<prefix>.<col>.col`` files."""
    col_bpms: dict[str, BufferPoolManager] = {}
    opened_dms: list[DiskManager] = []
    try:
        for col in schema.columns:
            dm = DiskManager(col_dir / f"{prefix}.{col.name}.col")
            opened_dms.append(dm)
            col_bpms[col.name] = BufferPoolManager(dm, pool_size=pool_size)
        return ColumnTableHeap(schema, col_bpms)
    except Exception:
        for dm in opened_dms:
            dm.close()
        raise


def _run_exp_1b(
    tmp_root: Path, runs: int, seed: int
) -> tuple[list[RawRunRecord], dict[str, Any]]:
    """Run Experiment 1B: ``RowTableHeap`` vs. ``ColumnTableHeap`` on 200k ``order_items``.

    Evaluates both layouts on OLAP aggregation (``SUM(qty * unit_price)``) and
    OLTP random point lookups (5,000 random ``RID``s), asserting exact sum and
    row equivalence before recording timings.

    Args:
        tmp_root: Temporary workspace directory cleaned up by caller.
        runs: Number of measured trials per configuration point.
        seed: Base deterministic PRNG seed.

    Returns:
        Tuple of ``(raw_run_records, metadata_dict)``.
    """
    exp_dir = tmp_root / "exp1b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    dataset = RetailDataGenerator(seed=seed).generate(scale="standard")
    items: list[dict[str, Any]] = dataset["order_items"]

    row_db_path = exp_dir / "order_items.db"
    dm_r = DiskManager(row_db_path)
    bpm_r = BufferPoolManager(dm_r, pool_size=DEFAULT_BUFFER_POOL_SIZE)
    row_heap_init = RowTableHeap(ORDER_ITEMS_SCHEMA, bpm_r)
    row_rids = [row_heap_init.insert(item) for item in items]
    row_heap_init.flush()
    row_pages = row_heap_init.page_count()
    row_heap_init.close()

    col_dir = exp_dir / "col"
    col_dir.mkdir(parents=True, exist_ok=True)
    col_heap_init = _open_column_heap(ORDER_ITEMS_SCHEMA, col_dir, "order_items")
    col_rids = col_heap_init.insert_batch(items)
    col_heap_init.flush()
    col_pages = col_heap_init.page_count()
    col_heap_init.close()

    expected_sum = math.fsum(item["qty"] * item["unit_price"] for item in items)
    oltp_lookups = 5_000
    records: list[RawRunRecord] = []

    # 1. OLAP Aggregation: SUM(qty * unit_price) across all 200,000 rows.
    for layout_name in ("RowTableHeap", "ColumnTableHeap"):
        for trial in range(-1, runs):
            is_warmup = trial < 0
            if layout_name == "RowTableHeap":
                heap_r = RowTableHeap(
                    ORDER_ITEMS_SCHEMA,
                    BufferPoolManager(
                        DiskManager(row_db_path), pool_size=DEFAULT_BUFFER_POOL_SIZE
                    ),
                )
                try:
                    t0 = time.perf_counter()
                    total_sum = 0.0
                    scanned_rows = 0
                    for _rid, row in heap_r.full_scan():
                        total_sum += row["qty"] * row["unit_price"]
                        scanned_rows += 1
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    stats = heap_r.bpm.get_stats()
                    pages_read = int(stats["disk_reads"])
                    assert scanned_rows == len(items)
                    assert math.isclose(
                        total_sum, expected_sum, rel_tol=1e-9, abs_tol=1e-6
                    )
                finally:
                    heap_r.close()
            else:
                heap_c = _open_column_heap(ORDER_ITEMS_SCHEMA, col_dir, "order_items")
                try:
                    t0 = time.perf_counter()
                    total_sum = 0.0
                    scanned_rows = 0
                    for batch in heap_c.iter_batches(columns=("qty", "unit_price")):
                        qty_arr = batch["qty"].astype(np.float64, copy=False)
                        price_arr = batch["unit_price"]
                        total_sum += float(np.sum(qty_arr * price_arr))
                        scanned_rows += len(qty_arr)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    pages_read = sum(
                        int(b.get_stats()["disk_reads"])
                        for b in heap_c.column_bpms.values()
                    )
                    assert scanned_rows == len(items)
                    assert math.isclose(
                        total_sum, expected_sum, rel_tol=1e-9, abs_tol=1e-6
                    )
                finally:
                    heap_c.close()

            if not is_warmup:
                records.append(
                    RawRunRecord(
                        experiment="1B",
                        config=f"{layout_name}_olap_aggregation",
                        access_method=layout_name,
                        workload="olap_aggregation",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=elapsed_ms,
                        ops_count=len(items),
                        throughput_ops_sec=len(items) / max(elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=elapsed_ms / len(items),
                        median_op_ms=elapsed_ms / len(items),
                        p95_op_ms=elapsed_ms / len(items),
                        disk_reads=pages_read,
                        pages_read=pages_read,
                    )
                )

    # 2. OLTP Point Lookup: 5,000 random RIDs fetching all 5 columns.
    for layout_name in ("RowTableHeap", "ColumnTableHeap"):
        for trial in range(-1, runs):
            is_warmup = trial < 0
            trial_seed = seed if is_warmup else seed + trial + 1
            np_rng = np.random.default_rng(trial_seed)
            sample_indices = np_rng.integers(0, len(items), size=oltp_lookups)

            if layout_name == "RowTableHeap":
                heap_r = RowTableHeap(
                    ORDER_ITEMS_SCHEMA,
                    BufferPoolManager(
                        DiskManager(row_db_path), pool_size=DEFAULT_BUFFER_POOL_SIZE
                    ),
                )
                try:
                    t0 = time.perf_counter()
                    for idx in sample_indices:
                        i = int(idx)
                        row = heap_r.get(row_rids[i])
                        assert row == items[i]
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    pages_read = int(heap_r.bpm.get_stats()["disk_reads"])
                finally:
                    heap_r.close()
            else:
                heap_c = _open_column_heap(ORDER_ITEMS_SCHEMA, col_dir, "order_items")
                try:
                    t0 = time.perf_counter()
                    for idx in sample_indices:
                        i = int(idx)
                        row = heap_c.get(col_rids[i])
                        assert row == items[i]
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    pages_read = sum(
                        int(b.get_stats()["disk_reads"])
                        for b in heap_c.column_bpms.values()
                    )
                finally:
                    heap_c.close()

            if not is_warmup:
                records.append(
                    RawRunRecord(
                        experiment="1B",
                        config=f"{layout_name}_oltp_point_lookup",
                        access_method=layout_name,
                        workload="oltp_point_lookup",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=elapsed_ms,
                        ops_count=oltp_lookups,
                        throughput_ops_sec=oltp_lookups / max(elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=elapsed_ms / oltp_lookups,
                        median_op_ms=elapsed_ms / oltp_lookups,
                        p95_op_ms=elapsed_ms / oltp_lookups,
                        disk_reads=pages_read,
                        pages_read=pages_read,
                    )
                )

    return records, {
        "order_items_rows": len(items),
        "row_heap_pages": row_pages,
        "column_heap_pages": col_pages,
    }


def _build_exp2a_ops(seed: int) -> list[tuple[str, int, dict[str, Any] | None]]:
    """Build deterministic 10,000-op 90/10 read-heavy workload for Experiment 2A."""
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    ops: list[tuple[str, int, dict[str, Any] | None]] = []
    for k in np_rng.integers(1, 50_001, size=7_000):
        ops.append(("point", int(k), None))
    for k in np_rng.integers(1, 49_981, size=2_000):
        ops.append(("range", int(k), None))
    for i in range(1, 1_001):
        pid = 50_000 + i
        ops.append(
            (
                "insert",
                pid,
                {
                    "product_id": pid,
                    "category": py_rng.choice(CATEGORIES),
                    "price": round(py_rng.uniform(5.0, 2500.0), 2),
                    "image_oid": 0,
                },
            )
        )
    perm = np_rng.permutation(len(ops))
    return [ops[int(idx)] for idx in perm]


def _select_fullscan_2a_sample(
    ops: list[tuple[str, int, dict[str, Any] | None]],
) -> list[tuple[int, tuple[str, int, dict[str, Any] | None]]]:
    """Select the exact 200-op stratified sample (140 point, 40 range, 20 insert)."""
    selected: list[tuple[int, tuple[str, int, dict[str, Any] | None]]] = []
    counts = {"point": 0, "range": 0, "insert": 0}
    targets = {"point": 140, "range": 40, "insert": 20}
    for op_idx, op in enumerate(ops):
        kind = op[0]
        if counts[kind] < targets[kind]:
            selected.append((op_idx, op))
            counts[kind] += 1
            if sum(counts.values()) == 200:
                break
    assert sum(counts.values()) == 200
    return selected


def _run_exp_2a(
    tmp_root: Path, runs: int, seed: int
) -> tuple[list[RawRunRecord], dict[str, Any]]:
    """Run Experiment 2A: Read-Heavy (90% Reads / 10% Writes) on 50,000 products.

    Compares ``FULL_SCAN`` (200-op representative sample) vs. ``BPLUS_TREE`` vs.
    ``LSM_TREE`` (10,000 ops each), asserting exact RID/result equivalence and
    measuring build time, per-op latencies, throughput, and on-disk index size.

    Args:
        tmp_root: Temporary workspace directory cleaned up by caller.
        runs: Number of measured trials per configuration point.
        seed: Base deterministic PRNG seed.

    Returns:
        Tuple of ``(raw_run_records, metadata_dict)``.
    """
    exp_dir = tmp_root / "exp2a"
    baseline_dir = exp_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    products = _generate_products(50_000, seed)
    base_db = baseline_dir / "products.db"
    heap_init = RowTableHeap(
        PRODUCTS_SCHEMA,
        BufferPoolManager(DiskManager(base_db), pool_size=DEFAULT_BUFFER_POOL_SIZE),
    )
    base_rids = [heap_init.insert(row) for row in products]
    heap_init.flush()
    heap_init.close()

    records: list[RawRunRecord] = []

    for trial in range(-1, runs):
        is_warmup = trial < 0
        trial_seed = seed if is_warmup else seed + trial + 1
        ops = _build_exp2a_ops(trial_seed)
        sample_200 = _select_fullscan_2a_sample(ops)
        sample_indices = {op_idx for op_idx, _ in sample_200}

        trial_dir = exp_dir / f"trial_{trial + 1}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        reference_results: dict[int, Any] = {}

        # --- BPLUS_TREE ---
        bpt_db = trial_dir / "bpt_products.db"
        shutil.copy2(base_db, bpt_db)
        bpt_idx_path = trial_dir / "products_pk.bpt"
        bpt_heap = RowTableHeap(
            PRODUCTS_SCHEMA,
            BufferPoolManager(DiskManager(bpt_db), pool_size=DEFAULT_BUFFER_POOL_SIZE),
        )
        bpt = BPlusTreeIndex(bpt_idx_path, DataType.INT64, key_size=8, unique=True)
        try:
            tb0 = time.perf_counter()
            for row, rid in zip(products, base_rids, strict=True):
                bpt.insert(row["product_id"], rid)
            bpt.flush()
            bpt_build_ms = (time.perf_counter() - tb0) * 1000.0
            bpt_size_mb = _dir_size_bytes(bpt_idx_path) / (1024.0 * 1024.0)

            bpt_latencies: list[float] = []
            t_start = time.perf_counter()
            for op_idx, (kind, key, new_row) in enumerate(ops):
                op_t0 = time.perf_counter()
                if kind == "point":
                    res = bpt.get(key)
                    assert len(res) == 1
                    fetched = bpt_heap.get(res[0])
                    assert fetched is not None and fetched["product_id"] == key
                    if op_idx in sample_indices:
                        reference_results[op_idx] = list(res)
                elif kind == "range":
                    res_rng = bpt.range_scan(key, key + 19, inclusive=True)
                    assert [k for k, _ in res_rng] == list(range(key, key + 20))
                    if op_idx in sample_indices:
                        reference_results[op_idx] = list(res_rng)
                else:
                    assert new_row is not None
                    new_rid = bpt_heap.insert(new_row)
                    bpt.insert(key, new_rid)
                    if op_idx in sample_indices:
                        reference_results[op_idx] = new_rid
                bpt_latencies.append((time.perf_counter() - op_t0) * 1000.0)
            bpt.flush()
            bpt_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            bpt_stats = bpt.get_stats()
            bpt_disk_reads = int(bpt_stats["disk_reads"])
            bpt_disk_writes = int(bpt_stats["disk_writes"])
            bpt_final_size_mb = _dir_size_bytes(bpt_idx_path) / (1024.0 * 1024.0)
        finally:
            bpt.close()
            bpt_heap.close()

        # --- LSM_TREE ---
        lsm_db = trial_dir / "lsm_products.db"
        shutil.copy2(base_db, lsm_db)
        lsm_idx_dir = trial_dir / "products_pk_lsm"
        lsm_heap = RowTableHeap(
            PRODUCTS_SCHEMA,
            BufferPoolManager(DiskManager(lsm_db), pool_size=DEFAULT_BUFFER_POOL_SIZE),
        )
        lsm = LSMTreeIndex(
            lsm_idx_dir,
            DataType.INT64,
            key_size=8,
            unique=True,
            memtable_size_bytes=DEFAULT_MEMTABLE_BYTES,
        )
        try:
            tl0 = time.perf_counter()
            for row, rid in zip(products, base_rids, strict=True):
                lsm.insert(row["product_id"], rid)
            lsm.flush()
            lsm_build_ms = (time.perf_counter() - tl0) * 1000.0
            lsm_size_mb = _dir_size_bytes(lsm_idx_dir) / (1024.0 * 1024.0)

            lsm_latencies: list[float] = []
            t_start = time.perf_counter()
            for op_idx, (kind, key, new_row) in enumerate(ops):
                op_t0 = time.perf_counter()
                if kind == "point":
                    res = lsm.get(key)
                    assert len(res) == 1
                    fetched = lsm_heap.get(res[0])
                    assert fetched is not None and fetched["product_id"] == key
                    if op_idx in sample_indices:
                        assert res == reference_results[op_idx]
                elif kind == "range":
                    res_rng = lsm.range_scan(key, key + 19, inclusive=True)
                    assert [k for k, _ in res_rng] == list(range(key, key + 20))
                    if op_idx in sample_indices:
                        assert res_rng == reference_results[op_idx]
                else:
                    assert new_row is not None
                    new_rid = lsm_heap.insert(new_row)
                    lsm.insert(key, new_rid)
                    if op_idx in sample_indices:
                        assert new_rid == reference_results[op_idx]
                lsm_latencies.append((time.perf_counter() - op_t0) * 1000.0)
            lsm.flush()
            lsm_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            lsm_stats = lsm.get_stats()
            lsm_block_reads = int(lsm_stats["block_reads"])
            lsm_flushes = int(lsm_stats["flushes"])
            lsm_compactions = int(lsm_stats["compactions"])
            lsm_final_size_mb = _dir_size_bytes(lsm_idx_dir) / (1024.0 * 1024.0)
        finally:
            lsm.close()
            lsm_heap.close()

        # --- FULL_SCAN (200-op representative sample) ---
        fs_db = trial_dir / "fs_products.db"
        shutil.copy2(base_db, fs_db)
        fs_heap = RowTableHeap(
            PRODUCTS_SCHEMA,
            BufferPoolManager(DiskManager(fs_db), pool_size=DEFAULT_BUFFER_POOL_SIZE),
        )
        try:
            fs_latencies: list[float] = []
            t_start = time.perf_counter()
            for op_idx, (kind, key, new_row) in sample_200:
                op_t0 = time.perf_counter()
                if kind == "point":
                    matches = [
                        rid
                        for rid, row in fs_heap.full_scan()
                        if row["product_id"] == key
                    ]
                    assert matches == reference_results[op_idx]
                elif kind == "range":
                    range_matches = sorted(
                        (
                            (row["product_id"], rid)
                            for rid, row in fs_heap.full_scan()
                            if key <= row["product_id"] <= key + 19
                        ),
                        key=lambda item: (item[0], item[1].page_id, item[1].slot_id),
                    )
                    assert range_matches == reference_results[op_idx]
                else:
                    assert new_row is not None
                    new_rid = fs_heap.insert(new_row)
                    got = fs_heap.get(new_rid)
                    assert got is not None and got["product_id"] == key
                fs_latencies.append((time.perf_counter() - op_t0) * 1000.0)
            fs_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            fs_stats = fs_heap.bpm.get_stats()
            fs_disk_reads = int(fs_stats["disk_reads"])
            fs_disk_writes = int(fs_stats["disk_writes"])
        finally:
            fs_heap.close()

        shutil.rmtree(trial_dir, ignore_errors=True)

        if not is_warmup:
            bpt_mean, bpt_med, bpt_p95 = _compute_latency_stats(bpt_latencies)
            lsm_mean, lsm_med, lsm_p95 = _compute_latency_stats(lsm_latencies)
            fs_mean, fs_med, fs_p95 = _compute_latency_stats(fs_latencies)

            records.extend(
                [
                    RawRunRecord(
                        experiment="2A",
                        config="BPLUS_TREE_read_heavy_90_10",
                        access_method="BPLUS_TREE",
                        workload="read_heavy_90_10",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=bpt_elapsed_ms,
                        ops_count=len(ops),
                        throughput_ops_sec=len(ops)
                        / max(bpt_elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=bpt_mean,
                        median_op_ms=bpt_med,
                        p95_op_ms=bpt_p95,
                        disk_reads=bpt_disk_reads,
                        disk_writes=bpt_disk_writes,
                        pages_read=bpt_disk_reads,
                        build_time_ms=bpt_build_ms,
                        disk_size_mb=bpt_final_size_mb,
                    ),
                    RawRunRecord(
                        experiment="2A",
                        config="LSM_TREE_read_heavy_90_10",
                        access_method="LSM_TREE",
                        workload="read_heavy_90_10",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=lsm_elapsed_ms,
                        ops_count=len(ops),
                        throughput_ops_sec=len(ops)
                        / max(lsm_elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=lsm_mean,
                        median_op_ms=lsm_med,
                        p95_op_ms=lsm_p95,
                        disk_reads=lsm_block_reads,
                        pages_read=lsm_block_reads,
                        build_time_ms=lsm_build_ms,
                        disk_size_mb=lsm_final_size_mb,
                        flushes=lsm_flushes,
                        compactions=lsm_compactions,
                    ),
                    RawRunRecord(
                        experiment="2A",
                        config="FULL_SCAN_read_heavy_90_10",
                        access_method="FULL_SCAN",
                        workload="read_heavy_90_10",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=fs_elapsed_ms,
                        ops_count=len(sample_200),
                        throughput_ops_sec=len(sample_200)
                        / max(fs_elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=fs_mean,
                        median_op_ms=fs_med,
                        p95_op_ms=fs_p95,
                        disk_reads=fs_disk_reads,
                        disk_writes=fs_disk_writes,
                        pages_read=fs_disk_reads,
                        build_time_ms=0.0,
                        disk_size_mb=0.0,
                    ),
                ]
            )

    return records, {
        "products_rows": len(products),
        "index_ops_per_run": 10_000,
        "full_scan_sample_ops_per_run": 200,
        "bpt_initial_size_mb": round(bpt_size_mb, 4),
        "lsm_initial_size_mb": round(lsm_size_mb, 4),
    }


def _build_exp2b_ops(seed: int) -> list[tuple[str, int, int, int]]:
    """Build deterministic 20,000-op 10/90 write-heavy workload for Experiment 2B."""
    np_rng = np.random.default_rng(seed)
    base_ts = 1_767_225_600_000 + 365 * 86_400_000
    ops: list[tuple[str, int, int, int]] = []
    upsert_keys = np_rng.integers(1, 50_001, size=18_000)
    upsert_qtys = np_rng.integers(0, 1_000, size=18_000)
    for idx, (k, q) in enumerate(zip(upsert_keys, upsert_qtys, strict=True)):
        ops.append(("upsert", int(k), int(q), base_ts + idx))
    for k in np_rng.integers(1, 50_001, size=2_000):
        ops.append(("point", int(k), 0, 0))
    perm = np_rng.permutation(len(ops))
    return [ops[int(idx)] for idx in perm]


def _select_fullscan_2b_sample(
    ops: list[tuple[str, int, int, int]],
) -> list[tuple[int, tuple[str, int, int, int]]]:
    """Select the exact 200-op stratified sample (180 updates, 20 point lookups)."""
    selected: list[tuple[int, tuple[str, int, int, int]]] = []
    counts = {"upsert": 0, "point": 0}
    targets = {"upsert": 180, "point": 20}
    for op_idx, op in enumerate(ops):
        kind = op[0]
        if counts[kind] < targets[kind]:
            selected.append((op_idx, op))
            counts[kind] += 1
            if sum(counts.values()) == 200:
                break
    assert sum(counts.values()) == 200
    return selected


def _run_exp_2b(
    tmp_root: Path, runs: int, seed: int
) -> tuple[list[RawRunRecord], dict[str, Any]]:
    """Run Experiment 2B: Write-Heavy (10% Reads / 90% Writes) on 50,000 inventory rows.

    Builds a clean 50k baseline (heap + B+ Tree + LSM Tree) once and copies it
    via ``shutil.copytree`` per trial so every run starts from identical state.
    Compares ``FULL_SCAN`` (200-op sample) vs. ``BPLUS_TREE`` vs. ``LSM_TREE``
    (20,000 ops each).

    Args:
        tmp_root: Temporary workspace directory cleaned up by caller.
        runs: Number of measured trials per configuration point.
        seed: Base deterministic PRNG seed.

    Returns:
        Tuple of ``(raw_run_records, metadata_dict)``.
    """
    exp_dir = tmp_root / "exp2b"
    baseline_dir = exp_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    inventory = _generate_inventory(50_000, seed)
    base_db = baseline_dir / "inventory.db"
    heap_init = RowTableHeap(
        INVENTORY_SCHEMA,
        BufferPoolManager(DiskManager(base_db), pool_size=DEFAULT_BUFFER_POOL_SIZE),
    )
    base_rids = [heap_init.insert(row) for row in inventory]
    heap_init.flush()
    heap_init.close()

    base_bpt_path = baseline_dir / "inventory_pk.bpt"
    bpt_init = BPlusTreeIndex(base_bpt_path, DataType.INT64, key_size=8, unique=True)
    for row, rid in zip(inventory, base_rids, strict=True):
        bpt_init.insert(row["product_id"], rid)
    bpt_init.flush()
    bpt_init.close()

    base_lsm_dir = baseline_dir / "inventory_pk_lsm"
    lsm_init = LSMTreeIndex(
        base_lsm_dir,
        DataType.INT64,
        key_size=8,
        unique=True,
        memtable_size_bytes=DEFAULT_MEMTABLE_BYTES,
    )
    for row, rid in zip(inventory, base_rids, strict=True):
        lsm_init.insert(row["product_id"], rid)
    lsm_init.flush()
    lsm_init.close()

    records: list[RawRunRecord] = []

    for trial in range(-1, runs):
        is_warmup = trial < 0
        trial_seed = seed if is_warmup else seed + trial + 1
        ops = _build_exp2b_ops(trial_seed)
        sample_200 = _select_fullscan_2b_sample(ops)

        bpt_dir = exp_dir / f"bpt_trial_{trial + 1}"
        lsm_dir = exp_dir / f"lsm_trial_{trial + 1}"
        fs_dir = exp_dir / f"fs_trial_{trial + 1}"
        for d in (bpt_dir, lsm_dir, fs_dir):
            if d.exists():
                shutil.rmtree(d)

        shutil.copytree(baseline_dir, bpt_dir)
        shutil.copytree(baseline_dir, lsm_dir)
        shutil.copytree(baseline_dir, fs_dir)

        bpt_point_results: dict[int, int] = {}

        # --- BPLUS_TREE ---
        bpt_heap = RowTableHeap(
            INVENTORY_SCHEMA,
            BufferPoolManager(
                DiskManager(bpt_dir / "inventory.db"),
                pool_size=DEFAULT_BUFFER_POOL_SIZE,
            ),
        )
        bpt = BPlusTreeIndex(
            bpt_dir / "inventory_pk.bpt", DataType.INT64, key_size=8, unique=True
        )
        try:
            bpt_latencies: list[float] = []
            t_start = time.perf_counter()
            for op_idx, (kind, key, qty, ts) in enumerate(ops):
                op_t0 = time.perf_counter()
                if kind == "point":
                    rids = bpt.get(key)
                    assert len(rids) == 1
                    row = bpt_heap.get(rids[0])
                    assert row is not None and row["product_id"] == key
                    bpt_point_results[op_idx] = int(row["stock_qty"])
                else:
                    rids = bpt.get(key)
                    assert len(rids) == 1
                    old_rid = rids[0]
                    old_row = bpt_heap.get(old_rid)
                    assert old_row is not None
                    new_row = {
                        "product_id": key,
                        "stock_qty": qty,
                        "warehouse_id": old_row["warehouse_id"],
                        "updated_ts": ts,
                    }
                    new_rid = bpt_heap.update(old_rid, new_row)
                    bpt.delete(key, old_rid)
                    bpt.insert(key, new_rid)
                bpt_latencies.append((time.perf_counter() - op_t0) * 1000.0)
            bpt.flush()
            bpt_heap.flush()
            bpt_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            bpt_stats = bpt.get_stats()
            bpt_disk_reads = int(bpt_stats["disk_reads"])
            bpt_disk_writes = int(bpt_stats["disk_writes"])
            bpt_size_mb = _dir_size_bytes(bpt_dir / "inventory_pk.bpt") / (
                1024.0 * 1024.0
            )
        finally:
            bpt.close()
            bpt_heap.close()

        # --- LSM_TREE ---
        lsm_heap = RowTableHeap(
            INVENTORY_SCHEMA,
            BufferPoolManager(
                DiskManager(lsm_dir / "inventory.db"),
                pool_size=DEFAULT_BUFFER_POOL_SIZE,
            ),
        )
        lsm = LSMTreeIndex(
            lsm_dir / "inventory_pk_lsm",
            DataType.INT64,
            key_size=8,
            unique=True,
            memtable_size_bytes=DEFAULT_MEMTABLE_BYTES,
        )
        try:
            lsm_latencies: list[float] = []
            t_start = time.perf_counter()
            for op_idx, (kind, key, qty, ts) in enumerate(ops):
                op_t0 = time.perf_counter()
                if kind == "point":
                    rids = lsm.get(key)
                    assert len(rids) == 1
                    row = lsm_heap.get(rids[0])
                    assert row is not None and row["product_id"] == key
                    assert int(row["stock_qty"]) == bpt_point_results[op_idx]
                else:
                    rids = lsm.get(key)
                    assert len(rids) == 1
                    old_rid = rids[0]
                    old_row = lsm_heap.get(old_rid)
                    assert old_row is not None
                    new_row = {
                        "product_id": key,
                        "stock_qty": qty,
                        "warehouse_id": old_row["warehouse_id"],
                        "updated_ts": ts,
                    }
                    new_rid = lsm_heap.update(old_rid, new_row)
                    lsm.insert(key, new_rid)
                lsm_latencies.append((time.perf_counter() - op_t0) * 1000.0)
            lsm.flush()
            lsm_heap.flush()
            lsm_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            lsm_stats = lsm.get_stats()
            lsm_block_reads = int(lsm_stats["block_reads"])
            lsm_flushes = int(lsm_stats["flushes"])
            lsm_compactions = int(lsm_stats["compactions"])
            lsm_size_mb = _dir_size_bytes(lsm_dir / "inventory_pk_lsm") / (
                1024.0 * 1024.0
            )
        finally:
            lsm.close()
            lsm_heap.close()

        # --- FULL_SCAN (200-op representative sample) ---
        fs_heap = RowTableHeap(
            INVENTORY_SCHEMA,
            BufferPoolManager(
                DiskManager(fs_dir / "inventory.db"),
                pool_size=DEFAULT_BUFFER_POOL_SIZE,
            ),
        )
        try:
            fs_latencies: list[float] = []
            t_start = time.perf_counter()
            for _op_idx, (kind, key, qty, ts) in sample_200:
                op_t0 = time.perf_counter()
                if kind == "point":
                    matches = [
                        (rid, row)
                        for rid, row in fs_heap.full_scan()
                        if row["product_id"] == key
                    ]
                    assert len(matches) == 1 and matches[0][1]["product_id"] == key
                else:
                    matches = [
                        (rid, row)
                        for rid, row in fs_heap.full_scan()
                        if row["product_id"] == key
                    ]
                    assert len(matches) == 1
                    old_rid, old_row = matches[0]
                    new_row = {
                        "product_id": key,
                        "stock_qty": qty,
                        "warehouse_id": old_row["warehouse_id"],
                        "updated_ts": ts,
                    }
                    new_rid = fs_heap.update(old_rid, new_row)
                    updated = fs_heap.get(new_rid)
                    assert updated is not None and updated["stock_qty"] == qty
                fs_latencies.append((time.perf_counter() - op_t0) * 1000.0)
            fs_heap.flush()
            fs_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            fs_stats = fs_heap.bpm.get_stats()
            fs_disk_reads = int(fs_stats["disk_reads"])
            fs_disk_writes = int(fs_stats["disk_writes"])
        finally:
            fs_heap.close()

        shutil.rmtree(bpt_dir, ignore_errors=True)
        shutil.rmtree(lsm_dir, ignore_errors=True)
        shutil.rmtree(fs_dir, ignore_errors=True)

        if not is_warmup:
            bpt_mean, bpt_med, bpt_p95 = _compute_latency_stats(bpt_latencies)
            lsm_mean, lsm_med, lsm_p95 = _compute_latency_stats(lsm_latencies)
            fs_mean, fs_med, fs_p95 = _compute_latency_stats(fs_latencies)

            records.extend(
                [
                    RawRunRecord(
                        experiment="2B",
                        config="BPLUS_TREE_write_heavy_10_90",
                        access_method="BPLUS_TREE",
                        workload="write_heavy_10_90",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=bpt_elapsed_ms,
                        ops_count=len(ops),
                        throughput_ops_sec=len(ops)
                        / max(bpt_elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=bpt_mean,
                        median_op_ms=bpt_med,
                        p95_op_ms=bpt_p95,
                        disk_reads=bpt_disk_reads,
                        disk_writes=bpt_disk_writes,
                        pages_read=bpt_disk_reads,
                        disk_size_mb=bpt_size_mb,
                    ),
                    RawRunRecord(
                        experiment="2B",
                        config="LSM_TREE_write_heavy_10_90",
                        access_method="LSM_TREE",
                        workload="write_heavy_10_90",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=lsm_elapsed_ms,
                        ops_count=len(ops),
                        throughput_ops_sec=len(ops)
                        / max(lsm_elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=lsm_mean,
                        median_op_ms=lsm_med,
                        p95_op_ms=lsm_p95,
                        disk_reads=lsm_block_reads,
                        pages_read=lsm_block_reads,
                        disk_size_mb=lsm_size_mb,
                        flushes=lsm_flushes,
                        compactions=lsm_compactions,
                    ),
                    RawRunRecord(
                        experiment="2B",
                        config="FULL_SCAN_write_heavy_10_90",
                        access_method="FULL_SCAN",
                        workload="write_heavy_10_90",
                        pool_size=DEFAULT_BUFFER_POOL_SIZE,
                        run_idx=trial,
                        elapsed_ms=fs_elapsed_ms,
                        ops_count=len(sample_200),
                        throughput_ops_sec=len(sample_200)
                        / max(fs_elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=fs_mean,
                        median_op_ms=fs_med,
                        p95_op_ms=fs_p95,
                        disk_reads=fs_disk_reads,
                        disk_writes=fs_disk_writes,
                        pages_read=fs_disk_reads,
                    ),
                ]
            )

    return records, {
        "inventory_rows": len(inventory),
        "index_ops_per_run": 20_000,
        "full_scan_sample_ops_per_run": 200,
    }


def _run_exp_2c(runs: int, seed: int) -> tuple[list[RawRunRecord], dict[str, Any]]:
    """Run Experiment 2C: ``WorkloadAdvisor`` 3-phase dynamic adaptation trace.

    Feeds a 3-phase trace (``150 cold ops`` -> ``2,000 90% inventory writes`` ->
    ``2,000 85% price range scans``) across 1 warm-up + ``runs`` trials,
    asserting exact ``FULL_SCAN -> LSM_TREE -> BPLUS_TREE`` strategy transitions.

    Args:
        runs: Number of measured trials.
        seed: Base deterministic PRNG seed.

    Returns:
        Tuple of ``(raw_run_records, metadata_dict)``.
    """
    records: list[RawRunRecord] = []
    trace_snapshot: list[dict[str, Any]] = []

    for trial in range(-1, runs):
        is_warmup = trial < 0
        trial_seed = seed if is_warmup else seed + trial + 1
        py_rng = random.Random(trial_seed)

        phase1 = ["POINT_READ" if i % 2 == 0 else "WRITE" for i in range(150)]
        phase2 = ["WRITE"] * 1_800 + ["POINT_READ"] * 200
        py_rng.shuffle(phase2)
        phase3 = ["RANGE_READ"] * 1_700 + ["POINT_READ"] * 200 + ["WRITE"] * 100
        py_rng.shuffle(phase3)

        advisor = WorkloadAdvisor()
        table_name = "retail_workload"
        col_name = "key"
        table_pages = 323

        phases = (
            ("phase1_cold_start", phase1, IndexStrategy.FULL_SCAN),
            ("phase2_write_heavy_90", phase2, IndexStrategy.LSM_TREE),
            ("phase3_range_heavy_85", phase3, IndexStrategy.BPLUS_TREE),
        )

        phase_results: list[tuple[str, float, int, float, float, IndexStrategy]] = []
        for phase_name, ops, expected_strategy in phases:
            t0 = time.perf_counter()
            for op in ops:
                advisor.record(op, table_name, col_name)
            rec = advisor.recommend(table_name, col_name, table_pages=table_pages)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert rec.strategy == expected_strategy, (
                f"{phase_name}: expected {expected_strategy}, got {rec.strategy} "
                f"(alpha={rec.alpha:.4f}, beta={rec.beta:.4f})"
            )
            phase_results.append(
                (phase_name, elapsed_ms, len(ops), rec.alpha, rec.beta, rec.strategy)
            )
            if trial == 0:
                trace_snapshot.append(
                    {
                        "phase": phase_name,
                        "ops": len(ops),
                        "window_n_ops": rec.n_ops,
                        "alpha": round(rec.alpha, 4),
                        "beta": round(rec.beta, 4),
                        "strategy": rec.strategy.value,
                    }
                )

        assert [r[5] for r in phase_results] == [
            IndexStrategy.FULL_SCAN,
            IndexStrategy.LSM_TREE,
            IndexStrategy.BPLUS_TREE,
        ]

        if not is_warmup:
            for phase_name, elapsed_ms, ops_cnt, alpha, beta, strategy in phase_results:
                records.append(
                    RawRunRecord(
                        experiment="2C",
                        config=f"WorkloadAdvisor_{phase_name}",
                        access_method=strategy.value,
                        workload=phase_name,
                        pool_size=0,
                        run_idx=trial,
                        elapsed_ms=elapsed_ms,
                        ops_count=ops_cnt,
                        throughput_ops_sec=ops_cnt / max(elapsed_ms / 1000.0, 1e-9),
                        mean_op_ms=elapsed_ms / ops_cnt,
                        median_op_ms=elapsed_ms / ops_cnt,
                        p95_op_ms=elapsed_ms / ops_cnt,
                        hit_ratio=alpha,
                        disk_size_mb=beta,
                    )
                )

    return records, {"trace_ops": 4_150, "transitions_trial_0": trace_snapshot}


def _run_exp_2d(
    tmp_root: Path, runs: int, seed: int
) -> tuple[list[RawRunRecord], dict[str, Any]]:
    """Run Experiment 2D: ``ObjectStore`` ``NoOpCipher`` vs. ``AES256GCMHook``.

    Measures ``put_object`` MB/s, ``get_object`` MB/s, and storage overhead %
    across 500 PNG-prefixed product images (16 KB–128 KB), asserting byte-for-byte
    equality on every read.

    Args:
        tmp_root: Temporary workspace directory cleaned up by caller.
        runs: Number of measured trials per cipher hook.
        seed: Base deterministic PRNG seed.

    Returns:
        Tuple of ``(raw_run_records, metadata_dict)``.
    """
    exp_dir = tmp_root / "exp2d"
    exp_dir.mkdir(parents=True, exist_ok=True)

    images = RetailDataGenerator(seed=seed).generate(scale="standard")["product_images"]
    total_raw_bytes = sum(len(payload) for _, payload in images)
    total_mb = total_raw_bytes / (1024.0 * 1024.0)
    aes_key = hashlib.sha256(b"csai302-retail-master-key").digest()

    ciphers = (
        ("NoOpCipher", lambda: NoOpCipher()),
        ("AES256GCMHook", lambda: AES256GCMHook(aes_key)),
    )
    records: list[RawRunRecord] = []

    for cipher_name, cipher_factory in ciphers:
        for trial in range(-1, runs):
            is_warmup = trial < 0
            store_dir = exp_dir / f"{cipher_name}_trial_{trial + 1}"
            if store_dir.exists():
                shutil.rmtree(store_dir)

            store = ObjectStore(store_dir, cipher=cipher_factory())
            try:
                t_put0 = time.perf_counter()
                for oid, payload in images:
                    store.put_object(oid, payload)
                store.flush()
                put_elapsed_ms = (time.perf_counter() - t_put0) * 1000.0

                t_get0 = time.perf_counter()
                for oid, payload in images:
                    retrieved = store.get_object(oid)
                    assert retrieved == payload
                get_elapsed_ms = (time.perf_counter() - t_get0) * 1000.0

                stats = store.get_stats()
                dir_bytes = (store_dir / "objects.dir").stat().st_size
                total_disk_bytes = int(stats["data_bytes"]) + dir_bytes
                overhead_pct = (
                    (total_disk_bytes - total_raw_bytes) / total_raw_bytes
                ) * 100.0
            finally:
                store.close()
                shutil.rmtree(store_dir, ignore_errors=True)

            if not is_warmup:
                records.extend(
                    [
                        RawRunRecord(
                            experiment="2D",
                            config=f"{cipher_name}_put_object",
                            access_method=cipher_name,
                            workload="put_object",
                            pool_size=0,
                            run_idx=trial,
                            elapsed_ms=put_elapsed_ms,
                            ops_count=len(images),
                            throughput_ops_sec=len(images)
                            / max(put_elapsed_ms / 1000.0, 1e-9),
                            mean_op_ms=put_elapsed_ms / len(images),
                            median_op_ms=put_elapsed_ms / len(images),
                            p95_op_ms=put_elapsed_ms / len(images),
                            disk_size_mb=total_disk_bytes / (1024.0 * 1024.0),
                            throughput_mb_s=total_mb / max(put_elapsed_ms / 1000.0, 1e-9),
                            storage_overhead_pct=overhead_pct,
                        ),
                        RawRunRecord(
                            experiment="2D",
                            config=f"{cipher_name}_get_object",
                            access_method=cipher_name,
                            workload="get_object",
                            pool_size=0,
                            run_idx=trial,
                            elapsed_ms=get_elapsed_ms,
                            ops_count=len(images),
                            throughput_ops_sec=len(images)
                            / max(get_elapsed_ms / 1000.0, 1e-9),
                            mean_op_ms=get_elapsed_ms / len(images),
                            median_op_ms=get_elapsed_ms / len(images),
                            p95_op_ms=get_elapsed_ms / len(images),
                            disk_size_mb=total_disk_bytes / (1024.0 * 1024.0),
                            throughput_mb_s=total_mb / max(get_elapsed_ms / 1000.0, 1e-9),
                            storage_overhead_pct=overhead_pct,
                        ),
                    ]
                )

    return records, {
        "product_images_count": len(images),
        "total_raw_mb": round(total_mb, 4),
    }


def _aggregate_summary(raw_records: Sequence[RawRunRecord]) -> list[dict[str, Any]]:
    """Group ``raw_records`` by configuration and compute summary statistics."""
    groups: dict[tuple[str, str, str, str, int], list[RawRunRecord]] = {}
    for rec in raw_records:
        key = (
            rec.experiment,
            rec.config,
            rec.access_method,
            rec.workload,
            rec.pool_size,
        )
        groups.setdefault(key, []).append(rec)

    summary_rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        exp, cfg, method, workload, pool_size = key
        elapsed = np.asarray([r.elapsed_ms for r in group], dtype=np.float64)
        tput = np.asarray([r.throughput_ops_sec for r in group], dtype=np.float64)
        mean_ops = np.asarray([r.mean_op_ms for r in group], dtype=np.float64)
        med_ops = np.asarray([r.median_op_ms for r in group], dtype=np.float64)
        p95_ops = np.asarray([r.p95_op_ms for r in group], dtype=np.float64)
        hits = np.asarray([r.hit_ratio for r in group], dtype=np.float64)
        reads = np.asarray([r.disk_reads for r in group], dtype=np.float64)
        writes = np.asarray([r.disk_writes for r in group], dtype=np.float64)
        pages = np.asarray([r.pages_read for r in group], dtype=np.float64)
        builds = np.asarray([r.build_time_ms for r in group], dtype=np.float64)
        sizes = np.asarray([r.disk_size_mb for r in group], dtype=np.float64)
        mb_s = np.asarray([r.throughput_mb_s for r in group], dtype=np.float64)
        overheads = np.asarray([r.storage_overhead_pct for r in group], dtype=np.float64)
        flushes = np.asarray([r.flushes for r in group], dtype=np.float64)
        compactions = np.asarray([r.compactions for r in group], dtype=np.float64)

        row = {
            "experiment": exp,
            "config": cfg,
            "access_method": method,
            "workload": workload,
            "pool_size": pool_size,
            "runs": len(group),
            "mean_ms": round(float(np.mean(elapsed)), 6),
            "median_ms": round(float(np.median(elapsed)), 6),
            "std_dev_ms": round(float(np.std(elapsed, ddof=1 if len(group) > 1 else 0)), 6),
            "p95_ms": round(float(np.percentile(elapsed, 95)), 6),
            "mean_op_ms": round(float(np.mean(mean_ops)), 6),
            "median_op_ms": round(float(np.median(med_ops)), 6),
            "p95_op_ms": round(float(np.mean(p95_ops)), 6),
            "std_p95_op_ms": round(
                float(np.std(p95_ops, ddof=1 if len(group) > 1 else 0)), 6
            ),
            "throughput_ops_sec": round(float(np.mean(tput)), 4),
            "std_throughput_ops_sec": round(
                float(np.std(tput, ddof=1 if len(group) > 1 else 0)), 4
            ),
            "mean_hit_ratio": round(float(np.mean(hits)), 6),
            "std_hit_ratio": round(
                float(np.std(hits, ddof=1 if len(group) > 1 else 0)), 6
            ),
            "mean_disk_reads": round(float(np.mean(reads)), 2),
            "mean_disk_writes": round(float(np.mean(writes)), 2),
            "mean_pages_read": round(float(np.mean(pages)), 2),
            "mean_build_time_ms": round(float(np.mean(builds)), 4),
            "mean_disk_size_mb": round(float(np.mean(sizes)), 6),
            "std_disk_size_mb": round(
                float(np.std(sizes, ddof=1 if len(group) > 1 else 0)), 6
            ),
            "mean_throughput_mb_s": round(float(np.mean(mb_s)), 4),
            "std_throughput_mb_s": round(
                float(np.std(mb_s, ddof=1 if len(group) > 1 else 0)), 4
            ),
            "storage_overhead_pct": round(float(np.mean(overheads)), 6),
            "mean_flushes": round(float(np.mean(flushes)), 2),
            "mean_compactions": round(float(np.mean(compactions)), 2),
        }
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                raise ValueError(f"Invalid numeric value {v} in summary {cfg}.{k}")
        summary_rows.append(row)

    return summary_rows


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    """Write rows to ``path`` with deterministic headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _lookup_summary(
    summary: Sequence[dict[str, Any]],
    *,
    experiment: str,
    access_method: str,
    workload: str,
    pool_size: int | None = None,
) -> dict[str, Any]:
    """Retrieve a unique summary row matching the specified keys."""
    for row in summary:
        if (
            row["experiment"] == experiment
            and row["access_method"] == access_method
            and row["workload"] == workload
            and (pool_size is None or row["pool_size"] == pool_size)
        ):
            return row
    raise KeyError(
        f"Missing summary row: exp={experiment}, method={access_method}, "
        f"workload={workload}, pool_size={pool_size}"
    )


def _plot_fig1(summary: Sequence[dict[str, Any]], out_path: Path) -> None:
    """Render ``fig1_buffer_pool_hit_ratio.png`` (300 DPI)."""
    pool_sizes = [16, 32, 64, 128]
    x = np.arange(len(pool_sizes))
    series = [
        ("LRU", "zipfian", "LRU — Zipfian (θ=0.9)", "#1f77b4", "o", "-"),
        ("FIFO", "zipfian", "FIFO — Zipfian (θ=0.9)", "#ff7f0e", "s", "--"),
        ("LRU", "sequential", "LRU — Sequential", "#2ca02c", "^", "-"),
        ("FIFO", "sequential", "FIFO — Sequential", "#d62728", "D", ":"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for method, workload, label, color, marker, ls in series:
        means = []
        stds = []
        for ps in pool_sizes:
            row = _lookup_summary(
                summary,
                experiment="1A",
                access_method=method,
                workload=workload,
                pool_size=ps,
            )
            means.append(row["mean_hit_ratio"] * 100.0)
            stds.append(row["std_hit_ratio"] * 100.0)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.0,
            markersize=7,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(ps) for ps in pool_sizes])
    ax.set_xlabel("Buffer Pool Size (frames of 8,192 bytes)", fontsize=11)
    ax.set_ylabel("Cache Hit Ratio (%)", fontsize=11)
    ax.set_ylim(-4, 108)
    ax.set_title(
        "Exp 1A: Buffer Pool Cache Hit Ratio — LRU vs. FIFO (10k Products, ±1 std)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fig2(summary: Sequence[dict[str, Any]], out_path: Path) -> None:
    """Render ``fig2_row_vs_column_latency.png`` (300 DPI)."""
    workloads = [
        ("olap_aggregation", "OLAP Aggregation\n(SUM qty*price, 200k rows)"),
        ("oltp_point_lookup", "OLTP Point Lookups\n(5,000 random RIDs)"),
    ]
    x = np.arange(len(workloads))
    width = 0.34

    row_means, row_stds = [], []
    col_means, col_stds = [], []
    for wl, _ in workloads:
        r = _lookup_summary(
            summary, experiment="1B", access_method="RowTableHeap", workload=wl
        )
        c = _lookup_summary(
            summary, experiment="1B", access_method="ColumnTableHeap", workload=wl
        )
        row_means.append(r["mean_ms"])
        row_stds.append(r["std_dev_ms"])
        col_means.append(c["mean_ms"])
        col_stds.append(c["std_dev_ms"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars_r = ax.bar(
        x - width / 2,
        row_means,
        width,
        yerr=row_stds,
        capsize=4,
        label="SlottedPage (RowTableHeap)",
        color="#2b5c8f",
    )
    bars_c = ax.bar(
        x + width / 2,
        col_means,
        width,
        yerr=col_stds,
        capsize=4,
        label="ColumnPage (ColumnTableHeap)",
        color="#e07a2f",
    )

    for bar, std in zip(bars_r, row_stds, strict=True):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + std + max(row_means) * 0.02,
            f"{h:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    for bar, std in zip(bars_c, col_stds, strict=True):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + std + max(row_means) * 0.02,
            f"{h:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in workloads], fontsize=10.5)
    ax.set_ylabel("Wall-Clock Latency (ms)", fontsize=11)
    ax.set_ylim(0, max(row_means) * 1.18)
    ax.set_title(
        "Exp 1B: Row vs. Column Storage Latency (200k Order Items, ±1 std)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fig3(summary: Sequence[dict[str, Any]], out_path: Path) -> None:
    """Render ``fig3_index_read_vs_write_throughput.png`` (300 DPI)."""
    methods = [
        ("FULL_SCAN", "Full Scan", "#7f7f7f"),
        ("BPLUS_TREE", "B+ Tree", "#1f77b4"),
        ("LSM_TREE", "LSM Tree", "#2ca02c"),
    ]
    panels = [
        ("2A", "read_heavy_90_10", "Exp 2A: Read-Heavy (90% Reads / 10% Writes)"),
        ("2B", "write_heavy_10_90", "Exp 2B: Write-Heavy (10% Reads / 90% Writes)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharey=True)
    x = np.arange(len(methods))

    for ax, (exp, wl, title) in zip(axes, panels, strict=True):
        tputs = []
        stds = []
        colors = []
        labels = []
        for method, label, color in methods:
            row = _lookup_summary(
                summary, experiment=exp, access_method=method, workload=wl
            )
            tputs.append(row["throughput_ops_sec"])
            stds.append(row["std_throughput_ops_sec"])
            colors.append(color)
            labels.append(label)

        bars = ax.bar(x, tputs, width=0.55, yerr=stds, capsize=4, color=colors)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
        for bar, val, std in zip(bars, tputs, stds, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                (val + std) * 1.18,
                f"{val:,.0f}" if val >= 100 else f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    axes[0].set_ylabel("Throughput (ops/sec, log scale)", fontsize=11)
    max_y = max(
        _lookup_summary(
            summary, experiment=exp, access_method=m, workload=wl
        )["throughput_ops_sec"]
        for exp, wl, _ in panels
        for m, _, _ in methods
    )
    axes[0].set_ylim(1.0, max_y * 2.8)
    fig.suptitle(
        "Exp 2A & 2B: Index Access Method Throughput Comparison (50k Rows, ±1 std)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fig4(summary: Sequence[dict[str, Any]], out_path: Path) -> None:
    """Render ``fig4_index_p95_latency_and_size.png`` (300 DPI)."""
    workloads = [
        ("2A", "read_heavy_90_10", "Read-Heavy\n(90/10, 50k products)"),
        ("2B", "write_heavy_10_90", "Write-Heavy\n(10/90, 50k inventory)"),
    ]
    x = np.arange(len(workloads))
    width = 0.34

    bpt_p95, bpt_p95_err, lsm_p95, lsm_p95_err = [], [], [], []
    bpt_sz, bpt_sz_err, lsm_sz, lsm_sz_err = [], [], [], []

    for exp, wl, _ in workloads:
        b = _lookup_summary(summary, experiment=exp, access_method="BPLUS_TREE", workload=wl)
        l = _lookup_summary(summary, experiment=exp, access_method="LSM_TREE", workload=wl)
        bpt_p95.append(b["p95_op_ms"])
        bpt_p95_err.append(b["std_p95_op_ms"])
        lsm_p95.append(l["p95_op_ms"])
        lsm_p95_err.append(l["std_p95_op_ms"])

        bpt_sz.append(b["mean_disk_size_mb"])
        bpt_sz_err.append(b["std_disk_size_mb"])
        lsm_sz.append(l["mean_disk_size_mb"])
        lsm_sz_err.append(l["std_disk_size_mb"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0))

    # Left: p95 per-op latency (ms)
    b1 = ax1.bar(
        x - width / 2, bpt_p95, width, yerr=bpt_p95_err, capsize=4, label="B+ Tree", color="#1f77b4"
    )
    l1 = ax1.bar(
        x + width / 2, lsm_p95, width, yerr=lsm_p95_err, capsize=4, label="LSM Tree", color="#2ca02c"
    )
    for bars in (b1, l1):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(bpt_p95 + lsm_p95) * 0.03,
                f"{h:.4f} ms",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, _, lbl in workloads], fontsize=10)
    ax1.set_ylabel("p95 Operation Latency (ms)", fontsize=11)
    ax1.set_ylim(0, max(bpt_p95 + lsm_p95) * 1.25)
    ax1.set_title("Per-Operation p95 Latency (ms)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True)

    # Right: On-disk footprint (MB)
    b2 = ax2.bar(
        x - width / 2, bpt_sz, width, yerr=bpt_sz_err, capsize=4, label="B+ Tree", color="#1f77b4"
    )
    l2 = ax2.bar(
        x + width / 2, lsm_sz, width, yerr=lsm_sz_err, capsize=4, label="LSM Tree", color="#2ca02c"
    )
    for bars in (b2, l2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(bpt_sz + lsm_sz) * 0.03,
                f"{h:.2f} MB",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, _, lbl in workloads], fontsize=10)
    ax2.set_ylabel("On-Disk Index Footprint (MB)", fontsize=11)
    ax2.set_ylim(0, max(bpt_sz + lsm_sz) * 1.25)
    ax2.set_title("On-Disk Index Footprint (MB)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle(
        "Exp 2A & 2B: B+ Tree vs. LSM Tree p95 Latency & Disk Footprint (±1 std)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fig5(summary: Sequence[dict[str, Any]], out_path: Path) -> None:
    """Render ``fig5_encryption_throughput.png`` (300 DPI)."""
    ops = [
        ("put_object", "Write (put_object + fsync)"),
        ("get_object", "Read (get_object + verify)"),
    ]
    x = np.arange(len(ops))
    width = 0.34

    noop_means, noop_stds = [], []
    aes_means, aes_stds = [], []
    for wl, _ in ops:
        n = _lookup_summary(
            summary, experiment="2D", access_method="NoOpCipher", workload=wl
        )
        a = _lookup_summary(
            summary, experiment="2D", access_method="AES256GCMHook", workload=wl
        )
        noop_means.append(n["mean_throughput_mb_s"])
        noop_stds.append(n["std_throughput_mb_s"])
        aes_means.append(a["mean_throughput_mb_s"])
        aes_stds.append(a["std_throughput_mb_s"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars_n = ax.bar(
        x - width / 2,
        noop_means,
        width,
        yerr=noop_stds,
        capsize=4,
        label="NoOpCipher (Unencrypted Baseline)",
        color="#4c72b0",
    )
    bars_a = ax.bar(
        x + width / 2,
        aes_means,
        width,
        yerr=aes_stds,
        capsize=4,
        label="AES256GCMHook (AEAD + OID AAD)",
        color="#c44e52",
    )

    top_y = max(m + s for m, s in zip(noop_means + aes_means, noop_stds + aes_stds, strict=True))
    for bars, stds in ((bars_n, noop_stds), (bars_a, aes_stds)):
        for bar, std in zip(bars, stds, strict=True):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + std + top_y * 0.02,
                f"{h:.1f} MB/s",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in ops], fontsize=10.5)
    ax.set_ylabel("Throughput (MB/s)", fontsize=11)
    ax.set_ylim(0, top_y * 1.2)
    ax.set_title(
        "Exp 2D: ObjectStore Encryption Throughput (500 Product Images, ±1 std)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_all_benchmarks(
    *, runs: int = 5, seed: int = 42, out_dir: str | Path = "results/m1_m2"
) -> dict[str, Any]:
    """Execute all 6 controlled experiments, export CSV/JSON metrics, and plot 5 charts.

    Args:
        runs: Number of measured trials per configuration point (plus 1 warmup).
        seed: Deterministic PRNG seed.
        out_dir: Output directory for ``env_metadata.json``, CSVs, and ``plots/``.

    Returns:
        Summary dictionary with output paths, row counts, and total runtime.

    Raises:
        ValueError: If ``runs < 1``.
    """
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError(f"runs must be an integer >= 1, got {runs!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    t_all = time.perf_counter()
    output_root = Path(out_dir)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_records: list[RawRunRecord] = []
    dataset_counts: dict[str, Any] = {}

    tmp_root = Path(tempfile.mkdtemp(prefix="csai302_bench_"))
    try:
        r1a, m1a = _run_exp_1a(tmp_root, runs=runs, seed=seed)
        raw_records.extend(r1a)
        dataset_counts["exp1a"] = m1a

        r1b, m1b = _run_exp_1b(tmp_root, runs=runs, seed=seed)
        raw_records.extend(r1b)
        dataset_counts["exp1b"] = m1b

        r2a, m2a = _run_exp_2a(tmp_root, runs=runs, seed=seed)
        raw_records.extend(r2a)
        dataset_counts["exp2a"] = m2a

        r2b, m2b = _run_exp_2b(tmp_root, runs=runs, seed=seed)
        raw_records.extend(r2b)
        dataset_counts["exp2b"] = m2b

        r2c, m2c = _run_exp_2c(runs=runs, seed=seed)
        raw_records.extend(r2c)
        dataset_counts["exp2c"] = m2c

        r2d, m2d = _run_exp_2d(tmp_root, runs=runs, seed=seed)
        raw_records.extend(r2d)
        dataset_counts["exp2d"] = m2d
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    summary_rows = _aggregate_summary(raw_records)

    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "PAGE_SIZE": PAGE_SIZE,
        "DEFAULT_BUFFER_POOL_SIZE": DEFAULT_BUFFER_POOL_SIZE,
        "DEFAULT_MEMTABLE_BYTES": DEFAULT_MEMTABLE_BYTES,
        "random_seed": seed,
        "runs": runs,
        "warmup_runs": 1,
        "dataset_counts": dataset_counts,
        "methodology": {
            "protocol": "1 untimed warmup + N measured trials per configuration point with correctness assertions prior to timing export.",
            "exp_2a_full_scan_sampling": (
                "BPLUS_TREE and LSM_TREE execute all 10,000 operations (7,000 point, 2,000 width-20 range, 1,000 inserts). "
                "FULL_SCAN executes a stratified 200-operation representative sample (140 point, 40 range, 20 inserts) from the "
                "identical shuffled workload and records per-operation latencies (mean, median, p95) and ops/sec for apples-to-apples comparison."
            ),
            "exp_2b_full_scan_sampling": (
                "BPLUS_TREE and LSM_TREE execute all 20,000 operations (18,000 stock updates/upserts, 2,000 point lookups) on a fresh "
                "copytree baseline per trial. FULL_SCAN executes a stratified 200-operation sample (180 updates, 20 point lookups) and "
                "reports exact per-operation latencies and ops/sec."
            ),
        },
        "total_elapsed_seconds": round(time.perf_counter() - t_all, 3),
    }

    env_path = output_root / "env_metadata.json"
    env_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    raw_csv_path = output_root / "raw_runs.csv"
    _write_csv(raw_csv_path, _RAW_CSV_FIELDS, [asdict(r) for r in raw_records])

    summary_csv_path = output_root / "summary_metrics.csv"
    _write_csv(summary_csv_path, _SUMMARY_CSV_FIELDS, summary_rows)

    plot_paths = [
        plots_dir / "fig1_buffer_pool_hit_ratio.png",
        plots_dir / "fig2_row_vs_column_latency.png",
        plots_dir / "fig3_index_read_vs_write_throughput.png",
        plots_dir / "fig4_index_p95_latency_and_size.png",
        plots_dir / "fig5_encryption_throughput.png",
    ]
    _plot_fig1(summary_rows, plot_paths[0])
    _plot_fig2(summary_rows, plot_paths[1])
    _plot_fig3(summary_rows, plot_paths[2])
    _plot_fig4(summary_rows, plot_paths[3])
    _plot_fig5(summary_rows, plot_paths[4])

    return {
        "env_metadata": str(env_path),
        "raw_runs_csv": str(raw_csv_path),
        "summary_metrics_csv": str(summary_csv_path),
        "raw_row_count": len(raw_records),
        "summary_row_count": len(summary_rows),
        "plots": [str(p) for p in plot_paths],
        "elapsed_seconds": metadata["total_elapsed_seconds"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``python -m bench.run_m1_m2_benchmarks``."""
    parser = argparse.ArgumentParser(
        description="Run CSAI 302 M1 + M2 reproducible benchmark suite and generate plots."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Measured trials per configuration point after 1 warmup (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic PRNG seed (default: 42).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/m1_m2"),
        help="Output directory for metadata, CSVs, and plots (default: results/m1_m2).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the M1 + M2 benchmark and visualization suite."""
    args = _parse_args(argv)
    summary = run_all_benchmarks(runs=args.runs, seed=args.seed, out_dir=args.out_dir)
    print(
        f"Completed M1/M2 benchmarks in {summary['elapsed_seconds']:.2f}s: "
        f"raw_rows={summary['raw_row_count']}, "
        f"summary_rows={summary['summary_row_count']}, "
        f"plots={len(summary['plots'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RawRunRecord", "main", "run_all_benchmarks"]
