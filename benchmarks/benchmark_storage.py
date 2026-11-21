"""Reproducible M1 storage engine benchmarks following the CSAI 302 evaluation protocol.

Suites:
  1. buffer_policy       - FIFO vs LRU on a mixed hot/cold page trace.
  2. projection_scan     - Row vs Column layout scanning 2 projected columns of 5.
  3. point_lookup_vs_scan - Random get_record() vs full scan on Row and Column.
  4. backend_insert      - PageFile vs LocalObjectStore vs PageFile+transform hook.
"""
from __future__ import annotations

import argparse
import csv
import math
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from smartdb.storage.buffer_pool import BufferPool
from smartdb.storage.page_file import PAGE_SIZE, PageFile
from smartdb.storage.table import StorageTable

SCHEMA = [
    ("product_id", "INT", False),
    ("name", "STR", False),
    ("category", "STR", False),
    ("price", "FLOAT", False),
    ("stock", "INT", False),
]


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[idx]


def _summarize(
    suite: str,
    variant: str,
    ops: int,
    timings_ms: list[float],
    stats: dict[str, Any],
) -> dict[str, Any]:
    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    stdev_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    ops_per_sec = (ops / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0
    return {
        "suite": suite,
        "variant": variant,
        "ops": ops,
        "ops_per_sec": round(ops_per_sec, 1),
        "mean_ms": round(mean_ms, 4),
        "median_ms": round(median_ms, 4),
        "p95_ms": round(_p95(timings_ms), 4),
        "stdev_ms": round(stdev_ms, 4),
        "hits": stats.get("hits", ""),
        "misses": stats.get("misses", ""),
        "hit_rate": stats.get("hit_rate", ""),
        "disk_reads": stats.get("disk_reads", ""),
        "evictions": stats.get("evictions", ""),
        "disk_writes": stats.get("disk_writes", ""),
    }


def _measure_repeated(
    repeats: int,
    fn: Callable[[], tuple[int, dict[str, Any]]],
) -> tuple[int, list[float], dict[str, Any]]:
    fn()  # Warm-up run
    timings_ms: list[float] = []
    last_stats: dict[str, Any] = {}
    ops = 0
    for _ in range(repeats):
        start = time.perf_counter()
        ops, last_stats = fn()
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return ops, timings_ms, last_stats


def _make_records(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    categories = ["Accessories", "Electronics", "Home", "Office", "Clothing"]
    return [
        {
            "product_id": i,
            "name": f"Product-{i:06d}-" + ("x" * 24),
            "category": rng.choice(categories),
            "price": round(rng.uniform(2.0, 500.0), 2),
            "stock": rng.randint(0, 250),
        }
        for i in range(count)
    ]


def bench_buffer_policy(
    num_pages: int, accesses: int, capacity: int, seed: int, repeats: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    hot = max(1, min(num_pages, capacity // 2 or 1))
    trace = [
        rng.randrange(hot) if rng.random() < 0.8 else rng.randrange(num_pages)
        for _ in range(accesses)
    ]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.pages"
        with PageFile(path) as pf:
            for page_id in range(num_pages):
                pf.allocate_page()
                pf.write_page(page_id, bytes([page_id % 251]) * PAGE_SIZE)

        for policy in ("FIFO", "LRU"):
            def _run() -> tuple[int, dict[str, Any]]:
                with PageFile(path) as backend:
                    pool = BufferPool(backend, capacity=capacity, policy=policy)
                    for pid in trace:
                        pool.read_page(pid)
                    snap = pool.snapshot()
                    pool.close()
                    return len(trace), snap

            ops, timings, snap = _measure_repeated(repeats, _run)
            rows.append(_summarize("buffer_policy", policy, ops, timings, snap))
    return rows


def bench_layouts(
    record_count: int, capacity: int, seed: int, repeats: int
) -> list[dict[str, Any]]:
    records = _make_records(record_count, seed)
    rng = random.Random(seed)
    sample_indices = [rng.randrange(record_count) for _ in range(min(200, record_count))]
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        locators_by_layout: dict[str, list[tuple[int, int]]] = {}
        table_paths: dict[str, Path] = {}
        for layout in ("row", "column"):
            tpath = Path(tmp) / layout / "Products.pages"
            table_paths[layout] = tpath
            table = StorageTable(
                tpath, fields=SCHEMA, layout=layout, buffer_frames=capacity, replacement_policy="LRU"
            )
            locators_by_layout[layout] = [table.insert(rec) for rec in records]
            table.close()

        for layout in ("row", "column"):
            def _proj_scan() -> tuple[int, dict[str, Any]]:
                tbl = StorageTable(
                    table_paths[layout], fields=SCHEMA, layout=layout, buffer_frames=capacity, replacement_policy="LRU"
                )
                result = tbl.scan(columns=["product_id", "price"])
                snap = tbl.pool.snapshot()
                tbl.close()
                return len(result), snap

            ops, timings, snap = _measure_repeated(repeats, _proj_scan)
            rows.append(_summarize("projection_scan", layout, ops, timings, snap))

        for layout in ("row", "column"):
            sample_locs = [locators_by_layout[layout][i] for i in sample_indices]

            def _point_lookup() -> tuple[int, dict[str, Any]]:
                tbl = StorageTable(
                    table_paths[layout], fields=SCHEMA, layout=layout, buffer_frames=capacity, replacement_policy="LRU"
                )
                for loc in sample_locs:
                    tbl.get_record(loc)
                snap = tbl.pool.snapshot()
                tbl.close()
                return len(sample_locs), snap

            ops, timings, snap = _measure_repeated(repeats, _point_lookup)
            rows.append(_summarize("point_lookup", layout, ops, timings, snap))

    return rows


def bench_backends(
    record_count: int, capacity: int, seed: int, repeats: int
) -> list[dict[str, Any]]:
    records = _make_records(record_count, seed)
    xor = lambda data: bytes(b ^ 0x5A for b in data)
    configs = [
        ("page_plain", "page", None, None),
        ("objects_plain", "objects", None, None),
        ("page_xor_hook", "page", xor, xor),
    ]
    rows: list[dict[str, Any]] = []
    for variant, backend_kind, enc, dec in configs:
        run_counter = 0

        def _insert_run() -> tuple[int, dict[str, Any]]:
            nonlocal run_counter
            run_counter += 1
            with tempfile.TemporaryDirectory() as tmp:
                tbl = StorageTable(
                    Path(tmp) / f"Products_{run_counter}.pages",
                    fields=SCHEMA,
                    layout="row",
                    buffer_frames=capacity,
                    backend_kind=backend_kind,
                    encrypt_page=enc,
                    decrypt_page=dec,
                )
                for rec in records:
                    tbl.insert(rec)
                tbl.flush()
                snap = tbl.pool.snapshot()
                tbl.close()
                return len(records), snap

        ops, timings, snap = _measure_repeated(repeats, _insert_run)
        rows.append(_summarize("backend_insert", variant, ops, timings, snap))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=128)
    parser.add_argument("--accesses", type=int, default=10000)
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--insert-records", type=int, default=500)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="benchmark_results/m1_storage.csv")
    args = parser.parse_args()

    print(
        f"Environment: OS={platform.platform()}, Python={sys.version.split()[0]}, "
        f"seed={args.seed}, repeats={args.repeats}, capacity={args.capacity}"
    )
    rows = (
        bench_buffer_policy(args.pages, args.accesses, args.capacity, args.seed, args.repeats)
        + bench_layouts(args.records, args.capacity, args.seed, args.repeats)
        + bench_backends(args.insert_records, args.capacity, args.seed, args.repeats)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
