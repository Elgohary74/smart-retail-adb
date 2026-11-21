"""M1 Storage Engine Benchmark: FIFO vs LRU, Row vs Column, and Page vs Object backends."""
from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from common import environment_metadata, run_repeated, save_bar_chart, write_csv
from smartdb.engine import SmartRetailDB
from smartdb.schema import ColumnSpec, DataType
from smartdb.storage.buffer_pool import BufferPool
from smartdb.storage.encryption import make_aes_gcm_hooks
from smartdb.storage.page_file import PAGE_SIZE, PageFile

SCHEMA = [
    ColumnSpec("product_id", DataType.INT, nullable=False),
    ColumnSpec("name", DataType.STR, nullable=False),
    ColumnSpec("category", DataType.STR, nullable=False),
    ColumnSpec("price", DataType.FLOAT, nullable=False),
    ColumnSpec("stock", DataType.INT, nullable=False),
    ColumnSpec("description", DataType.STR, nullable=False),
]


def bench_buffer_policies(pages: int, accesses: int, capacity: int, seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    hot = max(1, min(pages, capacity // 2 or 1))
    trace = [rng.randrange(hot) if rng.random() < 0.8 else rng.randrange(pages) for _ in range(accesses)]
    rows = []
    for policy in ("FIFO", "LRU"):
        def _once() -> dict[str, Any]:
            with tempfile.TemporaryDirectory() as tmp:
                backend = PageFile(Path(tmp) / "bench.pages")
                for pid in range(pages):
                    backend.allocate_page()
                    backend.write_page(pid, bytes([pid % 251]) * PAGE_SIZE)
                pool = BufferPool(backend, capacity=capacity, policy=policy)
                for pid in trace:
                    with pool.pin_page(pid):
                        pass
                snap = pool.snapshot()
                pool.close()
                return {
                    "hits": snap["hits"],
                    "misses": snap["misses"],
                    "hit_rate": snap["hit_rate"],
                    "disk_reads": snap["disk_reads"],
                    "evictions": snap["evictions"],
                }

        runs, summary = run_repeated(_once, repeats=repeats)
        last = runs[-1]
        rows.append({
            "suite": "buffer_policy",
            "variant": policy,
            "ops": accesses,
            "ops_per_sec": round(accesses / (summary["median_ms"] / 1000.0), 1),
            **summary,
            **{k: last[k] for k in ("hits", "misses", "hit_rate", "disk_reads", "evictions")},
        })
    return rows


def bench_layouts_and_backends(records: int, seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    data = [
        {
            "product_id": i,
            "name": f"Product-{i:05d}",
            "category": f"Cat-{i % 5}",
            "price": round(rng.uniform(5.0, 500.0), 2),
            "stock": i % 100,
            "description": "D" * 160,
        }
        for i in range(records)
    ]
    rows: list[dict[str, Any]] = []
    for layout in ("row", "column"):
        with tempfile.TemporaryDirectory() as tmp:
            db = SmartRetailDB(tmp, buffer_frames=32)
            db.create_table("Products", SCHEMA, layout=layout)
            for rec in data:
                db.insert("Products", rec)
            db.close()

            def _proj_scan() -> dict[str, Any]:
                reopened = SmartRetailDB(tmp, buffer_frames=16)
                res = reopened.execute("SELECT product_id, price FROM Products")
                assert len(res) == records
                snap = reopened.buffer_stats()
                reopened.close()
                return {"disk_reads": snap["disk_reads"], "hit_rate": snap["hit_rate"]}

            runs, summary = run_repeated(_proj_scan, repeats=repeats)
            rows.append({
                "suite": "projection_scan",
                "variant": layout,
                "ops": records,
                "ops_per_sec": round(records / (summary["median_ms"] / 1000.0), 1),
                **summary,
                "disk_reads": runs[-1]["disk_reads"],
                "hit_rate": runs[-1]["hit_rate"],
            })

    # Page vs Objects vs AES-GCM
    enc, dec, ovh, rec_enc, rec_dec = make_aes_gcm_hooks(os.urandom(32))
    configs = [
        ("page_plain", "page", None, None, 0, None, None),
        ("objects_plain", "objects", None, None, 0, None, None),
        ("page_aes_gcm", "page", enc, dec, ovh, rec_enc, rec_dec),
    ]
    for label, bkind, ep, dp, co, er, dr in configs:
        def _insert_once() -> dict[str, Any]:
            with tempfile.TemporaryDirectory() as tmp:
                db = SmartRetailDB(
                    tmp, buffer_frames=32, backend_kind=bkind,
                    encrypt_page=ep, decrypt_page=dp, crypto_overhead=co,
                    encrypt_record=er, decrypt_record=dr,
                )
                db.create_table("Products", SCHEMA, layout="row")
                for rec in data[: min(records, 500)]:
                    db.insert("Products", rec)
                snap = db.buffer_stats()
                db.close()
                return {"disk_reads": snap["disk_reads"], "disk_writes": snap["disk_writes"]}

        n_ins = min(records, 500)
        runs, summary = run_repeated(_insert_once, repeats=repeats)
        rows.append({
            "suite": "backend_insert",
            "variant": label,
            "ops": n_ins,
            "ops_per_sec": round(n_ins / (summary["median_ms"] / 1000.0), 1),
            **summary,
            "disk_writes": runs[-1]["disk_writes"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=128)
    parser.add_argument("--accesses", type=int, default=10000)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/m1_storage.csv"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    print("Environment:", environment_metadata(), f"seed={args.seed} repeats={args.repeats}")
    rows = bench_buffer_policies(args.pages, args.accesses, args.capacity, args.seed, args.repeats)
    rows.extend(bench_layouts_and_backends(args.records, args.seed, args.repeats))
    write_csv(args.out, rows)
    for row in rows:
        print(row)
    if args.plots:
        proj = [r for r in rows if r["suite"] == "projection_scan"]
        save_bar_chart(
            args.out.parent / "plots/m1_row_vs_col.png",
            "M1: 2-Column Projection Scan (Row vs Column)",
            [r["variant"] for r in proj],
            [r["median_ms"] for r in proj],
            "Median Latency (ms)",
        )
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
