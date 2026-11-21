"""M2 Indexing Benchmark: Indexed vs Full Scan, Read-Heavy vs Write-Heavy, Range & Index Size."""
from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path
from typing import Any

from common import environment_metadata, run_repeated, save_bar_chart, write_csv
from smartdb.engine import SmartRetailDB
from smartdb.schema import ColumnSpec, DataType

SCHEMA = [
    ColumnSpec("product_id", DataType.INT, nullable=False),
    ColumnSpec("name", DataType.STR, nullable=False),
    ColumnSpec("price", DataType.FLOAT, nullable=False),
]


def _setup_db(tmp: str, records: int, index_type: str | None, seed: int) -> tuple[SmartRetailDB, list[int]]:
    rng = random.Random(seed)
    keys = list(range(records))
    rng.shuffle(keys)
    db = SmartRetailDB(tmp, buffer_frames=32)
    db.create_table("Products", SCHEMA, layout="row")
    if index_type is not None:
        db.create_index("Products", "product_id", index_type)
    for k in keys:
        db.insert("Products", {"product_id": k, "name": f"Item-{k}", "price": float(k % 500)})
    db.pool.flush_all()
    return db, keys


def bench_scan_vs_index(records: int, queries: int, seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    qkeys = [rng.randrange(records) for _ in range(queries)]
    rows: list[dict[str, Any]] = []
    for variant, idx_type in (("full_scan", None), ("bplus_index", "BPLUS"), ("lsm_index", "LSM")):
        with tempfile.TemporaryDirectory() as tmp:
            db, _ = _setup_db(tmp, records, idx_type, seed)
            idx_bytes = db.index_manager.disk_bytes("Products", "product_id") if idx_type else 0

            def _lookup() -> dict[str, Any]:
                for k in qkeys:
                    res = db.execute(f"SELECT product_id, name FROM Products WHERE product_id = {k}")
                    assert len(res) == 1 and res[0]["product_id"] == k
                return {"index_bytes": idx_bytes}

            _, summary = run_repeated(_lookup, repeats=repeats)
            db.close()
            rows.append({
                "suite": "point_lookup_vs_scan",
                "variant": variant,
                "ops": queries,
                "ops_per_sec": round(queries / (summary["median_ms"] / 1000.0), 1),
                "index_bytes": idx_bytes,
                **summary,
            })
    return rows


def bench_workloads(records: int, ops: int, seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    # 1. Write-heavy (80W/20R) vs Read-heavy (90R/10W)
    for workload, write_prob in (("write_heavy_80W_20R", 0.8), ("read_heavy_90R_10W", 0.1)):
        for idx_type in ("BPLUS", "LSM"):
            def _run_mixed() -> dict[str, Any]:
                with tempfile.TemporaryDirectory() as tmp:
                    db, _ = _setup_db(tmp, records // 2, idx_type, seed)
                    next_id = records // 2
                    for _ in range(ops):
                        if rng.random() < write_prob:
                            db.insert("Products", {"product_id": next_id, "name": f"N-{next_id}", "price": 10.0})
                            next_id += 1
                        else:
                            k = rng.randrange(next_id)
                            res = db.execute(f"SELECT product_id FROM Products WHERE product_id = {k}")
                            assert len(res) == 1
                    idx_bytes = db.index_manager.disk_bytes("Products", "product_id")
                    db.close()
                    return {"index_bytes": idx_bytes}

            runs, summary = run_repeated(_run_mixed, repeats=repeats)
            rows.append({
                "suite": workload,
                "variant": idx_type,
                "ops": ops,
                "ops_per_sec": round(ops / (summary["median_ms"] / 1000.0), 1),
                "index_bytes": runs[-1]["index_bytes"],
                **summary,
            })

    # 2. Range scan comparison
    for idx_type in ("BPLUS", "LSM"):
        with tempfile.TemporaryDirectory() as tmp:
            db, _ = _setup_db(tmp, records, idx_type, seed)
            def _range() -> dict[str, Any]:
                res = db.execute(f"SELECT product_id FROM Products WHERE product_id BETWEEN 100 AND {100 + min(200, records // 2)}")
                assert len(res) > 0
                return {"rows_returned": len(res)}

            _, summary = run_repeated(_range, repeats=repeats)
            db.close()
            rows.append({
                "suite": "range_scan",
                "variant": idx_type,
                "ops": 1,
                "ops_per_sec": round(1000.0 / summary["median_ms"], 1),
                **summary,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--ops", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/m2_indexes.csv"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    print("Environment:", environment_metadata(), f"seed={args.seed} repeats={args.repeats}")
    rows = bench_scan_vs_index(args.records, args.queries, args.seed, args.repeats)
    rows.extend(bench_workloads(args.records, args.ops, args.seed, args.repeats))
    write_csv(args.out, rows)
    for row in rows:
        print(row)
    if args.plots:
        pt = [r for r in rows if r["suite"] == "point_lookup_vs_scan"]
        save_bar_chart(
            args.out.parent / "plots/m2_scan_vs_index.png",
            "M2: Full Scan vs Indexed Point Lookup",
            [r["variant"] for r in pt],
            [r["median_ms"] for r in pt],
            "Median Latency (ms)",
        )
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
