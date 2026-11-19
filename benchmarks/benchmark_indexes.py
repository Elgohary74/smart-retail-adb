"""Compare B+ Tree and LSM index write and point-lookup performance."""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from smartdb.indexes.bplus_tree import BPlusTree
from smartdb.indexes.lsm_tree import LSMTree


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _measure(index_type: str, keys: list[int], query_keys: list[int],
             repeat: int, seed: int) -> list[dict[str, Any]]:
    expected = {key: (key, 0) for key in keys}
    with tempfile.TemporaryDirectory() as temp_dir:
        if index_type == "BPLUS":
            index = BPlusTree()
        else:
            index = LSMTree(Path(temp_dir) / "lsm")

        start = time.perf_counter()
        for key in keys:
            index.insert(key, expected[key])
        if index_type == "LSM":
            index.close()
        write_seconds = time.perf_counter() - start

        start = time.perf_counter()
        results = [index.search(key) for key in query_keys]
        read_seconds = time.perf_counter() - start

        for key, result in zip(query_keys, results):
            if result != [expected[key]]:
                raise AssertionError(
                    f"{index_type} returned an incorrect result for key {key}: {result}"
                )

    return [
        {
            "index": index_type,
            "operation": "write",
            "records": len(keys),
            "queries": len(query_keys),
            "seed": seed,
            "repeat": repeat,
            "elapsed_seconds": write_seconds,
            "operations_per_second": len(keys) / write_seconds,
        },
        {
            "index": index_type,
            "operation": "point_lookup",
            "records": len(keys),
            "queries": len(query_keys),
            "seed": seed,
            "repeat": repeat,
            "elapsed_seconds": read_seconds,
            "operations_per_second": len(query_keys) / read_seconds,
        },
    ]


def run(records: int, seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(range(records))
    rng.shuffle(keys)
    query_keys = keys.copy()
    rng.shuffle(query_keys)

    measurements = []
    for repeat in range(1, repeats + 1):
        for index_type in ("BPLUS", "LSM"):
            measurements.extend(
                _measure(index_type, keys, query_keys, repeat, seed)
            )

    summary = []
    for index_type in ("BPLUS", "LSM"):
        for operation in ("write", "point_lookup"):
            samples = [
                row["elapsed_seconds"]
                for row in measurements
                if row["index"] == index_type and row["operation"] == operation
            ]
            summary.append(
                {
                    "index": index_type,
                    "operation": operation,
                    "records": records,
                    "queries": records,
                    "seed": seed,
                    "repeat": "median",
                    "elapsed_seconds": statistics.median(samples),
                    "operations_per_second": records / statistics.median(samples),
                }
            )
    return measurements + summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=_positive_int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/m2_indexes.csv"))
    args = parser.parse_args()

    rows = run(args.records, args.seed, args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["repeat"] == "median":
            print(
                f"{row['index']} {row['operation']}: median "
                f"{row['elapsed_seconds']:.6f}s, "
                f"{row['operations_per_second']:.0f} ops/s"
            )
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
