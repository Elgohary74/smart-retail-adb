"""Compare FIFO and LRU on the same reproducible page-access trace."""
from __future__ import annotations
import argparse
import csv
import random
import tempfile
import time
from pathlib import Path
from smartdb.storage.page_file import PageFile, PAGE_SIZE
from smartdb.storage.buffer_pool import BufferPool


def run(policy: str, num_pages: int, accesses: int, capacity: int, seed: int) -> dict:
    rng = random.Random(seed)
    # A hot set mixed with occasional cold pages makes replacement differences visible.
    hot = max(1, min(num_pages, capacity // 2 or 1))
    trace = [rng.randrange(hot) if rng.random() < 0.8 else rng.randrange(num_pages) for _ in range(accesses)]
    with tempfile.TemporaryDirectory() as tmp:
        backend = PageFile(Path(tmp) / "bench.pages")
        for page_id in range(num_pages):
            backend.allocate_page()
            backend.write_page(page_id, bytes([page_id % 251]) * PAGE_SIZE)
        pool = BufferPool(backend, capacity=capacity, policy=policy)
        start = time.perf_counter()
        for page_id in trace:
            pool.read_page(page_id)
        elapsed = time.perf_counter() - start
        pool.close()
        stats = pool.snapshot()
        return {"policy": policy, "pages": num_pages, "accesses": accesses,
                "capacity": capacity, "seed": seed, "elapsed_seconds": round(elapsed, 6), **stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=128)
    parser.add_argument("--accesses", type=int, default=10000)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="benchmark_results/m1_buffer_pool.csv")
    args = parser.parse_args()
    rows = [run(policy, args.pages, args.accesses, args.capacity, args.seed) for policy in ("FIFO", "LRU")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
