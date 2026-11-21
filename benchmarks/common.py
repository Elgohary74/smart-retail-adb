"""Shared reproducible measurement utilities for M1 & M2 benchmarks."""
from __future__ import annotations

import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from smartdb.core import DEFAULT_BENCH_REPEATS, DEFAULT_BENCH_WARMUP


def environment_metadata() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "machine": platform.machine(),
    }


def summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95_idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1) + 0.5))
    return {
        "mean_ms": round(statistics.mean(ordered), 4),
        "median_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[p95_idx], 4),
        "stdev_ms": round(statistics.stdev(ordered), 4) if len(ordered) > 1 else 0.0,
    }


def run_repeated(
    fn: Callable[[], dict[str, Any]],
    repeats: int = DEFAULT_BENCH_REPEATS,
    warmup: int = DEFAULT_BENCH_WARMUP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for _ in range(warmup):
        fn()
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        metrics = fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        runs.append({"elapsed_ms": round(elapsed_ms, 4), **metrics})
    summary = summarize_ms([r["elapsed_ms"] for r in runs])
    return runs, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_bar_chart(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
