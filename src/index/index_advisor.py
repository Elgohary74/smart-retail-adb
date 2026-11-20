"""
Module: src/index/index_advisor.py
Layer: Layer 2 — Access Methods & Workload Intelligence
Lock Domain: WorkloadAdvisor._lock (RLock; leaf metadata lock, never acquires storage or index locks)

Purpose:
    Implements the sliding-window ``WorkloadAdvisor`` auto-selector that monitors
    per-column operation mixes (``POINT_READ``, ``RANGE_READ``, ``WRITE``) and
    deterministically selects between ``FULL_SCAN``, ``LSM_TREE``, and
    ``BPLUS_TREE`` using quantitative ``(alpha, beta)`` thresholds.

Key Invariants:
    1. Each ``(table_name, column_name)`` pair maintains an independent bounded
       ``collections.deque(maxlen=window_size)`` so evicted operations outside
       the sliding window never bias current recommendations.
    2. Decision boundaries are strictly ordered: small tables / cold starts
       (``n_ops < min_ops`` or ``page_count < min_pages``) always choose
       ``FULL_SCAN``; strict inequalities ``alpha > alpha_threshold`` and
       ``beta < beta_threshold`` select ``LSM_TREE``; all other workloads fall
       back to ``BPLUS_TREE``.
    3. Every ``recommend()`` invocation appends an immutable ``Recommendation``
       audit entry to ``decision_log`` under ``_lock``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any

from src.common.config import (
    ADVISOR_ALPHA_THRESHOLD,
    ADVISOR_BETA_THRESHOLD,
    ADVISOR_MIN_OPS,
    ADVISOR_MIN_PAGES,
    ADVISOR_WINDOW_SIZE,
)

_VALID_OP_TYPES: frozenset[str] = frozenset({"POINT_READ", "RANGE_READ", "WRITE"})


class IndexStrategy(str, Enum):
    """Supported physical access-path strategies recommended by WorkloadAdvisor."""

    FULL_SCAN = "FULL_SCAN"
    BPLUS_TREE = "BPLUS_TREE"
    LSM_TREE = "LSM_TREE"


@dataclass(frozen=True)
class Recommendation:
    """Immutable audit record capturing a single WorkloadAdvisor recommendation."""

    strategy: IndexStrategy
    alpha: float
    beta: float
    n_ops: int
    page_count: int
    reason: str
    timestamp: float
    table_name: str = ""
    column_name: str = ""


class WorkloadAdvisor:
    """Thread-safe sliding-window index selector driven by ``(alpha, beta)`` ratios.

    Tracks recent operations per ``(table_name, column_name)`` in a fixed-capacity
    sliding window and evaluates deterministic Rules 1–3 to recommend ``FULL_SCAN``,
    ``LSM_TREE``, or ``BPLUS_TREE``.
    """

    def __init__(
        self,
        window_size: int = ADVISOR_WINDOW_SIZE,
        min_ops: int = ADVISOR_MIN_OPS,
        min_pages: int = ADVISOR_MIN_PAGES,
        alpha_threshold: float = ADVISOR_ALPHA_THRESHOLD,
        beta_threshold: float = ADVISOR_BETA_THRESHOLD,
    ) -> None:
        """Initialize the workload advisor with sliding-window and rule thresholds.

        Args:
            window_size: Maximum number of recent operations retained per column.
            min_ops: Minimum windowed operations required before recommending an index.
            min_pages: Minimum table page count required before recommending an index.
            alpha_threshold: Write-intensity ratio threshold above which ``LSM_TREE``
                is eligible.
            beta_threshold: Range-query ratio threshold below which ``LSM_TREE`` is
                eligible.

        Raises:
            ValueError: If ``window_size < 1``, ``min_ops < 0``, or ``min_pages < 0``.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_ops < 0:
            raise ValueError(f"min_ops must be >= 0, got {min_ops}")
        if min_pages < 0:
            raise ValueError(f"min_pages must be >= 0, got {min_pages}")

        self.window_size: int = window_size
        self.min_ops: int = min_ops
        self.min_pages: int = min_pages
        self.alpha_threshold: float = alpha_threshold
        self.beta_threshold: float = beta_threshold

        self._lock: threading.RLock = threading.RLock()
        self._windows: dict[tuple[str, str], deque[str]] = {}
        self.decision_log: list[Recommendation] = []

    def record(self, op_type: str, table_name: str, column_name: str) -> None:
        """Record a single workload operation into the column's sliding window.

        Args:
            op_type: Operation classification in ``{"POINT_READ", "RANGE_READ", "WRITE"}``.
            table_name: Target table name.
            column_name: Target column name.

        Raises:
            ValueError: If ``op_type`` is not one of the supported operation types.

        Complexity & Concurrency:
            $O(1)$ amortized time and bounded $O(\\text{window\\_size})$ space per
            column; synchronized via ``self._lock``.
        """
        if op_type not in _VALID_OP_TYPES:
            raise ValueError(
                f"Invalid op_type {op_type!r}; expected one of {sorted(_VALID_OP_TYPES)}"
            )

        key = (table_name, column_name)
        with self._lock:
            window = self._windows.get(key)
            if window is None:
                window = deque(maxlen=self.window_size)
                self._windows[key] = window
            window.append(op_type)

    def _compute_metrics_unlocked(
        self, table_name: str, column_name: str
    ) -> dict[str, Any]:
        """Compute windowed counts and ``(alpha, beta)`` ratios while ``_lock`` is held."""
        window = self._windows.get((table_name, column_name))
        r_eq = 0
        r_rng = 0
        u = 0
        if window is not None:
            for op in window:
                if op == "POINT_READ":
                    r_eq += 1
                elif op == "RANGE_READ":
                    r_rng += 1
                else:
                    u += 1

        n_ops = r_eq + r_rng + u
        # Guard against zero division on empty or write-only windows.
        alpha = u / max(1, n_ops)
        beta = r_rng / max(1, r_eq + r_rng)
        return {
            "n_ops": n_ops,
            "point_reads": r_eq,
            "range_reads": r_rng,
            "writes": u,
            "alpha": alpha,
            "beta": beta,
        }

    def get_metrics(self, table_name: str, column_name: str) -> dict[str, Any]:
        """Return current sliding-window operation counts and ``(alpha, beta)`` ratios.

        Args:
            table_name: Target table name.
            column_name: Target column name.

        Returns:
            Dictionary with ``n_ops``, ``point_reads``, ``range_reads``, ``writes``,
            ``alpha``, and ``beta``.

        Complexity & Concurrency:
            $O(\\text{window\\_size})$ scan over the column deque under ``self._lock``.
        """
        with self._lock:
            return self._compute_metrics_unlocked(table_name, column_name)

    def recommend(
        self,
        table_name: str,
        column_name: str,
        table_pages: int = 100,
        page_count: int | None = None,
    ) -> Recommendation:
        """Evaluate Rules 1–3 on a column's sliding window and log a recommendation.

        Args:
            table_name: Target table name.
            column_name: Target column name.
            table_pages: Current page count of ``table_name`` (primary parameter).
            page_count: Optional alias for ``table_pages`` when callers use the
                spec-prose keyword name.

        Returns:
            Immutable ``Recommendation`` containing the selected ``IndexStrategy``,
            computed ``alpha``/``beta`` metrics, operation count, page count,
            rationale string, and Unix timestamp.

        Raises:
            ValueError: If both ``table_pages`` (non-default) and ``page_count`` are
                supplied with conflicting values, or if the resolved page count is
                negative.

        Complexity & Concurrency:
            $O(\\text{window\\_size})$ under ``self._lock``.
        """
        if page_count is not None:
            if table_pages != 100 and table_pages != page_count:
                raise ValueError(
                    f"Conflicting page arguments: table_pages={table_pages}, "
                    f"page_count={page_count}"
                )
            pages = page_count
        else:
            pages = table_pages

        if pages < 0:
            raise ValueError(f"page_count must be >= 0, got {pages}")

        with self._lock:
            metrics = self._compute_metrics_unlocked(table_name, column_name)
            n_ops: int = metrics["n_ops"]
            alpha: float = metrics["alpha"]
            beta: float = metrics["beta"]

            if n_ops < self.min_ops or pages < self.min_pages:
                strategy = IndexStrategy.FULL_SCAN
                reasons: list[str] = []
                if n_ops < self.min_ops:
                    reasons.append(f"n_ops={n_ops} < min_ops={self.min_ops}")
                if pages < self.min_pages:
                    reasons.append(f"page_count={pages} < min_pages={self.min_pages}")
                reason = f"Rule 1 (Cold start / small table): {', '.join(reasons)}"
            elif alpha > self.alpha_threshold and beta < self.beta_threshold:
                strategy = IndexStrategy.LSM_TREE
                reason = (
                    f"Rule 2 (Write-heavy point workload): "
                    f"alpha={alpha:.4f} > {self.alpha_threshold} and "
                    f"beta={beta:.4f} < {self.beta_threshold}"
                )
            else:
                strategy = IndexStrategy.BPLUS_TREE
                reason = (
                    f"Rule 3 (Read-heavy or range-scan workload): "
                    f"alpha={alpha:.4f}, beta={beta:.4f}"
                )

            rec = Recommendation(
                strategy=strategy,
                alpha=alpha,
                beta=beta,
                n_ops=n_ops,
                page_count=pages,
                reason=reason,
                timestamp=time.time(),
                table_name=table_name,
                column_name=column_name,
            )
            self.decision_log.append(rec)
            return rec

    def reset(
        self, table_name: str | None = None, column_name: str | None = None
    ) -> None:
        """Clear recorded workload history for a column, a table, or the entire advisor.

        Args:
            table_name: Optional table name. When both ``table_name`` and
                ``column_name`` are ``None``, clears all windows and ``decision_log``.
            column_name: Optional column name (requires ``table_name``).

        Raises:
            ValueError: If ``column_name`` is provided without ``table_name``.
        """
        if table_name is None and column_name is not None:
            raise ValueError("table_name is required when column_name is specified")

        with self._lock:
            if table_name is None:
                self._windows.clear()
                self.decision_log.clear()
            elif column_name is None:
                keys_to_remove = [k for k in self._windows if k[0] == table_name]
                for key in keys_to_remove:
                    del self._windows[key]
            else:
                self._windows.pop((table_name, column_name), None)
