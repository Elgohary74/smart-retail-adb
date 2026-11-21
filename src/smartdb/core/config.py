"""Central engine constants and pure-stdlib .env / environment settings."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- Engine constants ---
PAGE_SIZE = 4096
DEFAULT_DB_DIR = "./retail_data"
DEFAULT_BUFFER_FRAMES = 64
DEFAULT_REPLACEMENT_POLICY = "LRU"
DEFAULT_BACKEND_KIND = "page"
DEFAULT_LAYOUT = "row"

BPLUS_INLINE_RID_CAP = 8
LSM_FLUSH_THRESHOLD = 128
LSM_COMPACTION_THRESHOLD = 4
BLOOM_DEFAULT_BITS = 1024
BLOOM_BITS_PER_KEY = 10
BLOOM_NUM_HASHES = 7

AUTO_DEFAULT_READS = 100
AUTO_DEFAULT_WRITES = 10
AUTO_WRITE_RATIO = 2
AES_GCM_OVERHEAD_BYTES = 28

DEFAULT_SEED = 42
DEFAULT_BENCH_REPEATS = 5
DEFAULT_BENCH_WARMUP = 1
DEFAULT_BENCH_OUT_DIR = "benchmark_results"


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file without external dependencies."""
    env_path = Path(path)
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        values[key] = val
    return values


@dataclass(frozen=True)
class Settings:
    """Editable runtime configuration resolved from environment variables and .env."""

    db_dir: str = DEFAULT_DB_DIR
    page_size: int = PAGE_SIZE
    buffer_frames: int = DEFAULT_BUFFER_FRAMES
    replacement_policy: str = DEFAULT_REPLACEMENT_POLICY
    backend_kind: str = DEFAULT_BACKEND_KIND
    lsm_flush_threshold: int = LSM_FLUSH_THRESHOLD
    lsm_compaction_threshold: int = LSM_COMPACTION_THRESHOLD
    seed: int = DEFAULT_SEED
    bench_repeats: int = DEFAULT_BENCH_REPEATS


def _validate_settings(settings: Settings) -> Settings:
    if settings.page_size < 128 or settings.buffer_frames < 1:
        raise ValueError("SMARTDB_PAGE_SIZE must be >= 128 and SMARTDB_BUFFER_FRAMES >= 1")
    if settings.replacement_policy not in {"FIFO", "LRU"}:
        raise ValueError("SMARTDB_POLICY must be 'FIFO' or 'LRU'")
    if settings.backend_kind not in {"page", "objects"}:
        raise ValueError("SMARTDB_BACKEND must be 'page' or 'objects'")
    if settings.lsm_flush_threshold < 1 or settings.lsm_compaction_threshold < 2:
        raise ValueError("SMARTDB_LSM_FLUSH_THRESHOLD >= 1 and SMARTDB_LSM_COMPACTION_THRESHOLD >= 2")
    return settings


def load_settings(env_path: str | Path = ".env") -> Settings:
    """Load editable settings (precedence: os.environ > .env file > constants)."""
    file_env = load_env(env_path)
    get = lambda k, d: os.environ.get(k, file_env.get(k, d))
    return _validate_settings(
        Settings(
            db_dir=get("SMARTDB_DIR", DEFAULT_DB_DIR),
            page_size=int(get("SMARTDB_PAGE_SIZE", str(PAGE_SIZE))),
            buffer_frames=int(get("SMARTDB_BUFFER_FRAMES", str(DEFAULT_BUFFER_FRAMES))),
            replacement_policy=get("SMARTDB_POLICY", DEFAULT_REPLACEMENT_POLICY).upper(),
            backend_kind=get("SMARTDB_BACKEND", DEFAULT_BACKEND_KIND).lower(),
            lsm_flush_threshold=int(get("SMARTDB_LSM_FLUSH_THRESHOLD", str(LSM_FLUSH_THRESHOLD))),
            lsm_compaction_threshold=int(get("SMARTDB_LSM_COMPACTION_THRESHOLD", str(LSM_COMPACTION_THRESHOLD))),
            seed=int(get("SMARTDB_SEED", str(DEFAULT_SEED))),
            bench_repeats=int(get("SMARTDB_BENCH_REPEATS", str(DEFAULT_BENCH_REPEATS))),
        )
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist JSON atomically via a temporary file and os.replace."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
