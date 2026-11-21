from __future__ import annotations
import json
from bisect import bisect_left
from pathlib import Path
from typing import Any


def _encode_key(key: Any) -> dict[str, Any]:
    if key is None or isinstance(key, (bool, int, float, str)):
        return {"type": type(key).__name__, "value": key}
    raise TypeError(f"LSM index keys must be JSON scalar values; got {type(key).__name__}")


def _decode_key(obj: dict[str, Any]) -> Any:
    return obj["value"]


class LSMTree:
    """Runs are sorted JSONL files; recent writes live in a memtable.

    This teaching implementation supports point and range queries. Compaction
    rewrites the current sorted runs into one run to reduce read amplification.
    """
    def __init__(self, directory: str | Path, flush_threshold: int = 128):
        if flush_threshold < 1:
            raise ValueError("flush_threshold must be >= 1")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_threshold = flush_threshold
        self.memtable: dict[Any, list[Any]] = {}
        self.runs: list[tuple[list[Any], list[list[Any]], Path]] = []
        self._load_runs()
        self.flushes = 0
        self.compactions = 0

    def _load_runs(self) -> None:
        for path in sorted(self.directory.glob("run_*.jsonl")):
            keys, values = [], []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    keys.append(_decode_key(item["key"]))
                    values.append([tuple(v) if isinstance(v, list) else v for v in item["values"]])
            self.runs.append((keys, values, path))

    def insert(self, key: Any, value: Any) -> None:
        # Locators are tuples; JSON turns them into lists, restored on load.
        self.memtable.setdefault(key, []).append(value)
        if len(self.memtable) >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self.memtable:
            return
        combined = sorted(self.memtable.items(), key=lambda item: item[0])
        existing_ids = [int(p.stem.split("_")[1]) for p in self.directory.glob("run_*.jsonl")]
        run_id = max(existing_ids, default=-1) + 1
        path = self.directory / f"run_{run_id:08d}.jsonl"
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as f:
            for key, values in combined:
                f.write(json.dumps({"key": _encode_key(key), "values": values}, separators=(",", ":")) + "\n")
        temp.replace(path)
        self.runs.append(([k for k, _ in combined], [v for _, v in combined], path))
        self.memtable.clear()
        self.flushes += 1
        if len(self.runs) >= 8:
            self.compact()

    def search(self, key: Any) -> list[Any]:
        found = list(self.memtable.get(key, []))
        for keys, values, _ in self.runs:
            pos = bisect_left(keys, key)
            if pos < len(keys) and keys[pos] == key:
                found.extend(values[pos])
        return found

    def range_search(self, low: Any | None = None, high: Any | None = None) -> list[tuple[Any, Any]]:
        items: list[tuple[Any, Any]] = []
        for key, values in self.memtable.items():
            if (low is None or key >= low) and (high is None or key <= high):
                items.extend((key, value) for value in values)
        for keys, values, _ in self.runs:
            for key, bucket in zip(keys, values):
                if (low is None or key >= low) and (high is None or key <= high):
                    items.extend((key, value) for value in bucket)
        items.sort(key=lambda item: item[0])
        return items

    def compact(self) -> None:
        if len(self.runs) < 2:
            return
        merged: dict[Any, list[Any]] = {}
        for keys, values, _ in self.runs:
            for key, bucket in zip(keys, values):
                merged.setdefault(key, []).extend(bucket)
        for key, bucket in self.memtable.items():
            merged.setdefault(key, []).extend(bucket)
        for _, _, path in self.runs:
            path.unlink(missing_ok=True)
        self.runs = []
        self.memtable.clear()
        self.flushes -= 0
        # Write one new sorted run without recursively triggering compaction.
        if merged:
            path = self.directory / "run_00000000.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for key, values in sorted(merged.items(), key=lambda item: item[0]):
                    f.write(json.dumps({"key": _encode_key(key), "values": values}, separators=(",", ":")) + "\n")
            ordered = sorted(merged.items(), key=lambda item: item[0])
            self.runs = [([k for k, _ in ordered], [v for _, v in ordered], path)]
        self.compactions += 1

    def close(self) -> None:
        self.flush()
