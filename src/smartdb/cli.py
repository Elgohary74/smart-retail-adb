"""Interactive shell for Smart Retail DB."""
from __future__ import annotations

import argparse
from typing import Any

from .core import load_settings
from .engine import SmartRetailDB


def _help() -> str:
    return (
        "Commands:\n"
        "  CREATE TABLE Products (product_id INT, name STR, category STR, price FLOAT, stock INT) LAYOUT ROW;\n"
        "  INSERT INTO Products VALUES (1, 'Keyboard', 'Accessories', 29.99, 20);\n"
        "  CREATE INDEX ON Products (price) USING AUTO;\n"
        "  SELECT product_id, name FROM Products WHERE price BETWEEN 10.0 AND 50.0;\n"
        "  UPDATE Products SET stock = 19 WHERE product_id = 1;\n"
        "  DELETE FROM Products WHERE product_id = 1;\n"
        "  :explain <SELECT ...> | :stats | :tables | :help | :quit\n"
    )


def _print_result(res: Any) -> None:
    if isinstance(res, list):
        for row in res:
            print(row)
        print(f"({len(res)} row(s))")
    elif isinstance(res, int):
        print(f"({res} row(s) affected)")
    else:
        print(res)


def _run_line(db: SmartRetailDB, line: str) -> bool:
    low = line.lower()
    if low in {":quit", "quit", "exit", "\\q"}:
        return False
    if low == ":help":
        print(_help())
    elif low == ":stats":
        print(db.buffer_stats())
    elif low == ":tables":
        print(list(db.tables.keys()))
    elif low.startswith(":explain "):
        print(db.explain(line[len(":explain "):]))
    else:
        _print_result(db.execute(line))
    return True


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Smart Retail DB CLI")
    parser.add_argument("--db", default=settings.db_dir, help="Database directory")
    parser.add_argument("--buffer-frames", type=int, default=settings.buffer_frames)
    parser.add_argument("--policy", choices=["FIFO", "LRU"], default=settings.replacement_policy)
    parser.add_argument("--backend", choices=["page", "objects"], default=settings.backend_kind)
    args = parser.parse_args()
    with SmartRetailDB(
        args.db, buffer_frames=args.buffer_frames, replacement_policy=args.policy, backend_kind=args.backend
    ) as db:
        print("Smart Retail DB | :help for commands | :quit to exit")
        while True:
            try:
                line = input("smartdb> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                if not _run_line(db, line):
                    break
            except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
