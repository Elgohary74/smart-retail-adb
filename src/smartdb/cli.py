"""Interactive shell for Smart Retail DB."""
from __future__ import annotations

import argparse
from .engine import SmartRetailDB


def _help() -> str:
    return """Commands:
  CREATE TABLE Products (product_id INT, name STR, category STR, price FLOAT, stock INT) LAYOUT ROW;
  INSERT INTO Products VALUES (1, 'Keyboard', 'Accessories', 29.99, 20);
  CREATE INDEX ON Products (product_id) USING AUTO;
  SELECT product_id, name FROM Products WHERE price BETWEEN 10.0 AND 50.0;
  UPDATE Products SET stock = 19 WHERE product_id = 1;
  DELETE FROM Products WHERE product_id = 1;
  :explain SELECT * FROM Products WHERE product_id = 1
  :stats
  :tables
  :help
  :quit
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Retail DB CLI")
    parser.add_argument("--db", default="./retail_data", help="Database directory")
    parser.add_argument("--buffer-frames", type=int, default=64)
    parser.add_argument("--policy", choices=["FIFO", "LRU"], default="LRU")
    parser.add_argument("--backend", choices=["page", "objects"], default="page")
    args = parser.parse_args()
    db = SmartRetailDB(
        args.db,
        buffer_frames=args.buffer_frames,
        replacement_policy=args.policy,
        backend_kind=args.backend,
    )
    print("Smart Retail DB | :help for commands | :quit to exit")
    try:
        while True:
            try:
                line = input("smartdb> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            low = line.lower()
            if low in {":quit", "quit", "exit", "\\q"}:
                break
            if low == ":help":
                print(_help())
                continue
            if low == ":stats":
                print(db.buffer_stats())
                continue
            if low == ":tables":
                print(list(db.tables.keys()))
                continue
            if low.startswith(":explain "):
                try:
                    print(db.explain(line[len(":explain "):]))
                except Exception as exc:
                    print(f"ERROR: {type(exc).__name__}: {exc}")
                continue
            try:
                result = db.execute(line)
                if isinstance(result, list):
                    for row in result:
                        print(row)
                    print(f"({len(result)} row(s))")
                elif isinstance(result, int):
                    print(f"({result} row(s) affected)")
                else:
                    print(result)
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
