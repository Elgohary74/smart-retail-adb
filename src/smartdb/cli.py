"""Interactive shell for the supported subset of the Smart Retail DB."""
from __future__ import annotations
import argparse
import ast
import re
from .engine import SmartRetailDB

_CREATE_TABLE = re.compile(
    r"^CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)\s*(?:LAYOUT\s+(ROW|COLUMN))?\s*;?$",
    re.IGNORECASE | re.DOTALL,
)
_INSERT = re.compile(r"^INSERT\s+INTO\s+(\w+)\s+VALUES\s*(\(.*\))\s*;?$", re.IGNORECASE | re.DOTALL)
_CREATE_INDEX = re.compile(
    r"^CREATE\s+INDEX\s+ON\s+(\w+)\s*\(\s*(\w+)\s*\)\s+USING\s+(AUTO|BPLUS|LSM)"
    r"(?:\s+READS\s+(\d+)\s+WRITES\s+(\d+))?\s*;?$", re.IGNORECASE,
)


def _help() -> str:
    return """Commands:
  CREATE TABLE Products (product_id, name, category, price, stock) LAYOUT ROW;
  INSERT INTO Products VALUES (1, 'Keyboard', 'Accessories', 29.99, 20);
  CREATE INDEX ON Products (product_id) USING AUTO READS 100 WRITES 10;
  SELECT * FROM Products;
  SELECT * FROM Products WHERE product_id = 1;
  :explain SELECT * FROM Products WHERE product_id = 1
  :stats
  :help
  :quit
Supported SQL is deliberately limited to table creation, row insertion,
SELECT * with an optional equality predicate, and index creation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Retail DB CLI")
    parser.add_argument("--db", default="./retail_data", help="Database directory")
    parser.add_argument("--buffer-frames", type=int, default=16)
    parser.add_argument("--policy", choices=["FIFO", "LRU"], default="LRU")
    parser.add_argument("--backend", choices=["page", "objects"], default="page")
    args = parser.parse_args()
    db = SmartRetailDB(args.db, buffer_frames=args.buffer_frames,
                       replacement_policy=args.policy, backend_kind=args.backend)
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
            if line.lower() in {":quit", "quit", "exit", "\\q"}:
                break
            if line.lower() == ":help":
                print(_help())
                continue
            if line.lower() == ":stats":
                print(db.buffer_stats())
                continue
            if line.lower().startswith(":explain "):
                try:
                    print(db.explain(line[len(":explain "):]))
                except Exception as exc:
                    print(f"ERROR: {exc}")
                continue
            try:
                match = _CREATE_TABLE.fullmatch(line)
                if match:
                    name, fields_text, layout = match.groups()
                    fields = [item.strip() for item in fields_text.split(",") if item.strip()]
                    db.create_table(name, fields, (layout or "ROW").lower())
                    print(f"Created table {name}")
                    continue
                match = _INSERT.fullmatch(line)
                if match:
                    table, values_text = match.groups()
                    values = ast.literal_eval(values_text)
                    if not isinstance(values, tuple):
                        raise ValueError("VALUES must be a tuple, e.g. (1, 'Book', 9.9)")
                    fields = db.tables[table].fields
                    if len(values) != len(fields):
                        raise ValueError(f"Expected {len(fields)} values, got {len(values)}")
                    locator = db.insert(table, dict(zip(fields, values)))
                    print(f"Inserted record at page/row {locator}")
                    continue
                match = _CREATE_INDEX.fullmatch(line)
                if match:
                    table, field, kind, reads, writes = match.groups()
                    selected = db.create_index(table, field, kind.lower(),
                                               int(reads or 100), int(writes or 10))
                    print(f"Created {selected} index on {table}.{field}")
                    continue
                if line.upper().startswith("SELECT "):
                    rows = db.execute(line)
                    for row in rows:
                        print(row)
                    print(f"({len(rows)} row(s))")
                    continue
                print("Unrecognized command. Type :help")
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
