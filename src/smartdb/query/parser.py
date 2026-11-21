"""Safe restricted SQL parser for DDL, DML, and equality/range SELECT queries."""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

from ..core.schema import ColumnSpec, DataType

_ID = r"[A-Za-z_][A-Za-z0-9_]*"
_RE_SELECT = re.compile(rf"^\s*SELECT\s+(\*|{_ID}(?:\s*,\s*{_ID})*)\s+FROM\s+({_ID})(?:\s+WHERE\s+(.+?))?\s*;?\s*$", re.I | re.S)
_RE_CREATE_TABLE = re.compile(rf"^\s*CREATE\s+TABLE\s+({_ID})\s*\((.+?)\)\s*(?:LAYOUT\s+(ROW|COLUMN))?\s*;?\s*$", re.I | re.S)
_RE_INSERT = re.compile(rf"^\s*INSERT\s+INTO\s+({_ID})\s+VALUES\s*\((.+)\)\s*;?\s*$", re.I | re.S)
_RE_CREATE_INDEX = re.compile(
    rf"^\s*CREATE\s+INDEX\s+ON\s+({_ID})\s*\(\s*({_ID})\s*\)\s+USING\s+(AUTO|BPLUS|LSM)(?:\s+READS\s+(\d+)\s+WRITES\s+(\d+))?\s*;?\s*$",
    re.I,
)
_RE_UPDATE = re.compile(rf"^\s*UPDATE\s+({_ID})\s+SET\s+(.+?)(?:\s+WHERE\s+(.+?))?\s*;?\s*$", re.I | re.S)
_RE_DELETE = re.compile(rf"^\s*DELETE\s+FROM\s+({_ID})(?:\s+WHERE\s+(.+?))?\s*;?\s*$", re.I | re.S)
_RE_BETWEEN = re.compile(rf"^\s*({_ID})\s+BETWEEN\s+(.+?)\s+AND\s+(.+?)\s*$", re.I | re.S)
_RE_IS_NULL = re.compile(rf"^\s*({_ID})\s+IS\s+(NOT\s+)?NULL\s*$", re.I)
_RE_CMP = re.compile(rf"^\s*({_ID})\s*(<=|>=|=|<|>)\s*(.+?)\s*$", re.I | re.S)


@dataclass(frozen=True)
class Predicate:
    """Single-column WHERE filter predicate."""

    field: str
    op: str
    value: Any = None
    high: Any = None


@dataclass(frozen=True)
class CreateTableStmt:
    """Parsed CREATE TABLE statement."""

    table: str
    columns: list[ColumnSpec]
    layout: str


@dataclass(frozen=True)
class InsertStmt:
    """Parsed INSERT INTO ... VALUES statement."""

    table: str
    values: list[Any]


@dataclass(frozen=True)
class CreateIndexStmt:
    """Parsed CREATE INDEX statement."""

    table: str
    field: str
    index_type: str
    reads: int | None = None
    writes: int | None = None


@dataclass(frozen=True)
class SelectStmt:
    """Parsed SELECT statement with optional column projection and WHERE predicate."""

    table: str
    columns: list[str] | None
    where: Predicate | None = None


@dataclass(frozen=True)
class UpdateStmt:
    """Parsed UPDATE ... SET statement."""

    table: str
    updates: dict[str, Any]
    where: Predicate | None = None


@dataclass(frozen=True)
class DeleteStmt:
    """Parsed DELETE FROM statement."""

    table: str
    where: Predicate | None = None


SQLStatement = CreateTableStmt | InsertStmt | CreateIndexStmt | SelectStmt | UpdateStmt | DeleteStmt


def _parse_literal(raw: str, allow_vector: bool = False) -> Any:
    text = raw.strip()
    if text.upper() in {"NULL", "TRUE", "FALSE"}:
        return {"NULL": None, "TRUE": True, "FALSE": False}[text.upper()]
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid literal: {text!r}") from exc
    if isinstance(val, (complex, dict, set)) or val is Ellipsis:
        raise ValueError(f"Unsupported literal: {text!r}")
    if isinstance(val, float) and not math.isfinite(val):
        raise ValueError("Literal float must be finite")
    if isinstance(val, (list, tuple)):
        if not allow_vector:
            raise ValueError("Scalar literal expected")
        return [float(x) for x in val]
    return val


def _split_top_level(text: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth, quote, escaped = 0, None, False
    for ch in text:
        buf.append(ch)
        if escaped:
            escaped = False
        elif quote is not None:
            escaped = (ch == "\\")
            quote = None if (ch == quote and not escaped) else quote
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            buf.pop()
            items.append("".join(buf).strip())
            buf = []
    if tail := "".join(buf).strip():
        items.append(tail)
    return items


def parse_where(text: str) -> Predicate:
    if m := _RE_BETWEEN.fullmatch(text):
        field, raw_low, raw_high = m.groups()
        return Predicate(field, "BETWEEN", _parse_literal(raw_low), _parse_literal(raw_high))
    if m := _RE_IS_NULL.fullmatch(text):
        return Predicate(m.group(1), "IS_NOT_NULL" if m.group(2) else "IS_NULL")
    if m := _RE_CMP.fullmatch(text):
        return Predicate(m.group(1), m.group(2), _parse_literal(m.group(3)))
    raise ValueError(f"Unsupported WHERE clause: {text!r}")


def _build_create_table(m: re.Match[str]) -> CreateTableStmt:
    cols = []
    for part in _split_top_level(m.group(2)):
        tokens = part.split(None, 1)
        if len(tokens) == 1:
            cols.append(ColumnSpec(tokens[0], DataType.ANY))
        else:
            dtype, dim = DataType.parse(tokens[1])
            cols.append(ColumnSpec(tokens[0], dtype, dim=dim))
    return CreateTableStmt(m.group(1), cols, (m.group(3) or "ROW").lower())


def _build_update(m: re.Match[str]) -> UpdateStmt:
    updates = {}
    for assign in _split_top_level(m.group(2)):
        if "=" not in assign:
            raise ValueError(f"Invalid SET assignment: {assign!r}")
        col, raw_val = assign.split("=", 1)
        updates[col.strip()] = _parse_literal(raw_val, allow_vector=True)
    return UpdateStmt(m.group(1), updates, parse_where(m.group(3)) if m.group(3) else None)


_DISPATCH = [
    (_RE_SELECT, lambda m: SelectStmt(m.group(2), None if m.group(1).strip() == "*" else [c.strip() for c in m.group(1).split(",")], parse_where(m.group(3)) if m.group(3) else None)),
    (_RE_CREATE_TABLE, _build_create_table),
    (_RE_INSERT, lambda m: InsertStmt(m.group(1), [_parse_literal(x, allow_vector=True) for x in _split_top_level(m.group(2))])),
    (_RE_CREATE_INDEX, lambda m: CreateIndexStmt(m.group(1), m.group(2), m.group(3).lower(), int(m.group(4)) if m.group(4) else None, int(m.group(5)) if m.group(5) else None)),
    (_RE_UPDATE, _build_update),
    (_RE_DELETE, lambda m: DeleteStmt(m.group(1), parse_where(m.group(2)) if m.group(2) else None)),
]


def parse_sql(sql: str) -> SQLStatement:
    for pattern, builder in _DISPATCH:
        if m := pattern.fullmatch(sql):
            return builder(m)
    raise ValueError("Unsupported or unsafe SQL statement")


def parse_select(sql: str) -> SelectStmt:
    stmt = parse_sql(sql)
    if not isinstance(stmt, SelectStmt):
        raise ValueError("Expected a SELECT statement")
    return stmt
