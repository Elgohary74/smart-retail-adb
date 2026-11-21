"""Safe restricted SQL parser for DDL, DML, and equality/range SELECT queries."""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

from .schema import ColumnSpec, DataType

_ID = r"[A-Za-z_][A-Za-z0-9_]*"

_CREATE_TABLE_RE = re.compile(
    rf"^\s*CREATE\s+TABLE\s+({_ID})\s*\((.+?)\)\s*(?:LAYOUT\s+(ROW|COLUMN))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_RE = re.compile(
    rf"^\s*INSERT\s+INTO\s+({_ID})\s+VALUES\s*\((.+)\)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_CREATE_INDEX_RE = re.compile(
    rf"^\s*CREATE\s+INDEX\s+ON\s+({_ID})\s*\(\s*({_ID})\s*\)\s+USING\s+(AUTO|BPLUS|LSM)"
    r"(?:\s+READS\s+(\d+)\s+WRITES\s+(\d+))?\s*;?\s*$",
    re.IGNORECASE,
)
_SELECT_RE = re.compile(
    rf"^\s*SELECT\s+(\*|{_ID}(?:\s*,\s*{_ID})*)\s+FROM\s+({_ID})(?:\s+WHERE\s+(.+?))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_UPDATE_RE = re.compile(
    rf"^\s*UPDATE\s+({_ID})\s+SET\s+(.+?)(?:\s+WHERE\s+(.+?))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_DELETE_RE = re.compile(
    rf"^\s*DELETE\s+FROM\s+({_ID})(?:\s+WHERE\s+(.+?))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)

_BETWEEN_RE = re.compile(rf"^\s*({_ID})\s+BETWEEN\s+(.+?)\s+AND\s+(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_IS_NULL_RE = re.compile(rf"^\s*({_ID})\s+IS\s+(NOT\s+)?NULL\s*$", re.IGNORECASE)
_CMP_RE = re.compile(rf"^\s*({_ID})\s*(<=|>=|=|<|>)\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Predicate:
    field: str
    op: str  # "=", "<", "<=", ">", ">=", "BETWEEN", "IS_NULL", "IS_NOT_NULL"
    value: Any = None
    high: Any = None


@dataclass(frozen=True)
class CreateTableStmt:
    table: str
    columns: list[ColumnSpec]
    layout: str


@dataclass(frozen=True)
class InsertStmt:
    table: str
    values: list[Any]


@dataclass(frozen=True)
class CreateIndexStmt:
    table: str
    field: str
    index_type: str
    reads: int | None = None
    writes: int | None = None


@dataclass(frozen=True)
class SelectStmt:
    table: str
    columns: list[str] | None
    where: Predicate | None = None


@dataclass(frozen=True)
class UpdateStmt:
    table: str
    updates: dict[str, Any]
    where: Predicate | None = None


@dataclass(frozen=True)
class DeleteStmt:
    table: str
    where: Predicate | None = None


SQLStatement = CreateTableStmt | InsertStmt | CreateIndexStmt | SelectStmt | UpdateStmt | DeleteStmt


def _parse_literal(raw: str, allow_vector: bool = False) -> Any:
    text = raw.strip()
    upper = text.upper()
    if upper == "NULL":
        return None
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid literal: {text!r}") from exc
    if isinstance(val, complex) or val is Ellipsis:
        raise ValueError(f"Unsupported literal: {text!r}")
    if isinstance(val, float) and not math.isfinite(val):
        raise ValueError("Literal float must be finite")
    if isinstance(val, (list, tuple)):
        if not allow_vector:
            raise ValueError("Scalar literal expected")
        return [float(x) for x in val]
    if isinstance(val, (dict, set)):
        raise ValueError("Scalar literal expected")
    return val


def _split_top_level(text: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if quote is not None:
            buf.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch in "([":
            depth += 1
            buf.append(ch)
        elif ch in ")]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def parse_where(text: str) -> Predicate:
    m = _BETWEEN_RE.fullmatch(text)
    if m:
        field, raw_low, raw_high = m.groups()
        return Predicate(field=field, op="BETWEEN", value=_parse_literal(raw_low), high=_parse_literal(raw_high))
    m = _IS_NULL_RE.fullmatch(text)
    if m:
        field, is_not = m.groups()
        return Predicate(field=field, op="IS_NOT_NULL" if is_not else "IS_NULL")
    m = _CMP_RE.fullmatch(text)
    if m:
        field, op, raw_val = m.groups()
        return Predicate(field=field, op=op, value=_parse_literal(raw_val))
    raise ValueError(f"Unsupported WHERE clause: {text!r}")


def parse_sql(sql: str) -> SQLStatement:
    m = _SELECT_RE.fullmatch(sql)
    if m:
        cols_text, table, where_text = m.groups()
        cols = None if cols_text.strip() == "*" else [c.strip() for c in cols_text.split(",")]
        where = parse_where(where_text) if where_text is not None else None
        return SelectStmt(table=table, columns=cols, where=where)

    m = _CREATE_TABLE_RE.fullmatch(sql)
    if m:
        table, cols_text, layout = m.groups()
        cols: list[ColumnSpec] = []
        for part in _split_top_level(cols_text):
            tokens = part.split(None, 1)
            if len(tokens) == 1:
                cols.append(ColumnSpec(tokens[0], DataType.ANY))
            else:
                dtype, dim = DataType.parse(tokens[1])
                cols.append(ColumnSpec(tokens[0], dtype, dim=dim))
        return CreateTableStmt(table=table, columns=cols, layout=(layout or "ROW").lower())

    m = _INSERT_RE.fullmatch(sql)
    if m:
        table, vals_text = m.groups()
        values = [_parse_literal(item, allow_vector=True) for item in _split_top_level(vals_text)]
        return InsertStmt(table=table, values=values)

    m = _CREATE_INDEX_RE.fullmatch(sql)
    if m:
        table, field, kind, reads, writes = m.groups()
        return CreateIndexStmt(
            table=table,
            field=field,
            index_type=kind.lower(),
            reads=int(reads) if reads is not None else None,
            writes=int(writes) if writes is not None else None,
        )

    m = _UPDATE_RE.fullmatch(sql)
    if m:
        table, set_text, where_text = m.groups()
        updates: dict[str, Any] = {}
        for assign in _split_top_level(set_text):
            if "=" not in assign:
                raise ValueError(f"Invalid SET assignment: {assign!r}")
            col, raw_val = assign.split("=", 1)
            updates[col.strip()] = _parse_literal(raw_val, allow_vector=True)
        where = parse_where(where_text) if where_text is not None else None
        return UpdateStmt(table=table, updates=updates, where=where)

    m = _DELETE_RE.fullmatch(sql)
    if m:
        table, where_text = m.groups()
        where = parse_where(where_text) if where_text is not None else None
        return DeleteStmt(table=table, where=where)

    raise ValueError("Unsupported or unsafe SQL statement")


def parse_select(sql: str) -> SelectStmt:
    stmt = parse_sql(sql)
    if not isinstance(stmt, SelectStmt):
        raise ValueError("Expected a SELECT statement")
    return stmt
