from __future__ import annotations
import ast
import re
from dataclasses import dataclass
from typing import Any

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_SELECT_RE = re.compile(
    rf"^\s*SELECT\s+\*\s+FROM\s+({_IDENTIFIER})(?:\s+WHERE\s+({_IDENTIFIER})\s*=\s*(.+?))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SelectQuery:
    table: str
    field: str | None = None
    value: Any = None


def parse_select(sql: str) -> SelectQuery:
    match = _SELECT_RE.fullmatch(sql)
    if not match:
        raise ValueError("Supported query: SELECT * FROM table [WHERE field = literal];")
    table, field, raw_value = match.groups()
    if field is None:
        return SelectQuery(table=table)
    try:
        value = ast.literal_eval(raw_value.strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError("WHERE value must be a literal such as 123, 9.5, True, None, or 'text'") from exc
    if isinstance(value, (list, dict, tuple, set)):
        raise ValueError("WHERE value must be a scalar literal")
    return SelectQuery(table=table, field=field, value=value)
