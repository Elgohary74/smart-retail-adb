"""SQL statement and predicate AST dataclasses shared by parser, predicates, and engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.schema import ColumnSpec


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
