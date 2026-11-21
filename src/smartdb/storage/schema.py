"""Lightweight typed column definitions and record validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

VALID_DTYPES = frozenset({"ANY", "INT", "FLOAT", "STR", "BOOL"})


@dataclass(frozen=True)
class Column:
    """Column specification with optional type and nullability constraints."""

    name: str
    dtype: str = "ANY"
    nullable: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Invalid column name: {self.name!r}")
        normalized = self.dtype.upper()
        if normalized not in VALID_DTYPES:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; expected one of {sorted(VALID_DTYPES)}")
        object.__setattr__(self, "dtype", normalized)

    def validate_value(self, value: Any) -> Any:
        if value is None:
            if not self.nullable:
                raise ValueError(f"Column {self.name!r} cannot be None")
            return None
        if self.dtype == "ANY":
            return value
        if self.dtype == "INT":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Column {self.name!r} expects INT, got {type(value).__name__}")
            return value
        if self.dtype == "FLOAT":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Column {self.name!r} expects FLOAT, got {type(value).__name__}")
            return float(value)
        if self.dtype == "STR":
            if not isinstance(value, str):
                raise TypeError(f"Column {self.name!r} expects STR, got {type(value).__name__}")
            return value
        if self.dtype == "BOOL":
            if not isinstance(value, bool):
                raise TypeError(f"Column {self.name!r} expects BOOL, got {type(value).__name__}")
            return value
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "nullable": self.nullable}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Column":
        return cls(
            name=data["name"],
            dtype=data.get("dtype", "ANY"),
            nullable=bool(data.get("nullable", True)),
        )


ColumnSpec = str | Column | tuple[str, str] | tuple[str, str, bool]


class TableSchema:
    """Ordered, uniquely named column schema for a table."""

    def __init__(self, columns: Sequence[Column]):
        if not columns:
            raise ValueError("columns must be non-empty")
        names = [col.name for col in columns]
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        self.columns: tuple[Column, ...] = tuple(columns)
        self._by_name: dict[str, Column] = {col.name: col for col in self.columns}

    @property
    def field_names(self) -> list[str]:
        return [col.name for col in self.columns]

    def validate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        expected = set(self._by_name)
        provided = set(record)
        missing = expected - provided
        unknown = provided - expected
        if missing or unknown:
            raise ValueError(
                f"Record fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return {col.name: col.validate_value(record[col.name]) for col in self.columns}

    def to_dict(self) -> list[dict[str, Any]]:
        return [col.to_dict() for col in self.columns]

    @classmethod
    def from_spec(cls, spec: Sequence[ColumnSpec]) -> "TableSchema":
        columns: list[Column] = []
        for item in spec:
            if isinstance(item, Column):
                columns.append(item)
            elif isinstance(item, str):
                columns.append(Column(name=item))
            elif isinstance(item, tuple) and len(item) == 2:
                columns.append(Column(name=item[0], dtype=item[1]))
            elif isinstance(item, tuple) and len(item) == 3:
                columns.append(Column(name=item[0], dtype=item[1], nullable=item[2]))
            else:
                raise ValueError(f"Invalid column specification: {item!r}")
        return cls(columns)

    @classmethod
    def from_dict(cls, data: Sequence[dict[str, Any]] | Sequence[str]) -> "TableSchema":
        if not data:
            raise ValueError("Schema definition cannot be empty")
        if isinstance(data[0], str):
            return cls.from_spec(data)  # type: ignore[arg-type]
        return cls([Column.from_dict(item) for item in data])  # type: ignore[arg-type]
