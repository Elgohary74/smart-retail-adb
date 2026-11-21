"""Typed table schemas and record validation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .key_codec import DataType, decode_key, encode_key
from .row_codec import deserialize_row, deserialize_value, serialize_row, serialize_value

RID = tuple[int, int]
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

__all__ = [
    "RID",
    "DataType",
    "ColumnSpec",
    "TableSchema",
    "encode_key",
    "decode_key",
    "serialize_value",
    "deserialize_value",
    "serialize_row",
    "deserialize_row",
]


@dataclass(frozen=True)
class ColumnSpec:
    """Column metadata: name, data type, nullability, and vector dimension."""

    name: str
    dtype: DataType = DataType.ANY
    nullable: bool = True
    dim: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Invalid column name: {self.name!r}")
        if self.dtype == DataType.VECTOR and (self.dim is None or self.dim < 1):
            raise ValueError(f"VECTOR column {self.name!r} requires positive dim")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "dtype": self.dtype.value, "nullable": self.nullable}
        if self.dim is not None:
            out["dim"] = self.dim
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnSpec":
        return cls(
            name=data["name"],
            dtype=DataType(data["dtype"]),
            nullable=bool(data.get("nullable", True)),
            dim=data.get("dim"),
        )


def _check_int(col: ColumnSpec, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Column {col.name!r} expects INT, got {type(value).__name__}")
    if value < _INT64_MIN or value > _INT64_MAX:
        raise OverflowError(f"INT value out of 64-bit range: {value}")
    return value


def _check_float(col: ColumnSpec, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Column {col.name!r} expects FLOAT, got {type(value).__name__}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Column {col.name!r} rejects non-finite float: {value}")
    return fval


def _check_vector(col: ColumnSpec, value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != col.dim:
        raise ValueError(f"Column {col.name!r} expects VECTOR({col.dim})")
    return [_check_float(col, item) for item in value]


def _check_any(col: ColumnSpec, value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _check_int(col, value)
    if isinstance(value, float):
        return _check_float(col, value)
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"Unsupported value type {type(value).__name__} for {col.name!r}")


_VALIDATORS = {
    DataType.BOOL: lambda col, v: v if isinstance(v, bool) else _raise_type(col, "BOOL", v),
    DataType.INT: _check_int,
    DataType.FLOAT: _check_float,
    DataType.STR: lambda col, v: v if isinstance(v, str) else _raise_type(col, "STR", v),
    DataType.BYTES: lambda col, v: bytes(v) if isinstance(v, (bytes, bytearray)) else _raise_type(col, "BYTES", v),
    DataType.VECTOR: _check_vector,
    DataType.ANY: _check_any,
}


def _raise_type(col: ColumnSpec, expected: str, value: Any) -> Any:
    raise TypeError(f"Column {col.name!r} expects {expected}, got {type(value).__name__}")


class TableSchema:
    """Ordered list of ColumnSpecs with record validation and literal coercion."""

    def __init__(self, columns: list[ColumnSpec]):
        if not columns:
            raise ValueError("TableSchema requires at least one column")
        names = [c.name for c in columns]
        if len(set(names)) != len(names):
            raise ValueError("Column names must be unique")
        self.columns = list(columns)
        self._by_name = {c.name: c for c in self.columns}
        self.field_names = names

    @classmethod
    def from_spec(cls, spec: list[str] | list[ColumnSpec] | dict[str, Any]) -> "TableSchema":
        if isinstance(spec, dict):
            cols = []
            for name, raw in spec.items():
                if isinstance(raw, ColumnSpec):
                    cols.append(raw)
                else:
                    dt, dim = DataType.parse(str(raw))
                    cols.append(ColumnSpec(name, dtype=dt, dim=dim))
            return cls(cols)
        if spec and isinstance(spec[0], ColumnSpec):
            return cls(list(spec))  # type: ignore[arg-type]
        return cls([ColumnSpec(name=str(name), dtype=DataType.ANY) for name in spec])

    def to_dict(self) -> dict[str, Any]:
        return {"columns": [c.to_dict() for c in self.columns]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableSchema":
        if "columns" in data:
            return cls([ColumnSpec.from_dict(c) for c in data["columns"]])
        return cls.from_spec(data.get("fields", []))

    def column(self, name: str) -> ColumnSpec:
        if name not in self._by_name:
            raise KeyError(f"Unknown field: {name!r}")
        return self._by_name[name]

    def validate_value(self, col: ColumnSpec, value: Any) -> Any:
        if value is None:
            if not col.nullable:
                raise ValueError(f"Column {col.name!r} is NOT NULL")
            return None
        return _VALIDATORS[col.dtype](col, value)

    def validate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        unknown = set(record) - set(self._by_name)
        missing = set(self._by_name) - set(record)
        if unknown or missing:
            raise ValueError(f"Record fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        return {c.name: self.validate_value(c, record[c.name]) for c in self.columns}

    def coerce_literal(self, field: str, value: Any) -> Any:
        col = self.column(field)
        if value is not None and col.dtype == DataType.FLOAT and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value
