"""Typed table schemas, order-preserving key encoding, and binary row codecs."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any

RID = tuple[int, int]

_INT64_BIAS = 1 << 63
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


class DataType(str, Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    STR = "STR"
    BYTES = "BYTES"
    VECTOR = "VECTOR"
    ANY = "ANY"

    @classmethod
    def parse(cls, spec: str) -> tuple["DataType", int | None]:
        text = spec.strip().upper()
        if text.startswith("VECTOR(") and text.endswith(")"):
            dim = int(text[7:-1].strip())
            if dim < 1:
                raise ValueError("VECTOR dimension must be >= 1")
            return cls.VECTOR, dim
        return cls(text), None


@dataclass(frozen=True)
class ColumnSpec:
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
            for name, raw_type in spec.items():
                if isinstance(raw_type, ColumnSpec):
                    cols.append(raw_type)
                else:
                    dtype, dim = DataType.parse(str(raw_type))
                    cols.append(ColumnSpec(name=name, dtype=dtype, dim=dim))
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
        dt = col.dtype
        if dt == DataType.BOOL:
            if not isinstance(value, bool):
                raise TypeError(f"Column {col.name!r} expects BOOL, got {type(value).__name__}")
            return value
        if dt == DataType.INT:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Column {col.name!r} expects INT, got {type(value).__name__}")
            if value < _INT64_MIN or value > _INT64_MAX:
                raise OverflowError(f"INT value out of 64-bit range: {value}")
            return value
        if dt == DataType.FLOAT:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Column {col.name!r} expects FLOAT, got {type(value).__name__}")
            fval = float(value)
            if not math.isfinite(fval):
                raise ValueError(f"Column {col.name!r} rejects non-finite float: {value}")
            return fval
        if dt == DataType.STR:
            if not isinstance(value, str):
                raise TypeError(f"Column {col.name!r} expects STR, got {type(value).__name__}")
            return value
        if dt == DataType.BYTES:
            if not isinstance(value, (bytes, bytearray)):
                raise TypeError(f"Column {col.name!r} expects BYTES, got {type(value).__name__}")
            return bytes(value)
        if dt == DataType.VECTOR:
            if not isinstance(value, (list, tuple)) or len(value) != col.dim:
                raise ValueError(f"Column {col.name!r} expects VECTOR({col.dim})")
            out: list[float] = []
            for item in value:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise TypeError(f"VECTOR elements must be numeric, got {type(item).__name__}")
                fval = float(item)
                if not math.isfinite(fval):
                    raise ValueError(f"VECTOR rejects non-finite float: {item}")
                out.append(fval)
            return out
        # DataType.ANY
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Column {col.name!r} rejects non-finite float: {value}")
        if isinstance(value, int) and not isinstance(value, bool) and (value < _INT64_MIN or value > _INT64_MAX):
            raise OverflowError(f"INT value out of 64-bit range: {value}")
        if not isinstance(value, (bool, int, float, str, bytes)):
            raise TypeError(f"Unsupported value type {type(value).__name__} for {col.name!r}")
        return value

    def validate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        unknown = set(record) - set(self._by_name)
        missing = set(self._by_name) - set(record)
        if unknown or missing:
            raise ValueError(f"Record fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        return {c.name: self.validate_value(c, record[c.name]) for c in self.columns}

    def coerce_literal(self, field: str, value: Any) -> Any:
        col = self.column(field)
        if value is None:
            return None
        if col.dtype == DataType.FLOAT and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value


def _enc_int64(val: int) -> bytes:
    return struct.pack(">Q", val + _INT64_BIAS)


def _dec_int64(raw: bytes | memoryview) -> int:
    return struct.unpack(">Q", raw)[0] - _INT64_BIAS


def _enc_float64(val: float) -> bytes:
    bits = struct.unpack(">Q", struct.pack(">d", 0.0 if val == 0.0 else val))[0]
    if bits & (1 << 63):
        bits ^= 0xFFFFFFFFFFFFFFFF
    else:
        bits ^= 1 << 63
    return struct.pack(">Q", bits)


def _dec_float64(raw: bytes | memoryview) -> float:
    bits = struct.unpack(">Q", raw)[0]
    if bits & (1 << 63):
        bits ^= 1 << 63
    else:
        bits ^= 0xFFFFFFFFFFFFFFFF
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _enc_any_num(val: int | float) -> bytes:
    if isinstance(val, int) or (-9223372036854775808.0 <= val < 9223372036854775808.0):
        floor_i = int(val) if isinstance(val, int) else math.floor(val)
        frac = 0.0 if isinstance(val, int) else (val - floor_i)
        return b"\x01" + _enc_int64(floor_i) + _enc_float64(frac)
    if val < 0:
        return b"\x00" + _enc_float64(val)
    return b"\x02" + _enc_float64(val)


def _dec_any_num(raw: bytes | memoryview) -> int | float:
    bucket = raw[0]
    if bucket == 1:
        floor_i = _dec_int64(raw[1:9])
        frac = _dec_float64(raw[9:17])
        return floor_i if frac == 0.0 else floor_i + frac
    return _dec_float64(raw[1:9])


def encode_key(dtype: DataType, value: Any) -> bytes:
    """Encode a scalar key into order-preserving bytes (NULL sorts first)."""
    if value is None:
        return b"\x00"
    if dtype == DataType.BOOL:
        if not isinstance(value, bool):
            raise TypeError(f"Expected BOOL key, got {type(value).__name__}")
        return b"\x01\x01" if value else b"\x01\x00"
    if dtype == DataType.INT:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Expected INT key, got {type(value).__name__}")
        return b"\x01" + _enc_int64(value)
    if dtype == DataType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Expected FLOAT key, got {type(value).__name__}")
        fval = float(value)
        if not math.isfinite(fval):
            raise ValueError("Key cannot be NaN or Inf")
        return b"\x01" + _enc_float64(fval)
    if dtype == DataType.STR:
        if not isinstance(value, str):
            raise TypeError(f"Expected STR key, got {type(value).__name__}")
        return b"\x01" + value.encode("utf-8")
    if dtype == DataType.BYTES:
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError(f"Expected BYTES key, got {type(value).__name__}")
        return b"\x01" + bytes(value)
    if dtype == DataType.ANY:
        if isinstance(value, bool):
            return b"\x01\x01\x01" if value else b"\x01\x01\x00"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Key cannot be NaN or Inf")
            return b"\x01\x02" + _enc_any_num(value)
        if isinstance(value, str):
            return b"\x01\x03" + value.encode("utf-8")
        if isinstance(value, (bytes, bytearray)):
            return b"\x01\x04" + bytes(value)
    raise TypeError(f"Unsupported index key type {dtype.value} for {type(value).__name__}")


def decode_key(dtype: DataType, raw: bytes | memoryview) -> Any:
    """Decode bytes produced by encode_key back to a Python scalar."""
    if raw[0] == 0:
        return None
    body = raw[1:]
    if dtype == DataType.BOOL:
        return body[0] == 1
    if dtype == DataType.INT:
        return _dec_int64(body)
    if dtype == DataType.FLOAT:
        return _dec_float64(body)
    if dtype == DataType.STR:
        return bytes(body).decode("utf-8")
    if dtype == DataType.BYTES:
        return bytes(body)
    if dtype == DataType.ANY:
        rank = body[0]
        payload = body[1:]
        if rank == 1:
            return payload[0] == 1
        if rank == 2:
            return _dec_any_num(payload)
        if rank == 3:
            return bytes(payload).decode("utf-8")
        if rank == 4:
            return bytes(payload)
    raise ValueError(f"Corrupt encoded key for {dtype.value}")


def serialize_value(dtype: DataType, value: Any, dim: int | None = None) -> bytes:
    """Serialize a non-null field value into compact binary bytes."""
    if dtype == DataType.INT:
        return struct.pack("<q", value)
    if dtype == DataType.FLOAT:
        return struct.pack("<d", float(value))
    if dtype == DataType.BOOL:
        return b"\x01" if value else b"\x00"
    if dtype == DataType.STR:
        encoded = value.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded
    if dtype == DataType.BYTES:
        return struct.pack("<I", len(value)) + value
    if dtype == DataType.VECTOR:
        assert dim is not None
        return struct.pack(f"<{dim}f", *value)
    # DataType.ANY
    if isinstance(value, bool):
        return b"\x01" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return b"\x02" + struct.pack("<q", value)
    if isinstance(value, float):
        return b"\x03" + struct.pack("<d", value)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"\x04" + struct.pack("<I", len(encoded)) + encoded
    if isinstance(value, (bytes, bytearray)):
        return b"\x05" + struct.pack("<I", len(value)) + bytes(value)
    raise TypeError(f"Unsupported value type for ANY: {type(value).__name__}")


def deserialize_value(dtype: DataType, buf: bytes | memoryview, offset: int = 0,
                      dim: int | None = None) -> tuple[Any, int]:
    """Deserialize a non-null field value, returning (value, new_offset)."""
    if dtype == DataType.INT:
        return struct.unpack_from("<q", buf, offset)[0], offset + 8
    if dtype == DataType.FLOAT:
        return struct.unpack_from("<d", buf, offset)[0], offset + 8
    if dtype == DataType.BOOL:
        return buf[offset] == 1, offset + 1
    if dtype == DataType.STR:
        length = struct.unpack_from("<I", buf, offset)[0]
        start = offset + 4
        return bytes(buf[start:start + length]).decode("utf-8"), start + length
    if dtype == DataType.BYTES:
        length = struct.unpack_from("<I", buf, offset)[0]
        start = offset + 4
        return bytes(buf[start:start + length]), start + length
    if dtype == DataType.VECTOR:
        assert dim is not None
        end = offset + 4 * dim
        return list(struct.unpack_from(f"<{dim}f", buf, offset)), end
    # DataType.ANY
    tag = buf[offset]
    pos = offset + 1
    if tag == 1:
        return buf[pos] == 1, pos + 1
    if tag == 2:
        return struct.unpack_from("<q", buf, pos)[0], pos + 8
    if tag == 3:
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    if tag == 4:
        length = struct.unpack_from("<I", buf, pos)[0]
        return bytes(buf[pos + 4:pos + 4 + length]).decode("utf-8"), pos + 4 + length
    if tag == 5:
        length = struct.unpack_from("<I", buf, pos)[0]
        return bytes(buf[pos + 4:pos + 4 + length]), pos + 4 + length
    raise ValueError(f"Unknown ANY value tag: {tag}")


def _skip_value(dtype: DataType, buf: bytes | memoryview, offset: int, dim: int | None = None) -> int:
    if dtype == DataType.INT or dtype == DataType.FLOAT:
        return offset + 8
    if dtype == DataType.BOOL:
        return offset + 1
    if dtype in (DataType.STR, DataType.BYTES):
        return offset + 4 + struct.unpack_from("<I", buf, offset)[0]
    if dtype == DataType.VECTOR:
        assert dim is not None
        return offset + 4 * dim
    tag = buf[offset]
    if tag == 1:
        return offset + 2
    if tag in (2, 3):
        return offset + 9
    if tag in (4, 5):
        return offset + 5 + struct.unpack_from("<I", buf, offset + 1)[0]
    raise ValueError(f"Unknown ANY value tag: {tag}")


def serialize_row(schema: TableSchema, record: dict[str, Any]) -> bytes:
    """Encode a validated record with a leading null bitmap."""
    n = len(schema.columns)
    bitmap_len = (n + 7) // 8
    bitmap = bytearray(bitmap_len)
    parts: list[bytes] = []
    for idx, col in enumerate(schema.columns):
        val = record[col.name]
        if val is None:
            bitmap[idx // 8] |= 1 << (idx % 8)
        else:
            parts.append(serialize_value(col.dtype, val, col.dim))
    return bytes(bitmap) + b"".join(parts)


def deserialize_row(schema: TableSchema, buf: bytes | memoryview,
                    projected: list[str] | None = None) -> dict[str, Any]:
    """Decode a binary row, optionally projecting only requested columns."""
    n = len(schema.columns)
    offset = (n + 7) // 8
    wanted = set(projected) if projected is not None else None
    out: dict[str, Any] = {}
    for idx, col in enumerate(schema.columns):
        is_null = bool(buf[idx // 8] & (1 << (idx % 8)))
        if is_null:
            if wanted is None or col.name in wanted:
                out[col.name] = None
        elif wanted is None or col.name in wanted:
            val, offset = deserialize_value(col.dtype, buf, offset, col.dim)
            out[col.name] = val
        else:
            offset = _skip_value(col.dtype, buf, offset, col.dim)
    return out
