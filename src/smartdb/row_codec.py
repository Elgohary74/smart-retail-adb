"""Binary value and row serialization/deserialization with projection skip."""
from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from .key_codec import DataType

if TYPE_CHECKING:
    from .schema import TableSchema


def _serialize_any(value: Any) -> bytes:
    if isinstance(value, bool):
        return b"\x01" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return b"\x02" + struct.pack("<q", value)
    if isinstance(value, float):
        return b"\x03" + struct.pack("<d", value)
    if isinstance(value, str):
        enc = value.encode("utf-8")
        return b"\x04" + struct.pack("<I", len(enc)) + enc
    if isinstance(value, (bytes, bytearray)):
        return b"\x05" + struct.pack("<I", len(value)) + bytes(value)
    raise TypeError(f"Unsupported value type for ANY: {type(value).__name__}")


def serialize_value(dtype: DataType, value: Any, dim: int | None = None) -> bytes:
    if dtype == DataType.INT:
        return struct.pack("<q", value)
    if dtype == DataType.FLOAT:
        return struct.pack("<d", float(value))
    if dtype == DataType.BOOL:
        return b"\x01" if value else b"\x00"
    if dtype == DataType.STR:
        enc = value.encode("utf-8")
        return struct.pack("<I", len(enc)) + enc
    if dtype == DataType.BYTES:
        return struct.pack("<I", len(value)) + value
    if dtype == DataType.VECTOR:
        assert dim is not None
        return struct.pack(f"<{dim}f", *value)
    return _serialize_any(value)


def _deserialize_any(buf: bytes | memoryview, offset: int) -> tuple[Any, int]:
    tag, pos = buf[offset], offset + 1
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


def deserialize_value(dtype: DataType, buf: bytes | memoryview, offset: int = 0,
                      dim: int | None = None) -> tuple[Any, int]:
    if dtype == DataType.INT:
        return struct.unpack_from("<q", buf, offset)[0], offset + 8
    if dtype == DataType.FLOAT:
        return struct.unpack_from("<d", buf, offset)[0], offset + 8
    if dtype == DataType.BOOL:
        return buf[offset] == 1, offset + 1
    if dtype == DataType.STR:
        length = struct.unpack_from("<I", buf, offset)[0]
        return bytes(buf[offset + 4:offset + 4 + length]).decode("utf-8"), offset + 4 + length
    if dtype == DataType.BYTES:
        length = struct.unpack_from("<I", buf, offset)[0]
        return bytes(buf[offset + 4:offset + 4 + length]), offset + 4 + length
    if dtype == DataType.VECTOR:
        assert dim is not None
        return list(struct.unpack_from(f"<{dim}f", buf, offset)), offset + 4 * dim
    return _deserialize_any(buf, offset)


def _skip_value(dtype: DataType, buf: bytes | memoryview, offset: int, dim: int | None = None) -> int:
    if dtype in (DataType.INT, DataType.FLOAT):
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


def serialize_row(schema: "TableSchema", record: dict[str, Any]) -> bytes:
    bitmap = bytearray((len(schema.columns) + 7) // 8)
    parts: list[bytes] = []
    for idx, col in enumerate(schema.columns):
        val = record[col.name]
        if val is None:
            bitmap[idx // 8] |= 1 << (idx % 8)
        else:
            parts.append(serialize_value(col.dtype, val, col.dim))
    return bytes(bitmap) + b"".join(parts)


def deserialize_row(schema: "TableSchema", buf: bytes | memoryview,
                    projected: list[str] | None = None) -> dict[str, Any]:
    offset = (len(schema.columns) + 7) // 8
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
