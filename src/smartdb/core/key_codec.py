"""Order-preserving binary key encoding and decoding for indexes."""
from __future__ import annotations

import math
import struct
from typing import Any

from .types import DataType

_INT64_BIAS = 1 << 63
_INT64_BOUND = 9223372036854775808.0


def _enc_int64(val: int) -> bytes:
    return struct.pack(">Q", val + _INT64_BIAS)


def _dec_int64(raw: bytes | memoryview) -> int:
    return struct.unpack(">Q", raw)[0] - _INT64_BIAS


def _enc_float64(val: float) -> bytes:
    bits = struct.unpack(">Q", struct.pack(">d", 0.0 if val == 0.0 else val))[0]
    bits ^= 0xFFFFFFFFFFFFFFFF if (bits & (1 << 63)) else (1 << 63)
    return struct.pack(">Q", bits)


def _dec_float64(raw: bytes | memoryview) -> float:
    bits = struct.unpack(">Q", raw)[0]
    bits ^= (1 << 63) if (bits & (1 << 63)) else 0xFFFFFFFFFFFFFFFF
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _enc_any_num(val: int | float) -> bytes:
    if isinstance(val, int) or (-_INT64_BOUND <= val < _INT64_BOUND):
        floor_i = val if isinstance(val, int) else math.floor(val)
        frac = 0.0 if isinstance(val, int) else (val - floor_i)
        return b"\x01" + _enc_int64(floor_i) + _enc_float64(frac)
    return (b"\x00" if val < 0 else b"\x02") + _enc_float64(val)


def _dec_any_num(raw: bytes | memoryview) -> int | float:
    if raw[0] == 1:
        floor_i = _dec_int64(raw[1:9])
        frac = _dec_float64(raw[9:17])
        return floor_i if frac == 0.0 else floor_i + frac
    return _dec_float64(raw[1:9])


def _encode_any(value: Any) -> bytes:
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
    raise TypeError(f"Unsupported ANY key type: {type(value).__name__}")


def encode_key(dtype: DataType, value: Any) -> bytes:
    """Encode a scalar key into order-preserving bytes (NULL sorts first)."""
    if value is None:
        return b"\x00"
    if dtype == DataType.ANY:
        return _encode_any(value)
    if dtype == DataType.BOOL:
        if not isinstance(value, bool):
            raise TypeError(f"Expected BOOL key, got {type(value).__name__}")
        return b"\x01\x01" if value else b"\x01\x00"
    if dtype == DataType.INT:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Expected INT key, got {type(value).__name__}")
        return b"\x01" + _enc_int64(value)
    if dtype == DataType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TypeError(f"Expected finite FLOAT key, got {value!r}")
        return b"\x01" + _enc_float64(float(value))
    if dtype == DataType.STR and isinstance(value, str):
        return b"\x01" + value.encode("utf-8")
    if dtype == DataType.BYTES and isinstance(value, (bytes, bytearray)):
        return b"\x01" + bytes(value)
    raise TypeError(f"Unsupported key {type(value).__name__} for {dtype.value}")


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
        rank, payload = body[0], body[1:]
        if rank == 1:
            return payload[0] == 1
        if rank == 2:
            return _dec_any_num(payload)
        if rank == 3:
            return bytes(payload).decode("utf-8")
        if rank == 4:
            return bytes(payload)
    raise ValueError(f"Corrupt encoded key for {dtype.value}")
