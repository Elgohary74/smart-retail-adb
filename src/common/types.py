"""
Module: src/common/types.py
Layer: Layer 0 — Common Primitives & Binary Codecs
Lock Domain: Stateless

Purpose:
    Implements canonical column types (``DataType``), storage layout selectors
    (``LayoutType``), immutable 64-bit record identifiers (``RID``), relational
    schemas (``ColumnDef``, ``Schema``), row serialization, and order-preserving
    binary key codecs for B+ Tree and LSM Tree indexes.

Key Invariants:
    1. ``RID`` packs as 8-byte little-endian ``<II`` on disk, while composite
       index keys append ``>II`` so lexicographical byte comparison preserves
       ascending ``(page_id, slot_id)`` order.
    2. Signed integer and IEEE-754 ``FLOAT64`` key encodings preserve exact total
       numeric order under raw big-endian unsigned byte comparison, normalizing
       ``-0.0`` to ``+0.0`` and rejecting ``NaN``.
    3. Row ``VARCHAR`` values pack as 2-byte little-endian length prefix ``<H``
       plus UTF-8 bytes, whereas fixed-width ``VARCHAR`` keys right-pad with
       ``b"\\x00"`` to ``key_size``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from src.common.config import INVALID_PAGE_ID, INVALID_SLOT_ID

_RID_STRUCT = struct.Struct("<II")
_COMP_RID_STRUCT = struct.Struct(">II")

_LE_I32 = struct.Struct("<i")
_LE_I64 = struct.Struct("<q")
_LE_U64 = struct.Struct("<Q")
_LE_F64 = struct.Struct("<d")
_LE_U16 = struct.Struct("<H")

_BE_U32 = struct.Struct(">I")
_BE_U64 = struct.Struct(">Q")
_BE_F64 = struct.Struct(">d")

_INT32_SIGN_BIT = 1 << 31
_INT32_MASK = 0xFFFFFFFF
_INT64_SIGN_BIT = 1 << 63
_INT64_MASK = 0xFFFFFFFFFFFFFFFF


class DataType(IntEnum):
    """Canonical column data types supported across row, columnar, and index engines."""

    INT32 = 1
    INT64 = 2
    UINT64 = 3
    FLOAT64 = 4
    VARCHAR = 5

    @property
    def fixed_size(self) -> int:
        """Return byte width for fixed-size types.

        Returns:
            Byte size of one element in binary page storage.

        Raises:
            ValueError: If called on variable-length ``DataType.VARCHAR``.
        """
        sizes = {
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.UINT64: 8,
            DataType.FLOAT64: 8,
        }
        if self not in sizes:
            raise ValueError(f"{self.name} does not have a fixed byte size")
        return sizes[self]

    @property
    def struct_fmt(self) -> str:
        """Return the little-endian ``struct`` format code for fixed-size types.

        Returns:
            Little-endian ``struct`` format string.

        Raises:
            ValueError: If called on variable-length ``DataType.VARCHAR``.
        """
        formats = {
            DataType.INT32: "<i",
            DataType.INT64: "<q",
            DataType.UINT64: "<Q",
            DataType.FLOAT64: "<d",
        }
        if self not in formats:
            raise ValueError(f"{self.name} does not have a single fixed struct format")
        return formats[self]

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        """Return the little-endian NumPy dtype for zero-copy ``ColumnPage`` views.

        Returns:
            Little-endian ``numpy.dtype`` matching the on-disk binary layout.

        Raises:
            ValueError: If called on variable-length ``DataType.VARCHAR``.
        """
        dtypes = {
            DataType.INT32: np.dtype("<i4"),
            DataType.INT64: np.dtype("<i8"),
            DataType.UINT64: np.dtype("<u8"),
            DataType.FLOAT64: np.dtype("<f8"),
        }
        if self not in dtypes:
            raise ValueError(f"{self.name} does not have a fixed NumPy dtype")
        return dtypes[self]


class LayoutType(str, Enum):
    """Physical table storage layout selector."""

    ROW = "ROW"
    COLUMN = "COLUMN"


@dataclass(frozen=True, order=True)
class RID:
    """Immutable 64-bit record identifier ``(page_id, slot_id)``.

    Binary Layout (8 bytes, little-endian ``<II``):
        * Offset ``0..3``: ``page_id`` (``uint32``)
        * Offset ``4..7``: ``slot_id`` (``uint32``)
    """

    page_id: int
    slot_id: int

    def pack(self) -> bytes:
        """Serialize this ``RID`` into 8 little-endian bytes.

        Returns:
            8-byte representation packed as ``<II``.
        """
        return _RID_STRUCT.pack(self.page_id, self.slot_id)

    @classmethod
    def unpack(cls, data: bytes | bytearray | memoryview, offset: int = 0) -> RID:
        """Deserialize an 8-byte little-endian ``RID`` from ``data``.

        Args:
            data: Buffer containing at least ``offset + 8`` bytes.
            offset: Byte offset at which the ``RID`` begins.

        Returns:
            Decoded ``RID`` instance.

        Raises:
            struct.error: If fewer than 8 bytes remain at ``offset``.
        """
        page_id, slot_id = _RID_STRUCT.unpack_from(data, offset)
        return cls(page_id=page_id, slot_id=slot_id)


NULL_RID: RID = RID(INVALID_PAGE_ID, INVALID_SLOT_ID)


@dataclass(frozen=True)
class ColumnDef:
    """Relational column definition within a table ``Schema``."""

    name: str
    dtype: DataType
    max_len: int = 8
    nullable: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Column name must be non-empty")
        if self.dtype == DataType.VARCHAR and self.max_len <= 0:
            raise ValueError(f"VARCHAR column {self.name!r} must have positive max_len")
        if self.max_len <= 0:
            raise ValueError(f"Column {self.name!r} must have positive max_len")

    @property
    def key_size(self) -> int:
        """Return fixed index key width in bytes for this column."""
        if self.dtype == DataType.VARCHAR:
            return self.max_len
        return self.dtype.fixed_size

    def to_dict(self) -> dict[str, Any]:
        """Serialize this column definition into a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype.name,
            "max_len": self.max_len,
            "nullable": self.nullable,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ColumnDef:
        """Reconstruct a ``ColumnDef`` from a catalog dictionary.

        Args:
            raw: Mapping containing ``name``, ``dtype``, and optional
                ``max_len`` and ``nullable`` fields.

        Returns:
            Parsed ``ColumnDef`` instance.
        """
        raw_dtype = raw["dtype"]
        if isinstance(raw_dtype, DataType):
            dtype = raw_dtype
        elif isinstance(raw_dtype, str):
            dtype = DataType[raw_dtype]
        else:
            dtype = DataType(int(raw_dtype))

        default_len = 64 if dtype == DataType.VARCHAR else dtype.fixed_size
        return cls(
            name=str(raw["name"]),
            dtype=dtype,
            max_len=int(raw.get("max_len", default_len)),
            nullable=bool(raw.get("nullable", False)),
        )


_ROW_STRUCT_MAP: dict[DataType, struct.Struct] = {
    DataType.INT32: _LE_I32,
    DataType.INT64: _LE_I64,
    DataType.UINT64: _LE_U64,
    DataType.FLOAT64: _LE_F64,
}


@dataclass(frozen=True)
class Schema:
    """Table schema defining ordered columns, optional primary key, and row codecs."""

    table_name: str
    columns: tuple[ColumnDef, ...]
    primary_key: str | None = None
    _col_index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.table_name:
            raise ValueError("Table name must be non-empty")
        cols = tuple(self.columns)
        if not cols:
            raise ValueError(f"Schema {self.table_name!r} must declare at least one column")
        object.__setattr__(self, "columns", cols)

        col_index: dict[str, int] = {}
        for idx, col in enumerate(cols):
            if col.name in col_index:
                raise ValueError(f"Duplicate column {col.name!r} in schema {self.table_name!r}")
            col_index[col.name] = idx
        if self.primary_key is not None and self.primary_key not in col_index:
            raise ValueError(
                f"Primary key {self.primary_key!r} is not present in schema {self.table_name!r}"
            )
        object.__setattr__(self, "_col_index", col_index)

    def get_column(self, name: str) -> ColumnDef:
        """Return the ``ColumnDef`` for ``name``.

        Args:
            name: Column name to look up.

        Returns:
            Matching ``ColumnDef``.

        Raises:
            KeyError: If ``name`` is not in this schema.
        """
        idx = self.column_index(name)
        return self.columns[idx]

    def column_index(self, name: str) -> int:
        """Return the zero-based position of column ``name``.

        Args:
            name: Column name to look up.

        Returns:
            Zero-based column index.

        Raises:
            KeyError: If ``name`` is not in this schema.
        """
        if name not in self._col_index:
            raise KeyError(f"Column {name!r} not found in schema {self.table_name!r}")
        return self._col_index[name]

    def encode_row(self, values: Mapping[str, Any] | Sequence[Any]) -> bytes:
        """Serialize a row mapping or sequence into sequential binary payload bytes.

        Args:
            values: Column-name mapping or ordered sequence matching ``self.columns``.

        Returns:
            Deterministic binary representation without the 16-byte MVCC header.

        Raises:
            KeyError: If a column is missing from a mapping input.
            ValueError: If positional length mismatches, a null value is supplied,
                or a ``VARCHAR`` exceeds ``col.max_len`` UTF-8 bytes.
            TypeError: If a column value has an incompatible Python type.
        """
        if isinstance(values, Mapping):
            ordered = [values[col.name] for col in self.columns]
        else:
            if len(values) != len(self.columns):
                raise ValueError(
                    f"Expected {len(self.columns)} values for {self.table_name!r}, got {len(values)}"
                )
            ordered = list(values)

        chunks: list[bytes] = []
        for col, val in zip(self.columns, ordered):
            if val is None:
                raise ValueError(f"Column {col.name!r} cannot be None in binary row encoding")
            if col.dtype == DataType.VARCHAR:
                if not isinstance(val, str):
                    raise TypeError(f"VARCHAR column {col.name!r} expects str, got {type(val).__name__}")
                encoded = val.encode("utf-8")
                if len(encoded) > col.max_len:
                    raise ValueError(
                        f"Column {col.name!r} length {len(encoded)} exceeds max_len={col.max_len}"
                    )
                chunks.append(_LE_U16.pack(len(encoded)))
                chunks.append(encoded)
            elif col.dtype == DataType.FLOAT64:
                chunks.append(_LE_F64.pack(float(val)))
            else:
                if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                    raise TypeError(
                        f"{col.dtype.name} column {col.name!r} expects int, got {type(val).__name__}"
                    )
                chunks.append(_ROW_STRUCT_MAP[col.dtype].pack(int(val)))
        return b"".join(chunks)

    def decode_row(self, data: bytes | bytearray | memoryview) -> dict[str, Any]:
        """Deserialize a binary row payload into a ``dict[str, Any]``.

        Args:
            data: Raw row payload bytes (excluding the 16-byte MVCC header).

        Returns:
            Dictionary mapping each column name to its decoded Python value.

        Raises:
            ValueError: If the payload is truncated or contains trailing bytes.
        """
        mv = memoryview(data)
        offset = 0
        result: dict[str, Any] = {}
        total_len = len(mv)

        for col in self.columns:
            if col.dtype == DataType.VARCHAR:
                if offset + 2 > total_len:
                    raise ValueError(f"Truncated VARCHAR length prefix for column {col.name!r}")
                str_len = _LE_U16.unpack_from(mv, offset)[0]
                offset += 2
                if str_len > col.max_len or offset + str_len > total_len:
                    raise ValueError(f"Invalid VARCHAR payload bounds for column {col.name!r}")
                result[col.name] = bytes(mv[offset : offset + str_len]).decode("utf-8")
                offset += str_len
            else:
                size = col.dtype.fixed_size
                if offset + size > total_len:
                    raise ValueError(f"Truncated fixed column {col.name!r}")
                result[col.name] = _ROW_STRUCT_MAP[col.dtype].unpack_from(mv, offset)[0]
                offset += size

        if offset != total_len:
            raise ValueError(
                f"Trailing bytes after decoding row for {self.table_name!r}: "
                f"consumed {offset} of {total_len}"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize this schema into a JSON-compatible dictionary."""
        return {
            "table_name": self.table_name,
            "primary_key": self.primary_key,
            "columns": [col.to_dict() for col in self.columns],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Schema:
        """Reconstruct a ``Schema`` from a catalog dictionary."""
        return cls(
            table_name=str(raw["table_name"]),
            columns=tuple(ColumnDef.from_dict(c) for c in raw["columns"]),
            primary_key=raw.get("primary_key"),
        )


def encode_key(value: Any, dtype: DataType, key_size: int) -> bytes:
    """Encode a key value into fixed-width order-preserving bytes.

    Raw lexicographical comparison (``bytes_a < bytes_b``) matches exact numeric
    or UTF-8 ordering for all supported ``DataType`` variants.

    Args:
        value: Key value to encode.
        dtype: Canonical ``DataType`` of the key column.
        key_size: Expected fixed byte width (4 for ``INT32``, 8 for 64-bit
            numerics, ``max_len`` for ``VARCHAR``).

    Returns:
        Fixed-length ``bytes`` of length ``key_size``.

    Raises:
        ValueError: If ``value`` is ``NaN``, out of integer range, exceeds
            ``key_size`` UTF-8 bytes, or ``key_size`` mismatches ``dtype``.
        TypeError: If ``value`` has an incompatible Python type.
    """
    if dtype == DataType.INT32:
        if key_size != 4:
            raise ValueError(f"INT32 key_size must be 4, got {key_size}")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"INT32 key expects int, got {type(value).__name__}")
        iv = int(value)
        if iv < -(1 << 31) or iv >= (1 << 31):
            raise ValueError(f"INT32 key {iv} out of range")
        # Mask before XOR so Python negative ints do not keep infinite leading sign bits.
        return _BE_U32.pack((iv & _INT32_MASK) ^ _INT32_SIGN_BIT)

    if dtype == DataType.INT64:
        if key_size != 8:
            raise ValueError(f"INT64 key_size must be 8, got {key_size}")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"INT64 key expects int, got {type(value).__name__}")
        iv = int(value)
        if iv < -(1 << 63) or iv >= (1 << 63):
            raise ValueError(f"INT64 key {iv} out of range")
        return _BE_U64.pack((iv & _INT64_MASK) ^ _INT64_SIGN_BIT)

    if dtype == DataType.UINT64:
        if key_size != 8:
            raise ValueError(f"UINT64 key_size must be 8, got {key_size}")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"UINT64 key expects int, got {type(value).__name__}")
        iv = int(value)
        if iv < 0 or iv >= (1 << 64):
            raise ValueError(f"UINT64 key {iv} out of range")
        return _BE_U64.pack(iv)

    if dtype == DataType.FLOAT64:
        if key_size != 8:
            raise ValueError(f"FLOAT64 key_size must be 8, got {key_size}")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"FLOAT64 key expects float or int, got {type(value).__name__}")
        fv = float(value)
        if math.isnan(fv):
            raise ValueError("NaN cannot be encoded as an ordered index key")
        # Collapse -0.0 into +0.0 so both zero encodings hash and compare identically.
        if fv == 0.0:
            fv = 0.0
        bits = _BE_U64.unpack(_BE_F64.pack(fv))[0]
        # Flip all bits for negative IEEE-754 floats; flip only sign bit for non-negatives.
        if bits >> 63:
            mapped = bits ^ _INT64_MASK
        else:
            mapped = bits ^ _INT64_SIGN_BIT
        return _BE_U64.pack(mapped)

    if dtype == DataType.VARCHAR:
        if key_size <= 0:
            raise ValueError(f"VARCHAR key_size must be positive, got {key_size}")
        if not isinstance(value, str):
            raise TypeError(f"VARCHAR key expects str, got {type(value).__name__}")
        encoded = value.encode("utf-8")
        if len(encoded) > key_size:
            raise ValueError(
                f"VARCHAR key length {len(encoded)} exceeds key_size={key_size}"
            )
        return encoded.ljust(key_size, b"\x00")

    raise ValueError(f"Unsupported DataType for key encoding: {dtype!r}")


def decode_key(raw: bytes | bytearray | memoryview, dtype: DataType) -> Any:
    """Decode an order-preserving key byte slice back into its Python value.

    Args:
        raw: Encoded fixed-width bytes or memoryview produced by ``encode_key``.
        dtype: Canonical ``DataType`` of the encoded key.

    Returns:
        Decoded Python scalar (``int``, ``float``, or ``str``).

    Raises:
        ValueError: If byte width is invalid for ``dtype`` or UTF-8 decoding fails.
    """
    mv = memoryview(raw)
    if dtype == DataType.INT32:
        if len(mv) != 4:
            raise ValueError(f"INT32 encoded key must be 4 bytes, got {len(mv)}")
        return _BE_U32.unpack_from(mv, 0)[0] - _INT32_SIGN_BIT

    if dtype == DataType.INT64:
        if len(mv) != 8:
            raise ValueError(f"INT64 encoded key must be 8 bytes, got {len(mv)}")
        return _BE_U64.unpack_from(mv, 0)[0] - _INT64_SIGN_BIT

    if dtype == DataType.UINT64:
        if len(mv) != 8:
            raise ValueError(f"UINT64 encoded key must be 8 bytes, got {len(mv)}")
        return _BE_U64.unpack_from(mv, 0)[0]

    if dtype == DataType.FLOAT64:
        if len(mv) != 8:
            raise ValueError(f"FLOAT64 encoded key must be 8 bytes, got {len(mv)}")
        mapped = _BE_U64.unpack_from(mv, 0)[0]
        if mapped >> 63:
            bits = mapped ^ _INT64_SIGN_BIT
        else:
            bits = mapped ^ _INT64_MASK
        return _BE_F64.unpack(_BE_U64.pack(bits))[0]

    if dtype == DataType.VARCHAR:
        return bytes(mv).rstrip(b"\x00").decode("utf-8")

    raise ValueError(f"Unsupported DataType for key decoding: {dtype!r}")


def make_composite_key(key_bytes: bytes, rid: RID) -> bytes:
    """Append an 8-byte big-endian ``RID`` suffix to form a unique composite key.

    Args:
        key_bytes: Fixed-width encoded key bytes from ``encode_key``.
        rid: Target ``RID`` appended as ``>II`` for ascending byte order.

    Returns:
        Composite key bytes of length ``len(key_bytes) + 8``.
    """
    return key_bytes + _COMP_RID_STRUCT.pack(rid.page_id, rid.slot_id)


def split_composite_key(comp: bytes | bytearray | memoryview, key_size: int) -> tuple[bytes, RID]:
    """Split a composite index key back into ``(key_bytes, RID)``.

    Args:
        comp: Composite byte sequence of length ``key_size + 8``.
        key_size: Fixed byte width of the base key component.

    Returns:
        Tuple of ``(raw_key_bytes, RID)``.

    Raises:
        ValueError: If ``len(comp) != key_size + 8``.
    """
    mv = memoryview(comp)
    expected_len = key_size + 8
    if len(mv) != expected_len:
        raise ValueError(f"Expected composite key of length {expected_len}, got {len(mv)}")
    page_id, slot_id = _COMP_RID_STRUCT.unpack_from(mv, key_size)
    return bytes(mv[:key_size]), RID(page_id=page_id, slot_id=slot_id)


__all__ = [
    "ColumnDef",
    "DataType",
    "LayoutType",
    "NULL_RID",
    "RID",
    "Schema",
    "decode_key",
    "encode_key",
    "make_composite_key",
    "split_composite_key",
]
