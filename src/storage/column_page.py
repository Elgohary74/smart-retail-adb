"""
Module: src/storage/column_page.py
Layer: Layer 1 — Column-Oriented Chunk Page Layout
Lock Domain: Caller must hold an active BufferPoolManager pin on the frame

Purpose:
    Implements an 8,192-byte column-oriented storage page
    (``PageType.COLUMN_CHUNK``) supporting contiguous fixed-width scalar
    arrays (``INT32``, ``INT64``, ``UINT64``, ``FLOAT64``), variable-length
    ``VARCHAR`` string columns, and arbitrary fixed-byte ``stride`` vectors
    (e.g., 384-d ``float32`` embeddings at ``stride=1536`` for Milestone 5)
    with zero-copy NumPy ``np.frombuffer`` execution views.

Key Invariants:
    1. Fixed-width and strided columns store C-contiguous little-endian
       elements starting at byte offset ``32`` (``HEADER_SIZE``), advancing
       ``free_space_ptr`` forward as ``32 + slot_count * item_size``.
    2. ``as_numpy_array()`` wraps the underlying ``bytearray`` or
       ``memoryview`` directly via ``np.frombuffer(..., offset=32)`` so
       ``arr.flags["OWNDATA"]`` is strictly ``False`` (zero-copy).
    3. ``DataType.VARCHAR`` columns store a forward-growing 4-byte ``<HH``
       offset directory from byte ``32`` and a backward-growing UTF-8 string
       heap from byte ``8192``, keeping ``free_space_ptr`` at the heap floor.
"""

from __future__ import annotations

from collections.abc import Sequence
import struct
from typing import Any

import numpy as np

from src.common.config import (
    HEADER_SIZE,
    INVALID_PAGE_ID,
    PAGE_SIZE,
    SLOT_ENTRY_SIZE,
)
from src.common.exceptions import RecordTooLargeError, StorageError
from src.common.types import DataType
from src.storage.page_header import PageFlags, PageHeader, PageType

_SLOT_STRUCT = struct.Struct("<HH")
_U16_STRUCT = struct.Struct("<H")
_U32_STRUCT = struct.Struct("<I")

_PACK_STRUCTS: dict[DataType, struct.Struct] = {
    DataType.INT32: struct.Struct("<i"),
    DataType.INT64: struct.Struct("<q"),
    DataType.UINT64: struct.Struct("<Q"),
    DataType.FLOAT64: struct.Struct("<d"),
}

_INT_BOUNDS: dict[DataType, tuple[int, int]] = {
    DataType.INT32: (-(1 << 31), (1 << 31) - 1),
    DataType.INT64: (-(1 << 63), (1 << 63) - 1),
    DataType.UINT64: (0, (1 << 64) - 1),
}

_PAGE_ID_OFFSET: int = 0
_SLOT_COUNT_OFFSET: int = 6
_FREE_SPACE_PTR_OFFSET: int = 16

_MAX_UINT32: int = 0xFFFFFFFF
_ZERO_PAGE: bytes = bytes(PAGE_SIZE)


def _validate_writable_page_buffer(buf: bytearray | memoryview) -> None:
    """Validate that ``buf`` is a writable 8,192-byte ``bytearray`` or ``memoryview``."""
    if not isinstance(buf, (bytearray, memoryview)):
        raise TypeError(
            f"ColumnPage requires a bytearray or memoryview, got {type(buf).__name__}"
        )
    if isinstance(buf, memoryview) and buf.readonly:
        raise ValueError("ColumnPage requires a writable memoryview")
    if len(buf) != PAGE_SIZE:
        raise ValueError(f"Page buffer must be {PAGE_SIZE} bytes, got {len(buf)}")


def _validate_positive_int(value: int | None, name: str) -> int | None:
    """Validate an optional strictly positive integer parameter."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int or None, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class ColumnPage:
    """8,192-byte columnar chunk page supporting zero-copy NumPy views and VARCHAR heaps.

    Binary Layout (8,192 bytes):
    ```text
    [A] Fixed-Width & Strided Layout (INT32, INT64, UINT64, FLOAT64, or stride != None):
    0                  32                                  32 + N * item_size         8192
    +------------------+-----------------------------------+--------------------------+
    | PageHeader (32B) | val[0] | val[1] | ... | val[N-1]  |   Unused Trailing Bytes  |
    | <IBBHQIIQ>       |  <--- Zero-Copy np.frombuffer() ->|   (free_space_ptr = end) |
    +------------------+-----------------------------------+--------------------------+
      item_size = stride if stride is not None else dtype.fixed_size
      physical_max = (8192 - 32) // item_size   (1020 for 8B, 2040 for 4B, 5 for 1536B)

    [B] Variable-Length VARCHAR Layout (dtype == DataType.VARCHAR and stride is None):
    0                  32                             32+4N              free_space_ptr   8192
    +------------------+------------------------------+------------------+----------------+
    | PageHeader (32B) | Offset Dir (4B * slot_count) |   Free Space     | UTF-8 Payloads |
    | <IBBHQIIQ>       | Entry i: <HH (offset, len)   |                  | (grow downward)|
    +------------------+------------------------------+------------------+----------------+
    ```
    """

    def __init__(
        self,
        buf: bytearray | memoryview,
        dtype: DataType,
        max_rows: int | None = None,
        stride: int | None = None,
    ) -> None:
        """Wrap an existing initialized 8,192-byte ``COLUMN_CHUNK`` page buffer.

        Args:
            buf: Writable ``bytearray`` or ``memoryview`` of exact length ``8192``.
            dtype: Canonical ``DataType`` stored in this column page.
            max_rows: Optional logical row cap per page (e.g., ``CHUNK_ROWS = 1020``
                so multi-column tables align ``(page_id, slot_id)`` across types).
            stride: Optional fixed byte width per entry (e.g., ``1536`` for
                384-dimensional ``float32`` vectors), overriding ``dtype.fixed_size``.

        Raises:
            TypeError: If ``buf`` is invalid, ``dtype`` is not a ``DataType``, or
                ``max_rows``/``stride`` has a non-integer type.
            ValueError: If ``buf`` is read-only or not ``8192`` bytes, header
                ``page_type`` is not ``PageType.COLUMN_CHUNK``, ``stride`` exceeds
                payload space, or header offsets are corrupt.

        Complexity:
            $O(1)$ time and space; operates zero-copy on ``buf``.
        """
        _validate_writable_page_buffer(buf)
        if not isinstance(dtype, DataType):
            raise TypeError(f"dtype must be a DataType, got {type(dtype).__name__}")

        max_rows = _validate_positive_int(max_rows, "max_rows")
        stride = _validate_positive_int(stride, "stride")

        payload_bytes = PAGE_SIZE - HEADER_SIZE
        if stride is not None and stride > payload_bytes:
            raise ValueError(
                f"stride {stride} exceeds single-page payload capacity {payload_bytes}"
            )

        header = PageHeader.unpack(buf, 0)
        if header.page_type != PageType.COLUMN_CHUNK:
            raise ValueError(
                f"Expected PageType.COLUMN_CHUNK, got {header.page_type!r}"
            )

        self._is_varchar: bool = stride is None and dtype == DataType.VARCHAR
        if self._is_varchar:
            self._item_size: int = 0
            physical_max = payload_bytes // SLOT_ENTRY_SIZE
            slot_dir_end = HEADER_SIZE + header.slot_count * SLOT_ENTRY_SIZE
            if header.free_space_ptr < slot_dir_end or header.free_space_ptr > PAGE_SIZE:
                raise ValueError(
                    f"Corrupt VARCHAR free_space_ptr {header.free_space_ptr} "
                    f"for slot_count {header.slot_count}"
                )
        else:
            item_size = stride if stride is not None else dtype.fixed_size
            self._item_size = item_size
            physical_max = payload_bytes // item_size
            expected_ptr = HEADER_SIZE + header.slot_count * item_size
            if header.slot_count > physical_max or header.free_space_ptr != expected_ptr:
                raise ValueError(
                    f"Corrupt fixed-width ColumnPage header: slot_count={header.slot_count}, "
                    f"free_space_ptr={header.free_space_ptr}, expected_ptr={expected_ptr}"
                )

        self._buf: bytearray | memoryview = buf
        self._dtype: DataType = dtype
        self._stride: int | None = stride
        self._physical_max: int = physical_max
        self._capacity: int = (
            min(max_rows, physical_max) if max_rows is not None else physical_max
        )

    @classmethod
    def init_page(
        cls,
        buf: bytearray | memoryview,
        page_id: int,
        dtype: DataType,
        max_rows: int | None = None,
        stride: int | None = None,
    ) -> ColumnPage:
        """Initialize ``buf`` as a fresh, empty ``COLUMN_CHUNK`` page.

        Args:
            buf: Writable ``8192``-byte ``bytearray`` or ``memoryview``.
            page_id: Zero-based physical page identifier stored in the header.
            dtype: Canonical ``DataType`` stored in this column page.
            max_rows: Optional logical row cap per page.
            stride: Optional fixed byte stride per vector entry.

        Returns:
            A ``ColumnPage`` wrapper bound directly to ``buf``.

        Raises:
            TypeError: If ``buf`` or parameters have invalid types.
            ValueError: If ``buf`` is invalid or ``page_id`` is outside ``uint32``.

        Complexity:
            ``O(PAGE_SIZE)`` to zero-initialize the page buffer.
        """
        _validate_writable_page_buffer(buf)
        if not isinstance(dtype, DataType):
            raise TypeError(f"dtype must be a DataType, got {type(dtype).__name__}")
        if isinstance(page_id, bool) or not isinstance(page_id, int):
            raise TypeError(f"page_id must be int, got {type(page_id).__name__}")
        if page_id < 0 or page_id > _MAX_UINT32:
            raise ValueError(
                f"page_id must be uint32 in [0, {_MAX_UINT32}], got {page_id}"
            )

        _validate_positive_int(max_rows, "max_rows")
        stride = _validate_positive_int(stride, "stride")
        if stride is not None and stride > PAGE_SIZE - HEADER_SIZE:
            raise ValueError(
                f"stride {stride} exceeds single-page payload capacity {PAGE_SIZE - HEADER_SIZE}"
            )

        buf[:] = _ZERO_PAGE
        is_varchar = stride is None and dtype == DataType.VARCHAR
        # Fixed/strided pages grow forward from byte 32; VARCHAR heap grows backward from 8192.
        initial_free_ptr = PAGE_SIZE if is_varchar else HEADER_SIZE
        PageHeader(
            page_id=page_id,
            page_type=PageType.COLUMN_CHUNK,
            flags=PageFlags.NONE,
            slot_count=0,
            page_lsn=0,
            free_space_ptr=initial_free_ptr,
            next_page_id=INVALID_PAGE_ID,
            checksum=0,
        ).pack_into(buf, 0)
        return cls(buf, dtype=dtype, max_rows=max_rows, stride=stride)

    @property
    def buffer(self) -> bytearray | memoryview:
        """Return the underlying 8,192-byte page buffer."""
        return self._buf

    @property
    def page_id(self) -> int:
        """Return the 32-bit physical page ID from header bytes ``0..3``."""
        return _U32_STRUCT.unpack_from(self._buf, _PAGE_ID_OFFSET)[0]

    @property
    def dtype(self) -> DataType:
        """Return the configured ``DataType`` for this column page."""
        return self._dtype

    @property
    def stride(self) -> int | None:
        """Return the configured fixed vector byte stride, or ``None`` for scalars/VARCHAR."""
        return self._stride

    @property
    def capacity(self) -> int:
        """Return the maximum number of rows this page can accept."""
        return self._capacity

    @property
    def slot_count(self) -> int:
        """Return the number of values currently stored on this page."""
        return _U16_STRUCT.unpack_from(self._buf, _SLOT_COUNT_OFFSET)[0]

    @property
    def free_space_ptr(self) -> int:
        """Return header ``free_space_ptr`` (data end for fixed-width; heap floor for VARCHAR)."""
        return _U32_STRUCT.unpack_from(self._buf, _FREE_SPACE_PTR_OFFSET)[0]

    @property
    def header(self) -> PageHeader:
        """Unpack and return a snapshot of the 32-byte ``PageHeader``."""
        return PageHeader.unpack(self._buf, 0)

    @property
    def is_full(self) -> bool:
        """Return ``True`` when the page has reached row or byte capacity."""
        count = self.slot_count
        if count >= self._capacity:
            return True
        if self._is_varchar:
            return self.contiguous_free_space() < SLOT_ENTRY_SIZE
        return False

    def contiguous_free_space(self) -> int:
        """Return remaining unallocated payload bytes on this page.

        Returns:
            Free byte count between directory and heap (for ``VARCHAR``) or
            trailing unused bytes ``PAGE_SIZE - free_space_ptr`` (for fixed-width).

        Complexity:
            $O(1)$ time and space.
        """
        if self._is_varchar:
            return self.free_space_ptr - (HEADER_SIZE + self.slot_count * SLOT_ENTRY_SIZE)
        return PAGE_SIZE - self.free_space_ptr

    def _set_slot_count(self, count: int) -> None:
        _U16_STRUCT.pack_into(self._buf, _SLOT_COUNT_OFFSET, count)

    def _set_free_space_ptr(self, ptr: int) -> None:
        _U32_STRUCT.pack_into(self._buf, _FREE_SPACE_PTR_OFFSET, ptr)

    def _coerce_strided_bytes(self, value: Any) -> memoryview | bytes:
        """Validate and extract exact ``stride`` bytes from a vector payload."""
        if isinstance(value, (bytes, bytearray, memoryview)):
            mv = memoryview(value)
            if mv.ndim != 1 or mv.itemsize != 1:
                mv = mv.cast("B")
            if len(mv) != self._stride:
                raise ValueError(
                    f"Expected {self._stride} bytes for strided vector, got {len(mv)}"
                )
            return mv
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            if arr.nbytes != self._stride:
                raise ValueError(
                    f"Expected {self._stride} bytes for strided ndarray, got {arr.nbytes}"
                )
            return memoryview(arr).cast("B")
        raise TypeError(
            f"Strided ColumnPage expects bytes-like or np.ndarray, got {type(value).__name__}"
        )

    def _pack_fixed_scalar(self, value: Any, offset: int) -> None:
        """Validate and pack a single fixed-width scalar at ``offset``."""
        if self._dtype == DataType.FLOAT64:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    f"FLOAT64 column expects float or int, got {type(value).__name__}"
                )
            _PACK_STRUCTS[DataType.FLOAT64].pack_into(self._buf, offset, float(value))
            return

        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"{self._dtype.name} column expects int, got {type(value).__name__}"
            )
        iv = int(value)
        lo, hi = _INT_BOUNDS[self._dtype]
        if iv < lo or iv > hi:
            raise ValueError(f"{self._dtype.name} value {iv} out of range [{lo}, {hi}]")
        _PACK_STRUCTS[self._dtype].pack_into(self._buf, offset, iv)

    @staticmethod
    def _encode_varchar(value: Any) -> bytes:
        """Encode ``str`` or validate UTF-8 bytes-like input for ``VARCHAR`` storage."""
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            raw.decode("utf-8")
            return raw
        raise TypeError(f"VARCHAR column expects str or UTF-8 bytes, got {type(value).__name__}")

    def append_value(self, value: Any) -> int | None:
        """Append a single scalar, ``VARCHAR`` string, or strided vector to this page.

        Args:
            value: Scalar (``int``/``float``), ``str``/UTF-8 ``bytes`` (for ``VARCHAR``),
                or ``stride``-byte payload / ``np.ndarray`` (for strided pages).

        Returns:
            Zero-based ``slot_id`` assigned to the value, or ``None`` if the page
            lacks row capacity or free byte space.

        Raises:
            TypeError: If ``value`` has an incompatible Python type.
            ValueError: If an integer overflows its bit width, a strided payload
                mismatches ``stride``, or raw bytes are invalid UTF-8.
            RecordTooLargeError: If a single ``VARCHAR`` value exceeds the maximum
                single-page payload capacity (``8156`` UTF-8 bytes).

        Complexity:
            $O(1)$ for fixed-width scalars; ``O(len(value))`` for ``VARCHAR`` and
            strided vectors.
        """
        count = self.slot_count

        if self._is_varchar:
            raw = self._encode_varchar(value)
            max_payload = PAGE_SIZE - HEADER_SIZE - SLOT_ENTRY_SIZE
            if len(raw) > max_payload:
                raise RecordTooLargeError(
                    f"VARCHAR value of {len(raw)} UTF-8 bytes exceeds max single-page "
                    f"capacity {max_payload}"
                )
            if count >= self._capacity:
                return None
            slot_pos = HEADER_SIZE + count * SLOT_ENTRY_SIZE
            free_ptr = self.free_space_ptr
            if free_ptr - slot_pos < SLOT_ENTRY_SIZE + len(raw):
                return None

            new_offset = free_ptr - len(raw)
            if raw:
                self._buf[new_offset:free_ptr] = raw
            _SLOT_STRUCT.pack_into(self._buf, slot_pos, new_offset, len(raw))
            self._set_slot_count(count + 1)
            self._set_free_space_ptr(new_offset)
            return count

        if count >= self._capacity:
            return None

        offset = HEADER_SIZE + count * self._item_size
        if self._stride is not None:
            raw_vec = self._coerce_strided_bytes(value)
            self._buf[offset : offset + self._item_size] = raw_vec
        else:
            self._pack_fixed_scalar(value, offset)

        self._set_slot_count(count + 1)
        self._set_free_space_ptr(offset + self._item_size)
        return count

    def append_batch(self, values: Sequence[Any] | np.ndarray[Any, Any]) -> int:
        """Append as many values from ``values`` as fit in this page.

        Args:
            values: Ordered sequence or NumPy array of column values.

        Returns:
            Number of leading elements from ``values`` successfully appended.

        Raises:
            TypeError: If any appended element has an incompatible type.
            ValueError: If any appended element overflows or mismatches ``stride``.
            RecordTooLargeError: If an encountered ``VARCHAR`` string exceeds
                single-page capacity (``8156`` UTF-8 bytes).

        Complexity:
            ``O(k)`` for ``k`` appended scalars, or vectorized C-contiguous copy
            when ``values`` is a matching-dtype ``np.ndarray``.
        """
        if len(values) == 0:
            return 0

        if self._is_varchar:
            appended = 0
            for val in values:
                if self.append_value(val) is None:
                    break
                appended += 1
            return appended

        count = self.slot_count
        remaining = self._capacity - count
        if remaining <= 0:
            return 0

        take = min(len(values), remaining)
        start_offset = HEADER_SIZE + count * self._item_size

        if self._stride is not None:
            if (
                isinstance(values, np.ndarray)
                and values.ndim == 2
                and values.shape[1] * values.itemsize == self._stride
            ):
                chunk = np.ascontiguousarray(values[:take])
                nbytes = take * self._stride
                self._buf[start_offset : start_offset + nbytes] = memoryview(chunk).cast("B")
            else:
                cursor = start_offset
                for idx in range(take):
                    raw_vec = self._coerce_strided_bytes(values[idx])
                    self._buf[cursor : cursor + self._stride] = raw_vec
                    cursor += self._stride
        else:
            target_dtype = self._dtype.numpy_dtype
            if isinstance(values, np.ndarray) and values.dtype == target_dtype and values.ndim == 1:
                chunk = np.ascontiguousarray(values[:take])
                nbytes = take * self._item_size
                self._buf[start_offset : start_offset + nbytes] = memoryview(chunk).cast("B")
            else:
                cursor = start_offset
                for idx in range(take):
                    self._pack_fixed_scalar(values[idx], cursor)
                    cursor += self._item_size

        self._set_slot_count(count + take)
        self._set_free_space_ptr(start_offset + take * self._item_size)
        return take

    def get_value(self, slot_id: int) -> Any:
        """Read and decode the value stored at ``slot_id``.

        Args:
            slot_id: Zero-based row ordinal within this page in ``[0, slot_count)``.

        Returns:
            Decoded Python ``int`` or ``float`` for scalar columns, ``str`` for
            ``VARCHAR`` columns, or ``bytes`` of length ``stride`` for strided
            vector pages.

        Raises:
            IndexError: If ``slot_id`` is outside ``[0, slot_count)``.

        Complexity:
            $O(1)$ for fixed-width scalars; ``O(length)`` for ``VARCHAR`` and
            strided vectors.
        """
        count = self.slot_count
        if slot_id < 0 or slot_id >= count:
            raise IndexError(f"slot_id {slot_id} out of range [0, {count})")

        if self._is_varchar:
            offset, length = _SLOT_STRUCT.unpack_from(
                self._buf, HEADER_SIZE + slot_id * SLOT_ENTRY_SIZE
            )
            return bytes(self._buf[offset : offset + length]).decode("utf-8")

        offset = HEADER_SIZE + slot_id * self._item_size
        if self._stride is not None:
            return bytes(self._buf[offset : offset + self._stride])
        return _PACK_STRUCTS[self._dtype].unpack_from(self._buf, offset)[0]

    def as_numpy_array(
        self, dtype: np.dtype[Any] | str | type[Any] | None = None
    ) -> np.ndarray[Any, Any]:
        """Return a zero-copy NumPy view directly over the page's column buffer.

        Args:
            dtype: Optional NumPy dtype override. For scalar columns, defaults to
                ``self.dtype.numpy_dtype`` (itemsize must match). For strided
                vector pages, defaults to ``np.uint8`` returning shape
                ``(slot_count, stride)``, or reshapes to
                ``(slot_count, stride // np.dtype(dtype).itemsize)`` (e.g.,
                ``dtype=np.float32`` with ``stride=1536`` yields ``(slot_count, 384)``).

        Returns:
            1D (scalar) or 2D (strided) ``numpy.ndarray`` sharing memory directly
            with ``self.buffer`` (``arr.flags["OWNDATA"] is False``).

        Raises:
            StorageError: If called on a variable-length ``DataType.VARCHAR`` page.
            ValueError: If ``dtype`` itemsize mismatches the scalar width or does
                not evenly divide ``stride``.

        Complexity:
            $O(1)$ metadata wrapper; zero bytes copied.
        """
        if self._is_varchar:
            raise StorageError(
                "ColumnPage with DataType.VARCHAR does not support zero-copy NumPy views"
            )

        count = self.slot_count
        if self._stride is not None:
            target_dtype = np.dtype(np.uint8) if dtype is None else np.dtype(dtype)
            if self._stride % target_dtype.itemsize != 0:
                raise ValueError(
                    f"stride {self._stride} is not divisible by dtype itemsize "
                    f"{target_dtype.itemsize}"
                )
            dims = self._stride // target_dtype.itemsize
            flat = np.frombuffer(
                self._buf,
                dtype=target_dtype,
                count=count * dims,
                offset=HEADER_SIZE,
            )
            return flat.reshape((count, dims))

        target_dtype = self._dtype.numpy_dtype if dtype is None else np.dtype(dtype)
        if target_dtype.itemsize != self._item_size:
            raise ValueError(
                f"NumPy dtype {target_dtype!r} (itemsize={target_dtype.itemsize}) "
                f"mismatches column item_size={self._item_size}"
            )
        return np.frombuffer(
            self._buf,
            dtype=target_dtype,
            count=count,
            offset=HEADER_SIZE,
        )


__all__ = ["ColumnPage"]
