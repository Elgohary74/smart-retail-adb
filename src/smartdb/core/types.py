"""Core database types shared across schemas, codecs, indexes, and tables."""
from __future__ import annotations

from enum import Enum

RID = tuple[int, int]


class DataType(str, Enum):
    """Supported column and index data types."""

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
