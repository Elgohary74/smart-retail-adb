"""
Module: src/common/exceptions.py
Layer: Layer 0 — Common Primitives & Exceptions
Lock Domain: Stateless

Purpose:
    Defines the standardized ``DatabaseError`` exception hierarchy used across
    storage, buffer management, indexing, object encryption, and query parsing.

Key Invariants:
    1. Every engine-specific error raised inside ``csai302_db`` inherits from
       ``DatabaseError`` so callers and tests can catch deterministic domain
       failures without catching generic Python exceptions.
"""


class DatabaseError(Exception):
    """Base exception for all ``csai302_db`` engine failures."""


class StorageError(DatabaseError):
    """Disk I/O, file boundary, or page allocation failure."""


class PageCorruptedError(StorageError):
    """Raised when page checksum validation or binary magic verification fails."""


class BufferPoolFullError(StorageError):
    """Raised on a cache miss when every frame in the buffer pool is pinned."""


class RecordTooLargeError(StorageError):
    """Raised when a serialized tuple exceeds single-page payload capacity."""


class IndexErrorBase(DatabaseError):
    """Base exception for B+ Tree and LSM Tree index failures."""


class DuplicateKeyError(IndexErrorBase):
    """Raised when inserting a duplicate key into a unique index."""


class KeyNotFoundError(IndexErrorBase):
    """Raised on strict lookup or deletion of a non-existent key."""


class QuerySyntaxError(DatabaseError):
    """Raised when parsing a malformed SQL-like query string."""


class SecurityViolationError(DatabaseError):
    """Raised on SQL injection attempts or AEAD authentication tag failure."""
