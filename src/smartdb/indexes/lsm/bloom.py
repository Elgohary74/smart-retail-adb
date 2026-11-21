"""Compact blake2b double-hashing Bloom filter."""
from __future__ import annotations

import hashlib
import struct
from typing import Iterator

from ...core.config import BLOOM_DEFAULT_BITS, BLOOM_NUM_HASHES

_HDR_STRUCT = "<II"
_HDR_SIZE = struct.calcsize(_HDR_STRUCT)


class BloomFilter:
    """Bitset Bloom filter for fast negative SSTable pruning."""

    def __init__(self, num_bits: int = BLOOM_DEFAULT_BITS, num_hashes: int = BLOOM_NUM_HASHES, bits: bytearray | None = None) -> None:
        self.num_bits = max(64, num_bits)
        self.num_hashes = max(1, num_hashes)
        expected_bytes = (self.num_bits + 7) // 8
        if bits is not None and len(bits) < expected_bytes:
            raise ValueError(f"BloomFilter bitset too short: expected {expected_bytes} bytes, got {len(bits)}")
        self.bits = bits if bits is not None else bytearray(expected_bytes)

    def _bit_positions(self, key_bytes: bytes) -> Iterator[int]:
        h1, h2 = struct.unpack("<QQ", hashlib.blake2b(key_bytes, digest_size=16).digest())
        step = h2 or 1
        for i in range(self.num_hashes):
            yield (h1 + i * step) % self.num_bits

    def add(self, key_bytes: bytes) -> None:
        """Record key_bytes in the Bloom filter."""
        for pos in self._bit_positions(key_bytes):
            self.bits[pos // 8] |= 1 << (pos % 8)

    def might_contain(self, key_bytes: bytes) -> bool:
        """Return False if key_bytes is definitely absent, True if possibly present."""
        return all(bool(self.bits[pos // 8] & (1 << (pos % 8))) for pos in self._bit_positions(key_bytes))

    def to_bytes(self) -> bytes:
        """Serialize filter header and bitset into bytes."""
        return struct.pack(_HDR_STRUCT, self.num_bits, self.num_hashes) + bytes(self.bits)

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview) -> "BloomFilter":
        """Deserialize a BloomFilter serialized by to_bytes()."""
        if len(raw) < _HDR_SIZE:
            raise ValueError("Corrupt BloomFilter header: too short")
        num_bits, num_hashes = struct.unpack_from(_HDR_STRUCT, raw, 0)
        if num_bits < 64 or num_hashes < 1:
            raise ValueError(f"Invalid BloomFilter parameters: num_bits={num_bits}, num_hashes={num_hashes}")
        return cls(num_bits=num_bits, num_hashes=num_hashes, bits=bytearray(raw[_HDR_SIZE:]))
