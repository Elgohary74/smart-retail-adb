"""Compact blake2b double-hashing Bloom filter."""
from __future__ import annotations

import hashlib
import struct

from ...core.config import BLOOM_DEFAULT_BITS, BLOOM_NUM_HASHES


class BloomFilter:
    """Bitset Bloom filter for fast negative SSTable pruning."""

    def __init__(self, num_bits: int = BLOOM_DEFAULT_BITS, num_hashes: int = BLOOM_NUM_HASHES, bits: bytearray | None = None):
        self.num_bits = max(64, num_bits)
        self.num_hashes = max(1, num_hashes)
        self.bits = bits if bits is not None else bytearray((self.num_bits + 7) // 8)

    def _hashes(self, key_bytes: bytes) -> tuple[int, int]:
        h1, h2 = struct.unpack("<QQ", hashlib.blake2b(key_bytes, digest_size=16).digest())
        return h1, h2 or 1

    def add(self, key_bytes: bytes) -> None:
        h1, h2 = self._hashes(key_bytes)
        for i in range(self.num_hashes):
            pos = (h1 + i * h2) % self.num_bits
            self.bits[pos // 8] |= 1 << (pos % 8)

    def might_contain(self, key_bytes: bytes) -> bool:
        h1, h2 = self._hashes(key_bytes)
        for i in range(self.num_hashes):
            pos = (h1 + i * h2) % self.num_bits
            if not (self.bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def to_bytes(self) -> bytes:
        return struct.pack("<II", self.num_bits, self.num_hashes) + bytes(self.bits)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "BloomFilter":
        num_bits, num_hashes = struct.unpack_from("<II", raw, 0)
        return cls(num_bits=num_bits, num_hashes=num_hashes, bits=bytearray(raw[8:]))
