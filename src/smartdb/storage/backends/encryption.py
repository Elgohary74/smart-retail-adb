"""Page-aware encryption/decryption hooks and optional AES-GCM helper."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from ...core.config import AES_GCM_OVERHEAD_BYTES
from .base import PageBackend

PageTransform = Callable[[int, bytes], bytes]
RecordTransform = Callable[[bytes], bytes]


@dataclass(frozen=True)
class CryptoHooks:
    """Paired page-level and optional record-level encryption/decryption hooks."""

    page_encrypt: PageTransform | None = None
    page_decrypt: PageTransform | None = None
    overhead_bytes: int = 0
    record_encrypt: RecordTransform | None = None
    record_decrypt: RecordTransform | None = None

    def __post_init__(self) -> None:
        if (self.page_encrypt is None) != (self.page_decrypt is None):
            raise ValueError("Provide both page_encrypt and page_decrypt, or neither")
        if (self.record_encrypt is None) != (self.record_decrypt is None):
            raise ValueError("Provide both record_encrypt and record_decrypt, or neither")


class TransformingBackend:
    """Wraps a PageBackend with paired (page_id, bytes) -> bytes transforms."""

    def __init__(
        self,
        backend: PageBackend,
        encrypt: PageTransform | None = None,
        decrypt: PageTransform | None = None,
        overhead_bytes: int = 0,
    ) -> None:
        if (encrypt is None) != (decrypt is None):
            raise ValueError("Provide both encrypt and decrypt hooks, or neither")
        if overhead_bytes < 0 or overhead_bytes >= backend.usable_page_size:
            raise ValueError("overhead_bytes must be in [0, usable_page_size)")
        self.backend = backend
        self.overhead_bytes = overhead_bytes
        self._encrypt: PageTransform = encrypt or (lambda _pid, data: data)
        self._decrypt: PageTransform = decrypt or (lambda _pid, data: data)

    @property
    def physical_page_size(self) -> int:
        return self.backend.physical_page_size

    @property
    def usable_page_size(self) -> int:
        return self.backend.usable_page_size - self.overhead_bytes

    @property
    def page_count(self) -> int:
        return self.backend.page_count

    def allocate_page(self) -> int:
        page_id = self.backend.allocate_page()
        transformed = self._encrypt(page_id, b"\x00" * self.usable_page_size)
        if len(transformed) != self.backend.usable_page_size:
            raise ValueError(f"Encryption hook must return {self.backend.usable_page_size} bytes")
        self.backend.write_page(page_id, transformed)
        return page_id

    def read_page(self, page_id: int) -> bytes:
        result = self._decrypt(page_id, self.backend.read_page(page_id))
        if len(result) != self.usable_page_size:
            raise ValueError(f"Decryption hook must return {self.usable_page_size} bytes")
        return result

    def write_page(self, page_id: int, data: bytes) -> None:
        if len(data) != self.usable_page_size:
            raise ValueError(f"Page must be exactly {self.usable_page_size} bytes; got {len(data)}")
        transformed = self._encrypt(page_id, bytes(data))
        if len(transformed) != self.backend.usable_page_size:
            raise ValueError(f"Encryption hook must return {self.backend.usable_page_size} bytes")
        self.backend.write_page(page_id, transformed)

    def close(self) -> None:
        self.backend.close()

    def destroy(self) -> None:
        self.backend.destroy()


def make_aes_gcm_crypto(key: bytes) -> CryptoHooks:
    """Build AES-256-GCM page and WAL record hooks (requires optional `cryptography`)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aes = AESGCM(key)

    def page_encrypt(page_id: int, plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + aes.encrypt(nonce, plaintext, page_id.to_bytes(8, "big"))

    def page_decrypt(page_id: int, raw: bytes) -> bytes:
        return aes.decrypt(raw[:12], raw[12:], page_id.to_bytes(8, "big"))

    def record_encrypt(plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + aes.encrypt(nonce, plaintext, b"wal")

    def record_decrypt(raw: bytes) -> bytes:
        return aes.decrypt(raw[:12], raw[12:], b"wal")

    return CryptoHooks(page_encrypt, page_decrypt, AES_GCM_OVERHEAD_BYTES, record_encrypt, record_decrypt)
