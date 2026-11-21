"""Page-aware encryption/decryption hooks and optional AES-GCM helper."""
from __future__ import annotations

import os
from typing import Callable
from .page_file import PageBackend

PageTransform = Callable[[int, bytes], bytes]
RecordTransform = Callable[[bytes], bytes]


class TransformingBackend:
    """Wraps a PageBackend with paired (page_id, bytes) -> bytes transforms."""

    def __init__(
        self,
        backend: PageBackend,
        encrypt: PageTransform | None = None,
        decrypt: PageTransform | None = None,
        overhead_bytes: int = 0,
    ):
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


def make_aes_gcm_hooks(
    key: bytes,
) -> tuple[PageTransform, PageTransform, int, RecordTransform, RecordTransform]:
    """Return (page_enc, page_dec, 28, rec_enc, rec_dec) using AES-256-GCM.

    Requires the optional `cryptography` package. Each page stores a 12-byte
    random nonce + ciphertext + 16-byte GCM tag (28 bytes total overhead),
    authenticated with page_id as AAD.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    aes = AESGCM(key)
    overhead = 28  # 12B nonce + 16B tag

    def page_encrypt(page_id: int, plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        aad = page_id.to_bytes(8, "big")
        return nonce + aes.encrypt(nonce, plaintext, aad)

    def page_decrypt(page_id: int, raw: bytes) -> bytes:
        nonce, ct = raw[:12], raw[12:]
        aad = page_id.to_bytes(8, "big")
        return aes.decrypt(nonce, ct, aad)

    def record_encrypt(plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + aes.encrypt(nonce, plaintext, b"wal")

    def record_decrypt(raw: bytes) -> bytes:
        return aes.decrypt(raw[:12], raw[12:], b"wal")

    return page_encrypt, page_decrypt, overhead, record_encrypt, record_decrypt
