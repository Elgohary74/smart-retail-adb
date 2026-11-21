"""Provide paired same-length transforms supplied by an approved crypto library
or course-approved implementation. The default is identity transformation;
this class does NOT claim to encrypt data by itself.
"""
from __future__ import annotations
from typing import Callable
from .page_file import PageBackend

Transform = Callable[[bytes], bytes]


class TransformingBackend:
    def __init__(self, backend: PageBackend, encrypt: Transform | None = None,
                 decrypt: Transform | None = None):
        if (encrypt is None) != (decrypt is None):
            raise ValueError("Provide both encrypt and decrypt hooks, or neither")
        self.backend = backend
        self.page_size = backend.page_size
        self._encrypt = encrypt or (lambda data: data)
        self._decrypt = decrypt or (lambda data: data)

    @property
    def page_count(self) -> int:
        return self.backend.page_count

    def allocate_page(self) -> int:
        # Zero bytes remain zero after common reversible transforms; write the
        # transformed zero page explicitly so custom hooks are respected.
        page_id = self.backend.allocate_page()
        transformed = self._encrypt(b"\x00" * self.page_size)
        if len(transformed) != self.page_size:
            raise ValueError(f"Encryption hook must return exactly {self.page_size} bytes")
        self.backend.write_page(page_id, transformed)
        return page_id

    def read_page(self, page_id: int) -> bytes:
        result = self._decrypt(self.backend.read_page(page_id))
        if len(result) != self.page_size:
            raise ValueError(f"Decryption hook must return exactly {self.page_size} bytes")
        return result

    def write_page(self, page_id: int, data: bytes) -> None:
        if len(data) != self.page_size:
            raise ValueError(f"Page must be exactly {self.page_size} bytes")
        transformed = self._encrypt(bytes(data))
        if len(transformed) != self.page_size:
            raise ValueError(f"Encryption hook must return exactly {self.page_size} bytes")
        self.backend.write_page(page_id, transformed)

    def close(self) -> None:
        self.backend.close()
