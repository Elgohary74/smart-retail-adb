"""
Module: src/storage/cipher_hooks.py
Layer: Layer 1 — Pluggable Object Store Encryption Interface
Lock Domain: Stateless

Purpose:
    Defines the ``CipherPayload`` container and ``CipherHook`` abstract base
    class used by ``ObjectStore`` to encrypt and authenticate binary extents
    with per-object Associated Authenticated Data (AAD).

Key Invariants:
    1. Every concrete cipher returns a 12-byte ``nonce``, 16-byte ``tag``, and
       ``ciphertext`` bound to ``aad`` (object ID).
    2. Authentication failure during decryption must raise
       ``SecurityViolationError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CipherPayload:
    """Authenticated encryption envelope returned by ``CipherHook.encrypt``."""

    nonce: bytes
    tag: bytes
    ciphertext: bytes


class CipherHook(ABC):
    """Abstract encryption/authentication hook for ``ObjectStore`` extents."""

    @property
    @abstractmethod
    def algorithm_id(self) -> int:
        """Return the numeric algorithm identifier stored or tracked by the engine."""

    @abstractmethod
    def encrypt(self, plaintext: bytes, aad: bytes) -> CipherPayload:
        """Encrypt ``plaintext`` and authenticate ``aad``.

        Args:
            plaintext: Raw unencrypted object payload bytes.
            aad: Additional authenticated data (e.g., packed ``uint64`` OID).

        Returns:
            ``CipherPayload`` containing 12-byte nonce, 16-byte tag, and ciphertext.
        """

    @abstractmethod
    def decrypt(self, payload: CipherPayload, aad: bytes) -> bytes:
        """Decrypt and verify ``payload`` against ``aad``.

        Args:
            payload: ``CipherPayload`` read from ``objects.dat``.
            aad: Expected additional authenticated data matching encryption time.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            SecurityViolationError: If authentication tag verification fails.
        """


__all__ = ["CipherHook", "CipherPayload"]
