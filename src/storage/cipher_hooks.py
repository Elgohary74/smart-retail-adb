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
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.common.exceptions import SecurityViolationError

_NONCE_SIZE: int = 12
_TAG_SIZE: int = 16
_AES256_KEY_SIZE: int = 32
_ZERO_NONCE: bytes = b"\x00" * _NONCE_SIZE
_ZERO_TAG: bytes = b"\x00" * _TAG_SIZE


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


def _validate_envelope_lengths(payload: CipherPayload) -> None:
    """Ensure ``nonce`` is 12 bytes and ``tag`` is 16 bytes before decryption."""
    if len(payload.nonce) != _NONCE_SIZE or len(payload.tag) != _TAG_SIZE:
        raise SecurityViolationError(
            f"Invalid AEAD envelope lengths: nonce={len(payload.nonce)}, tag={len(payload.tag)}"
        )


class NoOpCipher(CipherHook):
    """Unencrypted passthrough cipher hook (``algorithm_id = 0``) for baseline runs."""

    @property
    def algorithm_id(self) -> int:
        """Return ``0`` identifying the unencrypted passthrough cipher."""
        return 0

    def encrypt(self, plaintext: bytes, aad: bytes) -> CipherPayload:
        """Wrap ``plaintext`` in a zero-nonce, zero-tag ``CipherPayload``.

        Args:
            plaintext: Raw unencrypted object payload bytes.
            aad: Unused by ``NoOpCipher``; accepted for interface uniformity.

        Returns:
            ``CipherPayload`` with 12 zero bytes for ``nonce``, 16 zero bytes
            for ``tag``, and unmodified ``plaintext``.

        Complexity:
            $O(N)$ byte copy where $N$ is ``len(plaintext)``; stateless.
        """
        _ = aad
        return CipherPayload(
            nonce=_ZERO_NONCE,
            tag=_ZERO_TAG,
            ciphertext=bytes(plaintext),
        )

    def decrypt(self, payload: CipherPayload, aad: bytes) -> bytes:
        """Validate envelope lengths and return ``payload.ciphertext`` verbatim.

        Args:
            payload: ``CipherPayload`` read from ``objects.dat``.
            aad: Unused by ``NoOpCipher``; accepted for interface uniformity.

        Returns:
            Unmodified plaintext bytes from ``payload.ciphertext``.

        Raises:
            SecurityViolationError: If ``nonce`` is not 12 bytes or ``tag`` is
                not 16 bytes.

        Complexity:
            $O(N)$ byte copy where $N$ is ``len(payload.ciphertext)``; stateless.
        """
        _ = aad
        _validate_envelope_lengths(payload)
        return bytes(payload.ciphertext)


class AES256GCMHook(CipherHook):
    """Authenticated AES-256-GCM cipher hook (``algorithm_id = 1``) bound to AAD."""

    def __init__(self, key: bytes) -> None:
        """Initialize the AES-256-GCM hook with a 32-byte symmetric key.

        Args:
            key: 256-bit (32-byte) raw AES key material.

        Raises:
            ValueError: If ``key`` is not ``bytes``/``bytearray`` of length 32.
        """
        if not isinstance(key, (bytes, bytearray)) or len(key) != _AES256_KEY_SIZE:
            raise ValueError(
                f"AES-256-GCM requires a {_AES256_KEY_SIZE}-byte key"
            )
        self._key: bytes = bytes(key)
        self._aesgcm: AESGCM = AESGCM(self._key)

    @property
    def algorithm_id(self) -> int:
        """Return ``1`` identifying AES-256-GCM authenticated encryption."""
        return 1

    def encrypt(self, plaintext: bytes, aad: bytes) -> CipherPayload:
        """Encrypt ``plaintext`` with a fresh CSPRNG 96-bit nonce and ``aad``.

        Args:
            plaintext: Raw unencrypted object payload bytes.
            aad: Additional authenticated data (packed ``uint64`` OID) bound
                into the GCM authentication tag without being stored in plaintext.

        Returns:
            ``CipherPayload`` containing the 12-byte random nonce, 16-byte GCM
            tag, and ciphertext.

        Complexity:
            $O(N)$ AES-GCM encryption where $N$ is ``len(plaintext)``; thread-safe.
        """
        nonce = os.urandom(_NONCE_SIZE)
        # cryptography's AESGCM.encrypt appends the 16-byte tag directly after ciphertext.
        raw = self._aesgcm.encrypt(nonce, bytes(plaintext), bytes(aad))
        return CipherPayload(
            nonce=nonce,
            tag=raw[-_TAG_SIZE:],
            ciphertext=raw[:-_TAG_SIZE],
        )

    def decrypt(self, payload: CipherPayload, aad: bytes) -> bytes:
        """Verify GCM tag against ``aad`` and decrypt ``payload.ciphertext``.

        Args:
            payload: ``CipherPayload`` read from ``objects.dat``.
            aad: Expected additional authenticated data (packed ``uint64`` OID).

        Returns:
            Authenticated plaintext bytes.

        Raises:
            SecurityViolationError: If envelope lengths are invalid, ciphertext/tag
                was tampered with, or ``aad`` mismatches (e.g. cross-OID swap).

        Complexity:
            $O(N)$ AES-GCM verification and decryption; thread-safe.
        """
        _validate_envelope_lengths(payload)
        try:
            return self._aesgcm.decrypt(
                payload.nonce,
                payload.ciphertext + payload.tag,
                bytes(aad),
            )
        except (InvalidTag, ValueError, TypeError) as exc:
            raise SecurityViolationError(
                "AES-256-GCM authentication tag verification failed"
            ) from exc


__all__ = ["AES256GCMHook", "CipherHook", "CipherPayload", "NoOpCipher"]
