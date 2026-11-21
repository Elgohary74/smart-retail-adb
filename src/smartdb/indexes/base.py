"""Shared index protocol and key-range boundary helpers."""
from __future__ import annotations

from typing import Any, Protocol

from ..core import RID


class IndexProtocol(Protocol):
    """Common interface shared by BPlusTree and LSMTree indexes."""

    def insert(self, key: Any, rid: RID) -> None: ...
    def delete(self, key: Any, rid: RID | None = None) -> Any: ...
    def search(self, key: Any) -> list[RID]: ...
    def range_search(
        self,
        low: Any | None = None,
        high: Any | None = None,
        include_low: bool = True,
        include_high: bool = True,
    ) -> list[tuple[Any, RID]]: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def destroy(self) -> None: ...
    def disk_bytes(self) -> int: ...


def key_in_range(
    key_bytes: bytes,
    low_b: bytes | None,
    high_b: bytes | None,
    include_low: bool,
    include_high: bool,
) -> bool:
    """Return True iff encoded key_bytes falls within the configured bounds."""
    if low_b is not None and (key_bytes < low_b or (not include_low and key_bytes == low_b)):
        return False
    if high_b is not None and (key_bytes > high_b or (not include_high and key_bytes == high_b)):
        return False
    return True
