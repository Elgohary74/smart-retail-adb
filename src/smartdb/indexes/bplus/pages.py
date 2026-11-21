"""Binary page codecs for B+ Tree meta, internal, and leaf nodes."""
from __future__ import annotations

import struct

from ...core import RID
from ...storage.pages import HEADER_SIZE, PageHeader, PageType, read_header, write_header

_META_STRUCT = "<iiii"
_RID_STRUCT = "<II"
_RID_SIZE = 8
LeafEntry = tuple[bytes, list[RID], int]


def read_meta(buf: bytearray) -> tuple[int, int, int, int]:
    return struct.unpack_from(_META_STRUCT, buf, HEADER_SIZE)


def write_meta(buf: bytearray, root_pid: int, first_leaf: int, size: int, max_keys: int) -> None:
    write_header(buf, PageHeader(int(PageType.BPLUS_META), 0, 1, HEADER_SIZE + 16, len(buf)))
    struct.pack_into(_META_STRUCT, buf, HEADER_SIZE, root_pid, first_leaf, size, max_keys)


def serialize_internal(keys: list[bytes], children: list[int]) -> bytes:
    parts = [struct.pack("<I", children[0])]
    for k, child in zip(keys, children[1:]):
        parts.append(struct.pack("<H", len(k)) + k + struct.pack("<I", child))
    return b"".join(parts)


def read_internal(buf: bytearray) -> tuple[list[bytes], list[int]]:
    hdr = read_header(buf)
    pos = HEADER_SIZE
    children = [struct.unpack_from("<I", buf, pos)[0]]
    pos += 4
    keys: list[bytes] = []
    for _ in range(hdr.item_count):
        klen = struct.unpack_from("<H", buf, pos)[0]
        pos += 2
        keys.append(bytes(buf[pos:pos + klen]))
        pos += klen
        children.append(struct.unpack_from("<I", buf, pos)[0])
        pos += 4
    return keys, children


def write_internal(buf: bytearray, keys: list[bytes], children: list[int]) -> None:
    payload = serialize_internal(keys, children)
    write_header(buf, PageHeader(int(PageType.BPLUS_INTERNAL), 0, len(keys), HEADER_SIZE + len(payload), len(buf)))
    buf[HEADER_SIZE:HEADER_SIZE + len(payload)] = payload


def serialize_leaf(entries: list[LeafEntry]) -> bytes:
    parts: list[bytes] = []
    for key_bytes, rids, bucket_pid in entries:
        parts.append(struct.pack("<HHi", len(key_bytes), len(rids), bucket_pid))
        parts.append(key_bytes)
        for pid, sid in rids:
            parts.append(struct.pack(_RID_STRUCT, pid, sid))
    return b"".join(parts)


def read_leaf(buf: bytearray) -> tuple[list[LeafEntry], PageHeader]:
    hdr = read_header(buf)
    pos = HEADER_SIZE
    entries: list[LeafEntry] = []
    for _ in range(hdr.item_count):
        klen, rcount, bucket_pid = struct.unpack_from("<HHi", buf, pos)
        pos += 8
        key_bytes = bytes(buf[pos:pos + klen])
        pos += klen
        rids = [struct.unpack_from(_RID_STRUCT, buf, pos + i * _RID_SIZE) for i in range(rcount)]
        pos += rcount * _RID_SIZE
        entries.append((key_bytes, rids, bucket_pid))
    return entries, hdr


def write_leaf(buf: bytearray, entries: list[LeafEntry], next_page: int, prev_page: int) -> None:
    payload = serialize_leaf(entries)
    write_header(
        buf,
        PageHeader(
            int(PageType.BPLUS_LEAF), 0, len(entries),
            HEADER_SIZE + len(payload), len(buf),
            next_page=next_page, prev_page=prev_page,
        ),
    )
    buf[HEADER_SIZE:HEADER_SIZE + len(payload)] = payload
