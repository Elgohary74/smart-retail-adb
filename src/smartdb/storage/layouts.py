"""JSON payload codecs for row-oriented and column-oriented page layouts."""
from __future__ import annotations
import json
import struct
from typing import Any, Iterable
from .page_file import PAGE_SIZE

_HEADER_SIZE = 4


def encode_records(records: list[dict[str, Any]], layout: str,
                   page_size: int = PAGE_SIZE) -> bytes:
    layout = layout.lower()
    if layout == "row":
        payload_obj = {"layout": "row", "records": records}
    elif layout == "column":
        keys = list(records[0].keys()) if records else []
        payload_obj = {
            "layout": "column",
            "columns": {key: [row.get(key) for row in records] for key in keys},
        }
    else:
        raise ValueError("layout must be 'row' or 'column'")
    payload = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    max_payload = page_size - _HEADER_SIZE
    if len(payload) > max_payload:
        raise OverflowError(f"Encoded records need {len(payload) + _HEADER_SIZE} bytes; page size is {page_size}")
    return struct.pack(">I", len(payload)) + payload + b"\x00" * (page_size - _HEADER_SIZE - len(payload))


def decode_records(page: bytes) -> list[dict[str, Any]]:
    if len(page) != PAGE_SIZE:
        raise ValueError(f"Expected a {PAGE_SIZE}-byte page")
    payload_len = struct.unpack(">I", page[:_HEADER_SIZE])[0]
    if payload_len == 0:
        return []
    if payload_len > len(page) - _HEADER_SIZE:
        raise ValueError("Corrupt page: invalid payload length")
    obj = json.loads(page[_HEADER_SIZE:_HEADER_SIZE + payload_len].decode("utf-8"))
    if obj.get("layout") == "row":
        return obj["records"]
    if obj.get("layout") == "column":
        columns = obj["columns"]
        if not columns:
            return []
        names = list(columns)
        count = len(columns[names[0]])
        if any(len(columns[n]) != count for n in names):
            raise ValueError("Corrupt columnar page: column lengths differ")
        return [{name: columns[name][i] for name in names} for i in range(count)]
    raise ValueError("Corrupt page: unknown layout")


def empty_page(page_size: int = PAGE_SIZE) -> bytes:
    return b"\x00" * page_size
