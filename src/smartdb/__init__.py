"""Smart Retail DB package."""
from __future__ import annotations

__all__ = ["SmartRetailDB", "PAGE_SIZE", "PageFile", "BufferPool", "BPlusTree", "LSMTree", "choose_index_type"]


def __getattr__(name: str):
    if name == "SmartRetailDB":
        from .engine import SmartRetailDB
        return SmartRetailDB
    if name in {"PAGE_SIZE", "PageFile", "BufferPool"}:
        from .storage import PAGE_SIZE, BufferPool, PageFile
        return {"PAGE_SIZE": PAGE_SIZE, "PageFile": PageFile, "BufferPool": BufferPool}[name]
    if name in {"BPlusTree", "LSMTree", "choose_index_type"}:
        from .indexes import BPlusTree, LSMTree, choose_index_type
        return {"BPlusTree": BPlusTree, "LSMTree": LSMTree, "choose_index_type": choose_index_type}[name]
    raise AttributeError(name)
