"""Small educational B+ Tree with leaf-level values and range scans."""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class _Node:
    leaf: bool
    keys: list[Any] = field(default_factory=list)
    values: list[list[Any]] = field(default_factory=list)  # leaf: locators per key
    children: list["_Node"] = field(default_factory=list)
    next: "_Node | None" = None


class BPlusTree:
    def __init__(self, order: int = 16):
        if order < 3:
            raise ValueError("order must be at least 3")
        self.order = order
        self.root = _Node(leaf=True)
        self.size = 0  # number of distinct keys

    def insert(self, key: Any, value: Any) -> None:
        split = self._insert(self.root, key, value)
        if split is not None:
            separator, right = split
            self.root = _Node(leaf=False, keys=[separator], children=[self.root, right])

    def _insert(self, node: _Node, key: Any, value: Any) -> tuple[Any, _Node] | None:
        if node.leaf:
            pos = bisect_left(node.keys, key)
            if pos < len(node.keys) and node.keys[pos] == key:
                node.values[pos].append(value)
                return None
            node.keys.insert(pos, key)
            node.values.insert(pos, [value])
            self.size += 1
            if len(node.keys) < self.order:
                return None
            mid = len(node.keys) // 2
            right = _Node(leaf=True, keys=node.keys[mid:], values=node.values[mid:])
            node.keys, node.values = node.keys[:mid], node.values[:mid]
            right.next = node.next
            node.next = right
            return right.keys[0], right

        child_idx = bisect_right(node.keys, key)
        split = self._insert(node.children[child_idx], key, value)
        if split is None:
            return None
        separator, right = split
        node.keys.insert(child_idx, separator)
        node.children.insert(child_idx + 1, right)
        if len(node.keys) < self.order:
            return None
        mid = len(node.keys) // 2
        promoted = node.keys[mid]
        new_right = _Node(leaf=False, keys=node.keys[mid + 1:], children=node.children[mid + 1:])
        node.keys, node.children = node.keys[:mid], node.children[:mid + 1]
        return promoted, new_right

    def _find_leaf(self, key: Any) -> _Node:
        node = self.root
        while not node.leaf:
            node = node.children[bisect_right(node.keys, key)]
        return node

    def search(self, key: Any) -> list[Any]:
        leaf = self._find_leaf(key)
        pos = bisect_left(leaf.keys, key)
        return list(leaf.values[pos]) if pos < len(leaf.keys) and leaf.keys[pos] == key else []

    def range_search(self, low: Any | None = None, high: Any | None = None) -> list[tuple[Any, Any]]:
        if low is None:
            leaf = self.root
            while not leaf.leaf:
                leaf = leaf.children[0]
            pos = 0
        else:
            leaf = self._find_leaf(low)
            pos = bisect_left(leaf.keys, low)
        found: list[tuple[Any, Any]] = []
        while leaf is not None:
            while pos < len(leaf.keys):
                key = leaf.keys[pos]
                if high is not None and key > high:
                    return found
                found.extend((key, value) for value in leaf.values[pos])
                pos += 1
            leaf, pos = leaf.next, 0
        return found

    def __len__(self) -> int:
        return self.size
