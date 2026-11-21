"""Predicate evaluation with SQL three-valued NULL and bool/int separation."""
from __future__ import annotations

import operator
from typing import Any

from .ast import Predicate
from ..core.schema import TableSchema

_CMP_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def matches_predicate(schema: TableSchema, record: dict[str, Any], pred: Predicate) -> bool:
    val = record.get(pred.field)
    if pred.op == "IS_NULL":
        return val is None
    if pred.op == "IS_NOT_NULL":
        return val is not None
    target = schema.coerce_literal(pred.field, pred.value)
    if pred.op == "=":
        if target is None:
            return val is None
        return val is not None and isinstance(val, bool) == isinstance(target, bool) and val == target
    if val is None or target is None or isinstance(val, bool) != isinstance(target, bool):
        return False
    try:
        if pred.op in _CMP_OPS:
            return _CMP_OPS[pred.op](val, target)
        if pred.op == "BETWEEN":
            high = schema.coerce_literal(pred.field, pred.high)
            return high is not None and isinstance(val, bool) == isinstance(high, bool) and target <= val <= high
    except TypeError:
        return False
    raise ValueError(f"Unsupported predicate operator: {pred.op}")
