from __future__ import annotations

"""Utility helpers for configuration handling."""

from typing import Any


_TRUE_VALUES = {"true", "1", "yes", "y", "on"}
_FALSE_VALUES = {"false", "0", "no", "n", "off"}


def parse_bool(value: Any, *, default: bool = False) -> bool:
    """Return ``value`` coerced to ``bool``.

    Parameters
    ----------
    value:
        Input value that may be ``bool``-like. Accepts booleans, integers,
        and strings (case-insensitive) such as ``"true"``, ``"1"``,
        ``"yes"``, ``"on"``. Any non-zero integer is treated as ``True``.

    default:
        Value returned when ``value`` is ``None`` or cannot be interpreted.
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False

    return default

