from __future__ import annotations

import re

__all__ = ["normalize_symbol"]


def normalize_symbol(symbol: str) -> str:
    """Return ``symbol`` uppercased without special characters.

    The function trims leading/trailing whitespace, uppercases the result and
    strips any character that is not an ASCII letter or digit. This makes the
    normalization idempotent and compatible with exchanges that expect symbols
    like ``BTCUSDT``.
    """

    return re.sub(r"[^A-Z0-9]", "", symbol.strip().upper())
