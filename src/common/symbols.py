from __future__ import annotations

__all__ = ["normalize_symbol"]


def normalize_symbol(symbol: str) -> str:
    """Return ``symbol`` uppercased without the ``/`` character."""
    return symbol.replace("/", "").upper()
