"""Strategies package.

Provides utilities to load concrete strategy implementations by name.
"""

from __future__ import annotations

from typing import Type

from .base import Strategy
from .breakout import BreakoutStrategy


def load_strategy(name: str | None) -> Strategy:
    """Instantiate a strategy by name.

    Parameters
    ----------
    name: str
        Strategy identifier.  At the moment only ``"breakout"`` is
        supported.  The comparison is case-insensitive.
    """
    name = (name or "breakout").lower()
    if name == "breakout":
        return BreakoutStrategy()
    raise ValueError(f"Estrategia no soportada: {name}")

__all__ = ["load_strategy", "BreakoutStrategy"]
