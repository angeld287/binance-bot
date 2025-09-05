from __future__ import annotations

from datetime import datetime
from typing import Any

from core.ports.strategy import Strategy


class LiquiditySweepStrategy(Strategy):
    """Placeholder strategy that currently produces no signals."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate_signal(self, now: datetime):  # type: ignore[override]
        return None

__all__ = ["LiquiditySweepStrategy"]
