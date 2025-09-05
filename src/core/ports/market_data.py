"""Market data port definition."""

from __future__ import annotations

from typing import Protocol


class MarketDataPort(Protocol):
    """Provides price and candle information."""

    def get_klines(self, symbol: str, interval: str, limit: int): ...

    def get_price(self, symbol: str): ...
