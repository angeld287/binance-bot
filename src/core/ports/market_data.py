from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from core.domain.models.Candle import Candle


class MarketData(Protocol):
    """Data provider interface."""

    def get_klines(self, symbol: str, interval: str, limit: int) -> list["Candle"]:
        ...

    def get_price(self, symbol: str) -> float:
        ...
