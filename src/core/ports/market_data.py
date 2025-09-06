"""Market data port definition."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from core.domain.models.Candle import Candle


class MarketDataPort(Protocol):
    """Data provider interface."""

    def get_klines(self, symbol: str, interval: str, lookback_min: int) -> list["Candle"]:
        """Return OHLC candles for ``symbol`` within ``lookback_min`` minutes."""

        ...

    def get_price(self, symbol: str) -> float:
        ...

    def get_server_time_ms(self) -> int:
        ...

