from __future__ import annotations

"""Binance market data adapter.

This module exposes a thin wrapper around the `python-binance` client that
satisfies the :class:`core.ports.market_data.MarketData` protocol. Only a small
subset of functionality is implemented as required by the breakout strategy.
"""

import logging
from typing import TYPE_CHECKING

from binance.client import Client

from config.settings import Settings
from core.ports.market_data import MarketData as MarketDataPort

if TYPE_CHECKING:  # pragma: no cover - domain models are not yet implemented
    from core.domain.models.Candle import Candle


logger = logging.getLogger(__name__)


class BinanceMarketData(MarketDataPort):
    """Market data provider backed by Binance REST endpoints."""

    def __init__(self, settings: Settings) -> None:
        """Create the underlying Binance client from the provided settings."""
        self._settings = settings
        self._client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
            testnet=settings.PAPER_TRADING,
        )

    def get_klines(self, symbol: str, interval: str, limit: int) -> list["Candle"]:
        """Return OHLC candles for a symbol.

        The response is currently the raw output from the Binance client. When
        domain models are introduced this should be mapped accordingly.
        """
        # TODO: replace with shared retry helper
        for attempt in range(3):
            try:
                return self._client.get_klines(symbol=symbol, interval=interval, limit=limit)  # type: ignore[return-value]
            except Exception as exc:  # pragma: no cover - network failures
                logger.warning(
                    "Binance get_klines failed (attempt %s/3): %s", attempt + 1, exc
                )
        raise RuntimeError("Failed to fetch klines after retries")

    def get_price(self, symbol: str) -> float:
        """Return the latest price for ``symbol``."""
        # TODO: replace with shared retry helper
        for attempt in range(3):
            try:
                ticker = self._client.get_symbol_ticker(symbol=symbol)
                return float(ticker["price"])
            except Exception as exc:  # pragma: no cover - network failures
                logger.warning(
                    "Binance get_price failed (attempt %s/3): %s", attempt + 1, exc
                )
        raise RuntimeError("Failed to fetch price after retries")


def make_market_data(settings: Settings) -> MarketDataPort:
    """Factory for a :class:`MarketDataPort` bound to Binance."""
    return BinanceMarketData(settings)
