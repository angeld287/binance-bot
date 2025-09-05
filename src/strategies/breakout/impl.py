from __future__ import annotations

"""Breakout trading strategy implementation.

This class wraps the previous breakout logic and exposes it through the
:class:`core.ports.strategy.Strategy` interface. Network or exchange access is
performed exclusively via the injected ports.
"""

from datetime import datetime
from typing import Any

from core.domain.models.Signal import Signal
from core.ports.broker import Broker as BrokerPort
from core.ports.market_data import MarketData as MarketDataPort
from core.ports.strategy import Strategy
from config.settings import Settings


class BreakoutStrategy(Strategy):
    """Simple breakout strategy based on the latest two candles."""

    def __init__(
        self,
        market_data: MarketDataPort,
        broker: BrokerPort,
        settings: Settings,
        repositories: Any | None = None,
    ) -> None:
        self._market_data = market_data
        self._broker = broker
        self._settings = settings
        self._repositories = repositories

    # ------------------------------------------------------------------
    def generate_signal(self, now: datetime) -> Signal | None:
        """Generate a trading signal at ``now``.

        The strategy compares the latest candle close with the previous
        candle's high and low. A breakout above the previous high produces a
        ``BUY`` signal; a breakdown below the previous low produces a ``SELL``
        signal. If neither condition is met, ``None`` is returned.
        """

        symbol = self._settings.SYMBOL
        interval = self._settings.INTERVAL

        candles = self._market_data.get_klines(symbol=symbol, interval=interval, limit=2)
        if len(candles) < 2:
            return None

        prev, last = candles[-2], candles[-1]
        prev_high = float(prev[2])
        prev_low = float(prev[3])
        last_close = float(last[4])

        action: str | None = None
        if last_close > prev_high:
            action = "BUY"
        elif last_close < prev_low:
            action = "SELL"

        if action is None:
            return None

        price = self._market_data.get_price(symbol)
        return Signal(action=action, price=price, time=now)
