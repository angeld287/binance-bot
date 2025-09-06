from __future__ import annotations

"""Breakout trading strategy implementation.

This class wraps the previous breakout logic and exposes it through the
:class:`core.ports.strategy.Strategy` interface. Network or exchange access is
performed exclusively via the injected ports.
"""

from datetime import datetime
from typing import Any
import logging

from core.domain.models.Signal import Signal
from core.ports.broker import Broker as BrokerPort
from core.ports.market_data import MarketData as MarketDataPort
from core.ports.strategy import Strategy
from core.ports.settings import SettingsProvider, get_symbol

logger = logging.getLogger("bot.strategy.breakout")


class BreakoutStrategy(Strategy):
    """Simple breakout strategy based on the latest two candles."""

    def __init__(
        self,
        market_data: MarketDataPort,
        broker: BrokerPort,
        settings: SettingsProvider,
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

        symbol = get_symbol(self._settings)
        interval = self._settings.get("INTERVAL", "1h")

        candles = self._market_data.get_klines(symbol=symbol, interval=interval, limit=2)
        logger.debug("Candles fetched: %s", candles)
        if len(candles) < 2:
            logger.info("🔎 OBS: %s", "skip:not_enough_candles")
            return None

        prev, last = candles[-2], candles[-1]
        prev_high = float(prev[2])
        prev_low = float(prev[3])
        last_close = float(last[4])

        supports = [(prev_low, 1.0)]
        resistances = [(prev_high, 1.0)]
        supports_str = ", ".join([f"{p:.6f} (score {s:.2f})" for p, s in supports])
        resistances_str = ", ".join([f"{p:.6f} (score {s:.2f})" for p, s in resistances])
        logger.info("🛡️ Soportes estimados: %s", supports_str)
        logger.info("📚 Resistencias estimadas: %s", resistances_str)

        support_price, support_score = supports[0]
        resistance_price, resistance_score = resistances[0]
        support_dist = abs((last_close - support_price) / support_price * 100)
        resistance_dist = abs((resistance_price - last_close) / resistance_price * 100)
        logger.info(
            "🛡️ Próximo soporte: %.6f (score %.2f, dist≈%.2f%%) | razones: %s",
            support_price,
            support_score,
            support_dist,
            "prev_low",
        )
        logger.info(
            "🧱 Próxima resistencia: %.6f (score %.2f, dist≈%.2f%%) | razones: %s",
            resistance_price,
            resistance_score,
            resistance_dist,
            "prev_high",
        )

        action: str | None = None
        if last_close > prev_high:
            action = "BUY"
        elif last_close < prev_low:
            action = "SELL"

        if action is None:
            logger.info("🔎 OBS: %s", "no_signal")
            return None

        price = self._market_data.get_price(symbol)
        return Signal(action=action, price=price, time=now)
