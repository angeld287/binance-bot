"""Binance market data adapter."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING
import requests

from binance.client import Client

from config.settings import Settings
from core.ports.market_data import MarketDataPort
from common.symbols import normalize_symbol

if TYPE_CHECKING:  # pragma: no cover - domain models are not yet implemented
    from core.domain.models.Candle import Candle


logger = logging.getLogger(__name__)


def _interval_to_minutes(interval: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440}
    return int(interval[:-1]) * units[interval[-1]]


def _calc_drift_ms(client: Client) -> int:
    now_ms = int(time.time() * 1000)
    try:  # pragma: no cover - network failures
        server_ms = client.futures_time().get("serverTime", now_ms)
    except Exception:  # pragma: no cover - network failures
        return 0
    return int(server_ms) - now_ms


class BinanceMarketData(MarketDataPort):
    """Market data provider backed by Binance REST endpoints."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
            testnet=settings.BINANCE_TESTNET,
        )

    def get_klines(self, symbol: str, interval: str, lookback_min: int) -> list["Candle"]:
        """Return OHLC candles for a symbol over ``lookback_min`` minutes."""

        sym = normalize_symbol(symbol)
        minutes = _interval_to_minutes(interval)
        limit = max(1, lookback_min // minutes)
        # TODO: replace with shared retry helper
        for attempt in range(3):
            try:
                return self._client.get_klines(symbol=sym, interval=interval, limit=limit)  # type: ignore[return-value]
            except Exception as exc:  # pragma: no cover - network failures
                logger.warning(
                    "Binance get_klines failed (attempt %s/3): %s", attempt + 1, exc
                )
        raise RuntimeError("Failed to fetch klines after retries")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Fetch OHLCV data using Binance klines endpoint.

        Parameters mirror :func:`ccxt`'s ``fetch_ohlcv``.  Returned candles are
        normalized to ``[ms, open, high, low, close, volume]``.
        """

        sym = normalize_symbol(symbol)
        for attempt in range(3):
            try:
                data = self._client.get_klines(symbol=sym, interval=timeframe, limit=limit)
                return [
                    [
                        float(k[0]),
                        float(k[1]),
                        float(k[2]),
                        float(k[3]),
                        float(k[4]),
                        float(k[5]),
                    ]
                    for k in data
                ]
            except Exception as exc:  # pragma: no cover - network failures
                logger.warning(
                    "Binance fetch_ohlcv failed (attempt %s/3): %s", attempt + 1, exc
                )
        raise RuntimeError("Failed to fetch klines after retries")

    def get_price(self, symbol: str) -> float:
        sym = normalize_symbol(symbol)
        for attempt in range(3):
            try:
                ticker = self._client.get_symbol_ticker(symbol=sym)
                return float(ticker["price"])
            except Exception as exc:  # pragma: no cover - network failures
                logger.warning(
                    "Binance get_price failed (attempt %s/3): %s", attempt + 1, exc
                )
        raise RuntimeError("Failed to fetch price after retries")

    def get_server_time_ms(self) -> int:
        try:
            with requests.Session() as session:
                session.trust_env = False
                session.proxies.clear()
                base_url = (
                    "https://testnet.binancefuture.com"
                    if self._settings.BINANCE_TESTNET
                    else "https://fapi.binance.com"
                )
                resp = session.get(f"{base_url}/fapi/v1/time", timeout=5)
                resp.raise_for_status()
                data = resp.json()
                return int(data["serverTime"])
        except Exception as exc:  # pragma: no cover - network failures
            raise RuntimeError(f"Failed to fetch Binance server time: {exc}") from exc


def make_market_data(settings: Settings) -> MarketDataPort:
    """Factory for a :class:`MarketDataPort` bound to Binance."""

    return BinanceMarketData(settings)

