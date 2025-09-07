"""Binance broker adapter."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict

from binance.client import Client
from requests import Session

from config.settings import Settings
from core.ports.broker import BrokerPort

logger = logging.getLogger(__name__)


def _to_binance_symbol(sym: str) -> str:
    return sym.replace("/", "")


class BinanceBroker(BrokerPort):
    """Broker implementation using Binance Futures REST API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Reuse an HTTP session to benefit from keep-alive and reduce latency
        self._session = Session()
        timeout = (
            settings.HTTP_TIMEOUT
            if hasattr(settings, "HTTP_TIMEOUT") and settings.HTTP_TIMEOUT
            else 30
        )
        requests_params = {"timeout": timeout}
        logger.warning("BINANCE_API_KEY=%r", settings.BINANCE_API_KEY)
        logger.warning("BINANCE_API_SECRET=%r", settings.BINANCE_API_SECRET)
        logger.warning("BINANCE_TESTNET=%r", settings.BINANCE_TESTNET)
        logger.warning("requests_params=%r", requests_params)

        self._client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
            testnet=settings.BINANCE_TESTNET,
            requests_params=requests_params,
        )

        logger.warning("self._client=%r", self._client.FUTURES_URL)
        self._client.session = getattr(self, "_session", None)
        # Cache for symbol filters to avoid repeated ``exchangeInfo`` calls
        self._filters_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Orders
    def open_orders(self, symbol: str) -> list[Any]:
        try:
            return self._client.futures_get_open_orders(symbol=_to_binance_symbol(symbol))  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch open orders: %s", exc)
            raise

    def get_order(
        self,
        symbol: str,
        clientOrderId: str | None = None,
        orderId: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": _to_binance_symbol(symbol)}
        if orderId is not None:
            params["orderId"] = orderId
        if clientOrderId is not None:
            params["origClientOrderId"] = clientOrderId
        try:
            return self._client.futures_get_order(**params)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch order: %s", exc)
            raise

    def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        clientOrderId: str,
        timeInForce: str = "GTC",
    ) -> dict[str, Any]:
        try:
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="LIMIT",
                price=price,
                quantity=qty,
                timeInForce=timeInForce,
                newClientOrderId=clientOrderId,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place limit order: %s", exc)
            raise

    def cancel_order(
        self,
        symbol: str,
        orderId: str | None = None,
        clientOrderId: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": _to_binance_symbol(symbol)}
        if orderId is not None:
            params["orderId"] = orderId
        if clientOrderId is not None:
            params["origClientOrderId"] = clientOrderId
        try:
            return self._client.futures_cancel_order(**params)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to cancel order: %s", exc)
            raise

    def place_sl_reduce_only(
        self,
        symbol: str,
        side: str,
        stopPrice: float,
        qty: float,
        clientOrderId: str,
    ) -> dict[str, Any]:
        try:
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="STOP_MARKET",
                stopPrice=stopPrice,
                quantity=qty,
                reduceOnly=True,
                newClientOrderId=clientOrderId,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place SL order: %s", exc)
            raise

    def place_tp_reduce_only(
        self,
        symbol: str,
        side: str,
        tpPrice: float,
        qty: float,
        clientOrderId: str,
    ) -> dict[str, Any]:
        """Place a take-profit limit reduce-only order.

        Binance does not expose a dedicated "TP-LIMIT" order type in the
        futures REST API, therefore we rely on a standard LIMIT order with the
        ``reduceOnly`` flag so the position size can only decrease.
        """

        try:
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="LIMIT",
                price=tpPrice,
                quantity=qty,
                timeInForce="GTC",
                reduceOnly=True,
                newClientOrderId=clientOrderId,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place TP order: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Helpers
    def get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Return and cache the exchange filters for ``symbol``.

        The first call populates a cache with the filters for all symbols to
        minimise subsequent HTTP requests.
        """

        sym = _to_binance_symbol(symbol)
        if not self._filters_cache:
            try:
                info = self._client.futures_exchange_info()
                for s in info.get("symbols", []):
                    self._filters_cache[s.get("symbol", "")] = {
                        f["filterType"]: f for f in s.get("filters", [])
                    }
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Failed to fetch symbol filters: %s", exc)
                raise
        try:
            return self._filters_cache[sym]
        except KeyError:
            raise ValueError(f"Symbol {symbol} not found") from None

    def round_price_to_tick(self, symbol: str, px: float) -> float:
        filters = self.get_symbol_filters(symbol)
        tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
        return float((Decimal(str(px)) // tick) * tick)

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        filters = self.get_symbol_filters(symbol)
        step = Decimal(filters["LOT_SIZE"]["stepSize"])
        return float((Decimal(str(qty)) // step) * step)

    def get_available_balance_usdt(self) -> float:
        """Return available USDT balance.

        If the call fails or the USDT asset is missing, the method falls back
        to ``settings.RISK_NOTIONAL_USDT`` (default ``0``).
        """

        try:
            balances = self._client.futures_account_balance()
            for bal in balances:
                if bal.get("asset") == "USDT":
                    return float(bal.get("availableBalance", 0.0))
        except Exception as exc:  # pragma: no cover - network failures
            logger.warning(
                "Failed to fetch balance, using RISK_NOTIONAL_USDT fallback: %s",
                exc,
            )
            return float(getattr(self._settings, "RISK_NOTIONAL_USDT", 0.0))

        logger.warning(
            "USDT balance not found, using RISK_NOTIONAL_USDT fallback",
        )
        return float(getattr(self._settings, "RISK_NOTIONAL_USDT", 0.0))


def make_broker(settings: Settings) -> BrokerPort:
    """Factory for a :class:`BrokerPort` bound to Binance."""

    return BinanceBroker(settings)

