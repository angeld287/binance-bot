"""Binance broker adapter."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from binance.client import Client

from config.settings import Settings
from core.ports.broker import BrokerPort

logger = logging.getLogger(__name__)


def _clean_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


class BinanceBroker(BrokerPort):
    """Broker implementation using Binance Futures REST API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
            testnet=settings.PAPER_TRADING,
        )

    # ------------------------------------------------------------------
    # Orders
    def open_orders(self, symbol: str) -> list[Any]:
        try:
            return self._client.futures_get_open_orders(symbol=_clean_symbol(symbol))  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch open orders: %s", exc)
            raise

    def get_order(
        self,
        symbol: str,
        clientOrderId: str | None = None,
        orderId: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": _clean_symbol(symbol)}
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
                symbol=_clean_symbol(symbol),
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
        params: dict[str, Any] = {"symbol": _clean_symbol(symbol)}
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
                symbol=_clean_symbol(symbol),
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
        try:
            return self._client.futures_create_order(
                symbol=_clean_symbol(symbol),
                side=side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tpPrice,
                quantity=qty,
                reduceOnly=True,
                newClientOrderId=clientOrderId,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place TP order: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Helpers
    def get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        try:
            info = self._client.futures_exchange_info()
            sym = _clean_symbol(symbol)
            for s in info.get("symbols", []):
                if s.get("symbol") == sym:
                    return {f["filterType"]: f for f in s.get("filters", [])}
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch symbol filters: %s", exc)
            raise
        raise ValueError(f"Symbol {symbol} not found")

    def round_price_to_tick(self, symbol: str, px: float) -> float:
        filters = self.get_symbol_filters(symbol)
        tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
        return float((Decimal(str(px)) // tick) * tick)

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        filters = self.get_symbol_filters(symbol)
        step = Decimal(filters["LOT_SIZE"]["stepSize"])
        return float((Decimal(str(qty)) // step) * step)

    def get_available_balance_usdt(self) -> float:
        try:
            balances = self._client.futures_account_balance()
            for bal in balances:
                if bal.get("asset") == "USDT":
                    return float(bal.get("availableBalance", 0.0))
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch balance: %s", exc)
            raise
        raise RuntimeError("USDT balance not found")


def make_broker(settings: Settings) -> BrokerPort:
    """Factory for a :class:`BrokerPort` bound to Binance."""

    return BinanceBroker(settings)

