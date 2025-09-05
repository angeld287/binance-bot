from __future__ import annotations

"""Binance broker adapter.

This module exposes a minimal wrapper for order and account management on
Binance Futures via the `python-binance` client. Only the methods required by
current strategies are implemented.
"""

import logging
from typing import TYPE_CHECKING, Any

from binance.client import Client

from config.settings import Settings
from core.ports.broker import Broker as BrokerPort

if TYPE_CHECKING:  # pragma: no cover - domain models are not yet implemented
    from core.domain.models.Order import Order
    from core.domain.models.Position import Position


logger = logging.getLogger(__name__)


class BinanceBroker(BrokerPort):
    """Broker implementation using Binance Futures REST API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
            testnet=settings.PAPER_TRADING,
        )

    def get_positions(self) -> list["Position"]:
        """Return current futures positions."""
        try:
            return self._client.futures_position_information()  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch positions: %s", exc)
            raise

    def get_open_orders(self) -> list["Order"]:
        """Return currently open orders for the configured symbol."""
        try:
            return self._client.futures_get_open_orders(symbol=self._settings.SYMBOL)  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch open orders: %s", exc)
            raise

    def place_order(self, order: dict[str, Any]) -> str:  # type: ignore[override]
        """Place a new futures order and return its id."""
        try:
            response = self._client.futures_create_order(**order)
            return str(response["orderId"])
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place order: %s", exc)
            raise

    def cancel_order(self, id: str) -> None:
        """Cancel an existing order by id."""
        try:
            self._client.futures_cancel_order(symbol=self._settings.SYMBOL, orderId=id)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to cancel order %s: %s", id, exc)
            raise

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set the desired leverage for ``symbol``."""
        try:
            self._client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to set leverage for %s: %s", symbol, exc)
            raise

    def set_margin_mode(self, symbol: str, mode: str) -> None:
        """Change the margin mode (cross/isolated)."""
        try:
            self._client.futures_change_margin_type(symbol=symbol, marginType=mode)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to set margin mode for %s: %s", symbol, exc)
            raise


def make_broker(settings: Settings) -> BrokerPort:
    """Factory for a :class:`BrokerPort` bound to Binance."""
    return BinanceBroker(settings)
