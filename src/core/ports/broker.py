"""Broker port definition."""

from __future__ import annotations

from typing import Any, Protocol


class BrokerPort(Protocol):
    """Abstract broker used by strategies to interact with the market."""

    # ------------------------------------------------------------------
    # Orders
    def open_orders(self, symbol: str) -> list[Any]:
        """Return the list of open orders for ``symbol``."""

        ...

    def get_order(
        self,
        symbol: str,
        clientOrderId: str | None = None,
        orderId: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a single order by id or client id."""

        ...

    def place_entry_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        clientOrderId: str,
        timeInForce: str = "GTC",
    ) -> dict[str, Any]:
        """Place a limit entry order."""

        ...

    def place_entry_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        clientOrderId: str,
    ) -> dict[str, Any]:
        """Place a market entry order."""

        ...

    def cancel_order(
        self,
        symbol: str,
        orderId: str | None = None,
        clientOrderId: str | None = None,
    ) -> dict[str, Any]:
        """Cancel an order by id or client id."""

        ...

    def place_stop_reduce_only(
        self,
        symbol: str,
        side: str,
        stopPrice: float,
        qty: float,
        clientOrderId: str,
    ) -> dict[str, Any]:
        """Place a stop-loss reduce-only order."""

        ...

    def place_tp_reduce_only(
        self,
        symbol: str,
        side: str,
        tpPrice: float,
        qty: float,
        clientOrderId: str,
    ) -> dict[str, Any]:
        """Place a take-profit reduce-only order."""

        ...

    # ------------------------------------------------------------------
    # Helpers
    def get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Return exchange filters for ``symbol``."""

        ...

    def round_price_to_tick(self, symbol: str, px: float) -> float:
        """Round ``px`` according to the tick size of ``symbol``."""

        ...

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        """Round ``qty`` according to the quantity step of ``symbol``."""

        ...

    def get_available_balance_usdt(self) -> float:
        """Return available USDT balance."""

        ...

