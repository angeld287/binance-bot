from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from core.domain.models.Order import Order
    from core.domain.models.Position import Position


class Broker(Protocol):
    """Abstract broker used by strategies to interact with the market."""

    # Reading state
    def get_positions(self) -> list["Position"]:
        ...

    def get_open_orders(self) -> list["Order"]:
        ...

    # Order management
    def place_order(self, order: "Order") -> str:
        """Place a new order and return the created order id."""
        ...

    def cancel_order(self, id: str) -> None:
        ...

    # Configuration helpers
    def set_leverage(self, symbol: str, leverage: int) -> None:
        ...

    def set_margin_mode(self, symbol: str, mode: str) -> None:
        ...
