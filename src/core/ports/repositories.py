from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from core.domain.models.Order import Order
    from core.domain.models.Position import Position


class OrderRepository(Protocol):
    def save(self, order: "Order") -> None:
        ...

    def get(self, id: str) -> "Order | None":
        ...


class PositionRepository(Protocol):
    def save(self, position: "Position") -> None:
        ...

    def get(self, id: str) -> "Position | None":
        ...
