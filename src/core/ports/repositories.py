"""Repository port definitions."""

from __future__ import annotations

from typing import Any, Protocol


class OrderRepository(Protocol):
    """Persists and retrieves order information."""

    def save(self, order: Any) -> None: ...

    def list(self, symbol: str): ...


class PositionRepository(Protocol):
    """Persists and retrieves position information."""

    def save(self, position: Any) -> None: ...

    def list(self, symbol: str): ...
