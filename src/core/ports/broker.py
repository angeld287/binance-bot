"""Broker port definition."""

from __future__ import annotations

from typing import Any, Protocol


class BrokerPort(Protocol):
    """Abstracts trading operations against a broker or exchange."""

    def get_positions(self): ...

    def place_order(self, order: Any) -> dict: ...

    def cancel_order(self, order_id: str) -> None: ...
