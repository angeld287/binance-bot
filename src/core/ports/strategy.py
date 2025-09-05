from __future__ import annotations

from datetime import datetime
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from core.domain.models.Signal import Signal


class Strategy(Protocol):
    """Interface for trading strategies."""

    def generate_signal(self, now: datetime) -> "Signal | None":
        """Generate a trading signal for the given time."""
        ...
