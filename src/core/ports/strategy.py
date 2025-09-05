"""Strategy port definition."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class Strategy(Protocol):
    """Generates signals or actions based on injected data and state."""

    def generate_signal(self, now: datetime) -> Any: ...
