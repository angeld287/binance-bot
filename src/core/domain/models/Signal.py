from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """Minimal trading signal produced by strategies."""

    action: str
    price: float
    time: datetime
