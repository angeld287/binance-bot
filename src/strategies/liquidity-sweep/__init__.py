from __future__ import annotations

from datetime import datetime
from typing import Any

from core.ports.strategy import Strategy


class LiquiditySweepStrategy(Strategy):
    """Placeholder strategy that currently produces no signals."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate_signal(self, now: datetime):  # type: ignore[override]
        import logging

        log = logging.getLogger("bot.strategy.liquidity")
        NL = "\n"
        supports: list[Any] = []
        resistances: list[Any] = []
        sup = max(supports, key=lambda x: x.score) if supports else None
        res = max(resistances, key=lambda x: x.score) if resistances else None
        log.info(
            "📊 Liquidity Sweep%s"
            "🛡️ Próximo soporte fuerte: %s (score %s)%s"
            "🧱 Próxima resistencia fuerte: %s (score %s)",
            NL,
            getattr(sup, "price", "N/A"), getattr(sup, "score", "N/A"), NL,
            getattr(res, "price", "N/A"), getattr(res, "score", "N/A"),
        )

        return None

__all__ = ["LiquiditySweepStrategy"]
