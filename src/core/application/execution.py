"""Application orchestrator leveraging the plugin registry."""

from __future__ import annotations

from typing import Any

from config.settings import settings
from plugins.registry import STRATEGIES
from strategies import _run_iteration


def run_iteration(exchange: Any, cfg: dict[str, Any]) -> Any:
    """Resolve dependencies and execute one trading iteration."""
    strategy_name = settings.strategy_name
    strategy_cls = STRATEGIES.get(strategy_name)
    if strategy_cls is None:
        raise ValueError(f"Strategy '{strategy_name}' not registered")

    symbol = cfg.get("symbol", "BTC/USDT")
    leverage = cfg.get("leverage")
    use_breakout_dynamic_stops = cfg.get("use_breakout_dynamic_stops", False)
    bot = strategy_cls(
        exchange,
        symbol,
        leverage=leverage,
        use_breakout_dynamic_stops=use_breakout_dynamic_stops,
    )
    testnet = cfg.get("testnet", False)
    return _run_iteration(exchange, bot, testnet, symbol, leverage)


__all__ = ["run_iteration"]
