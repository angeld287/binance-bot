from __future__ import annotations

from .breakout import BreakoutStrategy
from .breakout_dual_tf import BreakoutDualTFStrategy
from .liquidity_sweep import LiquiditySweepStrategy

STRATEGY_REGISTRY: dict[str, type] = {}
STRATEGY_REGISTRY["breakout"] = BreakoutStrategy
STRATEGY_REGISTRY["breakout_dual_tf"] = BreakoutDualTFStrategy
STRATEGY_REGISTRY["liquidity-sweep"] = LiquiditySweepStrategy

__all__ = [
    "STRATEGY_REGISTRY",
    "BreakoutStrategy",
    "BreakoutDualTFStrategy",
    "LiquiditySweepStrategy",
]
