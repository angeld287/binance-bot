from __future__ import annotations

from .breakout import BreakoutStrategy
from .breakout_dual_tf import BreakoutDualTFStrategy, factory as BreakoutDualTFFactory
from .liquidity_sweep import LiquiditySweepStrategy
from .wedge_formation import WedgeFormationStrategy
from .ParallelChannelFormation import ParallelChannelFormationStrategy

STRATEGY_REGISTRY: dict[str, type] = {}
STRATEGY_REGISTRY["breakout"] = BreakoutStrategy
STRATEGY_REGISTRY["breakout_dual_tf"] = BreakoutDualTFStrategy
STRATEGY_REGISTRY["liquidity-sweep"] = LiquiditySweepStrategy
STRATEGY_REGISTRY["wedge-formation"] = WedgeFormationStrategy
STRATEGY_REGISTRY["parallel-channel-formation"] = ParallelChannelFormationStrategy

__all__ = [
    "STRATEGY_REGISTRY",
    "BreakoutStrategy",
    "BreakoutDualTFStrategy",
    "BreakoutDualTFFactory",
    "LiquiditySweepStrategy",
    "WedgeFormationStrategy",
    "ParallelChannelFormationStrategy",
]
