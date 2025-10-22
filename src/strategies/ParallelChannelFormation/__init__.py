"""Parallel Channel Formation strategy package."""

from .channel_detector import (
    ParallelChannelFormationStrategy,
    STRATEGY_NAME,
    run,
)

__all__ = [
    "ParallelChannelFormationStrategy",
    "STRATEGY_NAME",
    "run",
]
