"""Public entry point for the :mod:`breakout_dual_tf` strategy package."""
from __future__ import annotations

from typing import Any

from .core import (
    BreakoutDualTFStrategy,
    BreakoutSignalPayload,
    CooldownEntry,
    Level,
    PendingBreakout,
    downscale_interval,
)

# NOTE: This module intentionally re-exports the strategy API for external users.


def factory(*args: Any, **kwargs: Any) -> BreakoutDualTFStrategy:
    """Factory helper that builds a :class:`BreakoutDualTFStrategy`."""
    return BreakoutDualTFStrategy(*args, **kwargs)


# Backwards compatible alias commonly used by tests and notebooks.
create_strategy = factory


def get_levels(
    strategy: BreakoutDualTFStrategy,
    *args: Any,
    **kwargs: Any,
) -> list[Level]:
    """Proxy to :meth:`BreakoutDualTFStrategy.get_levels`."""
    return strategy.get_levels(*args, **kwargs)


def should_trigger_breakout(
    strategy: BreakoutDualTFStrategy,
    *args: Any,
    **kwargs: Any,
) -> BreakoutSignalPayload | None:
    """Proxy to :meth:`BreakoutDualTFStrategy.should_trigger_breakout`."""
    return strategy.should_trigger_breakout(*args, **kwargs)


def compute_orders(
    strategy: BreakoutDualTFStrategy,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Proxy to :meth:`BreakoutDualTFStrategy.compute_orders`."""
    return strategy.compute_orders(*args, **kwargs)


__all__ = [
    "BreakoutDualTFStrategy",
    "BreakoutSignalPayload",
    "CooldownEntry",
    "Level",
    "PendingBreakout",
    "downscale_interval",
    "factory",
    "create_strategy",
    "get_levels",
    "should_trigger_breakout",
    "compute_orders",
]
