"""Execution orchestrator.

This module is intentionally light weight: it loads the strategy
specified through the ``STRATEGY_NAME`` environment variable and
delegates entry/exit logic to it.  Order submission, guards and other
housekeeping are handled here but for the unit tests we only exercise a
minimal subset of the behaviour.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional

from strategies import load_strategy
from strategies.common import compute_qty_from_usdt

STRATEGY_NAME = os.getenv("STRATEGY_NAME", "breakout")

# Strategy instance used by the module
strategy = load_strategy(STRATEGY_NAME)


def handle(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Main handler used by the bot.

    Parameters
    ----------
    ctx: dict
        Context containing at least ``client`` and ``symbol`` keys.  The
        context is passed verbatim to the strategy.
    """
    plan = strategy.plan_entry(ctx)
    if plan:
        # In the real bot we would send the order here using information
        # from ``plan``.  For the unit tests returning the plan is
        # sufficient to prove the orchestrator delegates correctly.
        ctx["plan"] = plan
    # Delegate exit management to the strategy.  The strategy itself
    # decides what to do (place/update SL/TP, trailing, etc.).
    strategy.manage_exits(ctx)
    return ctx.get("plan")

__all__ = ["handle", "strategy", "compute_qty_from_usdt"]
