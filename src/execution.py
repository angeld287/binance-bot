"""Lambda entrypoint and trading orchestrator.

This module exposes two public callables:

``lambda_handler``
    AWS Lambda entry point that prepares the runtime context and
    delegates the actual trading logic to :func:`handle`.

``handle``
    Strategy orchestrator that selects the concrete strategy based on
    the ``STRATEGY_NAME`` environment variable and invokes its
    ``plan_entry`` and ``manage_exits`` methods.

Utility helpers that used to live in the legacy execution module (such
as :func:`compute_qty_from_usdt`) remain publicly exported to preserve
backwards compatibility for modules importing them from ``execution``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from core import config_loader, exchange, logging_utils
from core.logging_utils import log
from strategies import load_strategy
from strategies.common import compute_qty_from_usdt


def lambda_handler(event: Any, context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point.

    It bootstraps configuration, logging and the exchange client before
    delegating the trading logic to :func:`handle`.
    """

    log("═══════════════════ 🚀🚀🚀 INICIO EJECUCIÓN LAMBDA 🚀🚀🚀 ═══════════════════")

    cfg = config_loader.get_runtime_config()
    logging_utils.DEBUG_MODE = cfg.get("debug_mode", False)
    client = exchange.build(cfg)

    ctx = {
        "event": event,
        "context": context,
        "client": client,
        "config": cfg,
        "symbol": cfg.get("symbol"),
        "log": log,
    }

    plan = handle(ctx)

    log("═══════════════════ 🛑🛑🛑 FIN EJECUCIÓN LAMBDA 🛑🛑🛑 ═══════════════════")

    return {"statusCode": 200, "body": json.dumps({"plan": plan})}


def handle(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Main trading orchestrator.

    Parameters
    ----------
    ctx: dict
        Context object containing at least ``client`` and ``symbol``
        keys.  It is passed verbatim to the strategy methods.
    """

    strategy_name = os.getenv("STRATEGY_NAME", "breakout")
    strategy = load_strategy(strategy_name)

    plan = strategy.plan_entry(ctx)
    if plan:
        ctx["plan"] = plan

    strategy.manage_exits(ctx)

    return ctx.get("plan")


__all__ = ["lambda_handler", "handle", "compute_qty_from_usdt"]

