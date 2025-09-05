"""Core execution helpers.

Compatibility layer exposing the original breakout strategy symbols while the
project migrates to a hexagonal architecture.
"""

from __future__ import annotations

import sys
import types

from strategies import _run_iteration, create_bot
import strategies.breakout as breakout
from strategies.breakout import (
    compute_qty_from_usdt,
    floor_to_step,
    FuturesBot,
    config_por_moneda,
    IDEMPOTENCY_REGISTRY,
    ORDER_META_BY_CID,
    ORDER_META_BY_OID,
)

# Proxy mutable configuration values so tests can tweak them via this module.
_PROXIED_VARS = {
    "PENDING_TTL_MIN",
    "PENDING_MAX_GAP_BPS",
    "PENDING_GAP_ATR_MULT",
    "PENDING_USE_SR3",
    "PENDING_SR_BUFFER_BPS",
    "PENDING_CANCEL_CONFIRM_BARS",
}

for _name in _PROXIED_VARS:
    globals()[_name] = getattr(breakout, _name)


class _ExecModule(types.ModuleType):
    def __setattr__(self, name: str, value):  # type: ignore[override]
        if name in _PROXIED_VARS:
            setattr(breakout, name, value)
        types.ModuleType.__setattr__(self, name, value)


sys.modules[__name__].__class__ = _ExecModule


def run_iteration(exchange, cfg):
    """Wrapper preserving the original runtime behaviour."""
    symbol = cfg.get("symbol", "BTC/USDT")
    leverage = cfg.get("leverage")
    use_breakout_dynamic_stops = cfg.get("use_breakout_dynamic_stops", False)
    testnet = cfg.get("testnet", False)
    bot = create_bot(
        exchange,
        symbol,
        leverage=leverage,
        use_breakout_dynamic_stops=use_breakout_dynamic_stops,
    )
    return _run_iteration(exchange, bot, testnet, symbol, leverage)


__all__ = [
    "FuturesBot",
    "config_por_moneda",
    "IDEMPOTENCY_REGISTRY",
    "compute_qty_from_usdt",
    "floor_to_step",
    "ORDER_META_BY_CID",
    "ORDER_META_BY_OID",
    "run_iteration",
]
