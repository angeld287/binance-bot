"""Compatibility wrappers for legacy imports.

Historically a number of utility functions lived in the ``execution``
module.  They have been moved under :mod:`strategies.common` but some
external modules may still import them from ``execution``.  Importing
from this module keeps those imports working without polluting the new
orchestrator.
"""

from strategies.common import (
    floor_to_step,
    compute_qty_from_usdt,
    get_position_mode,
    get_symbol_positions,
    has_open_position,
    list_open_orders,
    split_orders,
    cancel_orders,
    now_ms,
    sync_time,
    OFFSET_MS,
    notional_guard,
    exclusive_entry_guard,
    recv_window_const,
)

__all__ = [
    "floor_to_step",
    "compute_qty_from_usdt",
    "get_position_mode",
    "get_symbol_positions",
    "has_open_position",
    "list_open_orders",
    "split_orders",
    "cancel_orders",
    "now_ms",
    "sync_time",
    "OFFSET_MS",
    "notional_guard",
    "exclusive_entry_guard",
    "recv_window_const",
]
