from __future__ import annotations
from typing import Protocol, Optional, Dict, Any

class Strategy(Protocol):
    """Trading strategy contract.

    Each strategy must expose a ``name`` attribute and implement
    ``plan_entry`` and ``manage_exits``. The orchestrator (``execution``)
    will call these methods to obtain entry plans and to let the
    strategy handle exit logic (stop loss, take profit, trailing, etc.).
    """

    name: str

    def plan_entry(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return an entry plan or ``None``.

        The plan is a dictionary with the information required by the
        execution layer to actually send the entry order.  The strategy
        **must not** send orders by itself.
        """

    def manage_exits(self, ctx: Dict[str, Any]) -> None:
        """Handle exit logic for open positions belonging to this strategy."""

    def close_mode(self) -> str:
        """Return ``'default'`` or ``'explicit'`` depending on how exits are managed."""
