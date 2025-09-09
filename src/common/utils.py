"""Helper functions shared across modules."""

from __future__ import annotations

import re

__all__ = ["sanitize_client_order_id"]


_CID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")


def sanitize_client_order_id(raw: str) -> str:
    """Sanitize ``raw`` to meet Binance client order id requirements.

    Any character outside ``[A-Za-z0-9_-]`` is replaced with ``-`` and the
    resulting string is truncated to 36 characters. The transformation is
    deterministic and does not generate new identifiers.
    """

    return _CID_SANITIZE_RE.sub("-", str(raw))[:36]
