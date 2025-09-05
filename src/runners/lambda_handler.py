from __future__ import annotations

from core.application.execution import run_iteration


def handler(event=None, context=None):  # pragma: no cover - entry point
    return run_iteration()
