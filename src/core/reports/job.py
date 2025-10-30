"""Resolve and execute the reporting job orchestrator."""

from __future__ import annotations

from datetime import datetime
from importlib import import_module
from typing import Any, Callable, Iterable, Tuple

Orchestrator = Callable[[dict | None, datetime | None], Any]


_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("core.reports.pipeline", "run"),
    ("core.reports.orchestrator", "run"),
    ("core.reports.jobs", "run"),
    ("core.reports.job_impl", "run"),
    ("exporters.daily_export", "run"),
    ("services.daily_export_handler", "run"),
)


def _iter_orchestrators() -> Iterable[tuple[Orchestrator, str]]:
    for module_name, attr_name in _CANDIDATES:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            continue
        orchestrator = getattr(module, attr_name, None)
        if callable(orchestrator):
            yield orchestrator, f"{module_name}.{attr_name}"


def _fallback_orchestrator(event_in: dict | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Fallback orchestrator used when no concrete implementation is present."""

    return {
        "status": "noop",
        "reason": "no_reports_orchestrator_found",
        "event": event_in or {},
        "timestamp": (now or datetime.utcnow()).isoformat(),
    }


def resolve_orchestrator() -> Tuple[Orchestrator, str]:
    """Return the first available reports orchestrator and its dotted path."""

    for orchestrator, path in _iter_orchestrators():
        return orchestrator, path
    return _fallback_orchestrator, "core.reports.job._fallback_orchestrator"


def run(event_in: dict | None = None, now: datetime | None = None):
    """Execute the resolved reporting job orchestrator."""

    orchestrator, _ = resolve_orchestrator()
    return orchestrator(event_in, now=now)
