"""Persistence repositories for external data stores."""

from __future__ import annotations

__all__ = [
    "DailyActivityStore",
    "ExecutionReportStore",
]

from .dynamo_store import DailyActivityStore, ExecutionReportStore
