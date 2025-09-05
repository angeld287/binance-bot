"""Settings provider port."""

from __future__ import annotations

from typing import Any, Protocol


class SettingsProvider(Protocol):
    """Reads configuration values from some source."""

    def get(self, key: str, default: Any | None = None) -> Any: ...
