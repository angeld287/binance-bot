from __future__ import annotations

from typing import Any, Protocol


class SettingsProvider(Protocol):
    """Generic provider for configuration values."""

    def get(self, key: str, default: Any | None = None) -> Any:
        ...
