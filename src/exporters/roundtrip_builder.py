"""Helpers to build enriched roundtrip records before persistence."""

from __future__ import annotations

from typing import Any

from analytics.market_analytics import enrich_roundtrip_with_market_data


def enrich_before_persist(
    roundtrip: dict[str, Any],
    *,
    client: Any,
    tz_name: str | None = None,
) -> dict[str, Any]:
    """Return ``roundtrip`` enriched with market analytics.

    This helper keeps backwards compatibility for existing pipelines that expect a
    dictionary to be returned before writing to DynamoDB or CSV exporters.
    """

    kwargs = {"client": client}
    if tz_name:
        kwargs["tz_name"] = tz_name
    return enrich_roundtrip_with_market_data(roundtrip, **kwargs)
