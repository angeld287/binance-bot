"""AWS Lambda handler for the reports/export job."""

from __future__ import annotations

import logging
from datetime import datetime, time as time_cls, timedelta, timezone
from typing import Any

from zoneinfo import ZoneInfo

from common.symbols import normalize_symbol
from core.reports.job import resolve_orchestrator

logger = logging.getLogger(__name__)

_DEFAULT_TZ = ZoneInfo("America/Santo_Domingo")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _parse_symbols(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [normalize_symbol(part) for part in raw.split(",")]
        return [val for val in values if val]
    if isinstance(raw, (list, tuple, set)):
        result: list[str] = []
        for item in raw:
            if item is None:
                continue
            symbol = normalize_symbol(str(item))
            if symbol:
                result.append(symbol)
        return result
    return []


def _parse_datetime_value(value: Any, tz: ZoneInfo) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            dt = datetime.fromtimestamp(float(text) / 1000, tz=timezone.utc)
        else:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
    else:  # pragma: no cover - defensive
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _resolve_previous_day(now: datetime) -> tuple[datetime, datetime]:
    local_now = now.astimezone(_DEFAULT_TZ)
    target_day = (local_now - timedelta(days=1)).date()
    start = datetime.combine(target_day, time_cls(0, 0), tzinfo=_DEFAULT_TZ)
    end = datetime.combine(target_day, time_cls(23, 59, 59, 999000), tzinfo=_DEFAULT_TZ)
    return start, end


def _to_epoch_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in seq:
        if item in seen or not item:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _resolve_overrides(event: dict | None, *, now: datetime | None = None) -> dict[str, Any]:
    event = event or {}
    detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}

    origin = "eventbridge" if event.get("source") == "aws.events" else "manual"

    def _get_param(name: str) -> Any:
        if isinstance(detail, dict) and name in detail:
            return detail[name]
        return event.get(name)

    tz_now = now or datetime.now(tz=_DEFAULT_TZ)
    tz_now = tz_now if tz_now.tzinfo else tz_now.replace(tzinfo=_DEFAULT_TZ)
    from_value = _get_param("fromTs") or _get_param("from")
    to_value = _get_param("toTs") or _get_param("to")
    start_dt = _parse_datetime_value(from_value, _DEFAULT_TZ)
    end_dt = _parse_datetime_value(to_value, _DEFAULT_TZ)

    if start_dt is None and end_dt is None:
        start_dt, end_dt = _resolve_previous_day(tz_now)
    elif start_dt is None and end_dt is not None:
        day = end_dt.astimezone(_DEFAULT_TZ).date()
        start_dt = datetime.combine(day, time_cls(0, 0), tzinfo=_DEFAULT_TZ)
    elif end_dt is None and start_dt is not None:
        day = start_dt.astimezone(_DEFAULT_TZ).date()
        end_dt = datetime.combine(day, time_cls(23, 59, 59, 999000), tzinfo=_DEFAULT_TZ)

    assert start_dt is not None and end_dt is not None

    if end_dt < start_dt:
        end_dt = start_dt

    symbols = _parse_symbols(_get_param("symbols"))
    symbols = _dedupe(symbols)
    dry_run = _parse_bool(_get_param("dryRun"))

    overrides = {
        "origin": origin,
        "fromTs": _to_epoch_ms(start_dt),
        "toTs": _to_epoch_ms(end_dt),
        "fromIso": start_dt.isoformat(),
        "toIso": end_dt.isoformat(),
        "symbols": symbols,
        "dryRun": dry_run,
    }
    if detail:
        overrides["detail"] = detail
    for key in ("note", "id", "trigger"):
        if key in event:
            overrides[key] = event[key]
    return overrides


def handler(event=None, context=None):  # pragma: no cover - entry point
    """AWS Lambda entry point for the reporting job."""

    orchestrator, path = resolve_orchestrator()
    overrides = _resolve_overrides(event)
    logger.info(
        "reports.handler.start",
        extra={
            "path": path,
            "origin": overrides.get("origin"),
            "from": overrides.get("fromIso"),
            "to": overrides.get("toIso"),
            "symbols": overrides.get("symbols"),
            "dryRun": overrides.get("dryRun"),
        },
    )
    result = orchestrator(overrides, now=None)
    logger.info(
        "reports.handler.done",
        extra={
            "status": result.get("status") if isinstance(result, dict) else None,
            "processed": result.get("processed") if isinstance(result, dict) else None,
        },
    )
    return {"overrides": overrides, "result": result}
