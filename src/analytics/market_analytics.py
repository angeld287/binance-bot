"""Market data enrichment for roundtrip records."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Callable

from zoneinfo import ZoneInfo

from .direction_classifier import EmaTrendThresholds, classify_trend
from .ema_utils import (
    compute_ema_map,
    find_candle_index_for_timestamp,
    parse_interval_to_ms,
)
from .mfe_utils import compute_mfe_mae

DEFAULT_TZ = "America/Santo_Domingo"
DEFAULT_EMA_TF = "15m"
DEFAULT_EMA_FAST = 7
DEFAULT_EMA_SLOW = 25
DEFAULT_K_SLOPE = 3
DEFAULT_TH_NEUTRO_FAST = 0.0005
DEFAULT_TH_NEUTRO_SLOW = 0.0002
DEFAULT_TH_FUERTE_FAST = 0.0020
DEFAULT_TH_FUERTE_SLOW = 0.0010
DEFAULT_TH_SPREAD_FUERTE = 0.0020


class _KlinesFetcher:
    def __init__(self, client: Any):
        self._client = client

    def fetch(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        limit: int = 1500,
    ) -> list[list[Any]]:
        """Fetch klines handling pagination if needed."""

        interval_ms = parse_interval_to_ms(interval)
        items: list[list[Any]] = []
        cursor = int(start_ms)
        while cursor < end_ms:
            batch = self._call_client(
                symbol=symbol,
                interval=interval,
                startTime=cursor,
                endTime=end_ms,
                limit=limit,
            )
            if not batch:
                break
            for candle in batch:
                open_time = int(candle[0])
                if open_time < start_ms:
                    continue
                if open_time > end_ms:
                    continue
                items.append(list(candle))
            last_open = int(batch[-1][0])
            cursor = last_open + interval_ms
            if last_open == batch[-1][0] and len(batch) < limit:
                break
            if cursor <= last_open:
                break
        return items

    def _call_client(self, **params) -> list[Any]:
        client = self._client
        for name in ("futures_klines", "get_klines", "klines"):
            if hasattr(client, name):
                method: Callable[..., Any] = getattr(client, name)
                return method(**params)
        raise AttributeError("Client does not expose a klines-compatible method")


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    elif isinstance(value, str):
        value_norm = value.strip()
        if value_norm.endswith("Z"):
            value_norm = value_norm[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(value_norm)
        except ValueError:
            dt = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    else:  # pragma: no cover - defensive
        raise TypeError("Unsupported datetime value")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


def _get_direction(roundtrip: dict[str, Any]) -> str:
    for key in ("direction", "positionSide", "side"):
        val = roundtrip.get(key)
        if not val:
            continue
        norm = str(val).upper()
        if norm in {"LONG", "BUY"}:
            return "LONG"
        if norm in {"SHORT", "SELL"}:
            return "SHORT"
    return "LONG"


def _read_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _read_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _calculate_slope(
    ema_values: list[tuple[int, float | None]],
    target_idx: int | None,
    k_slope: int,
) -> float | None:
    if target_idx is None or target_idx >= len(ema_values):
        return None
    if k_slope <= 0:
        return None
    prev_idx = target_idx - k_slope
    if prev_idx < 0:
        return None
    current = ema_values[target_idx][1]
    previous = ema_values[prev_idx][1]
    if current is None or previous is None or current == 0:
        return None
    return (current - previous) / (k_slope * current)


def _extract_spread(
    ema_fast: list[tuple[int, float | None]],
    ema_slow: list[tuple[int, float | None]],
    idx: int | None,
) -> float | None:
    if idx is None:
        return None
    if idx >= len(ema_fast) or idx >= len(ema_slow):
        return None
    fast_val = ema_fast[idx][1]
    slow_val = ema_slow[idx][1]
    if fast_val is None or slow_val is None or slow_val == 0:
        return None
    return (fast_val - slow_val) / slow_val


def enrich_roundtrip_with_market_data(
    roundtrip: dict[str, Any],
    *,
    client: Any,
    tz_name: str = DEFAULT_TZ,
) -> dict[str, Any]:
    """Enrich ``roundtrip`` with contextual market analytics."""

    enriched = dict(roundtrip)

    tz = ZoneInfo(tz_name)

    open_dt = _to_datetime(roundtrip.get("openAt"))
    close_dt = _to_datetime(roundtrip.get("closeAt"))
    if close_dt < open_dt:
        close_dt = open_dt

    duration_min = math.ceil((close_dt - open_dt).total_seconds() / 60)
    enriched["durationMin"] = int(max(duration_min, 0))

    roi_net = _to_float(roundtrip.get("roiNetPct"), default=_to_float(roundtrip.get("roiPct")))
    enriched["resultado"] = "GANADA" if roi_net >= 0 else "PERDIDA"

    fetcher = _KlinesFetcher(client)

    direction = _get_direction(roundtrip)
    entry_price = _to_float(roundtrip.get("entryPrice"))

    start_1m = int(open_dt.timestamp() * 1000) - 120_000
    end_1m = int(close_dt.timestamp() * 1000) + 120_000
    candles_1m = fetcher.fetch(roundtrip["symbol"], "1m", start_1m, end_1m)

    mfe_pct, mae_pct, mfe_ts = compute_mfe_mae(candles_1m, direction, entry_price)
    enriched["mfePct"] = round(mfe_pct, 6)
    enriched["maePct"] = round(mae_pct, 6)
    if mfe_ts is not None:
        mfe_dt = datetime.fromtimestamp(mfe_ts / 1000, tz=timezone.utc).astimezone(tz)
        enriched["mfeTs"] = mfe_dt.isoformat()

    ema_tf = os.getenv("EMA_TF", DEFAULT_EMA_TF)
    ema_interval_ms = parse_interval_to_ms(ema_tf)
    ema_fast_period = _read_env_int("EMA_FAST", DEFAULT_EMA_FAST)
    ema_slow_period = _read_env_int("EMA_SLOW", DEFAULT_EMA_SLOW)
    k_slope = _read_env_int("K_SLOPE", DEFAULT_K_SLOPE)

    lookback_candles = ema_slow_period + k_slope + 5
    start_ema = int(open_dt.timestamp() * 1000) - lookback_candles * ema_interval_ms
    end_ema = int(open_dt.timestamp() * 1000) + ema_interval_ms
    candles_ema = fetcher.fetch(roundtrip["symbol"], ema_tf, start_ema, end_ema)

    ema_fast = compute_ema_map(candles_ema, ema_fast_period)
    ema_slow = compute_ema_map(candles_ema, ema_slow_period)

    candle_idx = find_candle_index_for_timestamp(
        candles_ema, int(open_dt.timestamp() * 1000), ema_interval_ms
    )
    ema_fast_val = None
    if candle_idx is not None and candle_idx < len(ema_fast):
        ema_fast_val = ema_fast[candle_idx][1]

    enriched["ema7ValueAtOpen"] = float(ema_fast_val) if ema_fast_val is not None else None
    enriched["emaTf"] = ema_tf

    price_vs_ema = None
    if ema_fast_val is not None and entry_price > 0:
        price_vs_ema = "ENCIMA" if entry_price >= ema_fast_val else "DEBAJO"
    enriched["priceVsEma7Open"] = price_vs_ema

    slope_fast = _calculate_slope(ema_fast, candle_idx, k_slope)
    slope_slow = _calculate_slope(ema_slow, candle_idx, k_slope)
    spread = _extract_spread(ema_fast, ema_slow, candle_idx)

    thresholds = EmaTrendThresholds(
        neutral_fast=_read_env_float("TH_NEUTRO_FAST", DEFAULT_TH_NEUTRO_FAST),
        neutral_slow=_read_env_float("TH_NEUTRO_SLOW", DEFAULT_TH_NEUTRO_SLOW),
        strong_fast=_read_env_float("TH_FUERTE_FAST", DEFAULT_TH_FUERTE_FAST),
        strong_slow=_read_env_float("TH_FUERTE_SLOW", DEFAULT_TH_FUERTE_SLOW),
        strong_spread=_read_env_float("TH_SPREAD_FUERTE", DEFAULT_TH_SPREAD_FUERTE),
    )
    trend_class, trend_notes = classify_trend(slope_fast, slope_slow, spread, thresholds)
    enriched["emaTrendClassAtOpen"] = trend_class
    if trend_notes:
        enriched["emaTrendNotesAtOpen"] = trend_notes

    enriched["tz"] = tz.key

    return enriched
