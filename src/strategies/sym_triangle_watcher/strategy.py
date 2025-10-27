from __future__ import annotations

"""Symmetrical triangle watcher strategy.

This strategy does not place orders. It monitors BTCUSDT 15m candles to
identify symmetrical triangles and dispatches email notifications while the
pattern remains relevant.
"""

from datetime import datetime, timezone, timedelta
import json
import logging
import math
import os
from typing import Any, Iterable, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config.settings import load_settings
from config.utils import parse_bool
from adapters.data_providers.binance import make_market_data
from core.ports.market_data import MarketDataPort
from core.ports.settings import SettingsProvider

from utils import tp_store_s3


logger = logging.getLogger("bot.strategy.sym_triangle")


STRATEGY_NAME = "SymTriangleWatcher"


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


SYMBOL = _get_env("SYMBOL", "BTCUSDT").upper()
TIMEFRAME = _get_env("TIMEFRAME", "15m")
LOOKBACK_CANDLES = int(_get_env("LOOKBACK_CANDLES", "40"))
MIN_TOUCHES_PER_SIDE = int(_get_env("MIN_TOUCHES_PER_SIDE", "2"))
NARROWING_THRESHOLD = float(_get_env("NARROWING_THRESHOLD", "0.4"))
ALERT_MODE = _get_env("ALERT_MODE", "PRE_BREAKOUT").upper()
ALERT_REMIND_INTERVAL_MIN = int(_get_env("ALERT_REMIND_INTERVAL_MIN", "5"))
REMIND_WHILE_ACTIVE = parse_bool(_get_env("REMIND_WHILE_ACTIVE", "true"), default=True)
ALERT_EMAIL_TO = _get_env("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = _get_env("ALERT_EMAIL_FROM", "")
AWS_REGION = _get_env("AWS_REGION", "")


STATE_KEY = f"sym-triangle-watcher/{SYMBOL.lower()}-{TIMEFRAME}.json"


CONFIG: dict[str, Any] = {
    "symbol": SYMBOL,
    "timeframe": TIMEFRAME,
    "lookback_candles": LOOKBACK_CANDLES,
    "min_touches_per_side": MIN_TOUCHES_PER_SIDE,
    "narrowing_threshold": NARROWING_THRESHOLD,
    "alert_mode": ALERT_MODE,
    "alert_remind_interval_min": ALERT_REMIND_INTERVAL_MIN,
    "remind_while_active": REMIND_WHILE_ACTIVE,
    "alert_email_to": ALERT_EMAIL_TO,
    "alert_email_from": ALERT_EMAIL_FROM,
    "aws_region": AWS_REGION,
}


def _timeframe_to_minutes(timeframe: str) -> int:
    if not timeframe:
        raise ValueError("Timeframe requerido")
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Unidad de timeframe no soportada: {timeframe}")


TIMEFRAME_MINUTES = _timeframe_to_minutes(TIMEFRAME)
TIMEFRAME_MS = TIMEFRAME_MINUTES * 60_000


def _ms_to_iso(ms: int | float | None) -> str | None:
    if ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
    except (ValueError, TypeError):
        return None
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc, microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1]
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _linear_regression(points: Sequence[tuple[int, float]]) -> tuple[float, float] | None:
    n = len(points)
    if n < 2:
        return None
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    denominator = n * sum_xx - sum_x * sum_x
    if math.isclose(denominator, 0.0):
        return None
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _estimate_apex_ms(candles: Sequence[dict[str, Any]], x_intersect: float) -> int | None:
    if not candles:
        return None
    last_index = len(candles) - 1
    clamped_index = int(round(x_intersect))
    if 0 <= clamped_index <= last_index:
        return int(candles[clamped_index]["close_time"])
    if x_intersect < 0:
        reference = candles[0]["open_time"]
        offset = x_intersect * TIMEFRAME_MS
        return int(reference + offset)
    reference = candles[-1]["close_time"]
    delta_indices = x_intersect - last_index
    return int(reference + delta_indices * TIMEFRAME_MS)


def _normalize_candle(entry: Sequence[Any]) -> dict[str, float] | None:
    try:
        open_time = float(entry[0])
        open_price = float(entry[1])
        high = float(entry[2])
        low = float(entry[3])
        close = float(entry[4])
        volume = float(entry[5])
        close_time = float(entry[6]) if len(entry) > 6 else open_time + TIMEFRAME_MS
    except (TypeError, ValueError, IndexError):
        return None
    return {
        "open_time": open_time,
        "close_time": close_time,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def detect_sym_triangle(
    candles: Sequence[dict[str, float]], config: Mapping[str, Any]
) -> dict[str, Any]:
    default_result = {
        "found": False,
        "pattern_id": None,
        "upper_slope": None,
        "lower_slope": None,
        "apex_time_iso": None,
        "last_close": candles[-1]["close"] if candles else None,
        "width_start": None,
        "width_now": None,
        "status": "NONE",
    }

    if len(candles) < 3:
        return default_result

    min_touches = int(config.get("min_touches_per_side", 2))
    narrowing_threshold = float(config.get("narrowing_threshold", 0.4))

    upper_pivots: list[tuple[int, float]] = []
    lower_pivots: list[tuple[int, float]] = []

    for idx in range(1, len(candles) - 1):
        prev_c = candles[idx - 1]
        current = candles[idx]
        next_c = candles[idx + 1]
        if current["high"] > prev_c["high"] and current["high"] > next_c["high"]:
            upper_pivots.append((idx, current["high"]))
        if current["low"] < prev_c["low"] and current["low"] < next_c["low"]:
            lower_pivots.append((idx, current["low"]))

    if len(upper_pivots) < min_touches or len(lower_pivots) < min_touches:
        return default_result

    upper_reg = _linear_regression(upper_pivots)
    lower_reg = _linear_regression(lower_pivots)
    if not upper_reg or not lower_reg:
        return default_result

    slope_upper, intercept_upper = upper_reg
    slope_lower, intercept_lower = lower_reg
    if slope_upper >= 0 or slope_lower <= 0:
        return default_result

    x_start = min(upper_pivots[0][0], lower_pivots[0][0])
    x_now = len(candles) - 1
    upper_start = slope_upper * x_start + intercept_upper
    lower_start = slope_lower * x_start + intercept_lower
    upper_now = slope_upper * x_now + intercept_upper
    lower_now = slope_lower * x_now + intercept_lower

    width_start = upper_start - lower_start
    width_now = upper_now - lower_now

    if width_start <= 0:
        return default_result

    if width_now <= 0:
        result = dict(default_result)
        result.update(
            {
                "upper_slope": slope_upper,
                "lower_slope": slope_lower,
                "width_start": width_start,
                "width_now": width_now,
                "status": "INVALIDATED",
            }
        )
        return result

    narrowing_ratio = width_now / width_start
    if narrowing_ratio > narrowing_threshold:
        return default_result

    try:
        x_intersect = (intercept_lower - intercept_upper) / (slope_upper - slope_lower)
    except ZeroDivisionError:
        return default_result

    apex_ms = _estimate_apex_ms(candles, x_intersect)
    apex_time_iso = _ms_to_iso(apex_ms)
    pattern_id = (
        f"{config.get('symbol', SYMBOL)}-{config.get('timeframe', TIMEFRAME)}-{apex_time_iso}"
        if apex_time_iso
        else None
    )

    last_close = candles[-1]["close"]
    margin = width_now * 0.1

    status = "ACTIVE"
    if last_close > upper_now + margin:
        status = "BROKEN_UP"
    elif last_close < lower_now - margin:
        status = "BROKEN_DOWN"
    elif not (lower_now <= last_close <= upper_now):
        status = "INVALIDATED"

    found = status in {"ACTIVE", "BROKEN_UP", "BROKEN_DOWN"}

    result = {
        "found": found,
        "pattern_id": pattern_id,
        "upper_slope": slope_upper,
        "lower_slope": slope_lower,
        "apex_time_iso": apex_time_iso,
        "last_close": last_close,
        "width_start": width_start,
        "width_now": width_now,
        "status": status if found else ("INVALIDATED" if status == "INVALIDATED" else "NONE"),
    }
    return result


def process_alert_logic(
    tri: Mapping[str, Any], prev_state: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    alert_mode = str(config.get("alert_mode", "PRE_BREAKOUT")).upper()
    remind_while_active = bool(config.get("remind_while_active", True))
    remind_interval = int(config.get("alert_remind_interval_min", 5))

    status = str(tri.get("status") or "NONE")
    pattern_id = tri.get("pattern_id")
    prev_pattern_id = prev_state.get("pattern_id") if prev_state else None
    prev_status = prev_state.get("status") if prev_state else None
    now_iso = _now_iso()

    should_send = False
    new_state = dict(prev_state) if prev_state else {}
    email_subject: str | None = None
    email_body: str | None = None
    message = "Sin cambios"

    if status == "NONE":
        return {
            "should_send": False,
            "new_state": new_state,
            "email_subject": None,
            "email_body": None,
            "status": "NONE",
            "pattern_id": pattern_id,
            "message": "Sin patrón válido",
        }

    if status == "ACTIVE":
        new_state.update(
            {
                "pattern_id": pattern_id,
                "status": "ACTIVE",
            }
        )
        if prev_pattern_id != pattern_id:
            new_state["first_alert_ts"] = now_iso
            new_state["last_alert_ts"] = now_iso
            message = "Nuevo triángulo activo"
            if alert_mode == "PRE_BREAKOUT":
                should_send = True
        else:
            last_alert_ts = new_state.get("last_alert_ts")
            message = "Triángulo sigue activo"
            if (
                alert_mode == "PRE_BREAKOUT"
                and remind_while_active
                and last_alert_ts
            ):
                last_alert_dt = _parse_iso(last_alert_ts)
                if last_alert_dt is not None:
                    delta = datetime.utcnow().replace(tzinfo=timezone.utc) - last_alert_dt
                    if delta >= timedelta(minutes=remind_interval):
                        should_send = True
                        new_state["last_alert_ts"] = now_iso
                else:
                    should_send = True
                    new_state["last_alert_ts"] = now_iso
        if should_send:
            if "first_alert_ts" not in new_state:
                new_state["first_alert_ts"] = prev_state.get("first_alert_ts", now_iso)
            new_state.setdefault("last_alert_ts", now_iso)

    elif status in {"BROKEN_UP", "BROKEN_DOWN"}:
        new_state.update({"status": status})
        if pattern_id:
            new_state.setdefault("pattern_id", pattern_id)
        if alert_mode == status.replace("BROKEN_", "BREAKOUT_"):
            if prev_status != status:
                should_send = True
                new_state["last_alert_ts"] = now_iso
            message = "Triángulo rompió arriba" if status == "BROKEN_UP" else "Triángulo rompió abajo"
        else:
            message = "Ruptura detectada sin alertar (modo diferente)"

    elif status == "INVALIDATED":
        new_state.update({"status": "INVALIDATED"})
        if alert_mode == "PRE_BREAKOUT" and prev_status == "ACTIVE":
            should_send = True
            message = "Triángulo invalidado"
            new_state["last_alert_ts"] = now_iso
        else:
            message = "Triángulo invalidado sin alerta"
    else:
        message = f"Estado {status} sin acción"

    if should_send:
        subject_status = {
            "ACTIVE": "Triángulo ACTIVO",
            "BROKEN_UP": "Triángulo ROMPIÓ ARRIBA",
            "BROKEN_DOWN": "Triángulo ROMPIÓ ABAJO",
            "INVALIDATED": "Triángulo INVALIDADO",
        }.get(status, status)
        email_subject = f"[{CONFIG['symbol']} {CONFIG['timeframe']}] {subject_status}"
        lines = [
            f"Status: {status}",
            f"Pattern ID: {pattern_id}",
            f"Timeframe: {CONFIG['timeframe']}",
        ]
        if tri.get("last_close") is not None:
            lines.append(f"Last close: {tri['last_close']:.2f}")
        if tri.get("width_start") is not None and tri.get("width_now") is not None:
            lines.append(
                f"Width start: {tri['width_start']:.4f} -> now: {tri['width_now']:.4f}"
            )
        if tri.get("apex_time_iso"):
            lines.append(f"Apex ETA: {tri['apex_time_iso']}")
        if tri.get("upper_slope") is not None and tri.get("lower_slope") is not None:
            lines.append(
                f"Slopes: upper={tri['upper_slope']:.6f}, lower={tri['lower_slope']:.6f}"
            )
        lines.append(f"Timestamp: {now_iso}")
        email_body = "\n".join(lines)

    return {
        "should_send": should_send,
        "new_state": new_state,
        "email_subject": email_subject,
        "email_body": email_body,
        "status": status,
        "pattern_id": pattern_id,
        "message": message,
    }


def _load_state() -> dict[str, Any]:
    client = boto3.client("s3")
    try:
        response = client.get_object(Bucket=tp_store_s3._S3_BUCKET, Key=STATE_KEY)  # type: ignore[attr-defined]
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
        if error_code in {"NoSuchKey", "404"}:
            logger.info("sym_triangle.state.missing %s", {"key": STATE_KEY, "reason": "no_object"})
            return {}
        logger.warning(
            "sym_triangle.state.error %s",
            {"key": STATE_KEY, "error": str(exc)},
        )
        return {}
    except BotoCoreError as exc:
        logger.warning("sym_triangle.state.error %s", {"key": STATE_KEY, "error": str(exc)})
        return {}

    body = response.get("Body")
    if body is None:
        logger.info("sym_triangle.state.missing %s", {"key": STATE_KEY, "reason": "empty_body"})
        return {}
    try:
        raw = body.read()
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        logger.warning(
            "sym_triangle.state.error %s",
            {"key": STATE_KEY, "error": str(exc)},
        )
        return {}
    if not isinstance(data, dict):
        logger.info("sym_triangle.state.missing %s", {"key": STATE_KEY, "reason": "invalid_payload"})
        return {}
    return data


def _save_state(state: Mapping[str, Any]) -> None:
    payload = json.dumps(state).encode("utf-8")
    client = boto3.client("s3")
    try:
        client.put_object(
            Bucket=tp_store_s3._S3_BUCKET,  # type: ignore[attr-defined]
            Key=STATE_KEY,
            Body=payload,
            ContentType="application/json",
        )
    except (ClientError, BotoCoreError) as exc:
        logger.warning(
            "sym_triangle.state.persist_error %s",
            {"key": STATE_KEY, "error": str(exc)},
        )
        return
    logger.info(
        "sym_triangle.state.persist_success %s",
        {"key": STATE_KEY, "status": state.get("status"), "pattern_id": state.get("pattern_id")},
    )


def send_email_alert(subject: str, body: str) -> bool:
    if not ALERT_EMAIL_FROM or not ALERT_EMAIL_TO or not AWS_REGION:
        logger.warning(
            "sym_triangle.email.disabled %s",
            {"reason": "missing_config", "from": ALERT_EMAIL_FROM, "to": ALERT_EMAIL_TO},
        )
        return False
    client = boto3.client("ses", region_name=AWS_REGION)
    try:
        client.send_email(
            Source=ALERT_EMAIL_FROM,
            Destination={"ToAddresses": [ALERT_EMAIL_TO]},
            Message={
                "Subject": {"Charset": "UTF-8", "Data": subject},
                "Body": {"Text": {"Charset": "UTF-8", "Data": body}},
            },
        )
    except (ClientError, BotoCoreError) as exc:
        logger.warning(
            "sym_triangle.email.error %s",
            {"subject": subject, "error": str(exc)},
        )
        return False

    logger.info(
        "sym_triangle.email.sent %s",
        {"subject": subject, "to": ALERT_EMAIL_TO},
    )
    return True


def _fetch_candles(market_data: MarketDataPort) -> list[dict[str, float]]:
    lookback_min = LOOKBACK_CANDLES * TIMEFRAME_MINUTES
    raw_candles: Iterable[Sequence[Any]]
    if hasattr(market_data, "get_klines"):
        raw_candles = market_data.get_klines(
            symbol=SYMBOL,
            interval=TIMEFRAME,
            lookback_min=lookback_min,
        )
    elif hasattr(market_data, "fetch_ohlcv"):
        raw_candles = market_data.fetch_ohlcv(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            limit=LOOKBACK_CANDLES,
        )
    else:  # pragma: no cover - defensive
        raise RuntimeError("Proveedor de datos no soporta klines")

    candles: list[dict[str, float]] = []
    for entry in raw_candles:
        normalized = _normalize_candle(entry)
        if normalized:
            candles.append(normalized)
    candles.sort(key=lambda c: c["open_time"])
    return candles


def run_sym_triangle_watcher(
    *,
    market_data: MarketDataPort | None = None,
    settings: SettingsProvider | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    logger.info(
        "sym_triangle.start %s",
        {"symbol": SYMBOL, "timeframe": TIMEFRAME, "strategy": STRATEGY_NAME},
    )

    md = market_data
    settings_provider = settings
    created_market_data = False
    if md is None:
        settings_provider = settings_provider or load_settings()
        md = make_market_data(settings_provider)
        created_market_data = True

    try:
        candles = _fetch_candles(md)
    except Exception as exc:  # pragma: no cover - network/adapter failures
        logger.error(
            "sym_triangle.fetch.error %s",
            {"symbol": SYMBOL, "timeframe": TIMEFRAME, "error": str(exc)},
        )
        return {
            "strategy": STRATEGY_NAME,
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "status": "NONE",
            "alert_sent": False,
            "pattern_id": None,
            "message": f"Error obteniendo velas: {exc}",
        }
    finally:
        if created_market_data and hasattr(md, "close"):
            try:
                md.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    tri = detect_sym_triangle(candles, CONFIG)
    prev_state = _load_state()
    result = process_alert_logic(tri, prev_state, CONFIG)

    _save_state(result["new_state"])

    alert_sent = False
    if result["should_send"] and result.get("email_subject") and result.get("email_body"):
        alert_sent = send_email_alert(result["email_subject"], result["email_body"])
        if alert_sent:
            logger.info(
                "sym_triangle.alert.sent %s",
                {
                    "pattern_id": result.get("pattern_id"),
                    "status": result.get("status"),
                },
            )
        else:
            logger.warning(
                "sym_triangle.alert.failed %s",
                {
                    "pattern_id": result.get("pattern_id"),
                    "status": result.get("status"),
                },
            )
    else:
        logger.info(
            "sym_triangle.alert.skipped %s",
            {
                "pattern_id": result.get("pattern_id"),
                "status": result.get("status"),
                "reason": "no_send",
            },
        )

    return {
        "strategy": STRATEGY_NAME,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "status": result.get("status", "NONE"),
        "alert_sent": alert_sent,
        "pattern_id": result.get("pattern_id"),
        "message": result.get("message"),
    }


class SymTriangleWatcherStrategy:
    """AWS Lambda compatible wrapper for :func:`run_sym_triangle_watcher`."""

    def __init__(
        self,
        market_data: MarketDataPort,
        broker: Any,  # broker is unused but kept for signature compatibility
        settings: SettingsProvider,
    ) -> None:
        self._market_data = market_data
        self._settings = settings

    def run(
        self,
        exchange: Any | None = None,
        market_data: MarketDataPort | None = None,
        settings: SettingsProvider | None = None,
        now_utc: datetime | None = None,
        event: Any | None = None,
    ) -> dict[str, Any]:
        del exchange, event  # unused
        return run_sym_triangle_watcher(
            market_data=market_data or self._market_data,
            settings=settings or self._settings,
            now=now_utc,
        )


__all__ = [
    "SymTriangleWatcherStrategy",
    "STRATEGY_NAME",
    "run_sym_triangle_watcher",
    "detect_sym_triangle",
    "process_alert_logic",
]

