"""Helpers to clean up stale pending entry orders for the PCF strategy."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Mapping

from core.ports.broker import BrokerPort
from utils.tp_store_s3 import load_tp_entry, persist_tp_value

logger = logging.getLogger("bot.strategy.parallel_channel.pending")

DEFAULT_MAX_WAIT_CANDLES = 5
DEFAULT_MAX_DRIFT_PCT = 0.005


def _log(payload: Mapping[str, Any]) -> None:
    try:
        logger.info(json.dumps(payload, default=str))
    except Exception:  # pragma: no cover - defensive
        logger.info(str(payload))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if math.isnan(value):  # type: ignore[arg-type]
        return default
    return abs(value)


def _persist_payload(symbol: str, payload: Mapping[str, Any]) -> None:
    timestamp_raw = payload.get("timestamp")
    if timestamp_raw is None:
        timestamp_raw = datetime.utcnow().timestamp()
    try:
        timestamp_value = float(timestamp_raw)
    except (TypeError, ValueError):
        timestamp_value = datetime.utcnow().timestamp()

    tp_value_raw = payload.get("tp_value", 0.0)
    try:
        tp_value = float(tp_value_raw)
    except (TypeError, ValueError):
        tp_value = 0.0

    extra = {
        key: value
        for key, value in payload.items()
        if key not in {"symbol", "tp_value", "timestamp"}
    }

    persist_tp_value(symbol, tp_value, timestamp_value, extra=extra or None)


def _normalize_pending_payload(data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(data, Mapping):
        return None
    payload = dict(data)
    client_id = payload.get("client_order_id")
    if not client_id:
        return None
    try:
        payload["candle_index_created"] = int(payload.get("candle_index_created", 0))
    except (TypeError, ValueError):
        payload["candle_index_created"] = 0
    try:
        payload["limit_price"] = float(payload.get("limit_price", 0.0))
    except (TypeError, ValueError):
        return None
    return payload


def _clear_pending(symbol: str, store_payload: dict[str, Any], *, status: str | None = None, reason: str | None = None) -> None:
    updated = dict(store_payload)
    updated.pop("pending_order", None)
    if status:
        updated["status"] = status
    if status == "CANCELLED":
        updated["closed_at"] = datetime.utcnow().isoformat(timespec="seconds")
        if reason:
            updated["cancel_reason"] = reason
    _persist_payload(symbol, updated)


def build_pending_order_payload(
    *,
    client_order_id: str,
    side: str,
    limit_price: float,
    qty: float,
    timeframe: str | None,
    candle_index_created: int,
    created_at: str,
    order_id: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_order_id": client_order_id,
        "side": side,
        "limit_price": float(limit_price),
        "qty": float(qty),
        "timeframe": timeframe,
        "candle_index_created": int(candle_index_created),
        "created_at": created_at,
    }
    if order_id is not None:
        payload["order_id"] = order_id
    return payload


def sweep_stale_pending_orders(
    *,
    exchange: BrokerPort,
    symbol: str,
    current_price: float | None,
    current_candle_index: int | None,
    timeframe: str,
) -> None:
    if current_price is None or current_candle_index is None:
        return

    store_payload = load_tp_entry(symbol)
    if not isinstance(store_payload, Mapping):
        return

    pending_payload = _normalize_pending_payload(store_payload.get("pending_order"))
    if not pending_payload:
        return

    status_raw = store_payload.get("status")
    if status_raw and str(status_raw).strip().upper() not in {"OPEN", "PENDING", "NEW"}:
        return

    limit_price = pending_payload.get("limit_price")
    if not limit_price:
        return

    candle_created = int(pending_payload.get("candle_index_created", 0))
    age_candles = max(current_candle_index - candle_created, 0)
    drift_pct = abs(current_price - limit_price) / limit_price

    max_wait = _env_int("MAX_WAIT_CANDLES", DEFAULT_MAX_WAIT_CANDLES)
    max_drift = _env_float("MAX_DRIFT_PCT", DEFAULT_MAX_DRIFT_PCT)

    try:
        open_orders = exchange.open_orders(symbol)
    except Exception as exc:  # pragma: no cover - defensive
        _log(
            {
                "action": "pending_sweep_error",
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(exc),
            }
        )
        return

    client_id = str(pending_payload.get("client_order_id"))
    order_id = pending_payload.get("order_id")
    matching_order: Mapping[str, Any] | None = None
    for order in open_orders or []:
        if not isinstance(order, Mapping):
            continue
        status = str(order.get("status", "")).upper()
        if status and status not in {"NEW", "PARTIALLY_FILLED"}:
            continue
        client_order_id = str(order.get("clientOrderId", ""))
        if client_order_id and client_order_id.upper() == client_id.upper():
            matching_order = order
            break
        order_id_raw = order.get("orderId")
        if order_id is not None and order_id_raw is not None and str(order_id_raw) == str(order_id):
            matching_order = order
            break

    if matching_order is None:
        _clear_pending(symbol, dict(store_payload), status=None, reason="missing_on_exchange")
        return

    exchange_order_id = matching_order.get("orderId")
    if exchange_order_id is not None and exchange_order_id != order_id:
        pending_payload["order_id"] = exchange_order_id
        updated_payload = dict(store_payload)
        updated_payload["pending_order"] = dict(pending_payload)
        _persist_payload(symbol, updated_payload)

    if age_candles < max_wait or drift_pct < max_drift:
        return

    cancel_kwargs: dict[str, Any] = {"symbol": symbol}
    if exchange_order_id is not None:
        cancel_kwargs["orderId"] = exchange_order_id
    if client_id:
        cancel_kwargs["clientOrderId"] = client_id

    try:
        exchange.cancel_order(**cancel_kwargs)
    except Exception as exc:  # pragma: no cover - exchange failure
        _log(
            {
                "action": "cancel_pending_order_failed",
                "symbol": symbol,
                "side": pending_payload.get("side"),
                "limit_price": limit_price,
                "current_price": current_price,
                "age_candles": age_candles,
                "drift_pct": drift_pct,
                "error": str(exc),
            }
        )
        return

    _clear_pending(symbol, dict(store_payload), status="CANCELLED", reason="expired")

    _log(
        {
            "action": "cancel_pending_order",
            "symbol": symbol,
            "side": pending_payload.get("side"),
            "limit_price": limit_price,
            "current_price": current_price,
            "age_candles": age_candles,
            "drift_pct": drift_pct,
            "timeframe": timeframe,
            "reason": "expired",
        }
    )

