"""Helpers to persist and retrieve take-profit values from S3."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

import boto3
from botocore.exceptions import BotoCoreError, ClientError


logger = logging.getLogger("bot.tp_store_s3")

_S3_BUCKET = "trading-bot-storage-aa"
_S3_PREFIX = "tp"


def _build_key(symbol: str) -> str:
    symbol_clean = str(symbol or "").strip()
    if not symbol_clean:
        raise ValueError("Symbol required to build S3 key")
    return f"{_S3_PREFIX}/{symbol_clean}.json"


def persist_tp_value(
    symbol: str,
    tp_value: float,
    timestamp: int | float,
    extra: Mapping[str, Any] | None = None,
) -> bool:
    """Persist the take-profit value for ``symbol`` in S3."""

    key = _build_key(symbol)
    payload: dict[str, Any] = {
        "symbol": str(symbol),
        "tp_value": float(tp_value),
        "timestamp": timestamp,
    }
    if extra:
        for k, v in extra.items():
            payload[k] = v

    body = json.dumps(payload).encode("utf-8")
    client = boto3.client("s3")
    try:
        client.put_object(
            Bucket=_S3_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except (ClientError, BotoCoreError) as exc:
        logger.warning(
            "tp_s3_persist.error %s",
            {"bucket": _S3_BUCKET, "key": key, "error": str(exc)},
        )
        return False

    logger.info(
        "tp_s3_persist.success %s",
        {
            "bucket": _S3_BUCKET,
            "key": key,
            "tp_value": payload["tp_value"],
            "extra_keys": sorted(set(payload.keys()) - {"symbol", "tp_value", "timestamp"}),
        },
    )
    return True


def load_tp_value(symbol: str) -> float | None:
    """Load a previously persisted take-profit value for ``symbol`` from S3."""

    payload = load_tp_entry(symbol)
    if payload is None:
        return None

    value = payload.get("tp_value")
    if value is None:
        logger.info(
            "tp_s3_load.missing %s",
            {"bucket": _S3_BUCKET, "key": _build_key(symbol), "reason": "no_value"},
        )
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        logger.info(
            "tp_s3_load.missing %s",
            {
                "bucket": _S3_BUCKET,
                "key": _build_key(symbol),
                "reason": "invalid_value",
                "value": value,
            },
        )
        return None


def load_tp_entry(symbol: str) -> dict[str, Any] | None:
    """Load the raw TP entry payload for ``symbol`` from S3."""

    key = _build_key(symbol)
    client = boto3.client("s3")
    try:
        response = client.get_object(Bucket=_S3_BUCKET, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        if error_code in {"NoSuchKey", "404"}:
            logger.info(
                "tp_s3_load.missing %s",
                {"bucket": _S3_BUCKET, "key": key, "reason": "no_object"},
            )
            return None
        logger.warning(
            "tp_s3_load.error %s",
            {"bucket": _S3_BUCKET, "key": key, "error": str(exc)},
        )
        return None
    except BotoCoreError as exc:
        logger.warning(
            "tp_s3_load.error %s",
            {"bucket": _S3_BUCKET, "key": key, "error": str(exc)},
        )
        return None

    body = response.get("Body")
    if body is None:
        logger.info(
            "tp_s3_load.missing %s",
            {"bucket": _S3_BUCKET, "key": key, "reason": "empty_body"},
        )
        return None

    try:
        raw = body.read()
        data: Any = json.loads(raw.decode("utf-8"))
    except (ValueError, AttributeError) as exc:
        logger.warning(
            "tp_s3_load.error %s",
            {"bucket": _S3_BUCKET, "key": key, "error": str(exc)},
        )
        return None

    if not isinstance(data, dict):
        logger.info(
            "tp_s3_load.missing %s",
            {"bucket": _S3_BUCKET, "key": key, "reason": "invalid_payload"},
        )
        return None

    logger.info(
        "tp_s3_load.success %s",
        {"bucket": _S3_BUCKET, "key": key, "fields": sorted(data.keys())},
    )
    return data


__all__ = ["persist_tp_value", "load_tp_value", "load_tp_entry"]
