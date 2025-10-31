"""DynamoDB-backed stores for bot persistence."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, MutableMapping

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import BotoCoreError, ClientError

try:  # pragma: no cover - ZoneInfo is available since Python 3.9
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


logger = logging.getLogger("bot.dynamo_store")

_DEFAULT_REGION = os.getenv("DDB_REGION")
_DAILY_ACTIVITY_TABLE = os.getenv("DDB_TABLE_DAILY_ACTIVITY", "")
_EXECUTION_REPORT_TABLE = os.getenv("DDB_TABLE_EXECUTION_REPORT", "")

_SERIALIZER = TypeSerializer()
_DESERIALIZER = TypeDeserializer()


def _normalize_number(value: int | float | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("Boolean values are not supported as numeric fields")
    return Decimal(str(value))


def _pythonify(value: Any) -> Any:
    if isinstance(value, list):
        return [_pythonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _pythonify(v) for k, v in value.items()}
    if isinstance(value, set):
        return {_pythonify(v) for v in value}
    if isinstance(value, tuple):
        return tuple(_pythonify(v) for v in value)
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    return value


def _coerce_epoch_millis(value: int | float | Decimal) -> int:
    if isinstance(value, bool):
        raise TypeError("Boolean values cannot be coerced into epoch millis")
    if isinstance(value, Decimal):
        return int(value)
    return int(_normalize_number(value))


class DailyActivityStore:
    """Persistence helper for daily activity operations.

    Example usage::

        exec_store = ExecutionReportStore()
        exec_store.start_run(run_id, now_ts_ms, "America/Santo_Domingo", {"symbols": ["BTCUSDT"], "from": 1, "to": 2})
        exec_store.inc_metrics(run_id, {"opsAnalyzed": +5, "apiWeightsUsed": +10})
        exec_store.finalize_run(run_id, end_ts_ms, {"opsClosed": 1, "opsOpen": 0}, "SUCCESS", None)

        ops_store = DailyActivityStore()
        ops_store.put_roundtrip({"PK": "...", "SK": "...", "symbol": "BTCUSDT"})
        ops_store.upsert_open({"PK": "...", "SK": "...", "symbol": "BTCUSDT"}, None)
        ops_store.close_open_and_put_roundtrip({"PK": "...", "SK": "..."}, {"PK": "...", "SK": "..."})
    """

    def __init__(
        self,
        table_name: str | None = None,
        *,
        dynamo_client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        self._table_name = table_name or _DAILY_ACTIVITY_TABLE
        if not self._table_name:
            raise ValueError("Daily activity table name must be provided via argument or DDB_TABLE_DAILY_ACTIVITY env var")

        self._client = dynamo_client or boto3.client(
            "dynamodb", region_name=region_name or _DEFAULT_REGION
        )

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: DailyActivityStore._normalize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DailyActivityStore._normalize_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(DailyActivityStore._normalize_value(v) for v in value)
        if isinstance(value, set):
            return {DailyActivityStore._normalize_value(v) for v in value}
        if isinstance(value, bool):
            return value
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return _normalize_number(value)
        return value

    def _serialize(self, item: Mapping[str, Any]) -> dict[str, Any]:
        return {k: self._normalize_value(v) for k, v in item.items()}

    def put_roundtrip(self, rt: Mapping[str, Any]) -> None:
        item = self._serialize(rt)
        try:
            self._client.put_item(
                TableName=self._table_name,
                Item={k: _SERIALIZER.serialize(v) for k, v in item.items()},
                ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
            )
        except ClientError:
            logger.warning("daily_activity.put_roundtrip.failure", extra={"keys": {"PK": rt.get("PK"), "SK": rt.get("SK")}})
            raise
        logger.info("daily_activity.put_roundtrip.success", extra={"keys": {"PK": rt.get("PK"), "SK": rt.get("SK")}})

    def upsert_open(self, open_item: MutableMapping[str, Any], prev_version: int | None) -> None:
        if "PK" not in open_item or "SK" not in open_item:
            raise ValueError("open_item must include PK and SK")
        now_ms = int(time.time() * 1000)
        expr_names: dict[str, str] = {"#version": "version", "#last": "lastUpdateTs"}
        expr_values: dict[str, Any] = {
            ":now": _SERIALIZER.serialize(Decimal(now_ms))
        }
        set_parts: list[str] = []
        for key, value in open_item.items():
            if key in {"PK", "SK", "version", "lastUpdateTs"}:
                continue
            normalized = self._normalize_value(value)
            expr_name = f"#attr_{key}"
            expr_value = f":val_{key}"
            expr_names[expr_name] = key
            expr_values[expr_value] = _SERIALIZER.serialize(normalized)
            set_parts.append(f"{expr_name} = {expr_value}")
        set_parts.append("#last = :now")
        if prev_version is None:
            expr_values[":initial_version"] = _SERIALIZER.serialize(Decimal(1))
            set_parts.append("#version = :initial_version")
            condition = "attribute_not_exists(PK) AND attribute_not_exists(SK)"
        else:
            expr_values[":prev_version"] = _SERIALIZER.serialize(Decimal(prev_version))
            expr_values[":inc_one"] = _SERIALIZER.serialize(Decimal(1))
            set_parts.append("#version = #version + :inc_one")
            condition = "attribute_exists(PK) AND attribute_exists(SK) AND #version = :prev_version"
        update_expression = "SET " + ", ".join(set_parts)
        try:
            self._client.update_item(
                TableName=self._table_name,
                Key={
                    "PK": _SERIALIZER.serialize(self._normalize_value(open_item["PK"])),
                    "SK": _SERIALIZER.serialize(self._normalize_value(open_item["SK"])),
                },
                UpdateExpression=update_expression,
                ConditionExpression=condition,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
            )
        except ClientError:
            logger.warning(
                "daily_activity.upsert_open.failure",
                extra={"keys": {"PK": open_item.get("PK"), "SK": open_item.get("SK")}},
            )
            raise
        logger.info(
            "daily_activity.upsert_open.success",
            extra={"keys": {"PK": open_item.get("PK"), "SK": open_item.get("SK")}},
        )

    def close_open_and_put_roundtrip(self, open_keys: Mapping[str, Any], rt: Mapping[str, Any]) -> None:
        if "PK" not in open_keys or "SK" not in open_keys:
            raise ValueError("open_keys must include PK and SK")
        serialized_keys = {
            key: _SERIALIZER.serialize(self._normalize_value(value))
            for key, value in open_keys.items()
        }
        rt_item = self._serialize(rt)
        try:
            self._client.transact_write_items(
                TransactItems=[
                    {
                        "Delete": {
                            "TableName": self._table_name,
                            "Key": serialized_keys,
                            "ConditionExpression": "attribute_exists(PK) AND attribute_exists(SK)",
                        }
                    },
                    {
                        "Put": {
                            "TableName": self._table_name,
                            "Item": {k: _SERIALIZER.serialize(v) for k, v in rt_item.items()},
                            "ConditionExpression": "attribute_not_exists(PK) AND attribute_not_exists(SK)",
                        }
                    },
                ]
            )
        except (ClientError, BotoCoreError):
            logger.warning(
                "daily_activity.close_and_put.failure",
                extra={
                    "open_keys": {"PK": open_keys.get("PK"), "SK": open_keys.get("SK")},
                    "roundtrip_keys": {"PK": rt.get("PK"), "SK": rt.get("SK")},
                },
            )
            raise
        logger.info(
            "daily_activity.close_and_put.success",
            extra={
                "open_keys": {"PK": open_keys.get("PK"), "SK": open_keys.get("SK")},
                "roundtrip_keys": {"PK": rt.get("PK"), "SK": rt.get("SK")},
            },
        )


@dataclass
class _RunKey:
    pk: str
    sk: str

    def as_key(self) -> dict[str, str]:
        return {"PK": self.pk, "SK": self.sk}


class ExecutionReportStore:
    """Store that keeps track of execution reports for report/export jobs.

    All timestamps are normalized to epoch milliseconds when numeric values are
    provided; ISO strings are persisted untouched. Example usage::

        exec_store = ExecutionReportStore()
        exec_store.start_run(run_id, now_ts_ms, "America/Santo_Domingo", {"symbols": ["BTCUSDT"], "from": 1, "to": 2})
        exec_store.inc_metrics(run_id, {"opsAnalyzed": +5, "apiWeightsUsed": +10})
        exec_store.finalize_run(run_id, end_ts_ms, {"opsClosed": 1, "opsOpen": 0}, "SUCCESS", None)

        ops_store = DailyActivityStore()
        ops_store.put_roundtrip({"PK": "...", "SK": "...", "symbol": "BTCUSDT"})
        ops_store.upsert_open({"PK": "...", "SK": "...", "symbol": "BTCUSDT"}, None)
        ops_store.close_open_and_put_roundtrip({"PK": "...", "SK": "..."}, {"PK": "...", "SK": "..."})
    """

    def __init__(
        self,
        table_name: str | None = None,
        *,
        dynamo_client: Any | None = None,
        region_name: str | None = None,
        status_index_name: str = "GSI1",
        run_lookup_index_name: str = "GSI2",
    ) -> None:
        self._table_name = table_name or _EXECUTION_REPORT_TABLE
        if not self._table_name:
            raise ValueError(
                "Execution report table name must be provided via argument or DDB_TABLE_EXECUTION_REPORT env var"
            )
        self._client = dynamo_client or boto3.client(
            "dynamodb", region_name=region_name or _DEFAULT_REGION
        )
        self._status_index_name = status_index_name
        self._run_lookup_index_name = run_lookup_index_name
        self._run_cache: dict[str, _RunKey] = {}

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: self._normalize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._normalize_value(v) for v in value)
        if isinstance(value, set):
            return {self._normalize_value(v) for v in value}
        if isinstance(value, bool):
            return value
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return _normalize_number(value)
        return value

    def _normalize_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        return {k: self._normalize_value(v) for k, v in item.items()}

    def _ensure_datetime(self, value: int | float | str, tz: str) -> datetime:
        if isinstance(value, (int, float, Decimal)):
            millis = _coerce_epoch_millis(value)
            seconds = millis / 1000.0
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
            return dt.astimezone(ZoneInfo(tz))
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo(tz))
            else:
                dt = dt.astimezone(ZoneInfo(tz))
            return dt
        raise TypeError(f"Unsupported timestamp type {type(value)!r}")

    def _format_day_key(self, dt: datetime) -> str:
        return dt.strftime("DAY#%Y-%m-%d")

    def _format_started_at(self, value: int | float | str) -> int | str:
        if isinstance(value, (int, float, Decimal)):
            return _coerce_epoch_millis(value)
        return value

    def _format_gsi_fields(self, status: str, started_at: Any, day_key: str, run_id: str) -> dict[str, Any]:
        started_value = started_at if isinstance(started_at, str) else str(started_at)
        return {
            "GSI1PK": f"STATUS#{status}",
            "GSI1SK": f"START#{started_value}",
            "GSI2PK": f"RUN#{run_id}",
            "GSI2SK": day_key,
        }

    def start_run(
        self,
        run_id: str,
        started_at: int | float | str,
        tz: str,
        meta: Mapping[str, Any] | None,
    ) -> None:
        local_dt = self._ensure_datetime(started_at, tz)
        day_key = self._format_day_key(local_dt)
        pk = f"{day_key}"
        sk = f"RUN#{run_id}"
        normalized_meta = self._normalize_item(meta or {})
        normalized_started = self._format_started_at(started_at)
        now_ms = int(time.time() * 1000)
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "runId": run_id,
            "startedAt": normalized_started,
            "timezone": tz,
            "status": "IN_PROGRESS",
            "opsAnalyzed": Decimal(0),
            "opsClosed": Decimal(0),
            "opsOpen": Decimal(0),
            "errorsCount": Decimal(0),
            "apiWeightsUsed": Decimal(0),
            "version": Decimal(1),
            "lastUpdateTs": Decimal(now_ms),
            "meta": normalized_meta,
            "totals": {},
        }
        item.update(self._format_gsi_fields("IN_PROGRESS", normalized_started, day_key, run_id))
        if "symbolsProcessed" not in item:
            item["symbolsProcessed"] = []
        try:
            self._client.put_item(
                TableName=self._table_name,
                Item={k: _SERIALIZER.serialize(self._normalize_value(v)) for k, v in item.items()},
                ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
            )
        except ClientError:
            logger.warning(
                "execution_report.start_run.failure",
                extra={"run_id": run_id, "pk": pk, "sk": sk},
            )
            raise
        self._run_cache[run_id] = _RunKey(pk=pk, sk=sk)
        logger.info(
            "execution_report.start_run.success",
            extra={"run_id": run_id, "pk": pk, "sk": sk},
        )

    def _get_key(self, run_id: str) -> _RunKey:
        cached = self._run_cache.get(run_id)
        if cached:
            return cached
        response = self._client.query(
            TableName=self._table_name,
            IndexName=self._run_lookup_index_name,
            KeyConditionExpression="#gsi2pk = :run_pk",
            ExpressionAttributeNames={"#gsi2pk": "GSI2PK"},
            ExpressionAttributeValues={":run_pk": {"S": f"RUN#{run_id}"}},
            Limit=1,
        )
        items = response.get("Items") or []
        if not items:
            raise KeyError(f"Run {run_id} not found")
        item = items[0]
        pk = _DESERIALIZER.deserialize(item["PK"])
        sk = _DESERIALIZER.deserialize(item["SK"])
        key = _RunKey(pk=pk, sk=sk)
        self._run_cache[run_id] = key
        return key

    def _build_update_attributes(
        self, deltas: Mapping[str, Any]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        set_parts: list[str] = []
        add_parts: list[str] = []
        expr_names: dict[str, str] = {"#last": "lastUpdateTs", "#version": "version"}
        expr_values: dict[str, Any] = {
            ":now": _SERIALIZER.serialize(Decimal(int(time.time() * 1000))),
            ":inc_one": _SERIALIZER.serialize(Decimal(1)),
        }
        for attr, raw_value in deltas.items():
            name_key = f"#attr_{attr}"
            expr_names[name_key] = attr
            if isinstance(raw_value, (int, float, Decimal)) and not isinstance(raw_value, bool):
                value_token = f":val_{attr}"
                expr_values[value_token] = _SERIALIZER.serialize(self._normalize_value(raw_value))
                add_parts.append(f"{name_key} {value_token}")
            else:
                value_token = f":val_{attr}"
                expr_values[value_token] = _SERIALIZER.serialize(self._normalize_value(raw_value))
                set_parts.append(f"{name_key} = {value_token}")
        set_parts.append("#last = :now")
        set_parts.append("#version = #version + :inc_one")
        update_expression = ""
        if set_parts:
            update_expression += "SET " + ", ".join(set_parts)
        if add_parts:
            update_expression += " ADD " + ", ".join(add_parts)
        return update_expression, expr_names, expr_values

    def inc_metrics(self, run_id: str, deltas: Mapping[str, Any]) -> None:
        if not deltas:
            return
        key = self._get_key(run_id)
        update_expression, expr_names, expr_values = self._build_update_attributes(deltas)
        try:
            self._client.update_item(
                TableName=self._table_name,
                Key={
                    "PK": _SERIALIZER.serialize(key.pk),
                    "SK": _SERIALIZER.serialize(key.sk),
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
            )
        except ClientError:
            logger.warning(
                "execution_report.inc_metrics.failure",
                extra={"run_id": run_id, "pk": key.pk, "sk": key.sk, "deltas_keys": sorted(deltas.keys())},
            )
            raise
        logger.info(
            "execution_report.inc_metrics.success",
            extra={"run_id": run_id, "pk": key.pk, "sk": key.sk, "deltas_keys": sorted(deltas.keys())},
        )

    def _get_item(self, run_id: str) -> dict[str, Any]:
        key = self._get_key(run_id)
        response = self._client.get_item(
            TableName=self._table_name,
            Key={
                "PK": _SERIALIZER.serialize(key.pk),
                "SK": _SERIALIZER.serialize(key.sk),
            },
        )
        item = response.get("Item")
        if not item:
            raise KeyError(f"Run {run_id} not found")
        deserialized = {k: _DESERIALIZER.deserialize(v) for k, v in item.items()}
        return _pythonify(deserialized)

    def finalize_run(
        self,
        run_id: str,
        ended_at: int | float | str,
        totals: Mapping[str, Any],
        status: str,
        errorSummary: str | None,
    ) -> None:
        key = self._get_key(run_id)
        current_item = self._get_item(run_id)
        started_at = current_item.get("startedAt")
        if isinstance(started_at, (int, float)):
            if isinstance(ended_at, (int, float, Decimal)):
                duration_ms = _coerce_epoch_millis(ended_at) - int(started_at)
            else:
                duration_ms = None
        else:
            duration_ms = None
        normalized_totals = self._normalize_item(totals)
        now_ms = int(time.time() * 1000)
        expr_names = {
            "#status": "status",
            "#endedAt": "endedAt",
            "#durationMs": "durationMs",
            "#totals": "totals",
            "#errorSummary": "errorSummary",
            "#last": "lastUpdateTs",
            "#version": "version",
        }
        expr_values = {
            ":status": _SERIALIZER.serialize(status),
            ":totals": _SERIALIZER.serialize(normalized_totals),
            ":now": _SERIALIZER.serialize(Decimal(now_ms)),
            ":inc_one": _SERIALIZER.serialize(Decimal(1)),
        }
        if isinstance(ended_at, (int, float, Decimal)):
            expr_values[":endedAt"] = _SERIALIZER.serialize(_coerce_epoch_millis(ended_at))
        else:
            expr_values[":endedAt"] = _SERIALIZER.serialize(ended_at)
        set_parts = [
            "#status = :status",
            "#endedAt = :endedAt",
            "#totals = :totals",
            "#last = :now",
            "#version = #version + :inc_one",
        ]
        if duration_ms is not None:
            expr_values[":duration"] = _SERIALIZER.serialize(_normalize_number(duration_ms))
            set_parts.append("#durationMs = :duration")
        if errorSummary is not None:
            expr_values[":errorSummary"] = _SERIALIZER.serialize(errorSummary)
            set_parts.append("#errorSummary = :errorSummary")
            update_expression = "SET " + ", ".join(set_parts)
        else:
            update_expression = "SET " + ", ".join(set_parts) + " REMOVE #errorSummary"
        try:
            self._client.update_item(
                TableName=self._table_name,
                Key={
                    "PK": _SERIALIZER.serialize(key.pk),
                    "SK": _SERIALIZER.serialize(key.sk),
                },
                UpdateExpression=update_expression,
                ConditionExpression="attribute_exists(PK) AND attribute_exists(SK) AND #status = :expected_status",
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues={**expr_values, ":expected_status": _SERIALIZER.serialize("IN_PROGRESS")},
            )
        except ClientError:
            logger.warning(
                "execution_report.finalize_run.failure",
                extra={"run_id": run_id, "pk": key.pk, "sk": key.sk, "status": status},
            )
            raise
        logger.info(
            "execution_report.finalize_run.success",
            extra={"run_id": run_id, "pk": key.pk, "sk": key.sk, "status": status},
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        item = self._get_item(run_id)
        return item
