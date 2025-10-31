from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import MagicMock

import pytest
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

from src.repositories.dynamo_store import ExecutionReportStore


@pytest.fixture(autouse=True)
def execution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DDB_TABLE_EXECUTION_REPORT", "execution_report")


def _make_client_error(operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "ConditionalCheckFailedException", "Message": "conditional failed"}
        },
        operation,
    )


def _patch_time(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    series = deque(values)

    def _tick() -> float:
        if not series:
            return values[-1]
        return series.popleft()

    monkeypatch.setattr("src.repositories.dynamo_store.time.time", _tick)


def test_start_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    _patch_time(monkeypatch, [1234.0])
    store = ExecutionReportStore(dynamo_client=client)

    store.start_run("run-1", 1701385200000, "America/Santo_Domingo", {"symbols": ["BTCUSDT"]})

    client.put_item.assert_called_once()
    call = client.put_item.call_args.kwargs
    assert call["TableName"] == "execution_report"
    assert call["ConditionExpression"] == "attribute_not_exists(PK) AND attribute_not_exists(SK)"
    item = call["Item"]
    assert item["PK"]["S"].startswith("DAY#")
    assert item["status"]["S"] == "IN_PROGRESS"
    assert item["GSI2PK"]["S"] == "RUN#run-1"


def test_start_run_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.put_item.side_effect = _make_client_error("PutItem")
    store = ExecutionReportStore(dynamo_client=client)

    with pytest.raises(ClientError):
        store.start_run("run-1", 1701385200000, "America/Santo_Domingo", {})


def test_inc_metrics_accumulates(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    _patch_time(monkeypatch, [1000.0, 1001.0, 1002.0])
    store = ExecutionReportStore(dynamo_client=client)

    store.start_run("run-1", 1701385200000, "UTC", {})

    store.inc_metrics("run-1", {"opsAnalyzed": 3, "apiWeightsUsed": 5})
    store.inc_metrics("run-1", {"opsAnalyzed": 2, "symbolsProcessed": ["BTCUSDT"]})

    assert client.update_item.call_count == 2
    first_call = client.update_item.call_args_list[0].kwargs
    assert "ADD" in first_call["UpdateExpression"]
    values = first_call["ExpressionAttributeValues"]
    assert values[":val_opsAnalyzed"]["N"] == "3"
    second_call = client.update_item.call_args_list[1].kwargs
    assert ":val_symbolsProcessed" in second_call["ExpressionAttributeValues"]
    assert second_call["ExpressionAttributeValues"][":inc_one"]["N"] == "1"


def test_finalize_run_updates_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    serializer = TypeSerializer()
    _patch_time(monkeypatch, [2000.0, 2001.0])
    store = ExecutionReportStore(dynamo_client=client)

    store.start_run("run-1", 1701385200000, "UTC", {})

    client.get_item.return_value = {
        "Item": {
            "PK": serializer.serialize("DAY#2023-11-30"),
            "SK": serializer.serialize("RUN#run-1"),
            "startedAt": serializer.serialize(1701385200000),
            "status": serializer.serialize("IN_PROGRESS"),
        }
    }

    store.finalize_run(
        "run-1",
        1701385201234,
        {"opsClosed": 1, "opsOpen": 0},
        "SUCCESS",
        None,
    )

    assert client.update_item.call_args.kwargs["ConditionExpression"].endswith("#status = :expected_status")
    values = client.update_item.call_args.kwargs["ExpressionAttributeValues"]
    assert values[":status"]["S"] == "SUCCESS"
    assert values[":totals"]["M"]["opsClosed"]["N"] == "1"
    assert values[":duration"]["N"] == "1234"
    assert client.update_item.call_args.kwargs["UpdateExpression"].endswith("REMOVE #errorSummary")


def test_get_run_returns_python_types(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    serializer = TypeSerializer()
    _patch_time(monkeypatch, [3000.0])
    store = ExecutionReportStore(dynamo_client=client)

    store.start_run("run-1", 1701385200000, "UTC", {})

    client.get_item.return_value = {
        "Item": {
            "PK": serializer.serialize("DAY#2023-11-30"),
            "SK": serializer.serialize("RUN#run-1"),
            "startedAt": serializer.serialize(1701385200000),
            "opsAnalyzed": serializer.serialize(5),
            "status": serializer.serialize("IN_PROGRESS"),
        }
    }

    result = store.get_run("run-1")
    assert result["startedAt"] == 1701385200000
    assert result["opsAnalyzed"] == 5
