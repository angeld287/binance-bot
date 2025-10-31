from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.repositories.dynamo_store import DailyActivityStore


@pytest.fixture(autouse=True)
def daily_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DDB_TABLE_DAILY_ACTIVITY", "daily_activity")


def _make_client_error(operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "ConditionalCheckFailedException", "Message": "conditional failed"}
        },
        operation,
    )


def test_put_roundtrip_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    store = DailyActivityStore(dynamo_client=client)

    roundtrip = {
        "PK": "DAY#2023-12-01#SYM#BTCUSDT",
        "SK": "RT#abc",
        "openAt": 1700000000000,
        "closeAt": 1700003600000,
        "symbol": "BTCUSDT",
    }

    store.put_roundtrip(roundtrip)

    client.put_item.assert_called_once()
    call = client.put_item.call_args.kwargs
    assert call["TableName"] == "daily_activity"
    assert call["ConditionExpression"] == "attribute_not_exists(PK) AND attribute_not_exists(SK)"
    item = call["Item"]
    assert item["PK"]["S"] == roundtrip["PK"]
    assert item["openAt"]["N"] == str(Decimal(roundtrip["openAt"]))


def test_put_roundtrip_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.put_item.side_effect = _make_client_error("PutItem")
    store = DailyActivityStore(dynamo_client=client)

    with pytest.raises(ClientError):
        store.put_roundtrip({"PK": "P", "SK": "S"})


def test_upsert_open_new_item(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    store = DailyActivityStore(dynamo_client=client)

    open_item = {
        "PK": "OPEN#SYM#BTCUSDT",
        "SK": "POS#abc",
        "symbol": "BTCUSDT",
        "openQty": 1,
        "version": 99,
    }

    store.upsert_open(open_item, prev_version=None)

    client.update_item.assert_called_once()
    call = client.update_item.call_args.kwargs
    assert call["ConditionExpression"] == "attribute_not_exists(PK) AND attribute_not_exists(SK)"
    expr_values = call["ExpressionAttributeValues"]
    assert expr_values[":initial_version"]["N"] == "1"
    assert ":val_symbol" in expr_values
    assert ":val_version" not in expr_values


def test_upsert_open_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.update_item.side_effect = _make_client_error("UpdateItem")
    store = DailyActivityStore(dynamo_client=client)

    open_item = {
        "PK": "OPEN#SYM#BTCUSDT",
        "SK": "POS#abc",
        "symbol": "BTCUSDT",
    }

    with pytest.raises(ClientError):
        store.upsert_open(open_item, prev_version=3)

    client.update_item.assert_called_once()
    call = client.update_item.call_args.kwargs
    assert call["ConditionExpression"].endswith("#version = :prev_version")
    assert call["ExpressionAttributeValues"][":prev_version"]["N"] == "3"


def test_close_open_and_put_roundtrip_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    store = DailyActivityStore(dynamo_client=client)

    open_keys = {"PK": "OPEN#SYM#BTCUSDT", "SK": "POS#abc"}
    roundtrip = {"PK": "DAY#2023-12-01#SYM#BTCUSDT", "SK": "RT#abc", "symbol": "BTCUSDT"}

    store.close_open_and_put_roundtrip(open_keys, roundtrip)

    client.transact_write_items.assert_called_once()
    call = client.transact_write_items.call_args.kwargs
    items = call["TransactItems"]
    delete = items[0]["Delete"]
    put = items[1]["Put"]
    assert delete["ConditionExpression"] == "attribute_exists(PK) AND attribute_exists(SK)"
    assert put["ConditionExpression"] == "attribute_not_exists(PK) AND attribute_not_exists(SK)"
    assert put["Item"]["symbol"]["S"] == "BTCUSDT"


def test_close_open_and_put_roundtrip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.transact_write_items.side_effect = _make_client_error("TransactWriteItems")
    store = DailyActivityStore(dynamo_client=client)

    with pytest.raises(ClientError):
        store.close_open_and_put_roundtrip({"PK": "P", "SK": "S"}, {"PK": "X", "SK": "Y"})
