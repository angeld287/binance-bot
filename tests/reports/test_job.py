from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.reports import job


class FakeClient:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {"orders": 0, "trades": 0, "income": 0, "klines": 0}

    def futures_all_orders(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls["orders"] += 1
        return [
            {
                "orderId": 1001,
                "type": "MARKET",
                "status": "FILLED",
                "avgPrice": "101.0",
                "updateTime": kwargs.get("startTime"),
                "clientOrderId": "entry-1",
            },
            {
                "orderId": 1002,
                "type": "MARKET",
                "status": "FILLED",
                "avgPrice": "102.0",
                "updateTime": kwargs.get("endTime"),
                "clientOrderId": "exit-1",
            },
        ]

    def futures_account_trades(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls["trades"] += 1
        base_time = kwargs.get("startTime", 0)
        return [
            {
                "id": 1,
                "orderId": 1001,
                "symbol": kwargs["symbol"],
                "side": "BUY",
                "price": "100.0",
                "qty": "0.01",
                "realizedPnl": "0",
                "commission": "-0.0003",
                "time": base_time + 60_000,
                "positionSide": "BOTH",
            },
            {
                "id": 2,
                "orderId": 1002,
                "symbol": kwargs["symbol"],
                "side": "SELL",
                "price": "101.0",
                "qty": "0.01",
                "realizedPnl": "0.01",
                "commission": "-0.0003",
                "time": base_time + 120_000,
                "positionSide": "BOTH",
            },
        ]

    def futures_income_history(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls["income"] += 1
        return [
            {
                "symbol": kwargs["symbol"],
                "incomeType": "REALIZED_PNL",
                "income": "0.01",
                "tradeId": 2,
                "time": kwargs.get("endTime"),
            }
        ]

    def futures_klines(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls["klines"] += 1
        start = kwargs.get("startTime", 0)
        return [
            [start, "100.0", "102.0", "99.0", "101.0", "10"],
            [start + 60_000, "101.0", "103.0", "100.0", "102.0", "12"],
        ]


def _fake_enrich(roundtrip, **_kwargs):  # type: ignore[no-untyped-def]
    enriched = dict(roundtrip)
    enriched["enriched"] = True
    return enriched


class FakeDailyStore:
    instances: list["FakeDailyStore"] = []

    def __init__(self) -> None:
        self.records: list[dict] = []
        FakeDailyStore.instances.append(self)

    def put_roundtrip(self, item):  # type: ignore[no-untyped-def]
        self.records.append(item)


class FakeExecStore:
    instances: list["FakeExecStore"] = []

    def __init__(self) -> None:
        self.started: list[tuple] = []
        self.metrics: list[tuple] = []
        self.finalized: list[tuple] = []
        FakeExecStore.instances.append(self)

    def start_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.started.append((args, kwargs))

    def inc_metrics(self, run_id, deltas):  # type: ignore[no-untyped-def]
        self.metrics.append((run_id, deltas))

    def finalize_run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.finalized.append((args, kwargs))


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeDailyStore.instances.clear()
    FakeExecStore.instances.clear()
    yield
    FakeDailyStore.instances.clear()
    FakeExecStore.instances.clear()


def _make_event(dry_run: bool) -> dict:
    start = datetime(2025, 10, 28, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 10, 28, 23, 59, tzinfo=timezone.utc)
    return {
        "origin": "manual",
        "symbols": ["BTCUSDT"],
        "dryRun": dry_run,
        "fromTs": int(start.timestamp() * 1000),
        "toTs": int(end.timestamp() * 1000),
    }


def test_run_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job, "_create_binance_client", lambda: FakeClient())
    monkeypatch.setattr(job, "enrich_before_persist", _fake_enrich)
    monkeypatch.setattr(job, "DailyActivityStore", lambda: (_ for _ in ()).throw(AssertionError("should not create store")))
    monkeypatch.setattr(job, "ExecutionReportStore", lambda: (_ for _ in ()).throw(AssertionError("should not create store")))
    monkeypatch.setattr(job, "_generate_run_id", lambda _dt: "run-001")

    summary = job.run(_make_event(True), now=datetime(2025, 10, 29, 12, 0, tzinfo=timezone.utc))

    assert summary["status"] == "success"
    assert summary["processed"] == 1
    assert summary["inserted"] == 0
    assert summary["dryRun"] is True
    assert summary["symbols"] == ["BTCUSDT"]
    assert summary["roundtrips"] and summary["roundtrips"][0]["enriched"] is True


def test_run_persists_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job, "_create_binance_client", lambda: FakeClient())
    monkeypatch.setattr(job, "enrich_before_persist", _fake_enrich)
    monkeypatch.setattr(job, "DailyActivityStore", FakeDailyStore)
    monkeypatch.setattr(job, "ExecutionReportStore", FakeExecStore)
    monkeypatch.setattr(job, "_generate_run_id", lambda _dt: "run-1234")

    summary = job.run(_make_event(False), now=datetime(2025, 10, 29, 12, 0, tzinfo=timezone.utc))

    assert summary["status"] == "success"
    assert summary["processed"] == 1
    assert summary["inserted"] == 1
    assert not summary["dryRun"]
    assert FakeDailyStore.instances
    assert FakeDailyStore.instances[0].records
    persisted = FakeDailyStore.instances[0].records[0]
    assert persisted["runId"] == "run-1234"
    assert persisted["enriched"] is True
    assert FakeExecStore.instances
    exec_store = FakeExecStore.instances[0]
    assert exec_store.started
    assert exec_store.metrics
    assert exec_store.finalized
