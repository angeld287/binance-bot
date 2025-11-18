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


def test_build_roundtrips_closes_every_cycle() -> None:
    symbol = "ADAUSDT"
    trades = []
    base_ts = 1_000_000
    trade_id = 1
    order_id = 100
    for cycle in range(3):
        open_trade = {
            "id": trade_id,
            "orderId": order_id,
            "symbol": symbol,
            "side": "BUY",
            "price": "0.25",
            "qty": "9",
            "realizedPnl": "0",
            "commission": "-0.0001",
            "time": base_ts + cycle * 10_000 + 1000,
        }
        close_trade = {
            "id": trade_id + 1,
            "orderId": order_id + 1,
            "symbol": symbol,
            "side": "SELL",
            "price": "0.26",
            "qty": "9",
            "realizedPnl": "0.09",
            "commission": "-0.0001",
            "time": base_ts + cycle * 10_000 + 5000,
        }
        trades.extend([open_trade, close_trade])
        trade_id += 2
        order_id += 2

    roundtrips, skipped, leftovers = job._build_roundtrips(
        [symbol],
        {symbol: trades},
        {},
        {},
    )

    assert skipped == 0
    assert not leftovers
    assert len(roundtrips) == 3
    for idx, rt in enumerate(roundtrips):
        assert rt["symbol"] == symbol
        assert rt["qty"] == pytest.approx(9.0)
        assert rt["direction"] == "LONG"
        assert rt["openTimestamp"] < rt["closeTimestamp"]
        assert rt["pnl"] == pytest.approx(rt["netPnl"] + rt.get("incomeRealized", 0.0))


def test_build_roundtrips_bootstrap_position_before_range() -> None:
    symbol = "SOLUSDT"
    start_ts = 2_000_000
    lookback_trade = {
        "id": 9001,
        "orderId": 5001,
        "symbol": symbol,
        "side": "BUY",
        "price": "10",
        "qty": "2",
        "realizedPnl": "0",
        "commission": "-0.0002",
        "time": start_ts - 1_000,
    }
    closing_trade = {
        "id": 9002,
        "orderId": 5002,
        "symbol": symbol,
        "side": "SELL",
        "price": "12",
        "qty": "2",
        "realizedPnl": "4",  # ensure retrospective is triggered
        "commission": "-0.0002",
        "time": start_ts + 1_000,
    }

    class RetroClient:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def futures_account_trades(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((kwargs.get("startTime"), kwargs.get("endTime")))
            end = kwargs.get("endTime")
            if end == start_ts:
                return [lookback_trade]
            return []

    roundtrips, skipped, leftovers = job._build_roundtrips(
        [symbol],
        {symbol: [closing_trade]},
        {},
        {},
        client=RetroClient(),
        range_start=start_ts,
    )

    assert skipped == 0
    assert not leftovers
    assert len(roundtrips) == 1
    rt = roundtrips[0]
    assert rt["openTimestamp"] == lookback_trade["time"]
    assert rt["closeTimestamp"] == closing_trade["time"]
    assert rt["qty"] == pytest.approx(2.0)
    assert rt["direction"] == "LONG"


def test_open_position_is_discarded_at_range_end() -> None:
    symbol = "XRPUSDT"
    base_ts = 3_000_000
    open_trade = {
        "id": 42,
        "orderId": 9001,
        "symbol": symbol,
        "side": "BUY",
        "price": "0.5",
        "qty": "100",
        "realizedPnl": "0",
        "commission": "-0.1",
        "time": base_ts + 1_000,
    }

    roundtrips, skipped, leftovers = job._build_roundtrips(
        [symbol],
        {symbol: [open_trade]},
        {},
        {},
    )

    assert skipped == 0
    assert not roundtrips
    assert leftovers == [{
        "symbol": symbol,
        "direction": "LONG",
        "openTrades": 1,
        "openQty": pytest.approx(100.0),
        "firstTs": open_trade["time"],
        "lastTs": open_trade["time"],
    }]


def test_trades_are_sorted_by_trade_id() -> None:
    symbol = "DOTUSDT"
    base_ts = 4_000_000
    trades = [
        {
            "id": 2,
            "orderId": 2002,
            "symbol": symbol,
            "side": "SELL",
            "price": "11",
            "qty": "1",
            "realizedPnl": "1",
            "commission": "-0.01",
            "time": base_ts + 5_000,  # later timestamp but lower tradeId
        },
        {
            "id": 1,
            "orderId": 2001,
            "symbol": symbol,
            "side": "BUY",
            "price": "10",
            "qty": "1",
            "realizedPnl": "0",
            "commission": "-0.01",
            "time": base_ts + 10_000,  # higher timestamp but should be first after sort
        },
    ]

    roundtrips, skipped, leftovers = job._build_roundtrips(
        [symbol],
        {symbol: trades},
        {},
        {},
    )

    assert skipped == 0
    assert not leftovers
    assert len(roundtrips) == 1
    rt = roundtrips[0]
    assert rt["openTimestamp"] == trades[1]["time"]
    assert rt["closeTimestamp"] == trades[0]["time"]
    assert rt["tradeIds"] == ["1", "2"]
