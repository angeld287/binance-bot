from __future__ import annotations

from datetime import datetime

import pytest

from runners import reports_handler


def test_resolve_overrides_previous_day(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_now = datetime(2025, 10, 29, 12, 0)
    overrides = reports_handler._resolve_overrides({"note": "manual"}, now=fixed_now)

    assert overrides["origin"] == "manual"
    assert overrides["dryRun"] is False
    assert overrides["symbols"] == []
    assert overrides["fromIso"].startswith("2025-10-28T00:00:00")
    assert overrides["toIso"].startswith("2025-10-28T23:59:59")


def test_resolve_overrides_eventbridge_detail() -> None:
    event = {
        "source": "aws.events",
        "detail-type": "Scheduled Event",
        "detail": {
            "fromTs": 1_761_700_800_000,
            "toTs": 1_761_787_199_000,
            "symbols": "BTCUSDT,ETHUSDT",
            "dryRun": True,
        },
    }

    overrides = reports_handler._resolve_overrides(event, now=datetime(2025, 10, 29, 12, 0))

    assert overrides["origin"] == "eventbridge"
    assert overrides["dryRun"] is True
    assert overrides["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert overrides["fromTs"] == 1_761_700_800_000
    assert overrides["detail"] == event["detail"]


def test_handler_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_event: dict | None = None

    def fake_orchestrator(event_in, now=None):  # type: ignore[no-untyped-def]
        nonlocal captured_event
        captured_event = event_in
        return {"status": "success", "processed": 0, "inserted": 0, "skipped": 0}

    monkeypatch.setattr(reports_handler, "resolve_orchestrator", lambda: (fake_orchestrator, "fake.path"))
    monkeypatch.setattr(reports_handler, "_resolve_overrides", lambda event, now=None: {"origin": "manual", "fromIso": "x", "toIso": "y", "fromTs": 1, "toTs": 2, "symbols": [], "dryRun": False})

    response = reports_handler.handler({"note": "manual"}, None)

    assert response["result"]["status"] == "success"
    assert response["overrides"]["origin"] == "manual"
    assert captured_event == response["overrides"]
