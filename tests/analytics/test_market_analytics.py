import os
import sys
from datetime import datetime, timedelta, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from analytics.market_analytics import enrich_roundtrip_with_market_data


class FakeClient:
    def __init__(self, klines_map):
        self._klines_map = klines_map

    def futures_klines(self, *, symbol, interval, startTime=None, endTime=None, limit=None):
        key = (symbol, interval)
        data = self._klines_map.get(key, [])
        result = []
        for candle in data:
            open_time = candle[0]
            if startTime is not None and open_time < startTime:
                continue
            if endTime is not None and open_time > endTime:
                continue
            result.append(list(candle))
        return result


def _build_test_data():
    symbol = "BTCUSDT"
    open_dt = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    close_dt = open_dt + timedelta(minutes=30)

    candles_1m = []
    start_1m = open_dt - timedelta(minutes=2)
    for i in range(34):
        ts = int((start_1m + timedelta(minutes=i)).timestamp() * 1000)
        high = 106.0 if i == 20 else 102.0
        low = 98.0 if i == 10 else 99.0
        candles_1m.append([ts, 100.0, high, low, 100.5, 50.0])

    candles_15m = []
    start_15m = open_dt - timedelta(minutes=15 * 59)
    price = 80.0
    for i in range(60):
        ts = int((start_15m + timedelta(minutes=15 * i)).timestamp() * 1000)
        price += 0.5
        candles_15m.append([ts, price - 1, price + 1, price - 2, price, 100.0])

    klines_map = {
        (symbol, "1m"): candles_1m,
        (symbol, "15m"): candles_15m,
    }

    roundtrip = {
        "symbol": symbol,
        "openAt": int(open_dt.timestamp() * 1000),
        "closeAt": int(close_dt.timestamp() * 1000),
        "entryPrice": 100.0,
        "roiNetPct": 3.5,
        "direction": "LONG",
    }
    return roundtrip, FakeClient(klines_map)


def test_enrich_roundtrip_with_market_data(monkeypatch):
    monkeypatch.setenv("EMA_TF", "15m")
    monkeypatch.setenv("EMA_FAST", "7")
    monkeypatch.setenv("EMA_SLOW", "45")
    monkeypatch.setenv("K_SLOPE", "3")
    monkeypatch.setenv("TH_NEUTRO_FAST", "0.0005")
    monkeypatch.setenv("TH_NEUTRO_SLOW", "0.0002")
    monkeypatch.setenv("TH_FUERTE_FAST", "0.0020")
    monkeypatch.setenv("TH_FUERTE_SLOW", "0.0010")
    monkeypatch.setenv("TH_SPREAD_FUERTE", "0.0020")

    roundtrip, client = _build_test_data()
    enriched = enrich_roundtrip_with_market_data(roundtrip, client=client)

    assert enriched["durationMin"] == 30
    assert enriched["resultado"] == "GANADA"
    assert enriched["priceVsEma7Open"] == "DEBAJO"
    assert enriched["emaTf"] == "15m"
    assert enriched["ema7ValueAtOpen"] is not None
    assert abs(enriched["mfePct"] - 6.0) < 1e-6
    assert abs(enriched["maePct"] - 2.0) < 1e-6
    assert enriched["emaTrendClassAtOpen"] == "ASC_FUERTE"
    assert "fast=" in enriched.get("emaTrendNotesAtOpen", "")
    assert enriched["tz"] == "America/Santo_Domingo"
    assert enriched["mfeTs"].endswith("-04:00")
