import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from analytics.mfe_utils import compute_mfe_mae


CANDLE = [0, 0, 0, 0, 0, 0]


def _make_candle(open_time, high, low):
    candle = CANDLE.copy()
    candle[0] = open_time
    candle[2] = high
    candle[3] = low
    return candle


def test_compute_mfe_mae_long():
    candles = [
        _make_candle(1_000, 102.0, 99.5),
        _make_candle(2_000, 105.0, 98.0),
    ]
    mfe, mae, ts = compute_mfe_mae(candles, "LONG", 100.0)
    assert round(mfe, 6) == 5.0
    assert round(mae, 6) == 2.0
    assert ts == 2_000


def test_compute_mfe_mae_short():
    candles = [
        _make_candle(1_000, 105.0, 98.0),
        _make_candle(2_000, 106.0, 94.0),
    ]
    mfe, mae, ts = compute_mfe_mae(candles, "SHORT", 100.0)
    assert round(mfe, 6) == 6.0
    assert round(mae, 6) == 6.0
    assert ts == 2_000


def test_compute_mfe_mae_no_data():
    mfe, mae, ts = compute_mfe_mae([], "LONG", 0.0)
    assert mfe == mae == 0.0
    assert ts is None
