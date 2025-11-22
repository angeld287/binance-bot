from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[3]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

strategies_pkg = types.ModuleType("strategies")
strategies_pkg.__path__ = [str(PROJECT_SRC / "strategies")]
sys.modules.setdefault("strategies", strategies_pkg)

pcf_pkg = types.ModuleType("strategies.ParallelChannelFormation")
pcf_pkg.__path__ = [str(PROJECT_SRC / "strategies" / "ParallelChannelFormation")]
sys.modules.setdefault("strategies.ParallelChannelFormation", pcf_pkg)

from strategies.ParallelChannelFormation.validators.compression_filter import (
    compression_filter_allows_entry,
)


def test_compression_filter_allows_trend():
    closes = [100 + i * 0.5 for i in range(20)]
    allowed, meta = compression_filter_allows_entry(
        enabled=True,
        closes=closes,
        ema_fast_length=7,
        ema_slow_length=25,
        ema_fast_value=None,
        ema_slow_value=None,
        symbol="TREND",
    )

    assert allowed is True
    assert meta is None


def test_compression_filter_blocks_range():
    closes = [100.0, 99.8, 100.2, 99.9, 100.1, 99.95, 100.05, 99.9, 100.0, 99.85, 100.1]
    allowed, meta = compression_filter_allows_entry(
        enabled=True,
        closes=closes,
        ema_fast_length=7,
        ema_slow_length=25,
        ema_fast_value=None,
        ema_slow_value=None,
        symbol="RANGE",
    )

    assert allowed is False
    assert meta is not None
    assert meta.get("reason") == "compression_filter"
