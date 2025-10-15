from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

stub_config = types.ModuleType("config")
stub_settings = types.ModuleType("config.settings")


def _sl_stub(settings):  # pragma: no cover - test helper
    return None


def _tp_stub(settings):  # pragma: no cover - test helper
    return None


stub_settings.get_stop_loss_pct = _sl_stub
stub_settings.get_take_profit_pct = _tp_stub
stub_config.settings = stub_settings
sys.modules.setdefault("config", stub_config)
sys.modules.setdefault("config.settings", stub_settings)

stub_utils = types.ModuleType("config.utils")


def _parse_bool(value, default=False):  # pragma: no cover - test helper
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


stub_utils.parse_bool = _parse_bool
sys.modules.setdefault("config.utils", stub_utils)

boto3_stub = types.ModuleType("boto3")


class _ClientError(Exception):
    def __init__(self, error_response=None, operation_name: str | None = None):
        super().__init__(str(error_response) or "client_error")
        self.response = error_response or {}
        self.operation_name = operation_name or ""


class _BotoCoreError(Exception):
    pass


class _S3Stub:
    storage: dict[str, dict[str, bytes]]

    def __init__(self):
        self.storage = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kwargs):  # pragma: no cover - simple stub
        self.storage[Key] = {"Bucket": Bucket, "Body": Body}
        return {}

    def get_object(self, *, Bucket: str, Key: str, **_kwargs):  # pragma: no cover - simple stub
        raise _ClientError({"Error": {"Code": "NoSuchKey"}}, "get_object")


def _client_factory(*_args, **_kwargs):  # pragma: no cover - simple stub
    return _S3Stub()


setattr(boto3_stub, "client", _client_factory)
sys.modules.setdefault("boto3", boto3_stub)

botocore_stub = types.ModuleType("botocore")
botocore_exceptions = types.ModuleType("botocore.exceptions")
setattr(botocore_exceptions, "BotoCoreError", _BotoCoreError)
setattr(botocore_exceptions, "ClientError", _ClientError)
sys.modules.setdefault("botocore", botocore_stub)
sys.modules.setdefault("botocore.exceptions", botocore_exceptions)

from common.notional import ensure_min_notional_with_buffers
from strategies.breakout_dual_tf import BreakoutDualTFStrategy, BreakoutSignalPayload, Level


class DummySettings(dict):
    def get(self, key, default=None):
        return super().get(key, default)

    def __getattr__(self, item):
        return self.get(item)


def _build_signal(
    *,
    symbol: str,
    action: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
) -> BreakoutSignalPayload:
    direction = "LONG" if action == "BUY" else "SHORT"
    level_type = "S" if direction == "LONG" else "R"
    level = Level(price=entry, level_type=level_type, timestamp=0)
    return BreakoutSignalPayload(
        symbol=symbol,
        action=action,
        direction=direction,
        level=level,
        entry_price=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        rr=3.0,
        atr=0.001,
        volume_rel=1.5,
        ema_fast=entry * 1.01,
        ema_slow=entry * 1.02,
        exec_tf="15m",
        candle=[0, entry, entry, entry, entry, 0],
        swing=entry,
    )


@pytest.mark.parametrize(
    "symbol,action,entry,sl,tp1,tp2,tick,step,min_qty,min_notional",
    [
        ("DOGEUSDT", "SELL", 0.24790000000000003, 0.24956, 0.24339, 0.23968, "0.0001", "1", "1", "5"),
        ("SOLUSDT", "BUY", 23.456789, 22.85, 24.912345, 26.789012, "0.001", "0.1", "0.1", "5"),
        ("XRPUSDT", "SELL", 0.557899999, 0.564321, 0.545678, 0.523456, "0.0001", "1", "1", "5"),
    ],
)
def test_compute_orders_respects_filters(
    caplog,
    symbol,
    action,
    entry,
    sl,
    tp1,
    tp2,
    tick,
    step,
    min_qty,
    min_notional,
):
    settings = DummySettings(
        {
            "SYMBOL": symbol,
            "STRICT_ROUNDING": True,
            "RISK_NOTIONAL_USDT": 150,
            "RISK_PCT": 0.01,
            "INTERVAL": "1h",
        }
    )
    strategy = BreakoutDualTFStrategy(None, None, settings)
    signal = _build_signal(symbol=symbol, action=action, entry=entry, sl=sl, tp1=tp1, tp2=tp2)

    ctx = {
        "exec_tf": signal.exec_tf,
        "tick_size": tick,
        "step_size": step,
        "min_qty": min_qty,
        "min_notional": min_notional,
        "available_balance_usdt": "1000",
    }

    with caplog.at_level("INFO", logger="bot.strategy.breakout_dual_tf"):
        orders = strategy.compute_orders(signal, context=ctx)

    entry_dec = Decimal(orders["entry"])
    stop_dec = Decimal(orders["stop_loss"])
    tp1_dec = Decimal(orders["take_profit_1"])
    tp2_dec = Decimal(orders["take_profit_2"])
    qty_dec = Decimal(orders["qty"])

    tick_dec = Decimal(tick)
    step_dec = Decimal(step)
    assert (entry_dec / tick_dec).quantize(Decimal("1")) * tick_dec == entry_dec
    assert (stop_dec / tick_dec).quantize(Decimal("1")) * tick_dec == stop_dec
    assert (tp1_dec / tick_dec).quantize(Decimal("1")) * tick_dec == tp1_dec
    assert (tp2_dec / tick_dec).quantize(Decimal("1")) * tick_dec == tp2_dec
    if step_dec > 0:
        assert (qty_dec / step_dec).quantize(Decimal("1")) * step_dec == qty_dec

    min_notional_dec = Decimal(min_notional)
    if min_notional_dec > 0:
        buffer_pct = Decimal("0.03")
        buffer_usd = Decimal("0.10")
        min_target = max(
            min_notional_dec,
            min_notional_dec * (Decimal("1") + buffer_pct),
            min_notional_dec + buffer_usd,
        )
        assert entry_dec * qty_dec >= min_target

    assert Decimal(orders["qty_tp1"]) + Decimal(orders["qty_tp2"]) == qty_dec
    assert orders["entry"].count("0.000000") == 0

    pre_logs = [json.loads(record.message) for record in caplog.records if "pre_order_check" in record.message]
    assert pre_logs, "pre_order_check log missing"
    latest = pre_logs[-1]["pre_order_check"]
    assert latest["entry"] == orders["entry"]
    assert latest["qty"] == orders["qty"]
    assert latest["validated"] is True


def test_signal_log_serialisation_has_no_binary_tail(caplog):
    settings = DummySettings(
        {
            "SYMBOL": "DOGEUSDT",
            "STRICT_ROUNDING": True,
            "RISK_NOTIONAL_USDT": 200,
            "INTERVAL": "1h",
        }
    )
    strategy = BreakoutDualTFStrategy(None, None, settings)
    signal = _build_signal(
        symbol="DOGEUSDT",
        action="SELL",
        entry=0.24790000000000003,
        sl=0.24956,
        tp1=0.24339,
        tp2=0.23968,
    )
    ctx = {
        "exec_tf": signal.exec_tf,
        "tick_size": "0.0001",
        "step_size": "1",
        "min_qty": "1",
        "min_notional": "5",
        "available_balance_usdt": "500",
    }

    with caplog.at_level("INFO", logger="bot.strategy.breakout_dual_tf"):
        strategy.compute_orders(signal, context=ctx)

    payloads = [record.message for record in caplog.records if "breakout_dual_tf" in record.message or "action" in record.message]
    assert payloads, "no logs captured"
    for payload in payloads:
        assert "0.24790000000000003" not in payload


def test_ensure_min_notional_with_buffers_increments_qty():
    result = ensure_min_notional_with_buffers(
        qty="25.5",
        price="0.19666",
        side="BUY",
        step_size="1",
        tick_size="0.00001",
        min_notional="5",
        buffer_pct="0.0",
        buffer_usd="0.10",
    )

    assert result.qty_raw == Decimal("25.5")
    assert result.qty_rounded == Decimal("26")
    assert result.notional >= result.min_notional_target
    assert "qty_increased_for_min_notional" in result.adjustments
