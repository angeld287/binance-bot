from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

stub_config = types.ModuleType("config")
stub_settings = types.ModuleType("config.settings")
stub_settings.get_stop_loss_pct = lambda _settings: None
stub_settings.get_take_profit_pct = lambda _settings: None
stub_config.settings = stub_settings
sys.modules.setdefault("config", stub_config)
sys.modules.setdefault("config.settings", stub_settings)
stub_utils = types.ModuleType("config.utils")
stub_utils.parse_bool = lambda value, default=None: default if value in (None, "") else bool(value)
sys.modules.setdefault("config.utils", stub_utils)
boto3_stub = types.ModuleType("boto3")
boto3_stub.client = lambda *_args, **_kwargs: None
sys.modules.setdefault("boto3", boto3_stub)
botocore_stub = types.ModuleType("botocore")
botocore_exceptions = types.ModuleType("botocore.exceptions")
botocore_exceptions.BotoCoreError = type("BotoCoreError", (Exception,), {})
botocore_exceptions.ClientError = type("ClientError", (Exception,), {})
sys.modules.setdefault("botocore", botocore_stub)
sys.modules.setdefault("botocore.exceptions", botocore_exceptions)

from strategies.breakout_dual_tf import BreakoutDualTFStrategy
from strategies.liquidity_sweep import strategy as ls_strategy


class DummySettings:
    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data = dict(data or {})

    def get(self, key: str, default: Any | None = None) -> Any:
        return self._data.get(key, default)

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        value = self._data.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"1", "true", "yes"}:
                return True
            if lowered in {"0", "false", "no"}:
                return False
        return bool(value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self._data[item]
        except KeyError as exc:  # pragma: no cover - helper mirrors pydantic behaviour
            raise AttributeError(item) from exc


class BreakoutStubBroker:
    def __init__(self, *, tick_size: float, step_size: float, min_notional: float) -> None:
        self._filters = {
            "PRICE_FILTER": {"tickSize": f"{Decimal(str(tick_size)):.8f}"},
            "LOT_SIZE": {
                "stepSize": f"{Decimal(str(step_size)):.8f}",
                "minQty": f"{Decimal(str(step_size)):.8f}",
            },
            "MIN_NOTIONAL": {"notional": f"{Decimal(str(min_notional)):.8f}"},
        }
        self.entry_calls: list[dict[str, Any]] = []

    def get_symbol_filters(self, symbol: str) -> Mapping[str, Any]:
        return self._filters

    def get_available_balance_usdt(self) -> float:
        return 1_000.0

    def place_entry_limit(
        self,
        symbol: str,
        side: str,
        price: Any,
        qty: Any,
        clientOrderId: str,
        timeInForce: str | None = None,
    ) -> Mapping[str, Any]:
        price_f = float(price)
        qty_f = float(qty)
        payload = {
            "symbol": symbol,
            "side": side,
            "price": price_f,
            "qty": qty_f,
            "clientOrderId": clientOrderId,
            "timeInForce": timeInForce,
        }
        self.entry_calls.append(payload)
        return {
            "orderId": len(self.entry_calls),
            "status": "NEW",
            "price": f"{price_f:.4f}",
            "origQty": f"{qty_f:.6f}",
            "executedQty": "0",
            "cumQuote": f"{price_f * qty_f:.6f}",
        }

    def place_stop_reduce_only(
        self,
        symbol: str,
        side: str,
        stopPrice: Any,
        qty: Any,
        clientOrderId: str,
    ) -> Mapping[str, Any]:  # pragma: no cover - simple stub
        return {
            "orderId": f"stop-{clientOrderId}",
            "status": "NEW",
        }


class SweepStubExchange:
    def __init__(
        self,
        *,
        tick_size: float,
        step_size: float,
        min_qty: float,
        min_notional: float,
    ) -> None:
        step_dec = Decimal(str(step_size))
        self._filters = {
            "PRICE_FILTER": {"tickSize": f"{Decimal(str(tick_size)):.8f}"},
            "LOT_SIZE": {
                "stepSize": f"{step_dec:.8f}",
                "minQty": f"{Decimal(str(min_qty)):.8f}",
            },
            "MIN_NOTIONAL": {"notional": f"{Decimal(str(min_notional)):.8f}"},
        }
        self.orders: list[dict[str, Any]] = []

    def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return list(self.orders)

    def cancel_order(self, symbol: str, clientOrderId: str) -> None:  # pragma: no cover
        self.orders = [o for o in self.orders if o.get("clientOrderId") != clientOrderId]

    def get_symbol_filters(self, symbol: str) -> Mapping[str, Any]:
        return self._filters

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        step = float(self._filters["LOT_SIZE"]["stepSize"])
        if step == 0:
            return qty
        return float(int(qty / step) * step)

    def round_price_to_tick(self, symbol: str, price: float) -> float:
        tick = float(self._filters["PRICE_FILTER"]["tickSize"])
        if tick == 0:
            return price
        return float(int(price / tick) * tick)

    def place_entry_limit(
        self,
        symbol: str,
        side: str,
        price: Any,
        qty: Any,
        clientOrderId: str,
        timeInForce: str | None = None,
    ) -> Mapping[str, Any]:
        price_f = float(price)
        qty_f = float(qty)
        order = {
            "orderId": len(self.orders) + 1,
            "clientOrderId": clientOrderId,
            "side": side,
            "price": f"{price_f:.4f}",
            "origQty": f"{qty_f:.6f}",
            "executedQty": "0",
            "status": "NEW",
            "cumQuote": f"{price_f * qty_f:.6f}",
        }
        self.orders.append(order)
        return order

    def get_available_balance_usdt(self) -> float:
        return 5_000.0


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "env-valid",
            "env": {"RISK_NOTIONAL_USDT": "6.6"},
            "settings": {"RISK_NOTIONAL_USDT": 6.6, "INTERVAL": "1h", "ACCOUNT_MODE": "CROSS"},
            "min_notional": 0.0,
            "expect_env_key": "RISK_NOTIONAL_USDT",
            "expect_source_default": False,
            "expect_min_adjust": False,
        },
        {
            "id": "env-absent",
            "env": {},
            "settings": {"RISK_NOTIONAL_USDT": 0.0, "INTERVAL": "1h", "ACCOUNT_MODE": "CROSS"},
            "min_notional": 0.0,
            "expect_env_key": None,
            "expect_source_default": True,
            "expect_min_adjust": False,
        },
        {
            "id": "env-invalid",
            "env": {"RISK_NOTIONAL_USDT": "xyz"},
            "settings": {"RISK_NOTIONAL_USDT": 0.0, "INTERVAL": "1h", "ACCOUNT_MODE": "CROSS"},
            "min_notional": 0.0,
            "expect_env_key": "RISK_NOTIONAL_USDT",
            "expect_env_value": "xyz",
            "expect_source_default": True,
            "expect_min_adjust": False,
        },
        {
            "id": "min-notional",
            "env": {"RISK_NOTIONAL_USDT": "6.6"},
            "settings": {"RISK_NOTIONAL_USDT": 6.6, "INTERVAL": "1h", "ACCOUNT_MODE": "CROSS"},
            "min_notional": 20.0,
            "expect_env_key": "RISK_NOTIONAL_USDT",
            "expect_source_default": False,
            "expect_min_adjust": True,
        },
    ],
    ids=lambda c: c["id"],
)
def test_breakout_dual_tf_order_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, case: Mapping[str, Any]) -> None:
    for key in [
        "RISK_NOTIONAL_USDT",
        "RISK_NOTIONAL_USD",
        "RISK_NOTIONAL",
        "RISKNATIONALUSDT",
        "RISK_NATIONAL_USDT",
        "RISK_NOTIONAL_USDT_PCT",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in case["env"].items():
        monkeypatch.setenv(key, value)

    settings = DummySettings(case["settings"])
    broker = BreakoutStubBroker(
        tick_size=0.1,
        step_size=0.001,
        min_notional=case["min_notional"],
    )
    strategy = BreakoutDualTFStrategy(broker, None, settings)
    strategy._exec_tf = "15m"

    orders = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "entry": "100.0",
        "stop_loss": "95.0",
        "qty": "0.05",
        "qty_tp1": "0.03",
        "qty_tp2": "0.02",
        "timeframe_exec": "15m",
    }

    caplog.clear()
    with caplog.at_level("INFO", logger="bot.strategy.breakout_dual_tf"):
        strategy.place_orders(orders, exchange=broker)

    pre_record = next(
        record for record in caplog.records if "bdtf.order.debug pre" in record.message
    )
    post_record = next(
        record for record in caplog.records if "bdtf.order.debug post" in record.message
    )

    pre_payload = json.loads(pre_record.message.split("bdtf.order.debug pre ", 1)[1])
    post_payload = json.loads(post_record.message.split("bdtf.order.debug post ", 1)[1])

    assert pre_payload["strategy"] == "breakout-dualtf"
    assert pre_payload["symbol"] == "BTCUSDT"
    assert pre_payload["env_key_used"] == case.get("expect_env_key")
    assert pre_payload["source_of_defaults"] is case["expect_source_default"]
    if "expect_env_value" in case:
        assert pre_payload["env_value_raw"] == case["expect_env_value"]
    if case["expect_min_adjust"]:
        assert pre_payload["min_notional_adjustments"]
        assert pre_payload["notional_after_checks"] >= pre_payload["min_notional_target"]
    else:
        assert not pre_payload["min_notional_adjustments"]

    assert pre_payload["price_reference"] == "entry"
    assert "qty_after_step" in pre_payload
    assert "notional_after_checks" in pre_payload

    assert post_payload["order_sent"] is True
    assert post_payload["exchange_response_ok"] is True
    assert post_payload["order_payload"]["type"] == "LIMIT"
    assert post_payload["final_notional_usdt"] >= pre_payload["notional_after_checks"]


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "env-valid",
            "env": {"RISK_NOTIONAL_USDT": "6.6"},
            "settings": {
                "RISK_NOTIONAL_USDT": 6.6,
                "SYMBOL": "ETHUSDT",
                "INTERVAL": "1m",
                "ACCOUNT_MODE": "CROSS",
                "MAX_LOOKBACK_MIN": 5,
                "RISK_PCT": 0.05,
            },
            "min_notional": 0.0,
            "expect_env_key": "RISK_NOTIONAL_USDT",
            "expect_source_default": False,
            "expect_min_qty_applied": False,
        },
        {
            "id": "env-absent",
            "env": {},
            "settings": {
                "RISK_NOTIONAL_USDT": 0.0,
                "SYMBOL": "ETHUSDT",
                "INTERVAL": "1m",
                "ACCOUNT_MODE": "CROSS",
                "MAX_LOOKBACK_MIN": 5,
                "RISK_PCT": 0.05,
            },
            "min_notional": 0.0,
            "expect_env_key": None,
            "expect_source_default": True,
            "expect_min_qty_applied": False,
        },
        {
            "id": "env-invalid",
            "env": {"RISK_NOTIONAL_USDT": "xyz"},
            "settings": {
                "RISK_NOTIONAL_USDT": 0.0,
                "SYMBOL": "ETHUSDT",
                "INTERVAL": "1m",
                "ACCOUNT_MODE": "CROSS",
                "MAX_LOOKBACK_MIN": 5,
                "RISK_PCT": 0.05,
            },
            "min_notional": 0.0,
            "expect_env_key": "RISK_NOTIONAL_USDT",
            "expect_env_value": "xyz",
            "expect_source_default": True,
            "expect_min_qty_applied": False,
        },
        {
            "id": "min-notional",
            "env": {"RISK_NOTIONAL_USDT": "6.6"},
            "settings": {
                "RISK_NOTIONAL_USDT": 6.6,
                "SYMBOL": "ETHUSDT",
                "INTERVAL": "1m",
                "ACCOUNT_MODE": "CROSS",
                "MAX_LOOKBACK_MIN": 5,
                "RISK_PCT": 0.05,
            },
            "min_notional": 10.0,
            "expect_env_key": "RISK_NOTIONAL_USDT",
            "expect_source_default": False,
            "expect_min_qty_applied": True,
        },
    ],
    ids=lambda c: c["id"],
)
def test_liquidity_sweep_order_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, case: Mapping[str, Any]) -> None:
    for key in [
        "RISK_NOTIONAL_USDT",
        "RISK_NOTIONAL_USD",
        "RISK_NOTIONAL",
        "RISKNATIONALUSDT",
        "RISK_NATIONAL_USDT",
        "RISK_NOTIONAL_USDT_PCT",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in case["env"].items():
        monkeypatch.setenv(key, value)

    settings = DummySettings(case["settings"])
    exchange = SweepStubExchange(
        tick_size=0.1,
        step_size=0.1,
        min_qty=0.1,
        min_notional=case["min_notional"],
    )

    def _fake_levels(*_args: Any, **_kwargs: Any) -> Mapping[str, float]:
        return {
            "S": 1.0,
            "R": 1.4,
            "microbuffer": 0.1,
            "buffer_sl": 0.2,
            "atr1m": 0.5,
        }

    def _fake_bracket(*_args: Any, **_kwargs: Any) -> Mapping[str, float]:
        side = _args[0] if _args else "BUY"
        price = _args[1] if len(_args) > 1 else 1.0
        if side == "BUY":
            return {"sl": price - 0.5}
        return {"sl": price + 0.5}

    monkeypatch.setattr(ls_strategy, "compute_levels", _fake_levels)
    monkeypatch.setattr(ls_strategy, "build_bracket", _fake_bracket)

    class _MarketData:
        def fetch_ohlcv(self, *_args: Any, **_kwargs: Any) -> list[list[float]]:  # pragma: no cover - deterministic
            return [[0.0, 1.0, 1.2, 0.8, 1.0, 100.0]] * 5

    market_data = _MarketData()

    caplog.clear()
    with caplog.at_level("INFO", logger="strategies.liquidity_sweep.strategy"):
        ls_strategy.do_preopen(exchange, market_data, settings.SYMBOL, settings)

    pre_record = next(
        record for record in caplog.records if "bdtf.order.debug pre" in record.message
    )
    post_record = next(
        record for record in caplog.records if "bdtf.order.debug post" in record.message
    )

    pre_payload = json.loads(pre_record.message.split("bdtf.order.debug pre ", 1)[1])
    post_payload = json.loads(post_record.message.split("bdtf.order.debug post ", 1)[1])

    assert pre_payload["strategy"] == "liquidity-sweep"
    assert pre_payload["env_key_used"] == case.get("expect_env_key")
    assert pre_payload["source_of_defaults"] is case["expect_source_default"]
    if "expect_env_value" in case:
        assert pre_payload["env_value_raw"] == case["expect_env_value"]

    assert pre_payload["price_reference"] == "entry"
    assert "qty_after_step" in pre_payload
    assert "notional_after_checks" in pre_payload
    if case["expect_min_qty_applied"]:
        assert pre_payload["min_qty_applied"] is True
    else:
        assert pre_payload["min_qty_applied"] in {False, None}

    assert post_payload["order_sent"] is True or post_payload["rejection_reason"] == "existing_order_within_tick"
    if post_payload["order_sent"]:
        assert post_payload["exchange_response_ok"] is True
        assert post_payload["order_payload"]["type"] == "LIMIT"
