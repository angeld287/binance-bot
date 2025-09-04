"""Thin wrapper for Binance API providing utility helpers.
This module only defines the interface required by the strategy. The
actual implementation should talk to Binance REST endpoints. For tests we
can provide fakes implementing the same methods."""

from typing import Any, Dict, List, Optional


class BinanceClient:
    def __init__(self, client: Any | None = None):
        self.client = client

    # ----- Helpers -----
    def round_price_to_tick(self, symbol: str, px: float) -> float:
        raise NotImplementedError

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        raise NotImplementedError

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_klines(self, symbol: str, interval: str, start_ms: int | None = None,
                   end_ms: int | None = None, limit: int | None = None):
        raise NotImplementedError

    def open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_order(self, symbol: str, clientOrderId: str | None = None,
                  orderId: str | None = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def place_limit(self, symbol: str, side: str, price: float, qty: float,
                     clientOrderId: str, timeInForce: str = "GTC"):
        raise NotImplementedError

    def cancel_order(self, symbol: str, orderId: str | None = None,
                     clientOrderId: str | None = None):
        raise NotImplementedError

    def place_sl_reduce_only(self, symbol: str, side: str, stopPrice: float,
                              qty: float, clientOrderId: str):
        raise NotImplementedError

    def place_tp_reduce_only(self, symbol: str, side: str, tpPrice: float,
                              qty: float, clientOrderId: str):
        raise NotImplementedError

    def get_available_balance_usdt(self) -> Optional[float]:
        raise NotImplementedError
