"""Binance broker adapter."""

from __future__ import annotations

import hmac, hashlib, logging, urllib.parse as _url
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, ROUND_UP
from typing import Any, Dict

from binance.client import Client
from requests import Session

from config.settings import Settings
from core.ports.broker import BrokerPort
from common.precision import (
    FiltersCache,
    format_decimal,
    round_price_for_side,
    round_qty,
    to_decimal,
)
from common.utils import sanitize_client_order_id
from common.symbols import normalize_symbol as _normalize_symbol

logger = logging.getLogger(__name__)


def _to_binance_symbol(sym: str) -> str:
    return _normalize_symbol(sym)


@dataclass(frozen=True)
class _SymbolPrecision:
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal


def _calc_drift_ms(client: Client) -> int:
    now_ms = int(time.time() * 1000)
    try:  # pragma: no cover - network failures
        server_ms = client.futures_time().get("serverTime", now_ms)
    except Exception:  # pragma: no cover - network failures
        return 0
    return int(server_ms) - now_ms

def attach_signature_audit(client, api_secret: str):
    key = (api_secret or "").strip().encode()

    if hasattr(client, "session") and not getattr(client.session, "_sig_audit2", False):
        def _hook(resp, *_, **__):
            req = resp.request
            if req.method != "POST" or "/fapi/v1/order" not in req.url:
                return

            # --- cuerpo enviado ---
            body = req.body or b""
            body_str = body.decode("utf-8", "ignore") if isinstance(body, (bytes, bytearray)) else str(body)

            # --- query enviado ---
            parsed = _url.urlsplit(req.url)
            qs = parsed.query  # puede contener signature

            # Tomar firma “enviada”: primero de body, si no, de la URL
            sent_sig = ""
            for s in (body_str, qs):
                if not s:
                    continue
                i = s.rfind("&signature=")
                if i >= 0:
                    sent_sig = s[i + len("&signature="):]
                    break

            # ¿Con qué payload firmar?
            # python-binance normalmente firma EXACTAMENTE el string que envía (cuerpo para POST).
            # Pero hay versiones que ponen todo en query. Probamos ambos y vemos si alguno cuadra.
            candidates = [p for p in (body_str, qs) if p]
            matches = []
            for payload in candidates:
                calc = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
                matches.append(calc == sent_sig)

            # lista de nombres de parámetros reales que viajaron (del body si existió; si no, del query)
            use = body_str or qs
            names = [p.split("=",1)[0] for p in use.split("&") if p]

            logger.warning(
                "SIG-AUDIT2 | sig_in_body=%s sig_in_url=%s | match_body=%s match_url=%s | params=%s | payload_prefix=%r",
                ("&signature=" in body_str), ("&signature=" in qs),
                (matches[0] if body_str else None), (matches[-1] if qs else None),
                names, use[:160]
            )
        client.session.hooks.setdefault("response", []).append(_hook)
        client.session._sig_audit2 = True


class BinanceBroker(BrokerPort):
    """Broker implementation using Binance Futures REST API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Reuse an HTTP session to benefit from keep-alive and reduce latency
        self._session = Session()
        timeout = (
            settings.HTTP_TIMEOUT
            if hasattr(settings, "HTTP_TIMEOUT") and settings.HTTP_TIMEOUT
            else 30
        )
        requests_params = {"timeout": timeout}

        self._client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_API_SECRET,
            testnet=settings.BINANCE_TESTNET,
            requests_params=requests_params,
        )

        # Cache for symbol filters to avoid repeated ``exchangeInfo`` calls
        ttl_minutes = getattr(settings, "FILTERS_CACHE_TTL_MIN", 5)
        try:
            ttl_value = max(float(ttl_minutes), 1.0)
        except (TypeError, ValueError):
            ttl_value = 5.0
        self._filters_cache = FiltersCache(ttl=timedelta(minutes=ttl_value))

    def _redact(self, s: str) -> str:
        if not s:
            return "<empty>"
        s = str(s)
        return s[:6] + "…" + s[-4:]

    def normalize_symbol(self, symbol: str) -> str:
        """Return ``symbol`` uppercased without the ``/`` character."""
        return _normalize_symbol(symbol)

    # ------------------------------------------------------------------
    # Positions
    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Return position information for ``symbol`` or ``None``.

        The method queries Binance USDT-M futures using the existing client
        and returns a simplified dictionary with at least ``positionAmt`` and
        ``entryPrice``. If no position is found for the symbol, ``None`` is
        returned and an ``INFO`` log is emitted. The ``symbol`` passed in is
        assumed to be already normalised.
        """

        try:
            data = self._client.futures_position_information(symbol=symbol)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch position information: %s", exc)
            raise

        positions = data if isinstance(data, list) else [data]
        positions = [p for p in positions if p.get("symbol") == symbol]
        if not positions:
            logger.info("No position data for %s", symbol)
            return None

        if len(positions) > 1:
            net_amt = sum(float(p.get("positionAmt", 0.0)) for p in positions)
            total_qty = sum(abs(float(p.get("positionAmt", 0.0))) for p in positions)
            entry_price = (
                sum(
                    float(p.get("entryPrice", 0.0))
                    * abs(float(p.get("positionAmt", 0.0)))
                    for p in positions
                )
                / total_qty
                if total_qty
                else 0.0
            )
            return {"positionAmt": str(net_amt), "entryPrice": str(entry_price)}

        pos = positions[0]
        return {
            "positionAmt": pos.get("positionAmt", "0"),
            "entryPrice": pos.get("entryPrice", "0"),
        }

    # ------------------------------------------------------------------
    # Orders
    def open_orders(self, symbol: str) -> list[Any]:
        try:
            return self._client.futures_get_open_orders(symbol=_to_binance_symbol(symbol))  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch open orders: %s", exc)
            raise

    def get_order(
        self,
        symbol: str,
        clientOrderId: str | None = None,
        orderId: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": _to_binance_symbol(symbol)}
        if orderId is not None:
            params["orderId"] = orderId
        if clientOrderId is not None:
            params["origClientOrderId"] = sanitize_client_order_id(clientOrderId)
        try:
            return self._client.futures_get_order(**params)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch order: %s", exc)
            raise

    def place_entry_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        clientOrderId: str,
        timeInForce: str = "GTC",
        **extra: Any,
    ) -> dict[str, Any]:
        """Place a limit entry order."""

        try:
            safe_id = sanitize_client_order_id(clientOrderId)
            precision = self._symbol_precision(symbol)
            price_dec = round_price_for_side(price, precision.tick_size, side, "LIMIT")
            qty_dec = round_qty(qty, precision.step_size)
            qty_dec = self._apply_qty_guards(
                symbol=symbol,
                side=side,
                order_type="LIMIT",
                price=price_dec,
                qty=qty_dec,
                precision=precision,
            )
            payload = {
                "symbol": _to_binance_symbol(symbol),
                "side": side,
                "type": "LIMIT",
                "price": format_decimal(price_dec),
                "quantity": format_decimal(qty_dec),
                "timeInForce": timeInForce,
                "newClientOrderId": safe_id,
                **extra,
            }
            return self._client.futures_create_order(
                **payload,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place limit order: %s", exc)
            raise

    def place_entry_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        clientOrderId: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """Place a market entry order."""

        try:
            safe_id = sanitize_client_order_id(clientOrderId)
            precision = self._symbol_precision(symbol)
            qty_dec = round_qty(qty, precision.step_size)
            qty_dec = self._apply_qty_guards(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                price=None,
                qty=qty_dec,
                precision=precision,
            )
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="MARKET",
                quantity=format_decimal(qty_dec),
                newClientOrderId=safe_id,
                **extra,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place market order: %s", exc)
            raise

    def cancel_order(
        self,
        symbol: str,
        orderId: str | None = None,
        clientOrderId: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": _to_binance_symbol(symbol)}
        if orderId is not None:
            params["orderId"] = orderId
        if clientOrderId is not None:
            params["origClientOrderId"] = sanitize_client_order_id(clientOrderId)
        try:
            return self._client.futures_cancel_order(**params)
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to cancel order: %s", exc)
            raise

    def place_stop_reduce_only(
        self,
        symbol: str,
        side: str,
        stopPrice: float,
        qty: float,
        clientOrderId: str,
    ) -> dict[str, Any]:
        try:
            safe_id = sanitize_client_order_id(clientOrderId)
            precision = self._symbol_precision(symbol)
            stop_price_dec = round_price_for_side(
                stopPrice, precision.tick_size, side, "STOP_MARKET"
            )
            qty_dec = round_qty(qty, precision.step_size)
            qty_dec = self._apply_qty_guards(
                symbol=symbol,
                side=side,
                order_type="STOP_MARKET",
                price=stop_price_dec,
                qty=qty_dec,
                precision=precision,
            )
            adjusted_stop_price = format_decimal(stop_price_dec)
            adjusted_qty = format_decimal(qty_dec)

            logger.info(
                "Stop reduce-only price adjust | symbol=%s side=%s requested=%s adjusted=%s tickSize=%s",
                symbol,
                side,
                format_decimal(stopPrice),
                adjusted_stop_price,
                format_decimal(precision.tick_size),
            )
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="STOP_MARKET",
                stopPrice=adjusted_stop_price,
                quantity=adjusted_qty,
                reduceOnly=True,
                newClientOrderId=safe_id,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place SL order: %s", exc)
            raise

    def place_tp_reduce_only(
        self,
        symbol: str,
        side: str,
        tpPrice: float,
        qty: float,
        clientOrderId: str,
    ) -> dict[str, Any]:
        """Place a take-profit limit reduce-only order.

        Binance does not expose a dedicated "TP-LIMIT" order type in the
        futures REST API, therefore we rely on a standard LIMIT order with the
        ``reduceOnly`` flag so the position size can only decrease.
        """

        try:
            safe_id = sanitize_client_order_id(clientOrderId)
            precision = self._symbol_precision(symbol)
            price_dec = round_price_for_side(tpPrice, precision.tick_size, side, "LIMIT")
            qty_dec = round_qty(qty, precision.step_size)
            qty_dec = self._apply_qty_guards(
                symbol=symbol,
                side=side,
                order_type="LIMIT",
                price=price_dec,
                qty=qty_dec,
                precision=precision,
            )
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="LIMIT",
                price=format_decimal(price_dec),
                quantity=format_decimal(qty_dec),
                timeInForce="GTC",
                reduceOnly=True,
                newClientOrderId=safe_id,
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to place TP order: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Helpers
    def get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Return and cache the exchange filters for ``symbol``.

        The first call populates a cache with the filters for all symbols to
        minimise subsequent HTTP requests.
        """

        sym = _to_binance_symbol(symbol)
        def _on_error(err: Exception) -> None:  # pragma: no cover - network failure
            logger.warning("exchangeInfo refresh failed: %s", err)

        try:
            return self._filters_cache.get(
                self._client,
                sym,
                on_refresh_error=_on_error,
            )
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch symbol filters: %s", exc)
            raise

    def round_price_to_tick(self, symbol: str, px: float) -> float:
        precision = self._symbol_precision(symbol)
        rounded = round_price_for_side(px, precision.tick_size, "BUY", "LIMIT")
        return float(rounded)

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        precision = self._symbol_precision(symbol)
        rounded = round_qty(qty, precision.step_size)
        return float(rounded)

    # ------------------------------------------------------------------
    # Precision helpers

    def _symbol_precision(self, symbol: str) -> _SymbolPrecision:
        filters = self.get_symbol_filters(symbol)
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        market_filter = filters.get("MARKET_LOT_SIZE", {})
        min_notional_filter = filters.get("MIN_NOTIONAL", {})

        def _safe_decimal(value: Any) -> Decimal:
            try:
                return to_decimal(value)
            except Exception:
                return Decimal("0")

        tick_size = _safe_decimal(price_filter.get("tickSize"))
        step_candidates = [
            _safe_decimal(lot_filter.get("stepSize")),
            _safe_decimal(market_filter.get("stepSize")),
        ]
        step_size = next((s for s in step_candidates if s > 0), step_candidates[0])
        min_qty_candidates = [
            _safe_decimal(lot_filter.get("minQty")),
            _safe_decimal(market_filter.get("minQty")),
        ]
        min_qty = max(min_qty_candidates)
        min_notional = _safe_decimal(
            min_notional_filter.get("notional") or min_notional_filter.get("minNotional")
        )
        return _SymbolPrecision(
            tick_size=tick_size,
            step_size=step_size,
            min_qty=min_qty,
            min_notional=min_notional,
        )

    def _apply_qty_guards(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        price: Decimal | None,
        qty: Decimal,
        precision: _SymbolPrecision,
    ) -> Decimal:
        qty_adj = round_qty(qty, precision.step_size)

        if precision.min_qty > 0 and qty_adj < precision.min_qty:
            qty_adj = self._ceil_to_step(precision.min_qty, precision.step_size)
            qty_adj = round_qty(qty_adj, precision.step_size)

        if price is not None and price > 0 and precision.min_notional > 0:
            notional = price * qty_adj
            if notional < precision.min_notional:
                required = precision.min_notional / price
                required = self._ceil_to_step(required, precision.step_size)
                if precision.min_qty > 0 and required < precision.min_qty:
                    required = self._ceil_to_step(precision.min_qty, precision.step_size)
                required = round_qty(required, precision.step_size)
                if required > qty_adj:
                    qty_adj = required
                if price * qty_adj < precision.min_notional:
                    logger.warning(
                        "Order rejected below minNotional | symbol=%s side=%s type=%s price=%s qty=%s minNotional=%s",
                        symbol,
                        side,
                        order_type,
                        format_decimal(price),
                        format_decimal(qty_adj),
                        format_decimal(precision.min_notional),
                    )
                    raise ValueError("Order notional below minimum for symbol")

        if precision.min_qty > 0 and qty_adj < precision.min_qty:
            qty_adj = self._ceil_to_step(precision.min_qty, precision.step_size)
            qty_adj = round_qty(qty_adj, precision.step_size)

        return qty_adj

    @staticmethod
    def _ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        return value.quantize(step, rounding=ROUND_UP)

    def get_available_balance_usdt(self) -> float:
        """Return available USDT balance.

        If the call fails or the USDT asset is missing, the method falls back
        to ``settings.RISK_NOTIONAL_USDT`` (default ``0``).
        """

        try:
            balances = self._client.futures_account_balance()
            for bal in balances:
                if bal.get("asset") == "USDT":
                    return float(bal.get("availableBalance", 0.0))
        except Exception as exc:  # pragma: no cover - network failures
            logger.warning(
                "Failed to fetch balance, using RISK_NOTIONAL_USDT fallback: %s",
                exc,
            )
            return float(getattr(self._settings, "RISK_NOTIONAL_USDT", 0.0))

        logger.warning(
            "USDT balance not found, using RISK_NOTIONAL_USDT fallback",
        )
        return float(getattr(self._settings, "RISK_NOTIONAL_USDT", 0.0))


def make_broker(settings: Settings) -> BrokerPort:
    """Factory for a :class:`BrokerPort` bound to Binance."""

    return BinanceBroker(settings)

