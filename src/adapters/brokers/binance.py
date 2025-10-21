"""Binance broker adapter."""

from __future__ import annotations

import hmac, hashlib, logging, urllib.parse as _url
import os
import time
from datetime import timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict

from binance.client import Client
from requests import Session

from config.settings import Settings
from core.ports.broker import BrokerPort
from common.precision import FiltersCache, format_decimal, round_to_step, round_to_tick
from common.rounding_diag import emit_rounding_diag, format_rounding_diag_number
from common.utils import sanitize_client_order_id
from common.symbols import normalize_symbol as _normalize_symbol

logger = logging.getLogger(__name__)


def _to_binance_symbol(sym: str) -> str:
    return _normalize_symbol(sym)


def to_decimal_or_none(value: Any) -> Decimal | None:
    try:
        if value is None:
            return None
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def is_multiple(value_dec: Decimal | None, step_dec: Decimal | None) -> bool | None:
    try:
        if value_dec is None or step_dec is None or step_dec == 0:
            return None
        return (value_dec % step_dec) == 0
    except Exception:
        return None


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
            filters = self.get_symbol_filters(symbol)
            tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
            step = Decimal(filters["LOT_SIZE"]["stepSize"])
            side_norm = (side or "").upper()
            price_dec = round_to_tick(price, tick, side=side_norm)
            qty_dec = round_to_step(qty, step)
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
            if os.getenv("ROUNDING_DIAG") == "1":
                diag_payload: dict[str, Any] = {
                    "tag": "D_payload_final",
                    "symbol": symbol,
                    "side": side,
                    "orderType": payload.get("type"),
                    "price_final": None,
                    "qty_final": None,
                    "stop_final": None,
                    "price_final_is_multiple": None,
                    "qty_final_is_multiple": None,
                    "stop_final_is_multiple": None,
                    "types": {
                        "price": None,
                        "qty": None,
                        "stop": None,
                    },
                    "reduceOnly": payload.get("reduceOnly"),
                    "workingType": payload.get("workingType"),
                    "timeInForce": payload.get("timeInForce"),
                }
                try:
                    tick_dec = to_decimal_or_none(tick)
                    step_dec = to_decimal_or_none(step)
                    price_final_dec = to_decimal_or_none(payload.get("price"))
                    qty_final_dec = to_decimal_or_none(payload.get("quantity"))
                    stop_final_dec = to_decimal_or_none(payload.get("stopPrice"))

                    if price_final_dec is not None:
                        diag_payload["price_final"] = format_rounding_diag_number(
                            price_final_dec
                        )
                    if qty_final_dec is not None:
                        diag_payload["qty_final"] = format_rounding_diag_number(
                            qty_final_dec
                        )
                    if stop_final_dec is not None:
                        diag_payload["stop_final"] = format_rounding_diag_number(
                            stop_final_dec
                        )

                    price_final_is_multiple = is_multiple(price_final_dec, tick_dec)
                    qty_final_is_multiple = is_multiple(qty_final_dec, step_dec)
                    stop_final_is_multiple = (
                        True
                        if stop_final_dec is None
                        else is_multiple(stop_final_dec, tick_dec)
                    )

                    diag_payload["price_final_is_multiple"] = (
                        bool(price_final_is_multiple)
                        if price_final_is_multiple is not None
                        else None
                    )
                    diag_payload["qty_final_is_multiple"] = (
                        bool(qty_final_is_multiple)
                        if qty_final_is_multiple is not None
                        else None
                    )
                    diag_payload["stop_final_is_multiple"] = (
                        True
                        if stop_final_is_multiple is True
                        else (
                            bool(stop_final_is_multiple)
                            if stop_final_is_multiple is not None
                            else None
                        )
                    )

                    if "price" in payload:
                        diag_payload["types"]["price"] = type(
                            payload.get("price")
                        ).__name__
                    if "quantity" in payload:
                        diag_payload["types"]["qty"] = type(
                            payload.get("quantity")
                        ).__name__
                    if "stopPrice" in payload:
                        diag_payload["types"]["stop"] = type(
                            payload.get("stopPrice")
                        ).__name__
                except Exception as exc:  # pragma: no cover - diagnostics only
                    diag_payload["warn"] = str(exc)
                emit_rounding_diag(diag_payload, logger=logger)
            return self._client.futures_create_order(
                **payload,
            )
        except Exception as exc:  # pragma: no cover - network failures
            if os.getenv("ROUNDING_DIAG") == "1":
                diag_payload_err: dict[str, Any] = {
                    "tag": "E_error",
                    "http_status": getattr(exc, "status_code", None)
                    or getattr(exc, "status", None),
                    "binance_code": getattr(exc, "code", None),
                    "binance_msg": getattr(exc, "message", None) or str(exc),
                    "echo_price": None,
                    "echo_qty": None,
                    "echo_stop": None,
                }
                try:
                    if "payload" in locals():
                        price_echo_dec = to_decimal_or_none(payload.get("price"))
                        qty_echo_dec = to_decimal_or_none(payload.get("quantity"))
                        stop_echo_dec = to_decimal_or_none(payload.get("stopPrice"))

                        if price_echo_dec is not None:
                            diag_payload_err["echo_price"] = (
                                format_rounding_diag_number(price_echo_dec)
                            )
                        if qty_echo_dec is not None:
                            diag_payload_err["echo_qty"] = format_rounding_diag_number(
                                qty_echo_dec
                            )
                        if stop_echo_dec is not None:
                            diag_payload_err["echo_stop"] = format_rounding_diag_number(
                                stop_echo_dec
                            )
                except Exception as diag_exc:  # pragma: no cover - diagnostics only
                    diag_payload_err["warn"] = str(diag_exc)
                emit_rounding_diag(diag_payload_err, logger=logger)
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
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="MARKET",
                quantity=qty,
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
            filters = self.get_symbol_filters(symbol)
            tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
            step = Decimal(filters["LOT_SIZE"]["stepSize"])
            stop_price_dec = round_to_tick(stopPrice, tick, side=side)
            qty_dec = round_to_step(qty, step, rounding=ROUND_DOWN)
            adjusted_stop_price = format_decimal(stop_price_dec)
            adjusted_qty = format_decimal(qty_dec)

            logger.info(
                "Stop reduce-only price adjust | symbol=%s side=%s requested=%s adjusted=%s tickSize=%s",
                symbol,
                side,
                format_decimal(stopPrice),
                adjusted_stop_price,
                tick,
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
            filters = self.get_symbol_filters(symbol)
            tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
            step = Decimal(filters["LOT_SIZE"]["stepSize"])
            price_dec = round_to_tick(tpPrice, tick, side=side)
            qty_dec = round_to_step(qty, step)
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
        filters = self.get_symbol_filters(symbol)
        tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
        rounded = round_to_tick(px, tick, side="BUY")
        return float(rounded)

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        filters = self.get_symbol_filters(symbol)
        step = Decimal(filters["LOT_SIZE"]["stepSize"])
        rounded = round_to_step(qty, step)
        return float(rounded)

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

