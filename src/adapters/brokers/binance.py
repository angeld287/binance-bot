"""Binance broker adapter."""

from __future__ import annotations

import hmac, hashlib, logging, urllib.parse as _url
import os
import time
from decimal import Decimal
from typing import Any, Dict

from binance.client import Client
from requests import Session

from config.settings import Settings
from core.ports.broker import BrokerPort
from common.utils import sanitize_client_order_id
from common.symbols import normalize_symbol as _normalize_symbol

logger = logging.getLogger(__name__)


def _to_binance_symbol(sym: str) -> str:
    return _normalize_symbol(sym)


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
        self._filters_cache: Dict[str, Dict[str, Any]] = {}

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
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="LIMIT",
                price=price,
                quantity=qty,
                timeInForce=timeInForce,
                newClientOrderId=safe_id,
                **extra,
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
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="STOP_MARKET",
                stopPrice=stopPrice,
                quantity=qty,
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
            return self._client.futures_create_order(
                symbol=_to_binance_symbol(symbol),
                side=side,
                type="LIMIT",
                price=tpPrice,
                quantity=qty,
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
        if not self._filters_cache:
            try:
                info = self._client.futures_exchange_info()
                for s in info.get("symbols", []):
                    self._filters_cache[s.get("symbol", "")] = {
                        f["filterType"]: f for f in s.get("filters", [])
                    }
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Failed to fetch symbol filters: %s", exc)
                raise
        try:
            return self._filters_cache[sym]
        except KeyError:
            raise ValueError(f"Symbol {symbol} not found") from None

    def round_price_to_tick(self, symbol: str, px: float) -> float:
        filters = self.get_symbol_filters(symbol)
        tick = Decimal(filters["PRICE_FILTER"]["tickSize"])
        return float((Decimal(str(px)) // tick) * tick)

    def round_qty_to_step(self, symbol: str, qty: float) -> float:
        filters = self.get_symbol_filters(symbol)
        step = Decimal(filters["LOT_SIZE"]["stepSize"])
        return float((Decimal(str(qty)) // step) * step)

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

