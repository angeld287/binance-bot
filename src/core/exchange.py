import os
import time
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

from .logging_utils import logger, LoggingSession


def get_proxies():
    """Devuelve diccionario de proxies o None si no se usa proxy."""
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    proxy = os.getenv("PROXY_URL")
    if not testnet and proxy:
        return {"http": proxy, "https": proxy}
    return None


class LoggingClient:
    """Envuelve un Client para registrar cada request."""

    def __init__(self, client, testnet):
        self._client = client
        self.testnet = testnet

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                proxies = get_proxies()
                env = "testnet" if self.testnet else "producción"
                proxy_msg = "sí" if proxies else "no"
                try:
                    return attr(*args, **kwargs)
                except BinanceAPIException as e:
                    if e.code == -1021:
                        self._client.timestamp_offset = server_drift_ms()
                        return attr(*args, **kwargs)
                    raise

            return wrapper
        return attr


def server_drift_ms() -> int:
    """Calcula la deriva de tiempo con el servidor de Binance en ms."""
    url = "https://fapi.binance.com/fapi/v1/time"
    local_ms = int(time.time() * 1000)
    server_ms = local_ms
    try:
        resp = requests.get(url, timeout=5, proxies=get_proxies())
        data = resp.json()
        server_ms = int(data.get("serverTime", server_ms))
    except Exception:
        pass
    drift = server_ms - local_ms
    safety_ms = int(os.getenv("SAFETY_MS", "300"))
    offset = drift - safety_ms
    logger.info(
        "Binance timing: serverTime=%d localTime=%d drift_ms=%+d safety_ms=%d offset_ms=%+d",
        server_ms,
        local_ms,
        drift,
        safety_ms,
        offset,
    )
    return offset


def build(cfg):
    key = cfg.get("api_key")
    secret = cfg.get("api_secret")
    testnet = cfg.get("testnet", False)
    proxies = get_proxies()
    req_params = {"proxies": proxies} if proxies else None
    client = Client(key, secret, testnet=testnet, requests_params=req_params)

    drift_ms = server_drift_ms()
    client.timestamp_offset = drift_ms  # quedamos levemente por detrás
    client.REQUEST_RECVWINDOW = int(os.getenv("RECV_WINDOW_MS", "5000"))
    # Retry único ante -1021
    session = LoggingSession(logger)
    session.headers.update(client.session.headers)
    if proxies:
        session.proxies.update(proxies)
    client.session = session
    return LoggingClient(client, testnet)
