import os
import time
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

from .logging_utils import logger, LoggingSession


class LoggingClient:
    """Envuelve un Client para registrar cada request."""

    def __init__(self, client, testnet):
        self._client = client
        self.testnet = testnet

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
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
        session = requests.Session()
        session.trust_env = False
        session.proxies.clear()
        resp = session.get(url, timeout=5)
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
    client = Client(key, secret, testnet=testnet)

    drift_ms = server_drift_ms()
    client.timestamp_offset = drift_ms  # quedamos levemente por detrás
    client.REQUEST_RECVWINDOW = int(os.getenv("RECV_WINDOW_MS", "5000"))
    # Retry único ante -1021
    session = LoggingSession(logger)
    session.headers.update(client.session.headers)
    client.session = session
    return LoggingClient(client, testnet)
