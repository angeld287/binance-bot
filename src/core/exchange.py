import os
import time
import requests

from .logging_utils import logger


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
                #log(f"Llamada {name} | entorno: {env} | usando proxy: {proxy_msg}")
                return attr(*args, **kwargs)

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
    logger.info(
        "Binance timing: serverTime=%d localTime=%d drift_ms=%+d",
        server_ms,
        local_ms,
        drift,
    )
    return drift
