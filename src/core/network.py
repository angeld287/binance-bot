import time
import requests
from .logging_utils import logger

_LAST_CHECK_TS = 0
_CHECK_INTERVAL = 1800  # 30 minutes


def _new_session():
    session = requests.Session()
    session.trust_env = False
    session.proxies.clear()
    return session


def network_sanity_check():
    global _LAST_CHECK_TS
    now = time.time()
    if now - _LAST_CHECK_TS < _CHECK_INTERVAL:
        return
    _LAST_CHECK_TS = now
    try:
        session = _new_session()
        ip_resp = session.get("https://api.ipify.org?format=json", timeout=5)
        ip = ip_resp.json().get("ip")
        logger.info("OUTBOUND_IP: %s", ip)
        time_resp = session.get("https://api.binance.com/api/v3/time", timeout=5)
        server_time = time_resp.json().get("serverTime")
        logger.info("Binance server time: %s", server_time)
    except Exception as e:
        logger.warning("Network sanity check failed: %s", e)
