import os
import logging
import requests

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


def _mask_key(key: str) -> str:
    """Devuelve la clave enmascarada mostrando solo los primeros y últimos 3 caracteres."""
    if not key:
        return ""
    if len(key) <= 6:
        return key[0] + "***" + key[-1]
    return f"{key[:3]}***{key[-3:]}"


class LoggingSession(requests.Session):
    """Sesión de requests que registra cada request saliente."""

    def __init__(self, logger):
        super().__init__()
        self._logger = logger

    def request(self, method, url, **kwargs):
        headers = kwargs.get("headers", {})
        data = kwargs.get("data")
        ctype = headers.get("Content-Type") or self.headers.get("Content-Type")
        api_key = headers.get("X-MBX-APIKEY") or self.headers.get("X-MBX-APIKEY")

        body = data or ""
        if isinstance(body, bytes):
            body = body.decode()

        has_sig_end = False
        if body and "signature=" in body:
            prefix, sig = body.rsplit("signature=", 1)
            has_sig_end = body.endswith("signature=" + sig)
            body = f"{prefix}signature=<hidden len={len(sig)}>"

        masked_key = _mask_key(api_key)
        is_form = ctype == "application/x-www-form-urlencoded"

        if DEBUG_MODE:
            self._logger.info(
                "DEBUG - Request %s %s | Content-Type: %s | X-MBX-APIKEY: %s | body: %s",
                method.upper(),
                url,
                ctype,
                masked_key,
                body,
            )
            self._logger.info(
                "DEBUG - Content-Type is application/x-www-form-urlencoded: %s | signature at end: %s",
                is_form,
                has_sig_end,
            )
        return super().request(method, url, **kwargs)


logger = logging.getLogger("bot")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.propagate = False


def log(msg: str):
    logger.info(msg)


def debug_log(msg: str):
    if DEBUG_MODE:
        logger.info(f"DEBUG - {msg}")
