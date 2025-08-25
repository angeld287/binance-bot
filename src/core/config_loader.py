import os
from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "y")


def get_runtime_config() -> dict:
    load_dotenv()
    return {
        "api_key": os.getenv("BINANCE_API_KEY"),
        "api_secret": os.getenv("BINANCE_API_SECRET"),
        "testnet": _env_bool("BINANCE_TESTNET", False),
        "symbol": os.getenv("SYMBOL", "BTC/USDT"),
        "sup_interval": os.getenv("SUP_INTERVAL", "5m"),
        "leverage": int(os.getenv("LEVERAGE", "5")),
        "use_breakout_dynamic_stops": _env_bool("USE_BREAKOUT_DYNAMIC_STOPS", False),
        "debug_mode": _env_bool("DEBUG_MODE", False),
    }
