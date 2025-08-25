from .logging_utils import log


def get_current_position_info(exchange, symbol) -> dict:
    """Obtiene información de la posición actual."""
    try:
        sym = symbol.replace("/", "")
        info = exchange.futures_position_information(symbol=sym)
        if isinstance(info, list):
            pos = info[0] if len(info) > 0 else None
        else:
            pos = info
        if pos is None:
            return None
        amt = float(pos.get("positionAmt", 0))
        if amt != 0:
            return pos
    except Exception as e:
        log(f"❌❌❌❌❌ Error consultando posición: {e}")
    return None


def has_active_position(exchange, symbol) -> bool:
    return get_current_position_info(exchange, symbol) is not None
