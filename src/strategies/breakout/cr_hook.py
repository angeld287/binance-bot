import logging


def run_cr_on_open_position(ctx, symbol: str, position: dict, logger=None) -> None:
    """Placeholder hook for collateral/CR management when a position is open.

    Parameters
    ----------
    ctx: Any
        Execution context, passed through for future use.
    symbol: str
        Trading symbol (e.g., ``"BTCUSDT"``).
    position: dict
        Information about the currently open position.
    logger: logging.Logger | None
        Optional logger instance. Defaults to the breakout strategy logger.
    """
    logger = logger or logging.getLogger("bot.strategy.breakout")

    side = None if position is None else position.get("side")
    qty = None
    entry = None
    if isinstance(position, dict):
        qty = position.get("positionAmt") or position.get("qty")
        entry = position.get("entryPrice") or position.get("entry")

    logger.info(
        "breakout.cr: hook ejecutado (posición abierta); sym=%s side=%s qty=%s entry=%s",
        symbol,
        side,
        qty,
        entry,
        extra={"strategy": "breakout", "hook": "cr_on_open", "sym": symbol},
    )

    # TODO: evaluar SL/TP existentes
    # TODO: decidir políticas reduceOnly
    # TODO: idempotencia y actualización
    return None
