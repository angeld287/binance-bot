# -*- coding: utf-8 -*-
from strategies import _run_iteration, create_bot

def run_iteration(exchange, cfg):
    symbol = cfg.get("symbol", "BTC/USDT")
    leverage = cfg.get("leverage")
    use_breakout_dynamic_stops = cfg.get("use_breakout_dynamic_stops", False)
    testnet = cfg.get("testnet", False)
    bot = create_bot(
        exchange,
        symbol,
        leverage=leverage,
        use_breakout_dynamic_stops=use_breakout_dynamic_stops,
    )
    return _run_iteration(exchange, bot, testnet, symbol, leverage)


