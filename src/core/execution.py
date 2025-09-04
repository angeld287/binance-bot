from datetime import datetime, timezone
import os

from strategies import get_strategy_class


def handler(event, context):
    """Main Lambda handler delegating to selected strategy."""
    strategy_name = os.getenv("STRATEGY", "liquidity_sweep")
    cls = get_strategy_class(strategy_name)
    strategy = cls()
    # Placeholder exchange; real implementation would provide a Binance client
    binance = None
    result = strategy.run(exchange=binance, now_utc=datetime.now(timezone.utc), event=event)
    return result
