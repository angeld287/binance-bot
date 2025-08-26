import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from strategies.common import compute_qty_from_usdt, floor_to_step


def test_compute_qty_doge():
    price = 0.217010
    target = 7.0
    exchange_info = {
        "symbols": [
            {
                "symbol": "DOGEUSDT",
                "filters": [
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "1",
                        "minQty": "1",
                        "maxQty": "1000000",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        ]
    }
    qty, notional = compute_qty_from_usdt("DOGEUSDT", price, target, exchange_info)
    expected_qty = floor_to_step(target / price, 1)
    assert qty == expected_qty
    assert notional == qty * price
    assert target * 0.95 <= notional <= target
