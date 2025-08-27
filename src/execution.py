import json

from core import config_loader, exchange, logging_utils
from core import execution as core_execution
from core.logging_utils import log


def handler(event, context):
    """AWS Lambda handler que ejecuta una iteración de trading."""
    log("═══════════════════ 🚀🚀🚀 INICIO EJECUCIÓN LAMBDA 🚀🚀🚀 ═══════════════════")
    cfg = config_loader.get_runtime_config()
    logging_utils.DEBUG_MODE = cfg.get("debug_mode", False)
    ex = exchange.build(cfg)
    price = core_execution.run_iteration(ex, cfg)
    log("═══════════════════ 🛑🛑🛑 FIN EJECUCIÓN LAMBDA 🛑🛑🛑 ═══════════════════")
    return {
        "statusCode": 200,
        "body": json.dumps({"price": price}),
    }
