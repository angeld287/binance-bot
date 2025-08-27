"""Compatibility shim for the legacy AWS Lambda handler.

The real entry point now lives in :mod:`execution.lambda_handler`.  This
module is kept to avoid breaking deployments that still reference
``core.bot_trading.handler``.
"""

from execution import lambda_handler as _lambda_handler


def handler(event, context):  # pragma: no cover - thin wrapper
    return _lambda_handler(event, context)

