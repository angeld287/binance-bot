from __future__ import annotations

import os
import sys
import types
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - import side-effect
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DDB_TABLE_DAILY_ACTIVITY", "daily_activity")
os.environ.setdefault("DDB_TABLE_EXECUTION_REPORT", "execution_report")


if "botocore" not in sys.modules:
    botocore_pkg = types.ModuleType("botocore")
    exceptions_pkg = types.ModuleType("botocore.exceptions")

    class _ClientError(Exception):
        def __init__(self, error_response=None, operation_name: str | None = None):
            super().__init__(str(error_response) or "client_error")
            self.response = error_response or {}
            self.operation_name = operation_name or ""

    class _BotoCoreError(Exception):
        pass

    exceptions_pkg.ClientError = _ClientError  # type: ignore[attr-defined]
    exceptions_pkg.BotoCoreError = _BotoCoreError  # type: ignore[attr-defined]
    botocore_pkg.exceptions = exceptions_pkg  # type: ignore[attr-defined]
    sys.modules["botocore"] = botocore_pkg
    sys.modules["botocore.exceptions"] = exceptions_pkg


if "boto3" not in sys.modules:
    boto3_pkg = types.ModuleType("boto3")
    dynamodb_pkg = types.ModuleType("boto3.dynamodb")
    types_pkg = types.ModuleType("boto3.dynamodb.types")

    class _FakeTypeSerializer:
        def serialize(self, value):  # pragma: no cover - trivial serializer
            if isinstance(value, dict):
                return {"M": {k: self.serialize(v) for k, v in value.items()}}
            if isinstance(value, list):
                return {"L": [self.serialize(v) for v in value]}
            if isinstance(value, set):
                return {"SS": sorted(str(v) for v in value)}
            if value is None:
                return {"NULL": True}
            if isinstance(value, bool):
                return {"BOOL": value}
            if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
                return {"N": str(value)}
            return {"S": str(value)}

    class _FakeTypeDeserializer:
        def deserialize(self, value):  # pragma: no cover - trivial deserializer
            if "M" in value:
                return {k: self.deserialize(v) for k, v in value["M"].items()}
            if "L" in value:
                return [self.deserialize(v) for v in value["L"]]
            if "SS" in value:
                return set(value["SS"])
            if "NULL" in value:
                return None
            if "BOOL" in value:
                return bool(value["BOOL"])
            if "N" in value:
                return Decimal(value["N"])
            return value.get("S")

    types_pkg.TypeSerializer = _FakeTypeSerializer  # type: ignore[attr-defined]
    types_pkg.TypeDeserializer = _FakeTypeDeserializer  # type: ignore[attr-defined]

    def _client_factory(*_args, **_kwargs):  # pragma: no cover - tests inject their own clients
        raise RuntimeError("boto3 client should be mocked in tests")

    boto3_pkg.client = _client_factory  # type: ignore[attr-defined]
    sys.modules["boto3"] = boto3_pkg
    sys.modules["boto3.dynamodb"] = dynamodb_pkg
    sys.modules["boto3.dynamodb.types"] = types_pkg
