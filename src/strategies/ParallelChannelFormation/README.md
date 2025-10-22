# Parallel Channel Formation Strategy

This module implements the **ParallelChannelFormation** strategy for the Binance
bot. It is designed to co-exist with the existing Wedge formation logic without
any modifications to the legacy implementation.

## Key characteristics

* Concurrency guards for both open orders and live positions to prevent
overlapping management.
* Channel geometry detection built on top of reusable helpers shared with other
  strategies.
* Precision enforcement delegated to the same utilities used by
  `WedgeFormation`, including deterministic `clientOrderId` generation and
  strict min-notional checks with configurable buffers.
* Persistent take-profit tracking stored in S3 using the shared `tp_store_s3`
  helper leveraged by `WedgeFormation`, keeping JSON entries keyed by symbol.
* Modular structure with dedicated helpers for geometry, filters, environment
  configuration and state persistence.

Refer to `config/env_variables.json` for all supported environment overrides.
