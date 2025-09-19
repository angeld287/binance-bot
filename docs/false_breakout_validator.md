# False Breakout Validator (FBV)

The breakout strategy can optionally run an additional "false breakout" validator
before creating the entry order.  The validator analyses the recent OHLCV series and
rejects weak signals such as wick-driven moves, low-volume pops or stale levels.

## Activation

Set the feature flag and restart the bot:

```bash
export BREAKOUT_FBV_ENABLED=true
```

When disabled (default), the strategy behaves exactly as before.

## Parameters

| Env var | Default | Description |
| --- | --- | --- |
| `FBV_LEVEL_LOOKBACK` | `48` | Candles to scan for the level (≈2 days on 1h). |
| `FBV_MIN_TOUCHES` | `2` | Minimum tolerated touches before trading the level. |
| `FBV_TOUCH_TOLERANCE_PCT` | `0.10` | Touch tolerance as percent of price. |
| `FBV_CLOSE_BUFFER_PCT` | `0.10` | Close confirmation buffer (percent). |
| `FBV_CLOSE_BUFFER_ATR` | `0.2` | ATR14 multiplier for close confirmation. |
| `FBV_VOL_MA_MULT` | `1.3` | Minimum volume ratio versus SMA(`FBV_VOL_MA_N`). |
| `FBV_VOL_MA_N` | `20` | Length of the simple volume moving average. |
| `FBV_WICK_MAX_RATIO` | `1.2` | Maximum wick-to-body ratio allowed. |
| `FBV_RETEST_WAIT` | `3` | Candles allowed between confirmation and retest. |
| `FBV_TIME_WINDOW` | `6` | Maximum candles between first touch and signal. |
| `FBV_USE_WICK_TOUCH` | `true` | Whether to count wick contacts as touches. |

All parameters are read from the environment on startup and logged with the
`fbv.settings` prefix.

## Behaviour summary

* If the breakout candle closes above/below the level by at least the buffer and
  confirms with strong volume and a controlled wick, the signal is allowed.
* Without a buffer close, a quick retest (≤ `FBV_RETEST_WAIT`) after a confirmed
  breakout is accepted.
* Signals are rejected when:
  - The level lacks the required number of touches.
  - The first touch happened more than `FBV_TIME_WINDOW` candles ago.
  - The breakout lacks confirmation (`close_buffer`), has excessive wick or
    insufficient volume.

## Log snippets

* `fbv.settings {...}` – configuration at startup.
* `fbv.result {...}` – validation outcome (allowed flag, reason, metrics).
* `fbv.blocked_by=reason` – counter emitted when a signal is rejected.
* `status=skipped_fbv` – status dictionary returned by the breakout strategy
  when the validator blocks an order.

## Examples

```bash
# Strict volume filter with a narrower tolerance
export BREAKOUT_FBV_ENABLED=true
export FBV_VOL_MA_MULT=1.5
export FBV_TOUCH_TOLERANCE_PCT=0.05

# Aggressive mode: allow touches based on closes only
export FBV_USE_WICK_TOUCH=false
```
