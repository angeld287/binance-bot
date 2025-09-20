# Breakout Dual Timeframe Strategy

La estrategia `breakout_dual_tf` combina el análisis de niveles en una
temporalidad mayor con ejecuciones en una temporalidad menor. Utiliza la
variable global `INTERVAL` como timeframe macro para la construcción de
niveles y deriva automáticamente el timeframe de ejecución mediante
`downscale_interval`.

## Configuración local por símbolo

El framework multi-estrategia puede inyectar overrides específicos por
símbolo a través del argumento `config` del constructor. Ejemplo en formato
JSON (o `dict`) sin introducir nuevas variables de entorno globales:

```json
{
  "BTCUSDT": {
    "strategy": "breakout_dual_tf",
    "config": {
      "K_ATR": 0.25,
      "VOL_REL_MIN": 1.8,
      "RR_MIN": 2.0,
      "RETEST_TIMEOUT": 4,
      "USE_RETEST": true
    }
  },
  "ETHUSDT": {
    "strategy": "breakout_dual_tf",
    "config": {
      "USE_RETEST": false,
      "MAX_RETRIES": 3,
      "COOLDOWN_BARS": 5
    }
  }
}
```

Los parámetros omitidos se resuelven con los defaults internos de la
estrategia. Se reaprovechan las variables globales existentes como
`STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `RISK_NOTIONAL_USDT`, `MAX_RETRIES` y
`COOLDOWN_BARS` cuando estén definidas en el entorno.

## Ejemplo de logs

La estrategia publica logs estructurados para cada decisión importante. A
continuación se muestran ejemplos típicos (formato JSON comprimido para
brevedad):

```text
{"action": "reject", "reason": "vol_rel", "level": {"price": 27450.0, "type": "R"}, "vol_rel": 0.9, "threshold": 1.5}
{"action": "pending_retest", "level": {"price": 27510.0, "type": "R"}, "direction": "LONG", "k_atr": 0.3, "atr": 45.2, "close": 27555.0}
{"action": "retest_detected", "level": {"price": 27510.0, "type": "R"}, "direction": "LONG", "tolerance_atr": 0.2}
{"action": "signal", "strategy": "breakout_dual_tf", "orders": {"symbol": "BTCUSDT", "side": "BUY", "entry": 27580.0, "stop_loss": 27460.0, "take_profit_1": 27680.0, "take_profit_2": 27780.0, "qty": 0.35, "rr": 2.1, "breakeven_on_tp1": true, "qty_target_src": "NOTIONAL_RISK", "timeframe_exec": "15m", "level": {"price": 27510.0, "type": "R"}}, "atr": 45.2, "volume_rel": 1.9, "ema_fast": 27540.0, "ema_slow": 27480.5}
```

En los logs anteriores se aprecian los motivos de rechazo (volumen relativo
bajo, timeout de retest, límites de reintento, etc.) y los parámetros de la
señal válida (entrada, SL, TPs, RR, qty calculada y timeframes utilizados).

