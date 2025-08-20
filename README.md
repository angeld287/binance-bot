# Binance Futures Bot

Bot de trading de futuros en Binance con patrones en 15m. Permite desactivar stops dinámicos.

## Variables de entorno

- `SYMBOL` (ej. `DOGEUSDT`): par a operar.
- `USE_BREAKOUT_DYNAMIC_STOPS` (por defecto `false`): si es `true` habilita el movimiento automático de stop a break-even/verde y otras lógicas de micro stops.
- `TAKE_PROFIT_PCT` (por defecto `2.0`): porcentaje de take profit.
- `STOP_LOSS_PCT` (por defecto `1.0`): porcentaje de stop loss.

Ejemplo de `.env`:

```
SYMBOL=DOGEUSDT
USE_BREAKOUT_DYNAMIC_STOPS=false
TAKE_PROFIT_PCT=2.0
STOP_LOSS_PCT=1.0
```

## Pruebas

1. `USE_BREAKOUT_DYNAMIC_STOPS=false` → no se moverá el SL a break-even ni habrá trailing micro.
2. `USE_BREAKOUT_DYNAMIC_STOPS=true` → mantiene el comportamiento anterior.

## Logs

Si los mini stops están desactivados:

```
Breakout/mini-SL/BE desactivados por configuración
```

