# Binance Futures Bot

Bot de trading de futuros en Binance con patrones en 15m. Permite desactivar stops dinámicos.

## Variables de entorno

- `SYMBOL` (ej. `DOGEUSDT`): par a operar.
- `USE_BREAKOUT_DYNAMIC_STOPS` (por defecto `false`): si es `true` habilita el movimiento automático de stop a break-even/verde y otras lógicas de micro stops.

Ejemplo de `.env`:

```
SYMBOL=DOGEUSDT
USE_BREAKOUT_DYNAMIC_STOPS=false
```

## Pruebas

1. `USE_BREAKOUT_DYNAMIC_STOPS=false` → no se moverá el SL a break-even ni habrá trailing micro.
2. `USE_BREAKOUT_DYNAMIC_STOPS=true` → mantiene el comportamiento anterior.

## Logs

Si los mini stops están desactivados:

```
Breakout/mini-SL/BE desactivados por configuración
```

