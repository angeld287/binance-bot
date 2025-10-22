# Estrategia WedgeFormation

La estrategia **WedgeFormation** detecta formaciones de cuña (ascendentes o descendentes) y coloca órdenes *limit* con `Take Profi
 t` basado en la línea opuesta proyectada. Está diseñada para ejecutarse como función stateless en AWS Lambda.

## Flujo general
1. Carga el símbolo y timeframe definidos por entorno (`WEDGE_TIMEFRAME`).
2. Verifica idempotencia: si existe posición abierta o alguna orden con prefijo `WEDGE{symbol}{timeframe}`, gestiona únicamente el
 estado (TP o cancelación por TTL) y finaliza.
3. Descarga velas recientes (`fetch_ohlcv`) y calcula un ATR simple (14 períodos).
4. Extrae pivots locales (swing highs/lows) para ajustar las líneas superior e inferior mediante regresión lineal.
5. Clasifica la figura:
   - **Cuña ascendente** → *bias* bajista → prepara short en la línea superior.
   - **Cuña descendente** → *bias* alcista → prepara long en la línea inferior.
6. (Opcional) Aplica filtros cuando `WEDGE_FILTERS_ENABLED=true`:
   - Tolerancia máxima de toque (`WEDGE_TOL_ATR_MULT` y `WEDGE_TOL_WIDTH_PCT`).
   - Convergencia mínima (`WEDGE_MIN_CONVERGENCE`).
   - Ventana de barras (`WEDGE_MIN_BARS` / `WEDGE_MAX_BARS`).
   - Relación riesgo/beneficio mínima (`RR_MIN`).
7. Calcula precio de entrada en la línea activa y TP en la línea opuesta, añadiendo un *buffer* configurable (`WEDGE_BUFFER_ATR`).
8. Revalida guardas de idempotencia y envía la orden `LIMIT` con `clientOrderId` prefijado.
9. Persiste el TP proyectado en S3 (utilidad `tp_store_s3`) para colocar el reduce-only una vez que la posición se abra.

## Client Order IDs y logging
Todos los logs y `clientOrderId` generados por la estrategia incluyen el prefijo `WEDGE{symbol}{timeframe}`. Ejemplo: `WEDGEBTCUSDT1
5m_28250992_ENTRY`.

## Variables de entorno

| Variable | Default | 1m | 5m | 30m | Rationale |
| --- | --- | --- | --- | --- | --- |
| `WEDGE_TIMEFRAME` | `15m` | `1m` | `5m` | `30m` | Ajustar el timeframe acorde al motor de velas objetivo. |
| `WEDGE_MARKET` | `futures` | `futures` | `futures` | `futures` | Mantener la misma ruta (spot/futuros) en todos los horizontes evita incoherencias operativas. |
| `WEDGE_FILTERS_ENABLED` | `false` | `true` | `true` | `false` | En marcos cortos conviene endurecer filtros para reducir falsas detecciones; en 30m la estructura es más limpia. |
| `WEDGE_MIN_TOUCHES_PER_SIDE` | `2` | `3` | `3` | `2` | Timeframes rápidos requieren más confirmaciones para validar la cuña. |
| `WEDGE_TOL_ATR_MULT` | `0.25` | `0.35` | `0.30` | `0.20` | Ajustar la tolerancia ATR al ruido típico de cada horizonte evita rechazos prematuros. |
| `WEDGE_TOL_WIDTH_PCT` | `0.10` | `0.15` | `0.12` | `0.08` | Aumentar el piso relativo en marcos volátiles mantiene la tolerancia mínima estable. |
| `WEDGE_MIN_CONVERGENCE` | `0.2` | `0.30` | `0.25` | `0.18` | Exigir mayor convergencia en 1m/5m ayuda a distinguir patrones reales del ruido; en 30m basta una reducción moderada. |
| `WEDGE_MIN_BARS` | `20` | `30` | `24` | `16` | Se requieren más velas para estructurar patrones confiables en marcos cortos y menos en marcos amplios. |
| `WEDGE_MAX_BARS` | `120` | `180` | `150` | `100` | Limitar la duración total evita patrones demasiado extensos para cada ventana temporal. |
| `WEDGE_PIVOT_WINDOW_HIGH` | `3` | `4` | `3` | `2` | Ventanas más amplias suavizan pivotes en marcos veloces; en 30m basta una ventana más estrecha. |
| `WEDGE_PIVOT_WINDOW_LOW` | `3` | `4` | `3` | `2` | Mantener simetría en la detección de pivotes altos/bajos por timeframe. |
| `RR_MIN` | `1.0` | `1.2` | `1.1` | `1.0` | Operaciones de 1m/5m necesitan mejor relación RR para compensar costos y ruido. |
| `WEDGE_ORDER_TTL_BARS` | `5` | `60` | `15` | `5` | Ajustar el TTL por barras para mantener una ventana temporal semejante (~1 h) en cada timeframe. |
| `WEDGE_BUFFER_ATR` | `0.15` | `0.25` | `0.20` | `0.12` | Incrementar el buffer en marcos ruidosos reduce la probabilidad de ejecuciones anticipadas. |

> **Nota:** la implementación actual sigue leyendo `WEDGE_TOUCH_TOL_ATR` y `ORDER_TTL_BARS` como alias históricos. Puedes exportar ambas variables con los mismos valores para garantizar compatibilidad.

## Gestión de TP
- Si existe posición abierta sin TP asociado, la estrategia busca un `clientOrderId` con sufijo `_TP`.
- Si no encuentra uno, carga el valor persistido en S3 (`tp_store_s3.load_tp_value`) y envía una orden reduce-only (`place_tp_reduce_only`).
- **Stop Loss**: se deja un *placeholder* en el código para implementar la lógica cuando se habilite (al cierre fuera del patrón).

## Idempotencia
- La función Lambda puede invocarse múltiples veces por minuto; los prefijos `WEDGE` y los guardas sobre posiciones/órdenes evitan
 duplicaciones.
- Si una orden permanece abierta más allá de `WEDGE_ORDER_TTL_BARS` (alias `ORDER_TTL_BARS`), se cancela automáticamente en el siguiente tick.

## Dependencias
- Reutiliza utilidades existentes del bot (`fetch_ohlcv`, `round_price_to_tick`, `tp_store_s3`).
- No introduce stop loss automático; sólo se deja un comentario `TODO` para la futura lógica basada en cierre de vela fuera de la
 cuña.
