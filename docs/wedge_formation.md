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
   - Tolerancia máxima de toque (`WEDGE_TOUCH_TOL_ATR`).
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
| Variable | Default | Descripción |
| --- | --- | --- |
| `WEDGE_TIMEFRAME` | `15m` | Timeframe utilizado para evaluar la figura. |
| `WEDGE_FILTERS_ENABLED` | `false` | Activa los filtros opcionales (tolerancia, convergencia, RR). |
| `WEDGE_MIN_TOUCHES_PER_SIDE` | `2` | Toques mínimos por lado para considerar la cuña válida. |
| `WEDGE_TOUCH_TOL_ATR` | `0.25` | Tolerancia de toque en múltiplos de ATR. |
| `WEDGE_MIN_CONVERGENCE` | `0.2` | Reducción relativa mínima del ancho entre líneas. |
| `WEDGE_MIN_BARS` / `WEDGE_MAX_BARS` | `20` / `120` | Número de velas permitido en la figura. |
| `RR_MIN` | `1.0` | Relación riesgo/beneficio mínima (sólo con filtros activos). |
| `ORDER_TTL_BARS` | `5` | TTL en barras para cancelar la limit pendiente. |
| `WEDGE_BUFFER_ATR` | `0.15` | Buffer aplicado a la entrada y al cálculo de RR teórico. |

## Gestión de TP
- Si existe posición abierta sin TP asociado, la estrategia busca un `clientOrderId` con sufijo `_TP`.
- Si no encuentra uno, carga el valor persistido en S3 (`tp_store_s3.load_tp_value`) y envía una orden reduce-only (`place_tp_reduce_only`).
- **Stop Loss**: se deja un *placeholder* en el código para implementar la lógica cuando se habilite (al cierre fuera del patrón).

## Idempotencia
- La función Lambda puede invocarse múltiples veces por minuto; los prefijos `WEDGE` y los guardas sobre posiciones/órdenes evitan
 duplicaciones.
- Si una orden permanece abierta más allá de `ORDER_TTL_BARS`, se cancela automáticamente en el siguiente tick.

## Dependencias
- Reutiliza utilidades existentes del bot (`fetch_ohlcv`, `round_price_to_tick`, `tp_store_s3`).
- No introduce stop loss automático; sólo se deja un comentario `TODO` para la futura lógica basada en cierre de vela fuera de la
 cuña.
