# Bot de Trading Automatizado en Binance Futures

## 🚀 Descripción
Bot en Python que ejecuta estrategias automáticas de trading sobre **Binance Futures**. Corre como función **AWS Lambda** programada por **EventBridge**, consulta precios vía API REST/WebSocket oficial y maneja órdenes *long/short* con **stop loss** y **take profit**. Todos los eventos se registran en **CloudWatch Logs**.

## 🏗️ Arquitectura
- **AWS Lambda**: ejecuta la lógica de trading.
- **Amazon EventBridge**: agenda ejecuciones periódicas.
- **Binance Futures API**: consulta de mercado y envío de órdenes.
- **CloudWatch Logs**: auditoría centralizada.
- **Estructura modular**: `core`, `strategies`, `analysis`, `config`.

```
EventBridge --> Lambda (bot_trading.py)
               |--> core/exchange.py (API Binance)
               |--> strategies/breakout.py (lógica de estrategia)
               |--> analysis/* (detección de patrones y niveles)
Logs --> CloudWatch
```

## 📂 Estructura del Proyecto
```
src/
├── core/        # Lógica base y conexión con Binance
├── strategies/  # Estrategias de trading (ej. breakout)
├── analysis/    # Detección de patrones y niveles
├── config/      # Configuración y helpers
└── tests/       # Pruebas unitarias e integración
```

## ⚙️ Configuración
1. Clonar el repositorio.
2. Crear archivo `.env` a partir de `.env.example` y completar:
   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET`
   - `SYMBOL` (ej. DOGEUSDT)
   - `SR_TIMEFRAME` (ej. 4h)
   - `STRATEGY_NAME` (alias: `STRATEGY`, ej. `wedge-formation`)
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Configurar AWS CLI con credenciales.
5. Desplegar empaquetando con el `Dockerfile` o zip para Lambda.

## 🔧 Tecnologías Usadas
- **Python 3.x**
- **AWS Lambda** + **EventBridge**
- **CloudWatch Logs**
- **Docker** (para build)
- **Binance Futures API** (REST y WebSocket)
- Librerías: `python-binance`, `requests`, `pandas` (si aplica).

## 📡 APIs y Endpoints
- `GET /fapi/v1/klines` → velas/candles.
- `POST /fapi/v1/order` → creación de órdenes.
- `GET /fapi/v2/positionRisk` → estado de posiciones.
- WebSocket User Data (si se usa) → fills y actualizaciones en tiempo real.

## 📝 Ejemplo de Uso
Ejecutar localmente en modo observador:
```bash
python src/core/bot_trading.py --observer true
```
Ejemplo de log en CloudWatch:
```json
{"event":"ORDER_DECISION","symbol":"DOGEUSDT","signal":"LONG","confidence":0.78}
```

## ✅ Testing
Ejecutar pruebas con `pytest`:
```bash
pytest
```
Para operar sin riesgo utilizar Binance Testnet:
```bash
export BINANCE_TESTNET=true
```

### Parámetros de la estrategia WedgeFormation

La estrategia `wedge-formation` consume las siguientes variables de entorno. Los valores indicados son los *defaults* cuando no
se provee la variable:

| Variable | Descripción | Default |
| --- | --- | --- |
| `WEDGE_TIMEFRAME` | Timeframe para analizar las velas. | `15m` |
| `WEDGE_FILTERS_ENABLED` | Activa los filtros opcionales (tolerancia, convergencia, RR). | `false` |
| `WEDGE_MIN_TOUCHES_PER_SIDE` | Toques mínimos por línea para validar la cuña. | `2` |
| `WEDGE_TOUCH_TOL_ATR` | Tolerancia de toque en múltiplos de ATR. | `0.25` |
| `WEDGE_MIN_CONVERGENCE` | Reducción mínima del ancho entre líneas. | `0.2` |
| `WEDGE_MIN_BARS` / `WEDGE_MAX_BARS` | Rango permitido de barras en la figura. | `20` / `120` |
| `RR_MIN` | Relación riesgo/beneficio mínima (sólo con filtros activos). | `1.0` |
| `ORDER_TTL_BARS` | Barras a esperar antes de cancelar la limit pendiente. | `5` |
| `WEDGE_BUFFER_ATR` | Buffer en múltiplos de ATR para entrada y RR teórico. | `0.15` |

## 🎯 Precisión y filtros de Binance
- El bot consulta dinámicamente el `exchangeInfo` de Binance Futures y cachea los filtros de cada símbolo para respetar `tickSize`, `stepSize` y `minNotional`.
- Todas las cifras críticas (entry, stop, take profits y cantidades) se procesan con `decimal.Decimal` y se redondean vía `round_to_tick` y `round_to_step` antes de enviarse al broker.
- La bandera `STRICT_ROUNDING` (habilitada por defecto) garantiza que cualquier precio/cantidad que viole los filtros se ajuste o se rechace con logs claros.
- En los logs de CloudWatch encontrarás un bloque de validación previo al envío de órdenes:
  ```json
  {"pre_order_check":{"symbol":"DOGEUSDT","side":"SELL","filters":{"tickSize":"0.0001","stepSize":"1","minNotional":"5"},"entry":"0.248","stop_loss":"0.2496","take_profit_1":"0.2434","take_profit_2":"0.2397","qty":"30200","notional_est":"7499.6","validated":true}}
  ```
- La suite de tests incluye verificaciones específicas para símbolos con distintos filtros (`DOGEUSDT`, `SOLUSDT`, `XRPUSDT`) y valida que nunca aparezcan colas binarias en los JSON serializados.

## 🔒 Seguridad
- Nunca exponer claves API en el repositorio.
- Usar AWS Secrets Manager o Parameter Store en producción.
- Asignar roles IAM de mínimo privilegio a la Lambda.

## 📈 Roadmap
- Añadir nuevas estrategias (soporte/resistencia, Wyckoff).
- Mejorar gestión de posiciones (idempotencia, locks).
- Métricas avanzadas con CloudWatch Dashboards.

## 🤝 Contribución
Se aceptan PRs y issues. Abre un fork, crea tu rama y envía la propuesta con pruebas.

## 📜 Licencia
[MIT](LICENSE)
