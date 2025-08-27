# Bot de Trading Automatizado en Binance Futures

## 🚀 Descripción.
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
   - `STRATEGY` (ej. breakout)
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
