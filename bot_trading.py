import os
import time
import ccxt
import pandas as pd
import logging
from dotenv import load_dotenv
from patterns import detect_engulfing

# Configurar logging
logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
INTERVAL = os.getenv("INTERVAL", "1h")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

def get_binance_client():
    config = {
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
    }
    if USE_TESTNET:
        config['options'] = {'defaultType': 'spot'}
        config['urls'] = {
            'api': {
                'public': 'https://testnet.binance.vision/api',
                'private': 'https://testnet.binance.vision/api',
            }
        }
    return ccxt.binance(config)

def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=100):
    exchange = get_binance_client()
    exchange.set_sandbox_mode(True)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_bot():
    try:
        logging.info("⏳ Ejecutando análisis...")
        df = fetch_data(SYMBOL, INTERVAL)
        df = detect_engulfing(df)

        if df['bullish_engulfing'].iloc[-1]:
            price = df['close'].iloc[-1]
            logging.info(f"📈 Señal LONG en {SYMBOL} - {df.index[-1]} - Precio: {price}")
        else:
            logging.info(f"❌ No hay señal para {SYMBOL} - {df.index[-1]}")
    except Exception as e:
        logging.exception("⚠️ Error ejecutando bot:")

if __name__ == "__main__":
    while True:
        run_bot()
        time.sleep(60 * 5)
