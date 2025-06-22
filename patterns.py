def detect_engulfing(df):
    df['bullish_engulfing'] = (
        (df['close'].shift(1) > df['open'].shift(1)) &  # vela anterior bajista
        (df['open'] < df['close']) &                    # vela actual alcista
        (df['open'] < df['close'].shift(1)) &           # apertura por debajo del cierre anterior
        (df['close'] > df['open'].shift(1))             # cierre por encima de apertura anterior
    )
    return df
