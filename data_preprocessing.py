# data_preprocessing.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# Feature engineering functions
def moving_average(df, period=10):
    return df['Close'].rolling(window=period).mean()

def rsi(df, period=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(df, short_period=12, long_period=26, signal_period=9):
    ema_short = df['Close'].ewm(span=short_period, adjust=False).mean()
    ema_long = df['Close'].ewm(span=long_period, adjust=False).mean()
    macd_line = ema_short - ema_long
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, macd_signal

def bollinger_bands(df, period=20):
    middle_band = df['Close'].rolling(window=period).mean()
    std_dev = df['Close'].rolling(window=period).std()
    upper_band = middle_band + (2 * std_dev)
    lower_band = middle_band - (2 * std_dev)
    return upper_band, lower_band

def atr(df, period=14):
    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    return tr.rolling(window=period).mean()

def obv(df):
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

# Adding Computed Indicators
def preprocess_data(df):
    df['MA_10'] = moving_average(df)
    df['RSI'] = rsi(df)
    df['MACD'], df['MACD_Signal'] = macd(df)
    df['BB_High'], df['BB_Low'] = bollinger_bands(df)
    df['ATR'] = atr(df)
    df['OBV'] = obv(df)
    df.dropna(inplace=True)
    return df

def scale_data(df, fixed_features, optional_features):
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    df[fixed_features + optional_features] = scaler_X.fit_transform(df[fixed_features + optional_features])
    df['Close'] = scaler_Y.fit_transform(df[['Close']])
    return df, scaler_X, scaler_Y
