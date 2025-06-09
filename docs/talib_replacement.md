# TA-Lib Replacement Documentation

## Background
The project originally used the TA-Lib library (Technical Analysis Library) for financial market technical indicators. However, TA-Lib requires C/C++ dependencies which can be difficult to install on some platforms, especially Windows.

## Solution
On June 8, 2025, we replaced all TA-Lib references with the `ta` library (Technical Analysis Library), which is a pure Python implementation. This makes installation much easier across all platforms.

## Changes Made
1. Replaced `import talib` with `import ta`
2. Replaced all direct calls to TA-Lib functions with equivalent functions from the `ta` library
3. Created approximations for candlestick patterns (Doji, Hammer, etc.) which don't have direct equivalents in the `ta` library

## Usage Examples
Before:
```python
df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
```

After:
```python
df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Hist'] = macd.macd_diff()
bollinger = ta.volatility.BollingerBands(close=df['Close'])
df['BB_Upper'] = bollinger.bollinger_hband()
df['BB_Middle'] = bollinger.bollinger_mavg()
df['BB_Lower'] = bollinger.bollinger_lband()
```

## Documentation
For more information on the `ta` library, visit:
https://technical-analysis-library-in-python.readthedocs.io/
