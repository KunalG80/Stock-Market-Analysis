"""
Amazon Daily Stock — Technical Analysis & Trading Signals
Dataset: AMAZON_daily_cleaned.csv

Run from the stock_analysis/ directory:
    python3 amazon_trading_analysis.py
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Load cleaned data
# ─────────────────────────────────────────────
# Resolve path relative to this script's location so it works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV  = os.path.join(_HERE, "data", "AMAZON_daily_cleaned.csv")
if not os.path.exists(_CSV):
    # Fallback: look for the file next to this script
    _CSV = os.path.join(_HERE, "AMAZON_daily_cleaned.csv")
if not os.path.exists(_CSV):
    raise FileNotFoundError(
        f"Could not find AMAZON_daily_cleaned.csv. "
        f"Run amazon_data_cleaning.py first, or check the path: {_CSV}"
    )

df = pd.read_csv(_CSV, parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# ─────────────────────────────────────────────
# 2. Technical Indicators
# ─────────────────────────────────────────────

# --- Moving Averages ---
df["SMA_20"]  = df["Close"].rolling(window=20).mean()
df["SMA_50"]  = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()
df["EMA_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"]  = df["Close"].ewm(span=26, adjust=False).mean()

# --- MACD ---
df["MACD"]        = df["EMA_12"] - df["EMA_26"]
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# --- RSI (14-period) ---
delta  = df["Close"].diff()
gain   = delta.clip(lower=0)
loss   = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs     = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# --- Bollinger Bands (20-period, 2σ) ---
df["BB_Mid"]   = df["SMA_20"]
df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()

# --- Daily Return & Volatility (30-day rolling) ---
df["Daily_Return"]  = df["Close"].pct_change()
df["Volatility_30"] = df["Daily_Return"].rolling(30).std() * np.sqrt(252)

# ─────────────────────────────────────────────
# 3. Trading Signal Generation
# ─────────────────────────────────────────────

# Golden/Death Cross: SMA_50 vs SMA_200
df["GoldenCross"] = (df["SMA_50"] > df["SMA_200"]) & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
df["DeathCross"]  = (df["SMA_50"] < df["SMA_200"]) & (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1))

# MACD Signal cross
df["MACD_Buy"]  = (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
df["MACD_Sell"] = (df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))

# RSI oversold/overbought
df["RSI_Buy"]  = df["RSI"] < 30
df["RSI_Sell"] = df["RSI"] > 70

# Bollinger Band breakout
df["BB_Buy"]  = df["Close"] < df["BB_Lower"]
df["BB_Sell"] = df["Close"] > df["BB_Upper"]

# ─────────────────────────────────────────────
# 4. Composite Signal  (last 252 trading days)
# ─────────────────────────────────────────────
recent = df.tail(252).copy()

buy_count  = recent[["MACD_Buy",  "RSI_Buy",  "BB_Buy",  "GoldenCross"]].sum(axis=1)
sell_count = recent[["MACD_Sell", "RSI_Sell", "BB_Sell", "DeathCross"]].sum(axis=1)
recent["Signal"] = np.where(buy_count >= 2, "BUY",
                   np.where(sell_count >= 2, "SELL", "HOLD"))

# ─────────────────────────────────────────────
# 5. Most Recent Trading Decision
# ─────────────────────────────────────────────
last = df.iloc[-1]

print("=" * 60)
print("LATEST TRADING SNAPSHOT")
print("=" * 60)
print(f"Date          : {last['Date'].date()}")
print(f"Close Price   : ${last['Close']:.2f}")
print(f"SMA 20        : ${last['SMA_20']:.2f}")
print(f"SMA 50        : ${last['SMA_50']:.2f}")
print(f"SMA 200       : ${last['SMA_200']:.2f}")
print(f"RSI (14)      : {last['RSI']:.1f}")
print(f"MACD          : {last['MACD']:.4f}")
print(f"MACD Signal   : {last['MACD_Signal']:.4f}")
print(f"BB Upper      : ${last['BB_Upper']:.2f}")
print(f"BB Lower      : ${last['BB_Lower']:.2f}")
print(f"Volatility 30d: {last['Volatility_30']:.1%}")

last_signal = recent["Signal"].iloc[-1]
print(f"\n>>> COMPOSITE SIGNAL : {last_signal} <<<")

# ─────────────────────────────────────────────
# 6. Export chart data (last 252 days for charts)
# ─────────────────────────────────────────────
chart_cols = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "SMA_200",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI",
    "BB_Upper", "BB_Mid", "BB_Lower",
    "Volatility_30", "Daily_Return",
    "GoldenCross", "DeathCross",
    "MACD_Buy", "MACD_Sell",
    "RSI_Buy", "RSI_Sell",
    "BB_Buy", "BB_Sell",
    "Signal"
]
recent[chart_cols].to_csv("amazon_chart_data.csv", index=False)
print("\n✓ Chart data exported to: amazon_chart_data.csv")
