# 📈 Stock Market Analysis, Prediction & Segmentation

> A full-stack Python application for stock market data cleaning, technical analysis, machine learning forecasting, anomaly detection, and market regime segmentation — demonstrated on Amazon (AMZN) daily data.

---

## 📁 Dataset

| Property | Details |
|---|---|
| **File** | `AMAZON_daily.csv` |
| **Source** | [Yahoo Finance — AMZN Historical Data](https://finance.yahoo.com/quote/AMZN/history/) |
| **Coverage** | May 15, 1997 → September 22, 2026 |
| **Rows** | 7,384 trading days |
| **Columns** | Date, Open, High, Low, Close, Adj Close, Volume |
| **Known Quirk** | `Adj Close` and `Volume` columns are swapped in the raw file — auto-corrected by the pipeline |

---

## 📝 Project Description

This project builds a **generic, reusable stock market analysis engine** that works with any standard OHLCV CSV file. It covers the full data science workflow:

1. **Data Cleaning** — 12-step pipeline: column-swap fix, type coercion, missing-value imputation, duplicate removal, logical validity checks, outlier detection (IQR + Z-score), date-gap analysis
2. **Technical Analysis** — 20+ indicators: SMA-10/20/50/200, EMA-12/26, MACD, RSI-14, Bollinger Bands, ATR, OBV, Volatility, Cumulative Return
3. **ML Forecasting** — Linear Regression, Random Forest (200 trees), LSTM (2-layer, seq-60)
4. **Market Segmentation** — K-Means clustering (k=4) into Bullish Trending / Sideways / Breakout / Bearish regimes
5. **Anomaly Detection** — Isolation Forest (contamination=3%) flagging extreme market events
6. **Trading Signals** — Composite BUY/SELL/HOLD signal combining MACD crossover, RSI, Bollinger Band breaks, and Golden/Death Cross
7. **REST API** — 13 FastAPI endpoints for all analyses
8. **Interactive Dashboard** — 6-page Streamlit dashboard with Plotly charts and CSV upload
9. **Auto-Generated Reports** — 17 matplotlib chart images + 30-page `.docx` project report

---

## 🗂️ Project Structure

```
Stock Market Analysis/
├── AMAZON_daily.csv                  ← Raw dataset
├── AMAZON_daily_cleaned.csv          ← Cleaned output
├── amazon_chart_data.csv             ← Last 252-day chart export
├── amazon_data_cleaning.py           ← Standalone 12-step cleaning script
├── amazon_trading_analysis.py        ← Standalone technical analysis script
├── generate_charts.py                ← Generates all 17 chart PNGs → charts/
├── generate_report.py                ← Generates the .docx project report
│
├── charts/                           ← 17 chart PNG images
│   ├── 01_full_price_history.png
│   ├── 02_ohlc_candlestick.png
│   ├── 03_price_moving_averages.png
│   ├── 04_rsi.png
│   ├── 05_macd.png
│   ├── 06_bollinger_bands.png
│   ├── 07_price_volume.png
│   ├── 08_returns_distribution.png
│   ├── 09_volatility.png
│   ├── 10_lr_forecast.png
│   ├── 11_rf_forecast.png
│   ├── 12_kmeans_segmentation.png
│   ├── 13_anomaly_detection.png
│   ├── 14_trading_signals.png
│   ├── 15_obv.png
│   ├── 16_signal_distribution.png
│   └── 17_atr.png
│
└── stock_analysis/                   ← Main application package
    ├── requirements.txt
    ├── run.sh                        ← One-command launcher
    ├── data/
    │   └── AMAZON_daily.csv
    ├── models/
    │   └── AMZN_engine.pkl           ← Trained ML engine
    ├── src/
    │   ├── pipeline.py               ← Generic OHLCV loader + indicators + signals
    │   ├── models.py                 ← StockMLEngine (6 models, save/load)
    │   ├── api.py                    ← FastAPI backend (13 endpoints)
    │   └── ml_analysis.py            ← Standalone ML training runner
    ├── app/
    │   └── dashboard.py              ← Streamlit 6-page dashboard
    ├── Amazon_Stock_Analysis_Report.docx
    ├── Amazon_Stock_Business_Report.docx
    └── Amazon_Stock_Market_Analysis_Report.docx  ← Full project report (17 charts, 18 tables)
```

---

## 🛠️ Technologies Used

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.9+ |
| Data Processing | pandas | 2.2.2 |
| Numerical Computing | NumPy | 1.26.4 |
| Machine Learning | scikit-learn | 1.5.0 |
| Deep Learning | TensorFlow / Keras | 2.16.1 |
| REST API | FastAPI + Uvicorn | 0.111.0 / 0.30.1 |
| Frontend Dashboard | Streamlit | 1.35.0 |
| Interactive Charts | Plotly | 5.22.0 |
| Static Charts | Matplotlib | 3.9+ |
| Report Generation | python-docx | latest |
| HTTP Client | requests / httpx | 2.32.3 / 0.27.0 |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd "Stock Market Analysis"
```

### 2. Install dependencies

```bash
pip install -r stock_analysis/requirements.txt
pip install matplotlib python-docx
```

### 3. (Optional) Run standalone data cleaning

```bash
python3 amazon_data_cleaning.py
```

Outputs `AMAZON_daily_cleaned.csv` with a full 12-step cleaning report printed to the console.

### 4. (Optional) Generate all chart images

```bash
python3 generate_charts.py
```

Saves 17 PNG files to the `charts/` folder.

### 5. (Optional) Generate the project report

```bash
python3 generate_report.py
```

Saves `stock_analysis/Amazon_Stock_Market_Analysis_Report.docx`.

---

## 🚀 Running the Full Application

### Option A — One-command launcher

```bash
cd stock_analysis
bash run.sh
```

This installs all dependencies, starts the FastAPI backend on port **8000**, and the Streamlit dashboard on port **8501**.

### Option B — Start services manually

**Terminal 1 — FastAPI backend:**
```bash
cd stock_analysis
uvicorn src.api:app --reload --port 8000
```

**Terminal 2 — Streamlit dashboard:**
```bash
cd stock_analysis
streamlit run app/dashboard.py
```

Then open **http://localhost:8501** in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/tickers` | List all loaded tickers |
| `POST` | `/upload?ticker=X` | Upload any OHLCV CSV |
| `GET` | `/data/summary?ticker=X` | KPI snapshot |
| `GET` | `/data/ohlcv?ticker=X` | OHLCV time series |
| `GET` | `/data/indicators?ticker=X` | All 20+ technical indicators |
| `GET` | `/predict/linear?ticker=X&horizon=30` | Linear Regression forecast |
| `GET` | `/predict/lstm?ticker=X&horizon=30` | LSTM forecast |
| `GET` | `/segment?ticker=X&n=4` | K-Means regime segmentation |
| `GET` | `/signal/latest?ticker=X` | Latest composite signal |
| `GET` | `/anomalies?ticker=X&top=20` | Top anomalous trading days |
| `GET` | `/models/train?ticker=X` | Train & save the ML engine |
| `GET` | `/models/status?ticker=X` | Model training status |

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPI cards, price history, volume chart |
| **Technical Analysis** | Moving averages, RSI, MACD, Bollinger Bands, ATR |
| **ML Prediction** | Linear Regression & LSTM 30-day forecasts with metrics |
| **Segmentation** | K-Means market regime scatter & timeline |
| **Anomaly Detection** | Isolation Forest anomaly overlay on price chart |
| **Trading Signal** | Composite BUY/SELL/HOLD signal with confidence scores |

Use the **sidebar CSV uploader** to load any stock — the entire pipeline runs automatically on the new file.

---

## 🤖 Machine Learning Models

| Model | Algorithm | Purpose |
|---|---|---|
| Linear Regression | OLS + StandardScaler | Baseline price forecast |
| Random Forest | 200 estimators, depth 12 | Non-linear forecast + feature importance |
| LSTM | 2-layer (64+32), Dropout 0.2, seq-60 | Deep sequence price forecast |
| K-Means | k=4, StandardScaler | Market regime clustering |
| Isolation Forest | 200 estimators, contamination=3% | Anomaly / extreme event detection |
| Logistic Regression | Multi-class, balanced weights | BUY / SELL / HOLD classifier |

All models use **scale-invariant features** (ratios, oscillators, percentage returns) so a model trained on Amazon works equally well on any other stock.

---

## 📈 Key Results (AMZN)

| Metric | Value |
|---|---|
| Date Range | 1997-05-15 → 2026-09-22 |
| Total Trading Days | 7,384 |
| Price Growth | $0.07 → $284.02 (~3,900× return) |
| Missing Values (raw) | 5 cells → 0 after cleaning |
| IQR Outliers | 292 rows (retained — genuine price history) |
| Anomalies Detected | ~222 events (Isolation Forest) |
| LR Model R² | > 0.99 (test set) |
| Signal System | BUY / SELL / HOLD — 4-indicator composite vote |

---

## 📋 Data Cleaning Summary

| Check | Result |
|---|---|
| Column swap (Adj Close ↔ Volume) | ✅ Auto-detected & fixed |
| Data type conversion | ✅ datetime64, float64, Int64 |
| Missing values | ✅ 5 found → forward-filled |
| Duplicate rows | ✅ None found |
| Negative/zero prices | ✅ None found |
| High < Low violations | ✅ None found |
| Close outside \[Low, High\] | ✅ None found |
| Date gaps > 4 days | ✅ 3 found (known market closures) |
| Zero-volume days | ✅ None found |

---

## 📄 License

This project is for educational purposes. The dataset is sourced from Yahoo Finance and is subject to their [Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html).

---

*Built with Python · pandas · scikit-learn · TensorFlow · FastAPI · Streamlit*
