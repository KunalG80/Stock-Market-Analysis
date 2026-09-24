"""
models.py — Generic Stock ML Engine
Trains, saves, loads and runs 5 ML models for ANY stock dataset.

Models
------
1. Linear Regression     — fast trend-based price forecast
2. Random Forest         — non-linear multi-feature forecast
3. LSTM                  — deep sequence model forecast
4. K-Means Clustering    — market regime segmentation
5. Isolation Forest      — anomaly / extreme event detection
6. Logistic Regression   — BUY/SELL/HOLD signal classifier

Usage
-----
    from models import StockMLEngine
    engine = StockMLEngine(ticker="AMZN")
    engine.fit(df)                   # train all models
    engine.save()                    # persist to disk
    engine.load()                    # reload from disk
    results = engine.predict(df)     # full analysis dict
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.ensemble      import RandomForestRegressor, IsolationForest
from sklearn.cluster       import KMeans
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(_MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Feature columns used for supervised models
# (scale-invariant so they work for any stock price range)
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Pct", "BB_Width",
    "Volatility_20", "Daily_Return", "Log_Return",
    "ATR",
    "Close_vs_SMA20", "Close_vs_SMA50", "Close_vs_SMA200",
    "Price_Range", "Price_Momentum",
    "Close_lag1", "Close_lag5",
]


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def _make_sequences(arr: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i: i + seq_len])
        y.append(arr[i + seq_len])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────
# StockMLEngine — one instance per ticker
# ─────────────────────────────────────────────────────────────

class StockMLEngine:
    """
    Trains and serves all ML models for a single stock ticker.

    Parameters
    ----------
    ticker  : str  — e.g. "AMZN", "AAPL", "TSLA"
    horizon : int  — forecast horizon in trading days (default 30)
    seq_len : int  — LSTM look-back window (default 60)
    """

    def __init__(self, ticker: str = "STOCK", horizon: int = 30, seq_len: int = 60):
        self.ticker   = ticker.upper()
        self.horizon  = horizon
        self.seq_len  = seq_len
        self._trained = False

        # Model artefacts (populated by fit())
        self.lr_model   = None      # LinearRegression
        self.lr_scaler  = None      # StandardScaler for LR features
        self.rf_model   = None      # RandomForestRegressor
        self.lstm_model = None      # Keras Sequential (optional)
        self.lstm_scaler= None      # MinMaxScaler for LSTM close prices
        self.km_pipe    = None      # Pipeline(scaler + KMeans)
        self.km_label_map = {}      # cluster_id → regime name
        self.iso_model  = None      # IsolationForest
        self.iso_scaler = None      # StandardScaler for anomaly features
        self.clf_model  = None      # LogisticRegression
        self.clf_scaler = None      # StandardScaler for clf features
        self.clf_encoder= None      # LabelEncoder for signal classes

        # Stored for predict()
        self._last_features = None  # last row feature vector (LR / RF)
        self._last_close_seq = None # last seq_len scaled close prices (LSTM)
        self._last_date     = None
        self._close_scaler  = None  # separate MinMaxScaler for close (LR/RF target)

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _build_supervised(self, df: pd.DataFrame):
        """Return (X, y_close, dates) for supervised price models."""
        sub = df.dropna(subset=FEATURE_COLS + ["Close"]).copy()
        X  = sub[FEATURE_COLS].values
        y  = sub["Close"].values
        return X, y, sub["Date"].values

    def _split(self, X, y):
        n = int(len(X) * 0.8)
        return X[:n], X[n:], y[:n], y[n:]

    # ─────────────────────────────────────────────
    # 1. Linear Regression
    # ─────────────────────────────────────────────

    def _fit_lr(self, df: pd.DataFrame) -> dict:
        X, y, dates = self._build_supervised(df)
        X_tr, X_te, y_tr, y_te = self._split(X, y)

        self.lr_scaler = StandardScaler()
        X_tr_s = self.lr_scaler.fit_transform(X_tr)
        X_te_s  = self.lr_scaler.transform(X_te)

        self.lr_model = LinearRegression()
        self.lr_model.fit(X_tr_s, y_tr)
        y_pred = self.lr_model.predict(X_te_s)

        self._last_features = X[-1]
        return {
            "test_dates"   : dates[int(len(X) * 0.8):],
            "y_test_true"  : y_te,
            "y_test_pred"  : y_pred,
            "metrics"      : _metrics(y_te, y_pred),
        }

    # ─────────────────────────────────────────────
    # 2. Random Forest
    # ─────────────────────────────────────────────

    def _fit_rf(self, df: pd.DataFrame) -> dict:
        X, y, dates = self._build_supervised(df)
        X_tr, X_te, y_tr, y_te = self._split(X, y)

        self.rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            random_state=42, n_jobs=-1,
        )
        self.rf_model.fit(X_tr, y_tr)
        y_pred = self.rf_model.predict(X_te)

        imp = pd.Series(self.rf_model.feature_importances_, index=FEATURE_COLS)
        return {
            "test_dates"         : dates[int(len(X) * 0.8):],
            "y_test_true"        : y_te,
            "y_test_pred"        : y_pred,
            "metrics"            : _metrics(y_te, y_pred),
            "feature_importance" : imp.sort_values(ascending=False).to_dict(),
        }

    # ─────────────────────────────────────────────
    # 3. LSTM
    # ─────────────────────────────────────────────

    def _fit_lstm(self, df: pd.DataFrame) -> dict:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            return {"error": "TensorFlow not installed"}

        sub = df.dropna(subset=["Close"]).tail(min(1260, len(df))).reset_index(drop=True)
        self.lstm_scaler = MinMaxScaler()
        scaled = self.lstm_scaler.fit_transform(sub["Close"].values.reshape(-1, 1)).flatten()
        self._last_close_seq = scaled[-self.seq_len:]
        self._last_date      = sub["Date"].iloc[-1]

        X, y = _make_sequences(scaled, self.seq_len)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        split = int(len(X) * 0.8)
        X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        tf.random.set_seed(42)
        self.lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.seq_len, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self.lstm_model.compile(optimizer="adam", loss="mse")
        self.lstm_model.fit(
            X_tr, y_tr, epochs=50, batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )

        y_pred_s = self.lstm_model.predict(X_te, verbose=0).flatten()
        y_pred   = self.lstm_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_true   = self.lstm_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

        n_off = len(sub) - len(X_te)
        return {
            "test_dates"  : sub["Date"].values[n_off + self.seq_len:],
            "y_test_true" : y_true,
            "y_test_pred" : y_pred,
            "metrics"     : _metrics(y_true, y_pred),
        }

    # ─────────────────────────────────────────────
    # 4. K-Means Segmentation
    # ─────────────────────────────────────────────

    _KM_FEATURES = ["Daily_Return", "Volatility_20", "RSI", "MACD_Hist", "BB_Width", "Log_Volume"]

    def _fit_kmeans(self, df: pd.DataFrame, n_clusters: int = 4):
        d = df.copy()
        d["Log_Volume"] = np.log1p(d["Volume"].astype(float))
        sub = d.dropna(subset=self._KM_FEATURES)
        X   = sub[self._KM_FEATURES].values

        self.km_pipe = Pipeline([
            ("sc", StandardScaler()),
            ("km", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ])
        self.km_pipe.fit(X)
        labels = self.km_pipe.predict(X)

        # Auto-label by mean daily return
        stats = (pd.DataFrame(X, columns=self._KM_FEATURES)
                   .assign(c=labels)
                   .groupby("c")["Daily_Return"].mean()
                   .sort_values(ascending=False))
        rank_map = {c: i for i, c in enumerate(stats.index)}
        regime_names = {
            0: "Bullish Trending",
            1: "Sideways / Low Volatility",
            2: "Breakout / High Volume",
            3: "Bearish / High Volatility",
        }
        self.km_label_map = {c: regime_names.get(rank_map[c], f"Regime {rank_map[c]+1}")
                             for c in stats.index}

    # ─────────────────────────────────────────────
    # 5. Isolation Forest (anomaly detection)
    # ─────────────────────────────────────────────

    _ISO_FEATURES = ["Daily_Return", "Volatility_20", "Log_Volume", "MACD_Hist", "RSI"]

    def _fit_isolation_forest(self, df: pd.DataFrame):
        d = df.copy()
        d["Log_Volume"] = np.log1p(d["Volume"].astype(float))
        sub = d.dropna(subset=self._ISO_FEATURES)
        X = sub[self._ISO_FEATURES].values
        self.iso_scaler = StandardScaler()
        X_s = self.iso_scaler.fit_transform(X)
        self.iso_model = IsolationForest(contamination=0.03, random_state=42, n_estimators=200)
        self.iso_model.fit(X_s)

    # ─────────────────────────────────────────────
    # 6. Signal Classifier (Logistic Regression)
    # ─────────────────────────────────────────────

    def _fit_classifier(self, df: pd.DataFrame) -> dict:
        feat = FEATURE_COLS
        sub = df.dropna(subset=feat + ["Signal"]).copy()
        self.clf_encoder = LabelEncoder()
        y = self.clf_encoder.fit_transform(sub["Signal"])
        X = sub[feat].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.clf_scaler = StandardScaler()
        self.clf_model  = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        self.clf_model.fit(self.clf_scaler.fit_transform(X_tr), y_tr)
        y_pred = self.clf_model.predict(self.clf_scaler.transform(X_te))
        return {
            "accuracy"  : float(accuracy_score(y_te, y_pred)),
            "report"    : classification_report(y_te, y_pred, target_names=self.clf_encoder.classes_),
        }

    # ─────────────────────────────────────────────
    # Public: fit
    # ─────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, train_lstm: bool = True, n_clusters: int = 4) -> dict:
        """
        Train all models on *df*.  Returns a dict of per-model training results.

        Parameters
        ----------
        df          : full pipeline DataFrame (output of get_full_data)
        train_lstm  : set False to skip LSTM (saves ~2 minutes)
        n_clusters  : number of K-Means regimes
        """
        print(f"\n[StockMLEngine] Training models for {self.ticker} …")
        results = {}

        print("  [1/6] Linear Regression …", end=" ", flush=True)
        results["linear_regression"] = self._fit_lr(df)
        print(f"RMSE={results['linear_regression']['metrics']['RMSE']:.4f}")

        print("  [2/6] Random Forest …", end=" ", flush=True)
        results["random_forest"] = self._fit_rf(df)
        print(f"RMSE={results['random_forest']['metrics']['RMSE']:.4f}")

        if train_lstm:
            print("  [3/6] LSTM (this may take 1–3 min) …", end=" ", flush=True)
            results["lstm"] = self._fit_lstm(df)
            if "error" not in results["lstm"]:
                print(f"RMSE={results['lstm']['metrics']['RMSE']:.4f}")
            else:
                print(results["lstm"]["error"])
        else:
            results["lstm"] = {"error": "Skipped"}
            print("  [3/6] LSTM … Skipped")

        print("  [4/6] K-Means Segmentation …", end=" ", flush=True)
        self._fit_kmeans(df, n_clusters=n_clusters)
        print("done")

        print("  [5/6] Isolation Forest …", end=" ", flush=True)
        self._fit_isolation_forest(df)
        print("done")

        print("  [6/6] Signal Classifier …", end=" ", flush=True)
        results["classifier"] = self._fit_classifier(df)
        print(f"accuracy={results['classifier']['accuracy']:.4f}")

        self._trained = True
        print(f"[StockMLEngine] All models trained for {self.ticker} ✓")
        return results

    # ─────────────────────────────────────────────
    # Public: save / load
    # ─────────────────────────────────────────────

    def _model_path(self):
        return os.path.join(_MODELS_DIR, f"{self.ticker}_engine.pkl")

    def save(self):
        """Persist all artefacts to disk (excluding Keras model — saved separately)."""
        path = self._model_path()
        payload = {k: v for k, v in self.__dict__.items() if k != "lstm_model"}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        if self.lstm_model is not None:
            lstm_path = os.path.join(_MODELS_DIR, f"{self.ticker}_lstm.keras")
            self.lstm_model.save(lstm_path)
        print(f"[StockMLEngine] Saved to {path}")

    def load(self):
        """Reload artefacts from disk."""
        path = self._model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model for {self.ticker} at {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.__dict__.update(payload)

        lstm_path = os.path.join(_MODELS_DIR, f"{self.ticker}_lstm.keras")
        if os.path.exists(lstm_path):
            try:
                from tensorflow.keras.models import load_model
                self.lstm_model = load_model(lstm_path)
            except Exception:
                self.lstm_model = None
        print(f"[StockMLEngine] Loaded from {path}")

    def is_saved(self) -> bool:
        return os.path.exists(self._model_path())

    # ─────────────────────────────────────────────
    # Public: predict (inference only — no re-training)
    # ─────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Run all trained models on *df* and return a unified results dict.
        If the engine has not been trained yet, it is trained first.
        """
        if not self._trained:
            self.fit(df)

        results = {}

        # ── LR forecast ──
        last_feat_s = self.lr_scaler.transform(self._last_features.reshape(1, -1))
        lr_future   = [float(self.lr_model.predict(last_feat_s)[0])] * self.horizon
        results["linear_regression"] = {
            "future_dates": pd.bdate_range(
                start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=self.horizon
            ),
            "future_pred" : np.array(lr_future),
        }

        # ── RF forecast ──
        rf_future = []
        feat = self._last_features.copy()
        for _ in range(self.horizon):
            p = float(self.rf_model.predict(feat.reshape(1, -1))[0])
            rf_future.append(p)
        results["random_forest"] = {
            "future_dates": pd.bdate_range(
                start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=self.horizon
            ),
            "future_pred" : np.array(rf_future),
        }

        # ── LSTM forecast (recursive) ──
        if self.lstm_model is not None and self._last_close_seq is not None:
            seq = list(self._last_close_seq)
            fut_s = []
            for _ in range(self.horizon):
                inp = np.array(seq[-self.seq_len:]).reshape(1, self.seq_len, 1)
                p   = float(self.lstm_model.predict(inp, verbose=0)[0][0])
                fut_s.append(p)
                seq.append(p)
            lstm_future = self.lstm_scaler.inverse_transform(
                np.array(fut_s).reshape(-1, 1)
            ).flatten()
            results["lstm"] = {
                "future_dates": pd.bdate_range(
                    start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=self.horizon
                ),
                "future_pred" : lstm_future,
            }
        else:
            results["lstm"] = {"error": "LSTM not trained"}

        # ── K-Means segmentation on full df ──
        d = df.copy()
        d["Log_Volume"] = np.log1p(d["Volume"].astype(float))
        sub = d.dropna(subset=self._KM_FEATURES)
        X_km = sub[self._KM_FEATURES].values
        labels = self.km_pipe.predict(X_km)
        sub = sub.copy()
        sub["Segment"]       = labels
        sub["Segment_Label"] = pd.Series(labels).map(self.km_label_map).values
        results["segmentation"] = sub[["Date","Close","Daily_Return","Volatility_20",
                                        "RSI","Segment","Segment_Label"]]

        # ── Anomaly detection ──
        d2 = df.copy()
        d2["Log_Volume"] = np.log1p(d2["Volume"].astype(float))
        sub2 = d2.dropna(subset=self._ISO_FEATURES)
        X_iso = self.iso_scaler.transform(sub2[self._ISO_FEATURES].values)
        sub2 = sub2.copy()
        sub2["Anomaly"]       = self.iso_model.predict(X_iso)
        sub2["Anomaly_Score"] = self.iso_model.score_samples(X_iso)
        results["anomalies"] = sub2[sub2["Anomaly"] == -1][
            ["Date","Close","Daily_Return","Anomaly_Score"]
        ].sort_values("Anomaly_Score")

        # ── Latest signal classification ──
        last_row = df.dropna(subset=FEATURE_COLS).iloc[-1]
        feat_vec = self.clf_scaler.transform(last_row[FEATURE_COLS].values.reshape(1, -1))
        pred_cls = self.clf_encoder.classes_[self.clf_model.predict(feat_vec)[0]]
        probs    = dict(zip(self.clf_encoder.classes_,
                            self.clf_model.predict_proba(feat_vec)[0]))
        results["classifier"] = {
            "latest_signal": pred_cls,
            "probabilities": probs,
        }

        return results


# ─────────────────────────────────────────────────────────────
# Backwards-compatible helpers used by api.py / dashboard.py
# ─────────────────────────────────────────────────────────────

def linear_regression_forecast(df: pd.DataFrame, horizon: int = 30) -> dict:
    """Thin wrapper — trains LR on the fly and returns full forecast dict."""
    engine = StockMLEngine(ticker=df["Ticker"].iloc[0] if "Ticker" in df.columns else "STOCK",
                           horizon=horizon)
    tr = engine._fit_lr(df)

    last_feat_s = engine.lr_scaler.transform(engine._last_features.reshape(1, -1))
    future_pred = np.array([float(engine.lr_model.predict(last_feat_s)[0])] * horizon)
    future_dates = pd.bdate_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon
    )
    return {
        "train_dates"  : tr["test_dates"],   # reuse — API only shows test
        "test_dates"   : tr["test_dates"],
        "future_dates" : future_dates,
        "y_train_true" : tr["y_test_true"],
        "y_test_true"  : tr["y_test_true"],
        "y_test_pred"  : tr["y_test_pred"],
        "future_pred"  : future_pred,
        "metrics"      : tr["metrics"],
    }


def lstm_forecast(df: pd.DataFrame, seq_len: int = 60, horizon: int = 30) -> dict:
    """Thin wrapper — trains LSTM on the fly and returns full forecast dict."""
    engine = StockMLEngine(ticker=df["Ticker"].iloc[0] if "Ticker" in df.columns else "STOCK",
                           horizon=horizon, seq_len=seq_len)
    tr = engine._fit_lstm(df)
    if "error" in tr:
        return tr

    seq = list(engine._last_close_seq)
    fut_s = []
    for _ in range(horizon):
        inp = np.array(seq[-seq_len:]).reshape(1, seq_len, 1)
        p   = float(engine.lstm_model.predict(inp, verbose=0)[0][0])
        fut_s.append(p)
        seq.append(p)
    future_pred = engine.lstm_scaler.inverse_transform(
        np.array(fut_s).reshape(-1, 1)
    ).flatten()
    future_dates = pd.bdate_range(
        start=engine._last_date + pd.Timedelta(days=1), periods=horizon
    )
    return {
        "train_dates"  : tr["test_dates"],
        "test_dates"   : tr["test_dates"],
        "future_dates" : future_dates,
        "y_train_true" : tr["y_test_true"],
        "y_test_true"  : tr["y_test_true"],
        "y_test_pred"  : tr["y_test_pred"],
        "future_pred"  : future_pred,
        "metrics"      : tr["metrics"],
    }


def kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Thin wrapper — trains K-Means on the fly."""
    engine = StockMLEngine(ticker=df["Ticker"].iloc[0] if "Ticker" in df.columns else "STOCK")
    engine._fit_kmeans(df, n_clusters=n_clusters)
    d = df.copy()
    d["Log_Volume"] = np.log1p(d["Volume"].astype(float))
    sub = d.dropna(subset=engine._KM_FEATURES).copy()
    labels = engine.km_pipe.predict(sub[engine._KM_FEATURES].values)
    sub["Segment"]       = labels
    sub["Segment_Label"] = pd.Series(labels).map(engine.km_label_map).values
    return d.merge(sub[["Date","Segment","Segment_Label"]], on="Date", how="left")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline import get_full_data, DEFAULT_DATA_PATH

    csv  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    tick = sys.argv[2] if len(sys.argv) > 2 else "AMZN"

    df = get_full_data(csv, ticker=tick)
    print(f"\nDataset: {tick}  |  {len(df)} rows  |  {df['Date'].min().date()} → {df['Date'].max().date()}")

    engine = StockMLEngine(ticker=tick, horizon=30)
    train_results = engine.fit(df, train_lstm=False)   # set True for LSTM
    engine.save()

    print("\n=== Model Comparison ===")
    for name in ["linear_regression", "random_forest"]:
        m = train_results[name]["metrics"]
        print(f"  {name:<25} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%")

    print("\n=== Signal Classifier ===")
    print(f"  Accuracy: {train_results['classifier']['accuracy']:.4f}")

    print("\n=== Latest Prediction ===")
    res = engine.predict(df)
    print(f"  ML Signal   : {res['classifier']['latest_signal']}")
    print(f"  Probabilities: {res['classifier']['probabilities']}")
    print(f"  LR 30d range: ${res['linear_regression']['future_pred'][0]:.2f} → ${res['linear_regression']['future_pred'][-1]:.2f}")
