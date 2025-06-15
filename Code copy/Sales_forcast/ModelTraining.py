import pandas as pd
import numpy as np
import joblib
import time
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === CONFIGURATION ===
INPUT_FILE = "Dataset/Dataset.csv"  # Or .xlsx
DATE_COLUMN = "Order Date"
SALES_COLUMN = "Sales"
MONTHS_TO_FORECAST = 12
ARIMA_ORDER = (5, 1, 0)
MODEL_OUTPUT_DIR = "DevelopedModels"

# === PREPROCESS ===
def detectFrequency(dates):
    diffs = pd.Series(dates).sort_values().diff().dt.days.dropna()
    if diffs.empty:
        return "MS"
    avg_diff = diffs.mode().iloc[0]
    return "W" if avg_diff <= 8 else "SM" if avg_diff <= 15 else "MS"

def loadData(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="ISO-8859-1")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to read file due to encoding issues: {e}")
    df = df[[DATE_COLUMN, SALES_COLUMN]].dropna()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna()
    return df

# === TRAIN & SAVE MODELS ===
def trainModels(df):
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    freq_code = detectFrequency(df[DATE_COLUMN])
    df = df.groupby(pd.Grouper(key=DATE_COLUMN, freq=freq_code)).sum().sort_index()
    series = df[SALES_COLUMN].astype(float)

    train = series[:-MONTHS_TO_FORECAST]

    # XGBoost feature creation
    df_feat = pd.DataFrame({'y': train})
    for lag in range(1, 13):
        df_feat[f"lag_{lag}"] = df_feat['y'].shift(lag)
    df_feat.dropna(inplace=True)

    X = df_feat.drop("y", axis=1).values
    y = df_feat['y'].values

    # XGBoost training
    print("🔧 Training XGBoost...")
    start_xgb = time.time()
    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)
    end_xgb = time.time()
    print(f"✅ XGBoost trained in {end_xgb - start_xgb:.2f}s")

    # ARIMA training
    print("🔧 Training ARIMA...")
    start_arima = time.time()
    arima_model = ARIMA(train, order=ARIMA_ORDER).fit()
    end_arima = time.time()
    print(f"✅ ARIMA trained in {end_arima - start_arima:.2f}s")

    # Ensure output directory exists before saving models
    import os
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    # Save models
    joblib.dump(xgb_model, MODEL_OUTPUT_DIR + "xgboost_model.pkl")
    arima_model.save(MODEL_OUTPUT_DIR + "arima_model.pkl")
    print(f"📦 Models saved to {MODEL_OUTPUT_DIR}")

# === MAIN ===
if __name__ == "__main__":
    df = loadData(INPUT_FILE)
    trainModels(df)