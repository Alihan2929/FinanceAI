
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# DOSYA YOLLARI VE AYARLAR
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "TUPRS_IS_dataset.csv"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"

HISTORY_LIMIT = 120
DEFAULT_HORIZON = 20

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="FinanceAI API",
    description=(
        "BIST hisse verileri, makine öğrenmesi tahminleri "
        "ve deneysel işlem sinyalleri API'si."
    ),
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_package = None


# ==================================================
# MODEL YÜKLEME
# ==================================================

def get_model_package():
    global model_package

    if model_package is not None:
        return model_package

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model dosyası bulunamadı. "
                "Önce backend/train_model.py dosyasını çalıştır."
            ),
        )

    try:
        loaded = joblib.load(MODEL_PATH)

        if not isinstance(loaded, dict):
            raise ValueError(
                "Model dosyası sözlük biçiminde değil."
            )

        classifier = loaded.get("model")
        features = loaded.get("features")
        horizon = loaded.get("horizon", DEFAULT_HORIZON)

        if classifier is None:
            raise ValueError(
                "Model paketinde 'model' alanı bulunamadı."
            )

        if not hasattr(classifier, "predict_proba"):
            raise ValueError(
                "Model predict_proba metodunu desteklemiyor."
            )

        if not isinstance(features, list) or not features:
            raise ValueError(
                "Model özellik listesi geçersiz."
            )

        model_package = {
            "model": classifier,
            "features": features,
            "horizon": int(horizon),
        }

        return model_package

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model yüklenemedi: {exc}",
        )


# ==================================================
# VERİ SETİNİ YÜKLEME
# ==================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "TUPRS_IS_dataset.csv bulunamadı. "
                "Veri setinin backend/data klasöründe "
                "olduğunu kontrol et."
            ),
        )

    try:
        df = pd.read_csv(DATA_PATH)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Veri seti okunamadı: {exc}",
        )

    required_columns = ["Date", "Close"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "Veri setinde eksik sütunlar: "
                + ", ".join(missing_columns)
            ),
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(
            df["Volume"],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="Kullanılabilir fiyat verisi bulunamadı.",
        )

    return df


# ==================================================
# MODEL İÇİN EN GÜNCEL SATIRI HAZIRLAMA
# ==================================================

def get_latest_model_row():
    package = get_model_package()
    df = load_dataset()

    features = package["features"]

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=(
                "Veri setinde model özellikleri eksik: "
                + ", ".join(missing_features)
            ),
        )

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_rows = df.dropna(
        subset=["Date", "Close"] + features
    )

    if valid_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Tahmin için uygun veri satırı bulunamadı."
            ),
        )

    latest = valid_rows.iloc[-1]
    X_latest = valid_rows[features].iloc[[-1]]

    return package, latest, X_latest


# ==================================================
# CSV SONUÇ DOSYALARINI OKUMA
# ==================================================

def read_result_file(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        return {
            "available": False,
            "message": f"{filename} dosyası bulunamadı.",
            "columns": [],
            "rows": [],
        }

    try:
        result_df = pd.read_csv(file_path)

        result_df = result_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        result_df = result_df.astype(object).where(
            pd.notna(result_df),
            None,
        )

        return {
            "available": True,
            "message": "Sonuçlar başarıyla yüklendi.",
            "columns": list(result_df.columns),
            "rows": result_df.to_dict(orient="records"),
        }

    except Exception as exc:
        return {
            "available": False,
            "message": f"{filename} okunamadı: {exc}",
            "columns": [],
            "rows": [],
        }


# ==================================================
# SAĞLIK KONTROLÜ
# ==================================================

@app.get("/")
def home():
    return {
        "name": "FinanceAI API",
        "status": "running",
        "version": "1.3.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_file_exists": MODEL_PATH.exists(),
        "dataset_exists": DATA_PATH.exists(),
    }


# ==================================================
# HİSSE LİSTESİ
# ==================================================

@app.get("/stocks")
def get_stocks():
    df = load_dataset()
    latest = df.iloc[-1]

    return {
        "stocks": [
            {
                "symbol": "TUPRS",
                "name": "Tüpraş",
                "latest_date": str(latest["Date"].date()),
                "latest_close": float(latest["Close"]),
            }
        ]
    }


# ==================================================
# GEÇMİŞ FİYATLAR
# ==================================================

@app.get("/stocks/{symbol}/history")
def get_stock_history(
    symbol: str,
    limit: int = Query(default=HISTORY_LIMIT, ge=1, le=1000),
):
    if symbol.upper() != "TUPRS":
        raise HTTPException(
            status_code=404,
            detail="Şimdilik yalnızca TUPRS destekleniyor.",
        )

    df = load_dataset()

    selected_columns = ["Date", "Close"]

    if "Volume" in df.columns:
        selected_columns.append("Volume")

    history_df = df[selected_columns].tail(limit).copy()

    prices = []

    for _, row in history_df.iterrows():
        item = {
            "date": str(row["Date"].date()),
            "close": float(row["Close"]),
        }

        if "Volume" in history_df.columns:
            volume = row["Volume"]

            item["volume"] = (
                float(volume)
                if pd.notna(volume)
                else None
            )

        prices.append(item)

    return {
        "symbol": "TUPRS",
        "count": len(prices),
        "history": prices,
        "prices": prices,
    }


# ==================================================
# AL / BEKLE / SAT SİNYALİ
# ==================================================

@app.get("/predict/TUPRS")
def predict_tuprs():
    package, latest, X_latest = get_latest_model_row()

    classifier = package["model"]

    try:
        probabilities = classifier.predict_proba(X_latest)[0]
        classes = list(classifier.classes_)

        if 1 not in classes:
            raise ValueError(
                "Modelde pozitif getiri sınıfı (1) bulunamadı."
            )

        probability_up = float(
            probabilities[classes.index(1)]
        )

        probability_not_up = 1.0 - probability_up

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin üretilemedi: {exc}",
        )

    # Sinyal eşikleri
    if probability_up >= BUY_THRESHOLD:
        signal = "AL"
        signal_reason = (
            "20 işlem günlük pozitif getiri olasılığı "
            "%60 veya üzerinde."
        )
        signal_strength = "Yüksek"

    elif probability_up <= SELL_THRESHOLD:
        signal = "SAT"
        signal_reason = (
            "20 işlem günlük pozitif getiri olasılığı "
            "%40 veya altında. Bu, deneysel bir risk "
            "azaltma sinyalidir."
        )
        signal_strength = "Düşük"

    else:
        signal = "BEKLE"
        signal_reason = (
            "Tahmin olasılığı AL veya SAT eşiklerini "
            "karşılamıyor."
        )
        signal_strength = "Belirsiz"

    return {
        "symbol": "TUPRS",
        "date": str(latest["Date"].date()),
        "close": float(latest["Close"]),
        "horizon_trading_days": package["horizon"],
        "probability_positive_return": round(
            probability_up, 4
        ),
        "probability_non_positive_return": round(
            probability_not_up, 4
        ),
        "signal": signal,
        "signal_reason": signal_reason,
        "signal_strength": signal_strength,
        "signal_thresholds": {
            "buy": BUY_THRESHOLD,
            "sell": SELL_THRESHOLD,
        },
        "model": "Random Forest",
        "disclaimer": (
            "Bu çıktı deneysel bir model tahminidir; "
            "yatırım tavsiyesi değildir. Son veri tarihi "
            "güncel piyasa tarihi olmayabilir. Olasılık, "
            "gerçekleşecek getiri veya kesinlik garantisi değildir."
        ),
    }


# ==================================================
# MODEL PERFORMANSI
# ==================================================

@app.get("/model-performance")
def get_model_performance():
    return {
        "walk_forward": read_result_file(
            "walk_forward_summary.csv"
        ),
        "backtest_comparison": read_result_file(
            "backtest_comparison.csv"
        ),
    }
