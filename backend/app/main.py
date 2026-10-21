from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# DOSYA YOLLARI VE AYARLAR
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

CRYPTO_DATA_DIR = DATA_DIR / "crypto_processed"
CRYPTO_MODEL_DIR = MODEL_DIR / "crypto"

HISTORY_LIMIT = 120
DEFAULT_HORIZON = 20

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40

# Dinamik eşik ayarları: AL/SAT sınırları, modelin son dönemdeki
# olasılık dağılımına göre hisse/kripto ve model başına hesaplanır.
DYNAMIC_LOOKBACK = 250
DYNAMIC_MIN_SAMPLES = 60
DYNAMIC_BUY_QUANTILE = 0.80
DYNAMIC_SELL_QUANTILE = 0.20
DYNAMIC_BUY_RANGE = (0.52, 0.70)
DYNAMIC_SELL_RANGE = (0.30, 0.48)
DYNAMIC_MIN_GAP = 0.08

AVAILABLE_MODELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

BINANCE_BASE_URL = "https://data-api.binance.vision"
REQUEST_TIMEOUT = 10

model_cache = {}
crypto_model_cache = {}


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="FinanceAI API",
    description=(
        "BIST hisseleri ve kripto paralar için "
        "makine öğrenmesi tahmin API'si."
    ),
    version="3.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# SEMBOL NORMALİZASYONU
# ==================================================

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()

    if symbol.endswith(".IS"):
        symbol = symbol[:-3]

    if not re.fullmatch(r"[A-Z0-9]+", symbol):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz hisse sembolü.",
        )

    return symbol


def normalize_crypto_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()

    if symbol.endswith(".CSV"):
        symbol = symbol[:-4]

    if not re.fullmatch(r"[A-Z0-9]+", symbol):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz kripto sembolü.",
        )

    return symbol


# ==================================================
# VERİ SETİ YOLLARI
# ==================================================

def get_dataset_path(symbol: str) -> Path:
    symbol = normalize_symbol(symbol)

    candidates = [
        DATA_DIR / f"{symbol}_IS_features.csv",
        DATA_DIR / f"{symbol}_IS_dataset.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise HTTPException(
        status_code=404,
        detail=f"{symbol} için BIST veri dosyası bulunamadı.",
    )


def normalize_model_name(model_name: str) -> str:
    model_name = (model_name or "random_forest").strip().lower()
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Geçersiz model. Kullanılabilir modeller: "
                + ", ".join(AVAILABLE_MODELS.keys())
            ),
        )
    return model_name


def get_model_path(symbol: str, model_name: str = "random_forest") -> Path:
    symbol = normalize_symbol(symbol)
    model_name = normalize_model_name(model_name)
    return MODEL_DIR / f"{symbol}_IS_{model_name}.joblib"


def get_available_models(symbol: str, market: str = "stocks") -> list[dict]:
    """Yalnızca ilgili varlık için gerçekten mevcut model dosyalarını döndürür."""
    available = []
    for key, label in AVAILABLE_MODELS.items():
        if market == "stocks":
            path = get_model_path(symbol, key)
        elif market == "crypto":
            path = CRYPTO_MODEL_DIR / f"{normalize_crypto_symbol(symbol)}_{key}.joblib"
        else:
            raise HTTPException(status_code=400, detail="market stocks veya crypto olmalı.")
        if path.exists():
            available.append({"value": key, "label": label})
    return available


def get_crypto_dataset_path(symbol: str) -> Path:
    symbol = normalize_crypto_symbol(symbol)

    # Grafik ve tahmin için güncel özellik dosyası öncelikli.
    candidates = [
        CRYPTO_DATA_DIR / f"{symbol}_features.csv",
        CRYPTO_DATA_DIR / f"{symbol}_dataset.csv",
        CRYPTO_DATA_DIR / f"{symbol}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise HTTPException(
        status_code=404,
        detail=f"{symbol} için kripto veri dosyası bulunamadı.",
    )


# ==================================================
# VERİ SETİ OKUMA
# ==================================================

def _load_csv(path: Path, symbol: str, asset_type: str):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} için veri dosyası bulunamadı.",
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Veri dosyası okunamadı: {exc}",
        )

    if not {"Date", "Close"}.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail=(
                f"{asset_type} veri setinde Date ve Close "
                "sütunları bulunmalı."
            ),
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    for column in df.columns:
        if column != "Date":
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    df = df[df["Close"] > 0].copy()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} için kullanılabilir veri yok.",
        )

    return df


def load_dataset(symbol: str):
    symbol = normalize_symbol(symbol)

    return _load_csv(
        get_dataset_path(symbol),
        symbol,
        "BIST",
    )


def load_crypto_dataset(symbol: str):
    symbol = normalize_crypto_symbol(symbol)

    return _load_csv(
        get_crypto_dataset_path(symbol),
        symbol,
        "Kripto",
    )


# ==================================================
# MODEL PAKETİ YÜKLEME
# ==================================================

def _load_model_package(
    symbol: str,
    model_path: Path,
    cache: dict,
    asset_type: str,
):
    if symbol in cache:
        return cache[symbol]

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} için {asset_type} modeli bulunamadı.",
        )

    try:
        loaded = joblib.load(model_path)

        if not isinstance(loaded, dict):
            raise ValueError(
                "Model paketi sözlük biçiminde değil."
            )

        classifier = loaded.get("model")
        features = loaded.get("features")
        horizon = loaded.get("horizon", DEFAULT_HORIZON)

        if classifier is None or not hasattr(
            classifier,
            "predict_proba",
        ):
            raise ValueError(
                "Model predict_proba desteklemiyor."
            )

        if not isinstance(features, list) or not features:
            raise ValueError(
                "Model özellik listesi geçersiz."
            )

        stored_model_name = loaded.get("model_name", "")
        display_model_name = AVAILABLE_MODELS.get(
            stored_model_name,
            stored_model_name.replace("_", " ").title() if stored_model_name else asset_type,
        )
        package = {
            "model": classifier,
            "features": features,
            "horizon": int(horizon),
            "model_name": display_model_name,
        }

        cache[symbol] = package
        return package

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{symbol} modeli yüklenemedi: {exc}",
        )


def get_model_package(symbol: str, model_name: str = "random_forest"):
    symbol = normalize_symbol(symbol)
    model_name = normalize_model_name(model_name)
    cache_key = f"{symbol}:{model_name}"
    return _load_model_package(
        cache_key,
        get_model_path(symbol, model_name),
        model_cache,
        AVAILABLE_MODELS[model_name],
    )


def get_crypto_model_package(symbol: str, model_name: str = "random_forest"):
    symbol = normalize_crypto_symbol(symbol)
    model_name = normalize_model_name(model_name)
    cache_key = f"{symbol}:{model_name}"
    model_path = CRYPTO_MODEL_DIR / f"{symbol}_{model_name}.joblib"
    return _load_model_package(
        cache_key,
        model_path,
        crypto_model_cache,
        AVAILABLE_MODELS[model_name],
    )


# ==================================================
# TEKNİK GÖSTERGELER
# ==================================================

def build_technical_indicators(df: pd.DataFrame) -> dict:
    data = df.copy()

    def safe_number(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def last_value(series):
        if series is None or len(series) == 0:
            return None
        return safe_number(series.iloc[-1])

    close = pd.to_numeric(data["Close"], errors="coerce")

    high = (
        pd.to_numeric(data["High"], errors="coerce")
        if "High" in data.columns
        else None
    )

    low = (
        pd.to_numeric(data["Low"], errors="coerce")
        if "Low" in data.columns
        else None
    )

    volume = (
        pd.to_numeric(data["Volume"], errors="coerce")
        if "Volume" in data.columns
        else None
    )

    returns = close.pct_change(fill_method=None)

    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    ema_12 = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()

    relative_strength = avg_gain.div(
        avg_loss.replace(0, np.nan)
    )

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100,
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50,
    )

    macd_line = ema_12 - ema_26

    macd_signal = macd_line.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    macd_histogram = macd_line - macd_signal

    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)

    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std

    bb_range = (bb_upper - bb_lower).replace(0, np.nan)

    bb_position = (close - bb_lower) / bb_range
    bb_width = (bb_upper - bb_lower) / bb_middle

    volume_change = None
    volume_ratio = None
    obv_change = None

    if volume is not None:
        volume_change = volume.pct_change(fill_method=None)

        volume_mean_20 = volume.rolling(20).mean()
        volume_ratio = volume / volume_mean_20

        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume.fillna(0)).cumsum()

        volume_sum_20 = volume.rolling(20).sum()

        obv_change = (
            (obv - obv.shift(20))
            / volume_sum_20.replace(0, np.nan)
        )

    volatility_20d = returns.rolling(20).std()

    atr_ratio = None

    if high is not None and low is not None:
        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr_14 = true_range.rolling(
            14,
            min_periods=14,
        ).mean()

        atr_ratio = atr_14 / close

    def ratio_percent(series):
        value = last_value(series)
        return None if value is None else round(value * 100, 4)

    def ratio_value(series):
        value = last_value(series)
        return None if value is None else round(value, 6)

    return {
        "rsi": (
            round(last_value(rsi), 2)
            if last_value(rsi) is not None
            else None
        ),
        "macd": ratio_percent(macd_line / close),
        "macd_signal": ratio_percent(macd_signal / close),
        "macd_histogram": ratio_percent(macd_histogram / close),
        "ema_12": (
            round(last_value(ema_12), 4)
            if last_value(ema_12) is not None
            else None
        ),
        "ema_26": (
            round(last_value(ema_26), 4)
            if last_value(ema_26) is not None
            else None
        ),
        "ema_cross": ratio_percent((ema_12 - ema_26) / close),
        "sma_5": (
            round(last_value(sma_5), 4)
            if last_value(sma_5) is not None
            else None
        ),
        "sma_20": (
            round(last_value(sma_20), 4)
            if last_value(sma_20) is not None
            else None
        ),
        "sma_50": (
            round(last_value(sma_50), 4)
            if last_value(sma_50) is not None
            else None
        ),
        "sma_20_ratio": ratio_value(sma_20 / close),
        "sma_50_ratio": ratio_value(sma_50 / close),
        "bb_position": ratio_value(bb_position),
        "bb_width": ratio_percent(bb_width),
        "volume_change": (
            ratio_percent(volume_change)
            if volume_change is not None
            else None
        ),
        "volume_ratio": (
            ratio_value(volume_ratio)
            if volume_ratio is not None
            else None
        ),
        "obv_change": (
            ratio_value(obv_change)
            if obv_change is not None
            else None
        ),
        "volatility_20d": ratio_percent(volatility_20d),
        "atr_ratio": (
            ratio_percent(atr_ratio)
            if atr_ratio is not None
            else None
        ),
        "indicator_data_available": {
            "volume": volume is not None,
            "high_low": high is not None and low is not None,
            "enough_history_for_rsi": len(data) >= 14,
            "enough_history_for_macd": len(data) >= 35,
            "enough_history_for_sma50": len(data) >= 50,
        },
    }


# ==================================================
# ORTAK TAHMİN FONKSİYONU
# ==================================================

def check_thresholds(buy_threshold, sell_threshold):
    if (buy_threshold is None) != (sell_threshold is None):
        raise HTTPException(
            status_code=400,
            detail="buy_threshold ve sell_threshold birlikte verilmelidir.",
        )

    if buy_threshold is not None and buy_threshold <= sell_threshold:
        raise HTTPException(
            status_code=400,
            detail="buy_threshold, sell_threshold değerinden büyük olmalıdır.",
        )


def compute_dynamic_thresholds(valid, features, model, classes):
    """
    Modelin son DYNAMIC_LOOKBACK gündeki pozitif getiri olasılıklarının
    dağılımından AL / SAT eşiklerini üretir. Yetersiz veri veya hata
    durumunda None döner (sabit eşiklere geri dönülür).
    """
    try:
        window = valid[features].iloc[-DYNAMIC_LOOKBACK:]

        if len(window) < DYNAMIC_MIN_SAMPLES:
            return None

        probs = model.predict_proba(window)[:, classes.index(1)]

        buy = float(np.quantile(probs, DYNAMIC_BUY_QUANTILE))
        sell = float(np.quantile(probs, DYNAMIC_SELL_QUANTILE))

        buy = min(max(buy, DYNAMIC_BUY_RANGE[0]), DYNAMIC_BUY_RANGE[1])
        sell = min(max(sell, DYNAMIC_SELL_RANGE[0]), DYNAMIC_SELL_RANGE[1])

        if buy - sell < DYNAMIC_MIN_GAP:
            mid = (buy + sell) / 2
            buy = mid + DYNAMIC_MIN_GAP / 2
            sell = mid - DYNAMIC_MIN_GAP / 2

        return round(buy, 4), round(sell, 4)

    except Exception:
        return None


def generate_prediction(
    symbol,
    df,
    package,
    buy_threshold=None,
    sell_threshold=None,
    dynamic=True,
):
    df = df.copy()
    features = package["features"]

    missing = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Eksik model özellikleri: {missing}",
        )

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    valid = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Date", "Close"] + features)
    )

    if valid.empty:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} için tahmin üretilebilecek veri bulunamadı.",
        )

    latest = valid.iloc[-1]
    model = package["model"]

    try:
        probabilities = model.predict_proba(
            valid[features].iloc[[-1]]
        )[0]

        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Modelde pozitif getiri sınıfı yok.")

        probability_up = float(
            probabilities[classes.index(1)]
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin üretilemedi: {exc}",
        )

    # Eşik önceliği: elle verilen > dinamik hesap > sabit varsayılan
    if buy_threshold is not None and sell_threshold is not None:
        buy_limit, sell_limit = buy_threshold, sell_threshold
        threshold_mode = "manual"
    else:
        dynamic_limits = (
            compute_dynamic_thresholds(valid, features, model, classes)
            if dynamic
            else None
        )

        if dynamic_limits is not None:
            buy_limit, sell_limit = dynamic_limits
            threshold_mode = "dynamic"
        else:
            buy_limit, sell_limit = BUY_THRESHOLD, SELL_THRESHOLD
            threshold_mode = "static"

    if probability_up >= buy_limit:
        signal = "AL"
        strength = "Yüksek"
    elif probability_up <= sell_limit:
        signal = "SAT"
        strength = "Düşük"
    else:
        signal = "BEKLE"
        strength = "Belirsiz"

    return {
        "symbol": symbol,
        "date": str(latest["Date"].date()),
        "close": float(latest["Close"]),
        "horizon_trading_days": package["horizon"],
        "probability_positive_return": round(probability_up, 4),
        "probability_non_positive_return": round(1 - probability_up, 4),
        "signal": signal,
        "signal_strength": strength,
        "signal_thresholds": {
            "buy": buy_limit,
            "sell": sell_limit,
            "mode": threshold_mode,
        },
        "model": package.get("model_name", "Bilinmeyen Model"),
        "technical_indicators": build_technical_indicators(df),
        "disclaimer": (
            "Bu deneysel bir model tahminidir, yatırım tavsiyesi değildir. "
            "Tahmin olasılığı gerçekleşecek getiriyi garanti etmez. "
            "Teknik göstergeler tek başlarına alım veya satım sinyali değildir."
        ),
    }


# ==================================================
# HİSSE VE KRİPTO LİSTELERİ
# ==================================================

def discover_stocks():
    stocks = []

    if not DATA_DIR.exists() or not MODEL_DIR.exists():
        return stocks

    symbols = set()

    for path in DATA_DIR.glob("*_IS_features.csv"):
        symbols.add(path.name.replace("_IS_features.csv", ""))

    for path in DATA_DIR.glob("*_IS_dataset.csv"):
        symbols.add(path.name.replace("_IS_dataset.csv", ""))

    for symbol in sorted(symbols):
        if not any(get_model_path(symbol, model_key).exists() for model_key in AVAILABLE_MODELS):
            continue

        try:
            df = load_dataset(symbol)
            latest = df.iloc[-1]

            stocks.append({
                "symbol": symbol,
                "ticker": f"{symbol}.IS",
                "latest_date": str(latest["Date"].date()),
                "latest_close": float(latest["Close"]),
                "observations": len(df),
            })

        except HTTPException:
            continue

    return stocks


def discover_cryptos():
    cryptos = []

    if not CRYPTO_DATA_DIR.exists() or not CRYPTO_MODEL_DIR.exists():
        return cryptos

    symbols = set()

    for path in CRYPTO_DATA_DIR.glob("*_features.csv"):
        symbols.add(path.name.replace("_features.csv", ""))

    for path in CRYPTO_DATA_DIR.glob("*_dataset.csv"):
        symbols.add(path.name.replace("_dataset.csv", ""))

    for symbol in sorted(symbols):
        if not any(
            (CRYPTO_MODEL_DIR / f"{symbol}_{model_key}.joblib").exists()
            for model_key in AVAILABLE_MODELS
        ):
            continue

        try:
            df = load_crypto_dataset(symbol)
            latest = df.iloc[-1]

            cryptos.append({
                "symbol": symbol,
                "ticker": symbol,
                "latest_date": str(latest["Date"].date()),
                "latest_close": float(latest["Close"]),
                "observations": len(df),
            })

        except HTTPException:
            continue

    return cryptos


# ==================================================
# TEMEL ENDPOINT'LER
# ==================================================

@app.get("/")
def home():
    return {
        "name": "FinanceAI API",
        "status": "running",
        "version": "3.2.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "data_directory_exists": DATA_DIR.exists(),
        "model_directory_exists": MODEL_DIR.exists(),
        "available_stock_count": len(discover_stocks()),
        "available_crypto_count": len(discover_cryptos()),
    }


# ==================================================
# BIST ENDPOINT'LERİ
# ==================================================

@app.get("/stocks")
def get_stocks():
    stocks = discover_stocks()

    return {
        "count": len(stocks),
        "stocks": stocks,
    }


@app.get("/stocks/{symbol}/history")
def get_stock_history(
    symbol: str,
    limit: int = Query(default=HISTORY_LIMIT, ge=1, le=1000),
):
    symbol = normalize_symbol(symbol)
    df = load_dataset(symbol)

    columns = ["Date", "Close"]

    if "Volume" in df.columns:
        columns.append("Volume")

    history = []

    for _, row in df[columns].tail(limit).iterrows():
        item = {
            "date": str(row["Date"].date()),
            "close": float(row["Close"]),
        }

        if "Volume" in columns:
            item["volume"] = (
                float(row["Volume"])
                if pd.notna(row["Volume"])
                else None
            )

        history.append(item)

    return {
        "symbol": symbol,
        "count": len(history),
        "history": history,
        "prices": history,
    }


@app.get("/models")
def list_models(
    symbol: str = Query(..., min_length=1),
    market: str = Query(default="stocks"),
):
    """Bir varlık için eğitilmiş ve kullanılabilir model seçenekleri."""
    if market == "stocks":
        normalized_symbol = normalize_symbol(symbol)
    elif market == "crypto":
        normalized_symbol = normalize_crypto_symbol(symbol)
    else:
        raise HTTPException(status_code=400, detail="market stocks veya crypto olmalı.")
    models = get_available_models(normalized_symbol, market)
    return {
        "symbol": normalized_symbol,
        "market": market,
        "models": models,
        "default": "random_forest" if any(m["value"] == "random_forest" for m in models) else (models[0]["value"] if models else None),
    }


@app.get("/predict/{symbol}")
def predict_stock(
    symbol: str,
    model: str = Query(default="random_forest"),
    dynamic: bool = Query(default=True),
    buy_threshold: float | None = Query(default=None, gt=0, lt=1),
    sell_threshold: float | None = Query(default=None, gt=0, lt=1),
):
    check_thresholds(buy_threshold, sell_threshold)
    symbol = normalize_symbol(symbol)
    package = get_model_package(symbol, model)
    df = load_dataset(symbol)
    return generate_prediction(
        symbol, df, package,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        dynamic=dynamic,
    )


# ==================================================
# KRİPTO ENDPOINT'LERİ
# ==================================================

@app.get("/crypto")
def get_cryptos():
    cryptos = discover_cryptos()

    return {
        "count": len(cryptos),
        "cryptos": cryptos,
    }


@app.get("/crypto/live/{symbol}")
def get_live_crypto_price(symbol: str):
    """
    Binance'ten canlı spot fiyatı getirir.
    Bu endpoint günlük mum geçmişinden bağımsızdır.
    """
    symbol = normalize_crypto_symbol(symbol)

    try:
        response = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()

        return {
            "symbol": symbol,
            "price": float(result["price"]),
            "source": "Binance",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Binance canlı fiyatına şu anda ulaşılamıyor.",
        )

    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Binance geçerli bir fiyat verisi döndürmedi.",
        )


@app.get("/crypto/{symbol}/history")
def get_crypto_history(
    symbol: str,
    limit: int = Query(default=HISTORY_LIMIT, ge=1, le=1000),
):
    symbol = normalize_crypto_symbol(symbol)
    df = load_crypto_dataset(symbol)

    columns = ["Date", "Close"]

    if "Volume" in df.columns:
        columns.append("Volume")

    history = []

    for _, row in df[columns].tail(limit).iterrows():
        item = {
            "date": str(row["Date"].date()),
            "close": float(row["Close"]),
        }

        if "Volume" in columns:
            item["volume"] = (
                float(row["Volume"])
                if pd.notna(row["Volume"])
                else None
            )

        history.append(item)

    return {
        "symbol": symbol,
        "count": len(history),
        "history": history,
        "prices": history,
    }


@app.get("/crypto/predict/{symbol}")
def predict_crypto(
    symbol: str,
    model: str = Query(default="random_forest"),
    dynamic: bool = Query(default=True),
    buy_threshold: float | None = Query(default=None, gt=0, lt=1),
    sell_threshold: float | None = Query(default=None, gt=0, lt=1),
):
    check_thresholds(buy_threshold, sell_threshold)
    symbol = normalize_crypto_symbol(symbol)
    package = get_crypto_model_package(symbol, model)
    df = load_crypto_dataset(symbol)
    return generate_prediction(
        symbol, df, package,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        dynamic=dynamic,
    )


# ==================================================
# MODEL PERFORMANSI
# ==================================================

def read_result_file(path: Path):
    if not path.exists():
        return {
            "available": False,
            "message": f"{path.name} dosyası bulunamadı.",
            "columns": [],
            "rows": [],
        }

    try:
        df = pd.read_csv(path)

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        df = df.astype(object).where(
            pd.notna(df),
            None,
        )

        return {
            "available": True,
            "message": "Sonuçlar başarıyla yüklendi.",
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
        }

    except Exception as exc:
        return {
            "available": False,
            "message": f"Dosya okunamadı: {exc}",
            "columns": [],
            "rows": [],
        }


def read_first_available_result(paths: list[Path]):
    """İlk mevcut CSV dosyasını okur; hiçbiri yoksa aday dosyaları bildirir."""
    for path in paths:
        if path.exists():
            result = read_result_file(path)
            result["source_file"] = path.name
            return result

    names = ", ".join(path.name for path in paths)
    return {
        "available": False,
        "message": f"Sonuç dosyası bulunamadı. Aranan dosyalar: {names}",
        "columns": [],
        "rows": [],
        "source_file": None,
    }


@app.get("/model-performance")
def get_model_performance():
    """BIST ve kripto model test sonuçlarını API üzerinden sunar."""
    return {
        # Klasik model karşılaştırması
        "model_comparison": read_result_file(
            DATA_DIR / "model_comparison.csv"
        ),

        # Dosya adı farklı olan walk-forward raporları için yedek yollar
        "walk_forward": read_first_available_result([
            DATA_DIR / "walk_forward_summary.csv",
            DATA_DIR / "walk_forward_feature_summary.csv",
            DATA_DIR / "all_stocks_feature_summary.csv",
            DATA_DIR / "feature_set_summary.csv",
        ]),

        # Özellik gruplarını karşılaştıran ayrıntılı raporlar
        "walk_forward_feature_comparison": read_result_file(
            DATA_DIR / "walk_forward_feature_comparison.csv"
        ),
        "walk_forward_feature_summary": read_result_file(
            DATA_DIR / "walk_forward_feature_summary.csv"
        ),
        "all_stocks_feature_comparison": read_result_file(
            DATA_DIR / "all_stocks_feature_comparison.csv"
        ),
        "all_stocks_feature_summary": read_result_file(
            DATA_DIR / "all_stocks_feature_summary.csv"
        ),
        "feature_set_comparison": read_result_file(
            DATA_DIR / "feature_set_comparison.csv"
        ),
        "feature_set_summary": read_result_file(
            DATA_DIR / "feature_set_summary.csv"
        ),

        # Geçmiş strateji ve al-tut karşılaştırması
        "backtest_comparison": read_result_file(
            DATA_DIR / "backtest_comparison.csv"
        ),
        "backtest_model_predictions": read_result_file(
            DATA_DIR / "backtest_model_predictions.csv"
        ),

        # Kripto model sonuçları
        "crypto_model_comparison": read_result_file(
            CRYPTO_DATA_DIR / "crypto_model_comparison.csv"
        ),
        "crypto_walk_forward": read_first_available_result([
            CRYPTO_DATA_DIR / "walk_forward" / "crypto_walk_forward_summary.csv",
        ]),
        "crypto_backtest": read_first_available_result([
            CRYPTO_DATA_DIR / "walk_forward" / "crypto_backtest_summary.csv",
        ]),
    }