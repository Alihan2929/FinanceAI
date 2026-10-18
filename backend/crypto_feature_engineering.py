
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# FINANCEAI - KRİPTO ÖZELLİK ÇIKARIMI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "crypto_raw"
PROCESSED_DIR = BASE_DIR / "data" / "crypto_processed"

HORIZON = 20
MINIMUM_ROWS = 200

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Return_60d",
    "SMA_5_ratio",
    "SMA_20_ratio",
    "SMA_50_ratio",
    "RSI_scaled",
    "Volume_Change",
    "Volatility_20d",
]


# =========================================================
# HAM VERİYİ OKU
# =========================================================

def load_crypto_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])

    required = [
        "Date", "Open", "High", "Low", "Close", "Volume"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    df = df[
        (df["Close"] > 0)
        & (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Volume"] >= 0)
    ].copy()

    if len(df) < MINIMUM_ROWS + HORIZON:
        raise ValueError(f"Yeterli veri yok: {len(df)} satır.")

    return df.reset_index(drop=True)


# =========================================================
# RSI
# =========================================================

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50,
    )

    return rsi


# =========================================================
# ÖZELLİKLERİ HESAPLA
# =========================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm tarihler için geçmişe dayalı özellikleri hesaplar.
    Burada henüz son 20 gün silinmez.
    """

    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["Return_1d"] = close.pct_change(1, fill_method=None)
    df["Return_5d"] = close.pct_change(5, fill_method=None)
    df["Return_20d"] = close.pct_change(20, fill_method=None)
    df["Return_60d"] = close.pct_change(60, fill_method=None)

    for period in [5, 20, 50]:
        moving_average = close.rolling(
            window=period,
            min_periods=period,
        ).mean()

        df[f"SMA_{period}_ratio"] = close / moving_average - 1

    df["RSI_scaled"] = calculate_rsi(close) / 100
    df["Volume_Change"] = volume.pct_change(fill_method=None)

    daily_return = close.pct_change(fill_method=None)
    df["Volatility_20d"] = daily_return.rolling(
        window=20,
        min_periods=20,
    ).std()

    df = df.replace([np.inf, -np.inf], np.nan)

    # Sadece teknik özellikleri hesaplanamayan erken satırlar çıkarılır.
    features_df = df.dropna(subset=FEATURES).copy()

    if len(features_df) < MINIMUM_ROWS:
        raise ValueError(
            f"Özelliklerden sonra yeterli satır kalmadı: "
            f"{len(features_df)}"
        )

    return features_df.reset_index(drop=True)


# =========================================================
# EĞİTİM HEDEFLERİNİ OLUŞTUR
# =========================================================

def create_training_dataset(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gelecekteki 20 günlük getirisi bilinen satırları oluşturur.
    Son 20 tamamlanmış mum eğitim etiketine sahip olmayabilir.
    """

    data = features_df.copy()

    data["Future_Close"] = data["Close"].shift(-HORIZON)
    data["Future_Return_20d"] = (
        data["Future_Close"] / data["Close"] - 1
    )

    data["Target"] = np.where(
        data["Future_Return_20d"].notna(),
        (data["Future_Return_20d"] > 0).astype(int),
        np.nan,
    )

    data = data.dropna(
        subset=["Future_Close", "Future_Return_20d", "Target"]
    ).copy()

    data["Target"] = data["Target"].astype(int)

    if len(data) < MINIMUM_ROWS:
        raise ValueError(
            f"Eğitim veri setinde yeterli satır yok: {len(data)}"
        )

    return data.reset_index(drop=True)


# =========================================================
# TÜM KRİPTOLARI İŞLE
# =========================================================

def process_all_cryptos():
    files = sorted(
        p for p in RAW_DIR.glob("*.csv")
        if p.name != "download_summary.csv"
    )

    if not files:
        raise FileNotFoundError(
            f"Ham kripto verisi bulunamadı: {RAW_DIR}"
        )

    results = []

    print("=" * 65)
    print("FINANCEAI - KRİPTO ÖZELLİK ÇIKARIMI")
    print("=" * 65)
    print(f"Bulunan dosya: {len(files)}")
    print(f"Tahmin ufku: {HORIZON} günlük mum")

    for index, path in enumerate(files, start=1):
        symbol = path.stem
        print(f"\n[{index}/{len(files)}] {symbol}")

        try:
            raw = load_crypto_data(path)
            features_df = create_features(raw)
            training_df = create_training_dataset(features_df)

            # Güncel tahmin ve grafik için: son mum dahil.
            features_path = PROCESSED_DIR / f"{symbol}_features.csv"
            features_df.to_csv(
                features_path,
                index=False,
                encoding="utf-8-sig",
                date_format="%Y-%m-%d",
            )

            # Model eğitimi için: hedefi bilinen satırlar.
            dataset_path = PROCESSED_DIR / f"{symbol}_dataset.csv"

            output_columns = [
                "Date", "Open", "High", "Low", "Close", "Volume",
                *FEATURES,
                "Future_Close", "Future_Return_20d", "Target",
            ]

            training_df[output_columns].to_csv(
                dataset_path,
                index=False,
                encoding="utf-8-sig",
                date_format="%Y-%m-%d",
            )

            latest_feature_date = features_df["Date"].max().date()
            latest_training_date = training_df["Date"].max().date()

            print(f"Ham satır: {len(raw)}")
            print(f"Güncel özellik satırı: {len(features_df)}")
            print(f"Eğitim satırı: {len(training_df)}")
            print(f"Güncel özellik tarihi: {latest_feature_date}")
            print(f"Eğitim etiketlerinin son tarihi: {latest_training_date}")
            print(f"Özellik dosyası: {features_path}")
            print(f"Eğitim dosyası: {dataset_path}")

            results.append({
                "Symbol": symbol,
                "Raw_Rows": len(raw),
                "Feature_Rows": len(features_df),
                "Training_Rows": len(training_df),
                "Latest_Feature_Date": str(latest_feature_date),
                "Latest_Training_Date": str(latest_training_date),
                "Positive_Target_Ratio": round(
                    training_df["Target"].mean(), 4
                ),
                "Status": "OK",
                "Error": "",
            })

        except Exception as error:
            print(f"HATA | {symbol} | {error}")

            results.append({
                "Symbol": symbol,
                "Raw_Rows": 0,
                "Feature_Rows": 0,
                "Training_Rows": 0,
                "Latest_Feature_Date": "",
                "Latest_Training_Date": "",
                "Positive_Target_Ratio": np.nan,
                "Status": "FAILED",
                "Error": str(error),
            })

    summary = pd.DataFrame(results)
    summary_path = PROCESSED_DIR / "feature_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("İŞLEM ÖZETİ")
    print("=" * 65)
    print(summary.to_string(index=False))
    print(f"\nBaşarılı: {(summary['Status'] == 'OK').sum()}")
    print(f"Başarısız: {(summary['Status'] == 'FAILED').sum()}")
    print(f"Özet: {summary_path}")


if __name__ == "__main__":
    process_all_cryptos()
