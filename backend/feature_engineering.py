
from pathlib import Path

import numpy as np
import pandas as pd

from data_fetcher import BIST100_TICKERS


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    # Getiriler
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Return_60d",

    # Basit hareketli ortalamalar
    "SMA_5_ratio",
    "SMA_20_ratio",
    "SMA_50_ratio",

    # Üssel hareketli ortalamalar
    "EMA_12_ratio",
    "EMA_26_ratio",
    "EMA_cross",

    # RSI
    "RSI_scaled",

    # MACD
    "MACD_ratio",
    "MACD_signal_ratio",
    "MACD_histogram_ratio",

    # Bollinger Bands
    "BB_position",
    "BB_width",

    # Hacim
    "Volume_Change",
    "Volume_Ratio",
    "OBV_Change",

    # Volatilite ve ATR
    "Volatility_20d",
    "ATR_ratio",
]


# =========================================================
# TEK HİSSE İÇİN ÖZELLİK HESAPLAMA
# =========================================================

def create_features(ticker: str) -> pd.DataFrame:
    """Bir hissenin teknik özelliklerini hesaplar ve CSV'ye kaydeder."""

    file_name = ticker.replace(".", "_")
    file_path = DATA_DIR / f"{file_name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Hisse verisi bulunamadı: {file_path}. "
            "Önce data_fetcher.py dosyasını çalıştır."
        )

    data = pd.read_csv(
        file_path,
        index_col="Date",
        parse_dates=True,
    )

    if data.empty:
        raise ValueError(f"{ticker} için veri seti boş.")

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]

    required_columns = {"Close", "Volume", "High", "Low"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{ticker} verisinde eksik sütunlar: "
            f"{sorted(missing_columns)}. "
            "ATR için High ve Low sütunları gereklidir."
        )

    # Sayısal veri kontrolü
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    data.loc[data["Close"] <= 0, "Close"] = np.nan
    data.loc[data["Volume"] < 0, "Volume"] = np.nan

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # =====================================================
    # 1. GETİRİLER
    # =====================================================

    data["Return_1d"] = close.pct_change(fill_method=None)
    data["Return_5d"] = close.pct_change(5, fill_method=None)
    data["Return_20d"] = close.pct_change(20, fill_method=None)
    data["Return_60d"] = close.pct_change(60, fill_method=None)

    # =====================================================
    # 2. SMA
    # =====================================================

    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    data["SMA_5_ratio"] = sma_5 / close
    data["SMA_20_ratio"] = sma_20 / close
    data["SMA_50_ratio"] = sma_50 / close

    # =====================================================
    # 3. EMA
    # =====================================================

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

    data["EMA_12_ratio"] = ema_12 / close
    data["EMA_26_ratio"] = ema_26 / close

    # EMA'lar arasındaki farkı fiyatla normalize et
    data["EMA_cross"] = (ema_12 - ema_26) / close

    # =====================================================
    # 4. RSI (14)
    # =====================================================

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

    data["RSI_scaled"] = rsi / 100

    # =====================================================
    # 5. MACD (12, 26, 9)
    # =====================================================

    macd_line = ema_12 - ema_26

    macd_signal = macd_line.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    macd_histogram = macd_line - macd_signal

    # Fiyat seviyesine göre normalize et
    data["MACD_ratio"] = macd_line / close
    data["MACD_signal_ratio"] = macd_signal / close
    data["MACD_histogram_ratio"] = macd_histogram / close

    # =====================================================
    # 6. BOLLINGER BANDS (20, 2)
    # =====================================================

    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)

    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)

    bb_range = (bb_upper - bb_lower).replace(0, np.nan)

    # Fiyatın bantlar arasındaki konumu
    data["BB_position"] = (close - bb_lower) / bb_range

    # Bant genişliği / orta bant
    data["BB_width"] = (
        (bb_upper - bb_lower) / bb_middle
    )

    # =====================================================
    # 7. HACİM
    # =====================================================

    data["Volume_Change"] = volume.pct_change(
        fill_method=None
    )

    volume_mean_20 = volume.rolling(20).mean()

    data["Volume_Ratio"] = volume / volume_mean_20

    # =====================================================
    # 8. OBV
    # =====================================================

    price_direction = np.sign(close.diff()).fillna(0)

    obv = (price_direction * volume).cumsum()

    # OBV seviyesini değil, 20 dönemlik değişimini kullan.
    # Mutlak OBV seviyesi geçmiş hacim birikimine bağımlıdır.
    obv_20_ago = obv.shift(20)
    volume_sum_20 = volume.rolling(20).sum()

    data["OBV_Change"] = (
        (obv - obv_20_ago) / volume_sum_20.replace(0, np.nan)
    )

    # =====================================================
    # 9. VOLATİLİTE
    # =====================================================

    data["Volatility_20d"] = (
        data["Return_1d"].rolling(20).std()
    )

    # =====================================================
    # 10. ATR (14)
    # =====================================================

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

    # Fiyat seviyesine göre normalize edilmiş ATR
    data["ATR_ratio"] = atr_14 / close

    # =====================================================
    # 11. TEMİZLİK VE KAYIT
    # =====================================================

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    if data.empty:
        raise ValueError(
            f"{ticker}: Özellik hesaplamasından sonra "
            "kullanılabilir veri kalmadı."
        )

    output_path = DATA_DIR / f"{file_name}_features.csv"

    data.index.name = "Date"
    data.to_csv(output_path)

    print(f"Kaydedildi: {output_path}")
    print(f"Satır sayısı: {len(data)}")
    print(f"Özellik sayısı: {len(FEATURE_COLUMNS)}")

    return data


# =========================================================
# BÜTÜN HİSSELERİ İŞLEME
# =========================================================

def create_features_for_all_stocks(tickers: list[str]):
    successful = []
    failed = []

    for index, ticker in enumerate(tickers, start=1):
        print("\n" + "=" * 50)
        print(f"[{index}/{len(tickers)}] {ticker}")
        print("=" * 50)

        try:
            create_features(ticker)
            successful.append(ticker)

        except Exception as error:
            print(f"Hata: {ticker} - {error}")
            failed.append(ticker)

    print("\n" + "=" * 50)
    print("ÖZELLİK HESAPLAMA RAPORU")
    print("=" * 50)
    print(f"Toplam hisse: {len(tickers)}")
    print(f"Başarılı: {len(successful)}")
    print(f"Başarısız: {len(failed)}")

    if failed:
        print("\nİşlenemeyen hisseler:")
        for ticker in failed:
            print(f"- {ticker}")

    print(f"\nÇıktı klasörü: {DATA_DIR}")

    return successful, failed


# =========================================================
# PROGRAM GİRİŞ NOKTASI
# =========================================================

if __name__ == "__main__":
    create_features_for_all_stocks(BIST100_TICKERS)
