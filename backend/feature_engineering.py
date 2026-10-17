import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def create_features(ticker: str):
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
        parse_dates=True
    )

    # Tarih ve veri kontrolleri
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]

    required_columns = {"Close", "Volume"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError(f"{ticker} için veri seti boş.")

    # Sayısal veri kontrolü
    for column in ["Close", "Volume"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data.loc[data["Close"] <= 0, "Close"] = float("nan")
    data.loc[data["Volume"] < 0, "Volume"] = float("nan")

    # 1. Farklı dönemlerdeki getiriler
    data["Return_1d"] = data["Close"].pct_change()
    data["Return_5d"] = data["Close"].pct_change(5)
    data["Return_20d"] = data["Close"].pct_change(20)
    data["Return_60d"] = data["Close"].pct_change(60)

    # 2. Hareketli ortalamalar
    data["SMA_5"] = data["Close"].rolling(5).mean()
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()

    # Fiyat seviyeleri yerine göreli oranlar
    data["SMA_5_ratio"] = data["SMA_5"] / data["Close"]
    data["SMA_20_ratio"] = data["SMA_20"] / data["Close"]
    data["SMA_50_ratio"] = data["SMA_50"] / data["Close"]

    # 3. RSI (14 dönem)
    change = data["Close"].diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()

    relative_strength = avg_gain.div(avg_loss)
    data["RSI_14"] = (
        100 - (100 / (1 + relative_strength))
    )

    # Özel RSI durumları
    data.loc[
        (avg_loss == 0) & (avg_gain > 0), "RSI_14"
    ] = 100

    data.loc[
        (avg_loss == 0) & (avg_gain == 0), "RSI_14"
    ] = 50

    data["RSI_scaled"] = data["RSI_14"] / 100

    # 4. İşlem hacmi değişimi
    data["Volume_Change"] = data["Volume"].pct_change()

    # 5. Fiyat oynaklığı (20 günlük getiri standart sapması)
    data["Volatility_20d"] = (
        data["Return_1d"].rolling(20).std()
    )

    # Sonsuz değerleri ve eksik hesaplamaları temizle
    data = data.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    data = data.dropna().copy()

    if data.empty:
        raise ValueError(
            "Özellik hesaplamasından sonra veri kalmadı."
        )

    # CSV dosyasına kaydet
    output_path = DATA_DIR / f"{file_name}_features.csv"
    data.to_csv(output_path)

    print(f"Özellikler kaydedildi: {output_path}")
    print(f"Toplam satır: {len(data)}")
    print("\nSon hesaplanan özellikler:")

    print(data[[
        "Close",
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
    ]].tail())


if __name__ == "__main__":
    create_features("TUPRS.IS")
