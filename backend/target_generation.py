
from pathlib import Path

import pandas as pd

from data_fetcher import BIST100_TICKERS


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HORIZON = 20


# =========================================================
# TEK HİSSE İÇİN HEDEF OLUŞTURMA
# =========================================================

def create_target(ticker: str) -> pd.DataFrame:
    """Özellik dosyasından 20 işlem günlük hedef değişkenini üretir."""

    file_name = ticker.replace(".", "_")
    input_path = DATA_DIR / f"{file_name}_features.csv"
    output_path = DATA_DIR / f"{file_name}_dataset.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Özellik dosyası bulunamadı: {input_path}\n"
            "Önce data_fetcher.py ve feature_engineering.py "
            "dosyalarını çalıştır."
        )

    data = pd.read_csv(input_path, parse_dates=["Date"])

    if data.empty:
        raise ValueError(f"{ticker} veri seti boş.")

    if "Close" not in data.columns:
        raise ValueError(f"{ticker}: Close sütunu bulunamadı.")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    data = (
        data.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(f"{ticker}: Geçerli tarih/fiyat verisi yok.")

    if (data["Close"] <= 0).any():
        raise ValueError(
            f"{ticker}: Sıfır veya negatif kapanış fiyatı bulundu."
        )

    # Hedef sütunları varsa yeniden hesaplanır.
    data = data.drop(
        columns=[
            "Future_Close",
            "Future_Return_20d",
            "Target",
        ],
        errors="ignore",
    )

    # Gelecekteki kapanış fiyatı ve getiri.
    data["Future_Close"] = data["Close"].shift(-HORIZON)

    data["Future_Return_20d"] = (
        data["Future_Close"] / data["Close"] - 1
    )

    # 1 = pozitif getiri, 0 = sıfır veya negatif getiri.
    data["Target"] = (
        data["Future_Return_20d"] > 0
    ).astype("int64")

    # Gelecekteki fiyatı bilinmeyen son 20 satırı eğitim
    # veri setinden çıkar.
    dataset = data.dropna(
        subset=["Future_Close", "Future_Return_20d"]
    ).copy()

    if dataset.empty:
        raise ValueError(
            f"{ticker}: Hedef üretmek için yeterli veri yok."
        )

    # Tarih bilgisi hedef üretiminde korunur.
    dataset["Date"] = pd.to_datetime(dataset["Date"])

    dataset.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    latest_feature_date = data["Date"].max().date()
    latest_labeled_date = dataset["Date"].max().date()

    print(f"\nHisse: {ticker}")
    print(f"Tahmin ufku: {HORIZON} işlem günü")
    print(f"Özellik verisinin son tarihi: {latest_feature_date}")
    print(f"Etiketli verinin son tarihi: {latest_labeled_date}")
    print(f"Örnek sayısı: {len(dataset)}")
    print(f"Çıktı: {output_path}")

    print("\nHedef dağılımı:")
    print(
        dataset["Target"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(
            index={
                0: "Yükseliş yok",
                1: "Yükseliş",
            }
        )
    )

    print("\n20 günlük getiri istatistikleri:")
    print(dataset["Future_Return_20d"].describe().round(4))

    return dataset


# =========================================================
# BÜTÜN HİSSELERİ İŞLEME
# =========================================================

def create_targets_for_all_stocks(tickers: list[str]):
    successful = []
    failed = []

    for index, ticker in enumerate(tickers, start=1):
        print("\n" + "=" * 50)
        print(f"[{index}/{len(tickers)}] {ticker}")
        print("=" * 50)

        try:
            create_target(ticker)
            successful.append(ticker)

        except Exception as error:
            print(f"Hata: {ticker} - {error}")
            failed.append(ticker)

    print("\n" + "=" * 50)
    print("HEDEF OLUŞTURMA RAPORU")
    print("=" * 50)
    print(f"Toplam hisse: {len(tickers)}")
    print(f"Başarılı: {len(successful)}")
    print(f"Başarısız: {len(failed)}")

    if failed:
        print("\nİşlenemeyen hisseler:")
        for ticker in failed:
            print(f"- {ticker}")

    print(f"\nVeri klasörü: {DATA_DIR}")

    return successful, failed


# =========================================================
# PROGRAM GİRİŞ NOKTASI
# =========================================================

if __name__ == "__main__":
    create_targets_for_all_stocks(BIST100_TICKERS)
