
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HORIZON = 20


def create_target(ticker: str):
    file_name = ticker.replace(".", "_")
    input_path = DATA_DIR / f"{file_name}_features.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Özellik dosyası bulunamadı: {input_path}\n"
            "Önce data_fetcher.py ve "
            "feature_engineering.py dosyalarını çalıştır."
        )

    data = pd.read_csv(
        input_path,
        index_col="Date",
        parse_dates=True,
    )

    if "Close" not in data.columns:
        raise ValueError(
            "Veri setinde Close sütunu bulunamadı."
        )

    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError(
            "Veri setinde yinelenen tarihler var."
        )

    # Geçerli kapanış fiyatlarını sayısal hâle getir
    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    if (data["Close"].dropna() <= 0).any():
        raise ValueError(
            "Kapanış fiyatlarında sıfır veya negatif değer var."
        )

    # Hedef oluşturulmadan önce eski hedefleri temizle
    data = data.drop(
        columns=[
            "Future_Close",
            "Future_Return_20d",
            "Target",
        ],
        errors="ignore",
    )

    # 20 işlem günü sonraki kapanış
    data["Future_Close"] = data["Close"].shift(-HORIZON)

    # 20 işlem günlük getiri
    data["Future_Return_20d"] = (
        data["Future_Close"] / data["Close"] - 1
    )

    # 1: pozitif getiri
    # 0: sıfır veya negatif getiri
    data["Target"] = (
        data["Future_Return_20d"] > 0
    ).astype("int64")

    # Gelecek fiyatı bilinmeyen son 20 işlem gününü çıkar
    data = data.dropna(
        subset=["Close", "Future_Close", "Future_Return_20d"]
    ).copy()

    if data.empty:
        raise ValueError(
            "Hedef oluşturmak için yeterli veri yok."
        )

    output_path = DATA_DIR / f"{file_name}_dataset.csv"
    data.to_csv(output_path, index_label="Date")

    print("=" * 50)
    print("FINANCE AI - HEDEF OLUŞTURMA")
    print("=" * 50)

    print(f"Hisse: {ticker}")
    print(f"Tahmin ufku: {HORIZON} işlem günü")
    print(f"Örnek sayısı: {len(data)}")
    print(f"Dosya: {output_path}")

    print("\nHedef dağılımı:")
    print(
        data["Target"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={
            0: "Yükseliş yok",
            1: "Yükseliş",
        })
    )

    print("\n20 günlük getiri istatistikleri:")
    print(
        data["Future_Return_20d"]
        .describe()
        .round(4)
    )

    print("\nSon kayıtlar:")
    print(
        data[
            [
                "Close",
                "Future_Close",
                "Future_Return_20d",
                "Target",
            ]
        ].tail()
    )


if __name__ == "__main__":
    create_target("TUPRS.IS")
