
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RESULTS_PATH = DATA_DIR / "all_stocks_feature_comparison.csv"
SUMMARY_PATH = DATA_DIR / "all_stocks_feature_summary.csv"
STOCK_SUMMARY_PATH = DATA_DIR / "feature_comparison_by_stock.csv"

HORIZON = 20
INITIAL_TRAIN_RATIO = 0.60
N_FOLDS = 8
RANDOM_STATE = 42
MINIMUM_ROWS = 300


# =========================================================
# ÖZELLİK GRUPLARI
# =========================================================

OLD_FEATURES = [
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

TOP_FEATURES = [
    "Return_20d",
    "Return_60d",
    "Volatility_20d",
]

FEATURE_SETS = {
    "Eski 10 Özellik": OLD_FEATURES,
    "Seçili 3 Özellik": TOP_FEATURES,
}


# =========================================================
# MODEL
# =========================================================

def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =========================================================
# VERİYİ YÜKLE
# =========================================================

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = list(
        dict.fromkeys(
            OLD_FEATURES + ["Date", "Target"]
        )
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Eksik sütunlar: " + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    for col in OLD_FEATURES + ["Target"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Her iki özellik grubu aynı tarihlerde test edilir.
    df = df.dropna(
        subset=required
    )

    df = (
        df.sort_values("Date")
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    df = df[
        df["Target"].isin([0, 1])
    ].copy()

    df["Target"] = df["Target"].astype(int)

    return df.reset_index(drop=True)


# =========================================================
# TEK HİSSE İÇİN WALK-FORWARD TESTİ
# =========================================================

def evaluate_stock(path: Path):
    symbol = path.name.replace(
        "_dataset.csv", ""
    ).replace("_", ".")

    print("\n" + "=" * 70)
    print(f"HİSSE: {symbol}")
    print("=" * 70)

    df = load_data(path)

    if len(df) < MINIMUM_ROWS:
        raise ValueError(
            f"Yeterli veri yok: {len(df)} satır."
        )

    initial_train_size = int(
        len(df) * INITIAL_TRAIN_RATIO
    )

    remaining = len(df) - initial_train_size

    if remaining < N_FOLDS:
        raise ValueError(
            "Sekiz test dönemi oluşturmak için yeterli veri yok."
        )

    # Fold sınırları bütün hisseler için aynı kuralla oluşturulur.
    boundaries = np.linspace(
        initial_train_size,
        len(df),
        N_FOLDS + 1,
        dtype=int,
    )

    stock_results = []

    print(f"Veri sayısı: {len(df)}")
    print(f"İlk tarih: {df['Date'].iloc[0].date()}")
    print(f"Son tarih: {df['Date'].iloc[-1].date()}")

    for fold in range(N_FOLDS):
        test_start = int(boundaries[fold])
        test_end = int(boundaries[fold + 1])

        # Hedef 20 işlem günü sonrasını kullandığı için
        # eğitim/test sınırında 20 satır boşluk bırakılır.
        train_end = test_start - HORIZON

        if train_end <= 0 or test_end <= test_start:
            continue

        train = df.iloc[:train_end].copy()
        test = df.iloc[test_start:test_end].copy()

        if train.empty or test.empty:
            continue

        y_train = train["Target"]
        y_test = test["Target"]

        if y_train.nunique() < 2:
            print(
                f"Fold {fold + 1}: eğitimde tek sınıf var, atlandı."
            )
            continue

        print(
            f"\nFold {fold + 1}/{N_FOLDS} | "
            f"Test: {test['Date'].iloc[0].date()} - "
            f"{test['Date'].iloc[-1].date()}"
        )

        for set_name, features in FEATURE_SETS.items():
            model = create_model()

            model.fit(
                train[features],
                y_train,
            )

            predictions = model.predict(
                test[features]
            )

            metrics = {
                "Accuracy": accuracy_score(
                    y_test, predictions
                ),
                "Balanced_Accuracy": balanced_accuracy_score(
                    y_test, predictions
                ),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }

            row = {
                "Ticker": symbol,
                "Fold": fold + 1,
                "Feature_Set": set_name,
                "Feature_Count": len(features),
                "Train_Rows": len(train),
                "Test_Rows": len(test),
                "Train_End": str(
                    train["Date"].iloc[-1].date()
                ),
                "Test_Start": str(
                    test["Date"].iloc[0].date()
                ),
                "Test_End": str(
                    test["Date"].iloc[-1].date()
                ),
                **metrics,
            }

            stock_results.append(row)

            print(
                f"  {set_name}: "
                f"Accuracy={metrics['Accuracy']:.3f}, "
                f"Balanced Accuracy="
                f"{metrics['Balanced_Accuracy']:.3f}, "
                f"F1={metrics['F1']:.3f}"
            )

    if not stock_results:
        raise RuntimeError(
            "Bu hisse için hiçbir fold sonuç üretmedi."
        )

    return stock_results


# =========================================================
# TÜM HİSSELER
# =========================================================

def run_all_stocks():
    paths = sorted(
        DATA_DIR.glob("*_dataset.csv")
    )

    if not paths:
        raise FileNotFoundError(
            f"{DATA_DIR} içinde *_dataset.csv dosyası bulunamadı."
        )

    print("=" * 70)
    print("FINANCEAI - TÜM HİSSELERDE ÖZELLİK KARŞILAŞTIRMASI")
    print("=" * 70)
    print(f"Bulunan veri seti: {len(paths)}")
    print(f"Test dönemi sayısı: {N_FOLDS}")
    print(f"Hedef ufku: {HORIZON} işlem günü")

    all_results = []
    failed = []

    for index, path in enumerate(paths, start=1):
        print(
            f"\nİlerleme: {index}/{len(paths)}"
        )

        try:
            all_results.extend(
                evaluate_stock(path)
            )
        except Exception as error:
            print(
                f"HATA: {path.name} - {error}"
            )
            failed.append({
                "File": path.name,
                "Error": str(error),
            })

    if not all_results:
        raise RuntimeError(
            "Hiçbir veri setinde karşılaştırma tamamlanamadı."
        )

    results = pd.DataFrame(all_results)

    # Her hisse ve özellik grubu için ortalama
    # metrikleri hesapla.
    by_stock = (
        results
        .groupby(
            ["Ticker", "Feature_Set"],
            as_index=False,
        )
        .agg(
            Fold_Count=("Fold", "nunique"),
            Accuracy=("Accuracy", "mean"),
            Balanced_Accuracy=(
                "Balanced_Accuracy", "mean"
            ),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
        )
    )

    # Hisselerin ortalama performansları:
    # her hisseye eşit ağırlık verilir.
    summary = (
        by_stock
        .groupby("Feature_Set", as_index=False)
        .agg(
            Stock_Count=("Ticker", "nunique"),
            Mean_Accuracy=("Accuracy", "mean"),
            Std_Accuracy=("Accuracy", "std"),
            Mean_Balanced_Accuracy=(
                "Balanced_Accuracy", "mean"
            ),
            Std_Balanced_Accuracy=(
                "Balanced_Accuracy", "std"
            ),
            Mean_Precision=("Precision", "mean"),
            Mean_Recall=("Recall", "mean"),
            Mean_F1=("F1", "mean"),
            Std_F1=("F1", "std"),
        )
    )

    # Aynı hissede 3 özellik grubunun eski gruba göre
    # Balanced Accuracy farkı.
    pivot = by_stock.pivot(
        index="Ticker",
        columns="Feature_Set",
        values="Balanced_Accuracy",
    )

    old_name = "Eski 10 Özellik"
    top_name = "Seçili 3 Özellik"

    if old_name in pivot.columns and top_name in pivot.columns:
        pivot["3_eksi_10_Balanced_Accuracy"] = (
            pivot[top_name] - pivot[old_name]
        )

        differences = pivot[
            "3_eksi_10_Balanced_Accuracy"
        ].dropna()

        print("\n" + "=" * 70)
        print("HİSSE BAZINDA KARŞILAŞTIRMA")
        print("=" * 70)
        print(
            f"3 özellikli grubun daha iyi olduğu hisse: "
            f"{int((differences > 0).sum())}"
        )
        print(
            f"10 özellikli grubun daha iyi olduğu hisse: "
            f"{int((differences < 0).sum())}"
        )
        print(
            f"Eşit sonuç veren hisse: "
            f"{int((differences == 0).sum())}"
        )

        pivot.reset_index().to_csv(
            DATA_DIR / "feature_group_differences.csv",
            index=False,
            encoding="utf-8-sig",
        )

    results.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    by_stock.to_csv(
        STOCK_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("GENEL SONUÇLAR")
    print("=" * 70)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    if failed:
        failed_path = DATA_DIR / "feature_comparison_failures.csv"
        pd.DataFrame(failed).to_csv(
            failed_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\nBaşarısız veri setleri: {len(failed)}")
        print(f"Hata raporu: {failed_path}")

    print("\nKaydedilen dosyalar:")
    print(RESULTS_PATH)
    print(SUMMARY_PATH)
    print(STOCK_SUMMARY_PATH)

    print(
        "\nNOT: Sonuçlar tarihsel test performansıdır. "
        "Yatırım kârlılığını garanti etmez. "
        "Hisseler arası ortalamalar, her hisseye eşit "
        "ağırlık verilerek hesaplanmıştır."
    )


if __name__ == "__main__":
    run_all_stocks()
