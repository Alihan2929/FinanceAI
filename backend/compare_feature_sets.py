
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

DATA_PATH = BASE_DIR / "data" / "TUPRS_IS_dataset.csv"

RESULTS_PATH = BASE_DIR / "data" / "feature_set_results.csv"
SUMMARY_PATH = BASE_DIR / "data" / "feature_set_summary.csv"

HORIZON = 20
INITIAL_TRAIN_RATIO = 0.60
TEST_WINDOW_SIZE = 60

# A: Mevcut 10 özellik
ALL_FEATURES = [
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

# B: Önceki analizde öne çıkan 3 özellik
TOP_FEATURES = [
    "Return_20d",
    "Return_60d",
    "Volatility_20d",
]


# =========================================================
# VERİYİ YÜKLE
# =========================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = list(set(
        ALL_FEATURES + ["Date", "Target"]
    ))

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Eksik sütunlar: " + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    for column in ALL_FEATURES + ["Target"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    df = df[df["Target"].isin([0, 1])].copy()
    df["Target"] = df["Target"].astype(int)

    return df.reset_index(drop=True)


# =========================================================
# MODEL
# =========================================================

def create_model():

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# =========================================================
# METRİKLER
# =========================================================

def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


# =========================================================
# WALK-FORWARD KARŞILAŞTIRMASI
# =========================================================

def run_comparison():

    df = load_data()

    if len(df) < 300:
        raise ValueError(
            "Karşılaştırma için yeterli veri bulunamadı."
        )

    first_test_start = int(
        len(df) * INITIAL_TRAIN_RATIO
    )

    all_results = []
    fold_number = 1
    test_start = first_test_start

    feature_sets = {
        "10 Özellik": ALL_FEATURES,
        "3 Özellik": TOP_FEATURES,
    }

    print("\n" + "=" * 65)
    print("FINANCEAI - ÖZELLİK GRUBU KARŞILAŞTIRMASI")
    print("=" * 65)

    print(f"Toplam örnek: {len(df)}")
    print(f"İlk test başlangıcı: {first_test_start}")
    print(f"Test pencere boyutu: {TEST_WINDOW_SIZE}")
    print(f"Hedef ufku: {HORIZON} işlem günü")

    while test_start < len(df):

        test_end = min(
            test_start + TEST_WINDOW_SIZE,
            len(df),
        )

        # 20 günlük hedef nedeniyle eğitim/test sınırında
        # hedef ufku kadar boşluk bırakılır.
        train_end = test_start - HORIZON

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if train_df.empty or test_df.empty:
            break

        y_train = train_df["Target"].astype(int)
        y_test = test_df["Target"].astype(int)

        print("\n" + "-" * 65)
        print(f"FOLD {fold_number}")
        print(
            f"Eğitim: {train_df['Date'].min().date()} - "
            f"{train_df['Date'].max().date()}"
        )
        print(
            f"Test: {test_df['Date'].min().date()} - "
            f"{test_df['Date'].max().date()}"
        )

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print("Fold atlandı: eğitim veya testte tek sınıf var.")
            test_start = test_end
            fold_number += 1
            continue

        for set_name, features in feature_sets.items():

            model = create_model()

            model.fit(
                train_df[features],
                y_train,
            )

            predictions = model.predict(
                test_df[features],
            )

            metrics = calculate_metrics(
                y_test,
                predictions,
            )

            row = {
                "Fold": fold_number,
                "Feature_Set": set_name,
                "Feature_Count": len(features),
                "Train_Start": train_df["Date"].min(),
                "Train_End": train_df["Date"].max(),
                "Test_Start": test_df["Date"].min(),
                "Test_End": test_df["Date"].max(),
                "Train_Samples": len(train_df),
                "Test_Samples": len(test_df),
                **metrics,
            }

            all_results.append(row)

            print(f"\n{set_name}")
            print(
                f"Accuracy: {metrics['Accuracy']:.3f}"
            )
            print(
                f"Balanced accuracy: "
                f"{metrics['Balanced_Accuracy']:.3f}"
            )
            print(
                f"F1: {metrics['F1']:.3f}"
            )

        test_start = test_end
        fold_number += 1

    if not all_results:
        raise RuntimeError(
            "Karşılaştırma sonucu üretilemedi."
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    summary_df = (
        results_df
        .groupby("Feature_Set")
        .agg(
            Fold_Count=("Fold", "nunique"),
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
        .reset_index()
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("GENEL ÖZELLİK GRUBU SONUÇLARI")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print(f"\nDönem sonuçları: {RESULTS_PATH}")
    print(f"Özet sonuçlar: {SUMMARY_PATH}")

    print(
        "\nNot: Bu analiz özellik gruplarını karşılaştırır; "
        "yatırım getirisi hesaplamaz. Üç özellikli grup, "
        "önceki tek dönemlik önem analizine göre seçilmiştir. "
        "Bu nedenle sonuçlar bağımsız nihai test olarak "
        "yorumlanmamalıdır."
    )


if __name__ == "__main__":
    run_comparison()
