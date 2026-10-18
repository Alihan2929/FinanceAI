
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

RESULTS_PATH = BASE_DIR / "data" / "walk_forward_feature_comparison.csv"
SUMMARY_PATH = BASE_DIR / "data" / "walk_forward_feature_summary.csv"

HORIZON = 20
INITIAL_TRAIN_RATIO = 0.60
N_FOLDS = 8
RANDOM_STATE = 42


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
# VERİYİ YÜKLE
# =========================================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

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
            f"Veri setinde eksik sütunlar: {missing}"
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

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    df = df[df["Target"].isin([0, 1])].copy()
    df["Target"] = df["Target"].astype(int)

    if len(df) < 300:
        raise ValueError(
            f"Yeterli veri yok: {len(df)} satır."
        )

    return df.reset_index(drop=True)


# =========================================================
# SABİT MODEL
# İki özellik grubu için aynı parametreler kullanılır.
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
# METRİKLER
# =========================================================

def calculate_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


# =========================================================
# WALK-FORWARD KARŞILAŞTIRMASI
# =========================================================

def run_comparison():
    df = load_data()

    n = len(df)
    initial_train_size = int(
        n * INITIAL_TRAIN_RATIO
    )

    remaining = n - initial_train_size
    test_size = remaining // N_FOLDS

    if test_size < 1:
        raise ValueError(
            "Test dönemleri oluşturmak için yeterli veri yok."
        )

    all_results = []

    print("=" * 70)
    print("FINANCEAI - WALK-FORWARD ÖZELLİK KARŞILAŞTIRMASI")
    print("=" * 70)
    print(f"Veri: {DATA_PATH.name}")
    print(f"Toplam örnek: {n}")
    print(f"Fold sayısı: {N_FOLDS}")
    print(f"Hedef ufku: {HORIZON} işlem günü")
    print("Model parametreleri iki grupta da aynıdır.")

    for fold in range(N_FOLDS):
        test_start = (
            initial_train_size + fold * test_size
        )

        if fold == N_FOLDS - 1:
            test_end = n
        else:
            test_end = min(
                test_start + test_size,
                n,
            )

        # İleriye dönük hedef pencerelerinin
        # sınırda çakışmasını azaltmak için boşluk.
        train_end = test_start - HORIZON

        if train_end <= 0 or test_end <= test_start:
            continue

        train = df.iloc[:train_end].copy()
        test = df.iloc[test_start:test_end].copy()

        if train.empty or test.empty:
            continue

        y_train = train["Target"]
        y_test = test["Target"]

        print("\n" + "-" * 70)
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print(
            f"Eğitim: {train['Date'].iloc[0].date()} - "
            f"{train['Date'].iloc[-1].date()}"
        )
        print(
            f"Test: {test['Date'].iloc[0].date()} - "
            f"{test['Date'].iloc[-1].date()}"
        )

        if y_train.nunique() < 2:
            print("Fold atlandı: eğitimde tek sınıf var.")
            continue

        for set_name, features in FEATURE_SETS.items():
            model = create_model()

            model.fit(
                train[features],
                y_train,
            )

            predictions = model.predict(
                test[features]
            )

            metrics = calculate_metrics(
                y_test,
                predictions,
            )

            result = {
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

            all_results.append(result)

            print(f"\n{set_name}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

    if not all_results:
        raise RuntimeError(
            "Hiçbir fold sonuç üretmedi."
        )

    results_df = pd.DataFrame(all_results)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
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
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("GENEL KARŞILAŞTIRMA")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best = summary_df.loc[
        summary_df["Mean_Balanced_Accuracy"].idxmax()
    ]

    print("\nEn yüksek ortalama Balanced Accuracy:")
    print(f"Grup: {best['Feature_Set']}")
    print(
        f"Değer: {best['Mean_Balanced_Accuracy']:.4f}"
    )

    print("\nKaydedilen dosyalar:")
    print(RESULTS_PATH)
    print(SUMMARY_PATH)

    print(
        "\nNot: Bu karşılaştırma yalnızca TUPRS.IS verisi "
        "içindir. Aynı model parametreleri kullanılır; "
        "sonuçlar yatırım kârlılığını garanti etmez."
    )


if __name__ == "__main__":
    run_comparison()
