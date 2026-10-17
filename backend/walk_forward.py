
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

# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "TUPRS_IS_dataset.csv"
RESULTS_PATH = BASE_DIR / "data" / "walk_forward_results.csv"
SUMMARY_PATH = BASE_DIR / "data" / "walk_forward_summary.csv"
TUNING_PATH = BASE_DIR / "data" / "rf_tuning_results.csv"

HORIZON = 20
INITIAL_TRAIN_RATIO = 0.60
N_FOLDS = 8
RANDOM_STATE = 42

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

# Önceden belirlenen, sınırlı bir ayar grubu.
# Test fold'u bu ayarları seçmek için kullanılmaz.
PARAMETER_GRID = [
    {
        "n_estimators": 200,
        "max_depth": 3,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_leaf": 5,
    },
    {
        "n_estimators": 400,
        "max_depth": 7,
        "min_samples_leaf": 5,
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 10,
    },
]


# --------------------------------------------------
# VERİYİ OKU
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = ["Date", "Target"] + FEATURES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Veri setinde eksik sütunlar: "
            + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in FEATURES + ["Target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    if not set(df["Target"].unique()).issubset({0, 1}):
        raise ValueError("Target yalnızca 0 ve 1 değerlerini içermeli.")

    df["Target"] = df["Target"].astype(int)

    if len(df) < 250:
        raise ValueError(
            f"Yeterli veri yok: {len(df)} satır. "
            "Daha uzun bir geçmiş veri seti kullan."
        )

    return df


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def create_model(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# --------------------------------------------------
# AYAR SEÇİMİ
# Yalnızca dış test döneminden önceki veriler kullanılır.
# --------------------------------------------------

def select_parameters(train_df, fold_number):
    n = len(train_df)

    # İç doğrulama için eğitim verisinin son %20'sini ayır.
    validation_start = int(n * 0.80)

    # 20 işlem günlük hedef aralığı nedeniyle araya boşluk koy.
    inner_train_end = validation_start - HORIZON

    if inner_train_end < 100 or validation_start >= n:
        raise ValueError(
            f"Fold {fold_number}: iç doğrulama için "
            "yeterli eğitim verisi yok."
        )

    inner_train = train_df.iloc[:inner_train_end]
    validation = train_df.iloc[validation_start:]

    X_inner = inner_train[FEATURES]
    y_inner = inner_train["Target"]

    X_validation = validation[FEATURES]
    y_validation = validation["Target"]

    if y_inner.nunique() < 2:
        raise ValueError(
            f"Fold {fold_number}: iç eğitimde iki sınıf da yok."
        )

    tuning_results = []

    for params in PARAMETER_GRID:
        model = create_model(params)
        model.fit(X_inner, y_inner)

        predictions = model.predict(X_validation)

        balanced = balanced_accuracy_score(
            y_validation,
            predictions,
        )

        tuning_results.append({
            **params,
            "validation_balanced_accuracy": balanced,
            "validation_accuracy": accuracy_score(
                y_validation,
                predictions,
            ),
        })

    tuning_df = pd.DataFrame(tuning_results)

    # Öncelik dengeli doğrulukta.
    # Eşitlikte daha küçük ağaçlar yerine önceden belirlenmiş
    # listedeki ilk en iyi aday seçilir.
    best_row = tuning_df.sort_values(
        "validation_balanced_accuracy",
        ascending=False,
    ).iloc[0]

    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": (
            None
            if pd.isna(best_row["max_depth"])
            else int(best_row["max_depth"])
        ),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
    }

    tuning_df["fold"] = fold_number

    return best_params, tuning_df


# --------------------------------------------------
# WALK-FORWARD DEĞERLENDİRMESİ
# --------------------------------------------------

def run_walk_forward():
    df = load_data()

    n = len(df)
    initial_train_size = int(n * INITIAL_TRAIN_RATIO)
    remaining = n - initial_train_size
    test_size = remaining // N_FOLDS

    if test_size < 1:
        raise ValueError(
            "Test fold'ları oluşturmak için yeterli veri yok."
        )

    fold_results = []
    tuning_results = []

    print(f"Temiz veri satırı: {n}")
    print(f"Özellik sayısı: {len(FEATURES)}")
    print(f"Fold sayısı: {N_FOLDS}")
    print(f"Hedef ufku: {HORIZON} işlem günü")
    print()

    for fold in range(N_FOLDS):
        test_start = initial_train_size + fold * test_size

        if fold == N_FOLDS - 1:
            test_end = n
        else:
            test_end = min(test_start + test_size, n)

        if test_start >= n or test_end <= test_start:
            continue

        # Test hedeflerinin geleceğe uzanmasını dikkate alan boşluk.
        train_end = test_start - HORIZON

        if train_end <= 0:
            continue

        train = df.iloc[:train_end].copy()
        test = df.iloc[test_start:test_end].copy()

        if train["Target"].nunique() < 2:
            print(f"Fold {fold + 1} atlandı: eğitimde tek sınıf var.")
            continue

        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(
            f"  Eğitim: {train['Date'].iloc[0].date()} "
            f"- {train['Date'].iloc[-1].date()}"
        )
        print(
            f"  Test:   {test['Date'].iloc[0].date()} "
            f"- {test['Date'].iloc[-1].date()}"
        )

        # Ayarlar yalnızca dış eğitim döneminden seçilir.
        best_params, fold_tuning = select_parameters(
            train,
            fold + 1,
        )
        tuning_results.append(fold_tuning)

        print(f"  Seçilen ayarlar: {best_params}")

        model = create_model(best_params)
        model.fit(train[FEATURES], train["Target"])

        predictions = model.predict(test[FEATURES])

        result = {
            "Model": "Random Forest Tuned",
            "Feature_Set": "10_features",
            "Fold": fold + 1,
            "Train_Rows": len(train),
            "Test_Rows": len(test),
            "Train_End": str(train["Date"].iloc[-1].date()),
            "Test_Start": str(test["Date"].iloc[0].date()),
            "Test_End": str(test["Date"].iloc[-1].date()),
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "min_samples_leaf": best_params["min_samples_leaf"],
            "Accuracy": accuracy_score(
                test["Target"], predictions
            ),
            "Balanced_Accuracy": balanced_accuracy_score(
                test["Target"], predictions
            ),
            "Precision": precision_score(
                test["Target"],
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                test["Target"],
                predictions,
                zero_division=0,
            ),
            "F1": f1_score(
                test["Target"],
                predictions,
                zero_division=0,
            ),
        }

        fold_results.append(result)

    if not fold_results:
        raise ValueError(
            "Hiçbir test fold'u sonuç üretmedi."
        )

    results = pd.DataFrame(fold_results)
    results.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metrics = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    summary = results[metrics].agg(["mean", "std"]).T
    summary.columns = ["Ortalama", "Standart_Sapma"]
    summary.index.name = "Metrik"
    summary = summary.reset_index()

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if tuning_results:
        pd.concat(
            tuning_results,
            ignore_index=True,
        ).to_csv(
            TUNING_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    print("\nWALK-FORWARD SONUÇLARI")
    print("=" * 70)
    print(
        results[
            [
                "Fold",
                "Accuracy",
                "Balanced_Accuracy",
                "Precision",
                "Recall",
                "F1",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
            ]
        ].to_string(index=False)
    )

    print("\nÖZET")
    print(summary.to_string(index=False))

    print("\nDosyalar kaydedildi:")
    print(RESULTS_PATH)
    print(SUMMARY_PATH)
    print(TUNING_PATH)


if __name__ == "__main__":
    run_walk_forward()
