
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================================================
# FINANCEAI - KRİPTO WALK-FORWARD DEĞERLENDİRMESİ
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "crypto_processed"
RESULTS_DIR = DATA_DIR / "walk_forward"

RESULTS_PATH = RESULTS_DIR / "crypto_walk_forward_results.csv"
SUMMARY_PATH = RESULTS_DIR / "crypto_walk_forward_summary.csv"
PREDICTIONS_PATH = RESULTS_DIR / "crypto_walk_forward_predictions.csv"
TUNING_PATH = RESULTS_DIR / "crypto_walk_forward_tuning.csv"

HORIZON = 20
INITIAL_TRAIN_RATIO = 0.60
N_FOLDS = 8
RANDOM_STATE = 42
MINIMUM_SAMPLES = 250

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

# Sınırlı parametre grubu.
# Her dış test dönemi için seçim yalnızca önceki verilerle yapılır.
PARAMETER_GRID = [
    {
        "n_estimators": 150,
        "max_depth": 4,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 15,
    },
]


# =========================================================
# VERİ OKUMA
# =========================================================

def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["Date"])

    required = FEATURES + [
        "Date",
        "Target",
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")

    for column in FEATURES + [
        "Target",
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    df = df[
        df["Target"].isin([0, 1])
        & (df["Close"] > 0)
        & (df["Future_Close"] > 0)
    ].copy()

    df["Target"] = df["Target"].astype(int)

    if len(df) < MINIMUM_SAMPLES:
        raise ValueError(
            f"Yetersiz veri: {len(df)} satır; "
            f"minimum {MINIMUM_SAMPLES} gerekli."
        )

    if df["Target"].nunique() < 2:
        raise ValueError("Hedef sütununda iki sınıf da bulunmalı.")

    return df.reset_index(drop=True)


# =========================================================
# MODELLER
# =========================================================

def create_random_forest(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_logistic_regression():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


# =========================================================
# METRİKLER
# =========================================================

def calculate_metrics(y_true, predictions, probabilities):
    result = {
        "Accuracy": accuracy_score(y_true, predictions),
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

    if len(np.unique(y_true)) == 2:
        result["ROC_AUC"] = roc_auc_score(
            y_true, probabilities
        )
    else:
        result["ROC_AUC"] = np.nan

    return result


# =========================================================
# İÇ DOĞRULAMA İLE PARAMETRE SEÇİMİ
# =========================================================

def select_parameters(train_df, symbol, fold_number):
    n = len(train_df)

    validation_start = int(n * 0.80)

    # İç eğitim hedefleriyle doğrulama hedeflerinin örtüşmesini azalt.
    inner_train_end = validation_start - HORIZON

    if inner_train_end < 100 or validation_start >= n:
        raise ValueError(
            f"{symbol}, fold {fold_number}: "
            "iç doğrulama için yeterli veri yok."
        )

    inner_train = train_df.iloc[:inner_train_end]
    validation = train_df.iloc[validation_start:]

    X_inner = inner_train[FEATURES]
    y_inner = inner_train["Target"]
    X_validation = validation[FEATURES]
    y_validation = validation["Target"]

    if y_inner.nunique() < 2:
        raise ValueError("İç eğitimde iki sınıf da bulunmalı.")

    tuning_results = []

    for params in PARAMETER_GRID:
        model = create_random_forest(params)
        model.fit(X_inner, y_inner)

        predictions = model.predict(X_validation)

        tuning_results.append({
            **params,
            "Validation_Accuracy": accuracy_score(
                y_validation, predictions
            ),
            "Validation_Balanced_Accuracy": balanced_accuracy_score(
                y_validation, predictions
            ),
        })

    tuning_df = pd.DataFrame(tuning_results)

    best = tuning_df.sort_values(
        ["Validation_Balanced_Accuracy", "Validation_Accuracy"],
        ascending=False,
    ).iloc[0]

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }

    tuning_df["Symbol"] = symbol
    tuning_df["Fold"] = fold_number

    return best_params, tuning_df


# =========================================================
# TEK KRİPTO İÇİN WALK-FORWARD
# =========================================================

def run_one_crypto(path):
    symbol = path.stem.replace("_dataset", "")
    df = load_dataset(path)

    n = len(df)
    initial_train_size = int(n * INITIAL_TRAIN_RATIO)
    remaining = n - initial_train_size
    test_size = remaining // N_FOLDS

    if test_size <= HORIZON:
        raise ValueError(
            f"{symbol}: Test dönemleri çok kısa."
        )

    print("\n" + "=" * 65)
    print(f"KRİPTO: {symbol}")
    print(f"Temiz veri satırı: {n}")
    print("=" * 65)

    fold_results = []
    prediction_results = []
    tuning_results = []

    for fold in range(N_FOLDS):
        test_start = initial_train_size + fold * test_size

        if fold == N_FOLDS - 1:
            test_end = n
        else:
            test_end = min(test_start + test_size, n)

        if test_start >= n or test_end <= test_start:
            continue

        # Test hedeflerinin geleceğe uzanması nedeniyle boşluk.
        train_end = test_start - HORIZON

        if train_end <= 0:
            continue

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if train_df["Target"].nunique() < 2:
            warnings.warn(
                f"{symbol}, fold {fold + 1}: "
                "eğitimde iki sınıf yok; atlandı."
            )
            continue

        if test_df["Target"].nunique() < 2:
            warnings.warn(
                f"{symbol}, fold {fold + 1}: "
                "testte tek sınıf var; ROC-AUC hesaplanamayabilir."
            )

        print(
            f"\nFold {fold + 1}/{N_FOLDS} | "
            f"Eğitim: {len(train_df)} | Test: {len(test_df)}"
        )
        print(
            f"Test tarihleri: "
            f"{test_df['Date'].iloc[0].date()} - "
            f"{test_df['Date'].iloc[-1].date()}"
        )

        best_params, tuning_df = select_parameters(
            train_df,
            symbol,
            fold + 1,
        )
        tuning_results.append(tuning_df)

        print(f"Seçilen Random Forest ayarları: {best_params}")

        models = {
            "random_forest_tuned": create_random_forest(best_params),
            "logistic_regression": create_logistic_regression(),
        }

        for model_name, model in models.items():
            model.fit(train_df[FEATURES], train_df["Target"])

            predictions = model.predict(test_df[FEATURES])
            probability_matrix = model.predict_proba(test_df[FEATURES])

            positive_index = list(model.classes_).index(1)
            probabilities = probability_matrix[:, positive_index]

            metrics = calculate_metrics(
                test_df["Target"].values,
                predictions,
                probabilities,
            )

            print(
                f"{model_name} | "
                f"Balanced Accuracy: {metrics['Balanced_Accuracy']:.4f} | "
                f"F1: {metrics['F1']:.4f} | "
                f"ROC-AUC: {metrics['ROC_AUC']:.4f}"
                if pd.notna(metrics["ROC_AUC"])
                else (
                    f"{model_name} | "
                    f"Balanced Accuracy: "
                    f"{metrics['Balanced_Accuracy']:.4f} | "
                    f"F1: {metrics['F1']:.4f} | ROC-AUC: hesaplanamadı"
                )
            )

            fold_results.append({
                "Symbol": symbol,
                "Model": model_name,
                "Fold": fold + 1,
                "Train_Rows": len(train_df),
                "Test_Rows": len(test_df),
                "Train_End": str(train_df["Date"].iloc[-1].date()),
                "Test_Start": str(test_df["Date"].iloc[0].date()),
                "Test_End": str(test_df["Date"].iloc[-1].date()),
                **metrics,
                "n_estimators": (
                    best_params["n_estimators"]
                    if model_name == "random_forest_tuned"
                    else np.nan
                ),
                "max_depth": (
                    best_params["max_depth"]
                    if model_name == "random_forest_tuned"
                    else np.nan
                ),
                "min_samples_leaf": (
                    best_params["min_samples_leaf"]
                    if model_name == "random_forest_tuned"
                    else np.nan
                ),
            })

            prediction_results.append(pd.DataFrame({
                "Symbol": symbol,
                "Model": model_name,
                "Fold": fold + 1,
                "Date": test_df["Date"].values,
                "Actual_Target": test_df["Target"].values,
                "Predicted_Target": predictions,
                "Predicted_Up_Probability": probabilities,
                "Close": test_df["Close"].values,
                "Future_Close": test_df["Future_Close"].values,
                "Future_Return_20d": test_df["Future_Return_20d"].values,
            }))

    return fold_results, prediction_results, tuning_results


# =========================================================
# TÜM KRİPTOLAR
# =========================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path
        for path in DATA_DIR.glob("*_dataset.csv")
        if path.name != "crypto_model_comparison.csv"
    )

    if not paths:
        raise FileNotFoundError(
            f"Kripto veri setleri bulunamadı: {DATA_DIR}"
        )

    print("=" * 65)
    print("FINANCEAI - TÜM KRİPTOLAR İÇİN WALK-FORWARD")
    print("=" * 65)
    print(f"Kripto sayısı: {len(paths)}")
    print(f"Fold sayısı: {N_FOLDS}")
    print(f"Tahmin ufku: {HORIZON} günlük mum")

    all_folds = []
    all_predictions = []
    all_tuning = []
    failures = []

    for index, path in enumerate(paths, start=1):
        print(f"\nİlerleme: {index}/{len(paths)}")

        try:
            fold_results, prediction_results, tuning_results = (
                run_one_crypto(path)
            )

            all_folds.extend(fold_results)
            all_predictions.extend(prediction_results)
            all_tuning.extend(tuning_results)

        except Exception as error:
            print(f"HATA | {path.name} | {error}")
            failures.append({
                "Symbol": path.stem,
                "Error": str(error),
            })

    if not all_folds:
        raise RuntimeError(
            "Hiçbir kripto için walk-forward sonucu üretilemedi."
        )

    results = pd.DataFrame(all_folds)
    predictions = pd.concat(all_predictions, ignore_index=True)

    results.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    summary = (
        results.groupby(["Symbol", "Model"])[metric_columns]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if all_tuning:
        pd.concat(all_tuning, ignore_index=True).to_csv(
            TUNING_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    print("\n" + "=" * 65)
    print("GENEL MODEL ORTALAMALARI")
    print("=" * 65)

    print(
        results.groupby("Model")[metric_columns]
        .mean()
        .round(4)
        .to_string()
    )

    print(f"\nBaşarılı test kayıtları: {len(results)}")
    print(f"Başarısız kripto sayısı: {len(failures)}")
    print(f"Sonuçlar: {RESULTS_PATH}")
    print(f"Özet: {SUMMARY_PATH}")
    print(f"Tahminler: {PREDICTIONS_PATH}")

    if failures:
        print("\nBaşarısız kriptolar:")
        print(pd.DataFrame(failures).to_string(index=False))

    print(
        "\nNOT: Bu metrikler geçmiş dönemlerdeki sınıflandırma "
        "performansını ölçer; kârlılık garantisi değildir."
    )


if __name__ == "__main__":
    main()
