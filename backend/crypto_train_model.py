
from pathlib import Path

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

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "crypto_processed"
MODEL_DIR = BASE_DIR / "models" / "crypto"
RESULTS_PATH = DATA_DIR / "crypto_model_comparison.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 20
TEST_RATIO = 0.20
MINIMUM_SAMPLES = 200
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


def create_models(y_train):
    """Üç sınıflandırıcıyı karşılaştırmak için oluşturur."""
    class_counts = y_train.value_counts()
    negatives = int(class_counts.get(0, 0))
    positives = int(class_counts.get(1, 0))

    scale_pos_weight = (
        negatives / positives if positives > 0 else 1.0
    )

    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["Date"])

    required = FEATURES + [
        "Date",
        "Target",
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")

    for column in FEATURES + [
        "Target",
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

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
            f"Yetersiz örnek sayısı: {len(df)}"
        )

    if df["Target"].nunique() < 2:
        raise ValueError(
            "Hedef sütununda iki sınıf da bulunmalı."
        )

    return df.reset_index(drop=True)


def evaluate_model(y_true, predictions, probabilities):
    metrics = {
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

    if len(np.unique(y_true)) == 2:
        metrics["ROC_AUC"] = roc_auc_score(
            y_true, probabilities
        )
    else:
        metrics["ROC_AUC"] = np.nan

    return metrics


def train_crypto(path):
    symbol = path.stem.replace("_dataset", "")

    print("\n" + "=" * 60)
    print(f"KRİPTO: {symbol}")
    print("=" * 60)

    df = load_dataset(path)

    split = int(len(df) * (1 - TEST_RATIO))

    # 20 günlük hedef pencerelerinin eğitim ve test
    # dönemleri arasında örtüşmesini azaltmak için boşluk.
    train_end = split - HORIZON

    if train_end <= 0 or split >= len(df):
        raise ValueError(
            "Eğitim/test ayrımı yapılamadı."
        )

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[split:].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["Target"]
    X_test = test_df[FEATURES]
    y_test = test_df["Target"]

    if y_train.nunique() < 2:
        raise ValueError(
            "Eğitim döneminde iki sınıf da bulunmalı."
        )

    print(f"Eğitim örneği: {len(train_df)}")
    print(f"Test örneği: {len(test_df)}")
    print(
        f"Eğitim sonu: "
        f"{train_df['Date'].iloc[-1].date()}"
    )
    print(
        f"Test başlangıcı: "
        f"{test_df['Date'].iloc[0].date()}"
    )

    # Baseline: her zaman pozitif getiri tahmini.
    baseline = np.ones(len(y_test), dtype=int)
    baseline_accuracy = accuracy_score(
        y_test, baseline
    )
    baseline_balanced = balanced_accuracy_score(
        y_test, baseline
    )

    results = []

    for model_name, model in create_models(y_train).items():
        print(f"\nModel: {model_name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        positive_index = list(model.classes_).index(1)
        probabilities_up = probabilities[:, positive_index]

        metrics = evaluate_model(
            y_test,
            predictions,
            probabilities_up,
        )

        print(
            " | ".join(
                f"{key}: {value:.4f}"
                for key, value in metrics.items()
            )
        )

        # Backend ile uyumlu model dosya adı:
        # BTCUSDT_logistic_regression.joblib
        # BTCUSDT_random_forest.joblib
        # BTCUSDT_xgboost.joblib
        model_path = (
            MODEL_DIR / f"{symbol}_{model_name}.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "horizon": HORIZON,
                "ticker": symbol,
                "model_name": model_name,
            },
            model_path,
        )

        print(f"Model kaydedildi: {model_path}")

        prediction_path = (
            DATA_DIR
            / f"{symbol}_{model_name}_test_predictions.csv"
        )

        pd.DataFrame({
            "Date": test_df["Date"].values,
            "Symbol": symbol,
            "Model": model_name,
            "Actual_Target": y_test.values,
            "Predicted_Target": predictions,
            "Predicted_Up_Probability": probabilities_up,
            "Close": test_df["Close"].values,
            "Future_Close": test_df["Future_Close"].values,
            "Future_Return_20d": (
                test_df["Future_Return_20d"].values
            ),
        }).to_csv(
            prediction_path,
            index=False,
            encoding="utf-8-sig",
        )

        results.append({
            "Symbol": symbol,
            "Model": model_name,
            "Samples": len(df),
            "Train_Samples": len(train_df),
            "Test_Samples": len(test_df),
            "Train_End": str(
                train_df["Date"].iloc[-1].date()
            ),
            "Test_Start": str(
                test_df["Date"].iloc[0].date()
            ),
            "Test_End": str(
                test_df["Date"].iloc[-1].date()
            ),
            **metrics,
            "Baseline_Accuracy": baseline_accuracy,
            "Baseline_Balanced_Accuracy": baseline_balanced,
            "Model_File": str(model_path),
        })

    return results


def main():
    paths = sorted(DATA_DIR.glob("*_dataset.csv"))

    if not paths:
        raise FileNotFoundError(
            f"Kripto veri seti bulunamadı: {DATA_DIR}"
        )

    print(f"Bulunan kripto veri seti: {len(paths)}")

    all_results = []
    failures = []

    for index, path in enumerate(paths, start=1):
        print(f"\nİlerleme: {index}/{len(paths)}")

        try:
            all_results.extend(train_crypto(path))
        except Exception as error:
            print(f"HATA | {path.name} | {error}")
            failures.append({
                "Symbol": path.stem,
                "Error": str(error),
            })

    if not all_results:
        raise RuntimeError(
            "Hiçbir kripto modeli eğitilemedi. "
            "Yukarıdaki hata mesajlarını kontrol et."
        )

    results = pd.DataFrame(all_results)

    results.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("KRİPTO MODEL SONUÇLARI")
    print("=" * 60)

    print(
        results.groupby("Model")[
            [
                "Accuracy",
                "Balanced_Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
            ]
        ].mean().round(4).to_string()
    )

    print(f"\nModel kayıtları: {len(all_results)}")
    print(f"Başarısız kripto veri seti: {len(failures)}")
    print(f"Sonuç dosyası: {RESULTS_PATH}")

    if failures:
        print("\nBaşarısız işlemler:")
        print(
            pd.DataFrame(failures).to_string(index=False)
        )


if __name__ == "__main__":
    main()
