
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Return_1d",
    "Return_5d",
    "Return_20d",
    "Return_60d",
    "SMA_5_ratio",
    "SMA_20_ratio",
    "SMA_50_ratio",
    "EMA_12_ratio",
    "EMA_26_ratio",
    "EMA_cross",
    "RSI_scaled",
    "MACD_ratio",
    "MACD_signal_ratio",
    "MACD_histogram_ratio",
    "BB_position",
    "BB_width",
    "Volume_Change",
    "Volume_Ratio",
    "OBV_Change",
    "Volatility_20d",
    "ATR_ratio",
]

TARGET = "Target"
HORIZON = 20
TEST_RATIO = 0.20
MINIMUM_SAMPLES = 200
RANDOM_STATE = 42


# =========================================================
# MODELLER
# =========================================================

def create_models():
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
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# =========================================================
# VERİYİ YÜKLE VE DOĞRULA
# =========================================================

def load_dataset(data_path: Path):
    data = pd.read_csv(
        data_path,
        parse_dates=["Date"],
    )

    required = FEATURES + [
        "Date",
        TARGET,
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Eksik sütunlar: {missing}. "
            "Özellik ve hedef dosyalarını yeniden oluştur."
        )

    numeric_columns = FEATURES + [
        TARGET,
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=required,
    ).copy()

    data = (
        data.sort_values("Date")
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    data = data[
        data[TARGET].isin([0, 1])
        & (data["Close"] > 0)
        & (data["Future_Close"] > 0)
    ].copy()

    data[TARGET] = data[TARGET].astype(int)

    if len(data) < MINIMUM_SAMPLES:
        raise ValueError(
            f"Yeterli örnek yok: {len(data)}; "
            f"minimum {MINIMUM_SAMPLES} gerekli."
        )

    if data[TARGET].nunique() != 2:
        raise ValueError(
            "Hedef değişkeninde iki sınıf da bulunmalı."
        )

    return data.reset_index(drop=True)


# =========================================================
# METRİKLER
# =========================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities,
):
    result = {
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["ROC AUC"] = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        result["ROC AUC"] = np.nan

    return result


# =========================================================
# TEK HİSSE İÇİN EĞİTİM
# =========================================================

def train_stock(data_path: Path):
    ticker = (
        data_path.name
        .replace("_dataset.csv", "")
        .replace("_", ".")
    )

    print("\n" + "=" * 65)
    print(f"HİSSE: {ticker}")
    print("=" * 65)

    data = load_dataset(data_path)

    # Kronolojik test ayrımı
    split = int(
        len(data) * (1 - TEST_RATIO)
    )

    # Eğitim ve test arasındaki hedef penceresi
    # çakışmasını azaltmak için 20 satır ayır.
    train_end = split - HORIZON
    test_start = split

    if train_end <= 0 or test_start >= len(data):
        raise ValueError(
            "Eğitim/test ayrımı yapılamadı."
        )

    train_data = data.iloc[:train_end].copy()
    test_data = data.iloc[test_start:].copy()

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    if y_train.nunique() != 2:
        raise ValueError(
            "Eğitim döneminde iki hedef sınıfı da bulunmalı."
        )

    print(f"Toplam örnek: {len(data)}")
    print(f"Eğitim örneği: {len(train_data)}")
    print(f"Test örneği: {len(test_data)}")
    print(f"Kullanılan özellik sayısı: {len(FEATURES)}")

    print(
        f"Eğitim dönemi: "
        f"{train_data['Date'].iloc[0].date()} - "
        f"{train_data['Date'].iloc[-1].date()}"
    )

    print(
        f"Test dönemi: "
        f"{test_data['Date'].iloc[0].date()} - "
        f"{test_data['Date'].iloc[-1].date()}"
    )

    print("\nEğitim hedef dağılımı:")
    print(
        y_train.value_counts().sort_index()
    )

    print("\nTest hedef dağılımı:")
    print(
        y_test.value_counts().sort_index()
    )

    # Baseline 1: Her zaman yükseliş tahmini
    always_up = np.ones(
        len(y_test),
        dtype=int,
    )

    always_up_accuracy = accuracy_score(
        y_test,
        always_up,
    )

    always_up_balanced = balanced_accuracy_score(
        y_test,
        always_up,
    )

    # Baseline 2: Eğitimdeki çoğunluk sınıfı
    majority_class = int(
        y_train.mode().iloc[0]
    )

    majority_predictions = np.full(
        len(y_test),
        majority_class,
        dtype=int,
    )

    majority_accuracy = accuracy_score(
        y_test,
        majority_predictions,
    )

    majority_balanced = balanced_accuracy_score(
        y_test,
        majority_predictions,
    )

    print("\nBaseline sonuçları:")

    print(
        f"Her zaman yükseliş accuracy: "
        f"{always_up_accuracy:.4f}"
    )

    print(
        f"Her zaman yükseliş balanced accuracy: "
        f"{always_up_balanced:.4f}"
    )

    print(
        f"Çoğunluk sınıfı accuracy: "
        f"{majority_accuracy:.4f}"
    )

    print(
        f"Çoğunluk sınıfı balanced accuracy: "
        f"{majority_balanced:.4f}"
    )

    results = []

    # Üç modeli aynı eğitim/test döneminde karşılaştır.
    for model_name, model in create_models().items():
        print("\n" + "-" * 55)
        print(f"MODEL: {model_name}")
        print("-" * 55)

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test,
        )

        probabilities = model.predict_proba(
            X_test,
        )

        # Her iki model ailesinde de pozitif sınıfı
        # açıkça 1 olarak belirle.
        if hasattr(model, "named_steps"):
            classes = list(
                model.named_steps[
                    "logisticregression"
                ].classes_
            )
        else:
            classes = list(model.classes_)

        positive_index = classes.index(1)

        probabilities_up = probabilities[
            :,
            positive_index,
        ]

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities_up,
        )

        for metric_name, value in metrics.items():
            print(
                f"{metric_name}: {value:.4f}"
            )

        print("\nSınıflandırma raporu:")

        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=[
                    "Yükseliş yok",
                    "Yükseliş",
                ],
                zero_division=0,
            )
        )

        print("Confusion matrix:")

        print(
            confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            )
        )

        # Modeli ayrı dosyaya kaydet.
        model_path = (
            MODEL_DIR
            / (
                f"{ticker.replace('.', '_')}_"
                f"{model_name}.joblib"
            )
        )

        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "horizon": HORIZON,
                "ticker": ticker,
                "model_name": model_name,
            },
            model_path,
        )

        print(
            f"Model kaydedildi: {model_path}"
        )

        # Test tahminlerini kaydet.
        prediction_data = pd.DataFrame({
            "Date": test_data["Date"].values,
            "Ticker": ticker,
            "Actual_Target": y_test.values,
            "Predicted_Target": predictions,
            "Predicted_Up_Probability": probabilities_up,
            "Close": test_data["Close"].values,
            "Future_Close": test_data["Future_Close"].values,
            "Future_Return_20d": (
                test_data["Future_Return_20d"].values
            ),
        })

        prediction_path = (
            DATA_DIR
            / (
                f"{ticker.replace('.', '_')}_"
                f"{model_name}_test_predictions.csv"
            )
        )

        prediction_data.to_csv(
            prediction_path,
            index=False,
        )

        print(
            f"Tahminler kaydedildi: {prediction_path}"
        )

        results.append({
            "Ticker": ticker,
            "Model": model_name,
            "Samples": len(data),
            "Train Samples": len(train_data),
            "Test Samples": len(test_data),
            "Train Start": (
                train_data["Date"]
                .iloc[0]
                .date()
                .isoformat()
            ),
            "Train End": (
                train_data["Date"]
                .iloc[-1]
                .date()
                .isoformat()
            ),
            "Test Start": (
                test_data["Date"]
                .iloc[0]
                .date()
                .isoformat()
            ),
            "Test End": (
                test_data["Date"]
                .iloc[-1]
                .date()
                .isoformat()
            ),
            **metrics,
            "Always Up Accuracy": always_up_accuracy,
            "Always Up Balanced Accuracy": always_up_balanced,
            "Majority Baseline Accuracy": majority_accuracy,
            "Majority Baseline Balanced Accuracy": majority_balanced,
        })

    return results


# =========================================================
# TÜM HİSSELERİ EĞİT
# =========================================================

def train_all_stocks():
    dataset_paths = sorted(
        DATA_DIR.glob("*_dataset.csv")
    )

    if not dataset_paths:
        raise FileNotFoundError(
            f"{DATA_DIR} içinde "
            "*_dataset.csv dosyası bulunamadı."
        )

    print("=" * 65)
    print("FINANCEAI - ÇOK HİSSELİ MODEL EĞİTİMİ")
    print("=" * 65)

    print(
        f"Bulunan veri seti: {len(dataset_paths)}"
    )

    print(
        f"Kullanılan özellik sayısı: {len(FEATURES)}"
    )

    print("Özellikler:")
    print(", ".join(FEATURES))

    all_results = []
    failed = []

    for index, data_path in enumerate(
        dataset_paths,
        start=1,
    ):
        print(
            f"\nİlerleme: {index}/{len(dataset_paths)}"
        )

        try:
            all_results.extend(
                train_stock(data_path)
            )

        except Exception as error:
            print(
                f"HATA: {data_path.name} - {error}"
            )

            failed.append(
                data_path.name
            )

    if not all_results:
        raise RuntimeError(
            "Hiçbir hisse için model eğitilemedi."
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        DATA_DIR / "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("MODEL KARŞILAŞTIRMASI")
    print("=" * 65)

    display_columns = [
        "Ticker",
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
        "Majority Baseline Accuracy",
    ]

    print(
        results_df.sort_values(
            ["Model", "Balanced Accuracy"],
            ascending=[True, False],
        )[display_columns].to_string(
            index=False
        )
    )

    metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
    ]

    print("\nMODEL ORTALAMALARI")

    print(
        results_df.groupby("Model")[
            metric_columns
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print(
        f"\nKarşılaştırma dosyası: {results_path}"
    )

    print(
        f"Başarılı model sayısı: {len(all_results)}"
    )

    print(
        f"Başarısız veri seti sayısı: {len(failed)}"
    )

    if failed:
        print("\nEğitilemeyen veri setleri:")

        for name in failed:
            print(f"- {name}")

    print(
        "\nUYARI: Sonuçlar tek bir kronolojik test "
        "dönemine dayanır. Kârlılık veya gelecekteki "
        "performans garantisi değildir."
    )


if __name__ == "__main__":
    train_all_stocks()
