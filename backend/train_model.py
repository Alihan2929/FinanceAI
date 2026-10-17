
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# DOSYA YOLLARI VE AYARLAR
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "TUPRS_IS_dataset.csv"

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

TARGET = "Target"
HORIZON = 20
TEST_RATIO = 0.20


# --------------------------------------------------
# MODEL EĞİTİMİ VE DEĞERLENDİRME
# --------------------------------------------------

def train_models():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri seti bulunamadı: {DATA_PATH}\n"
            "Önce veri hazırlama dosyalarını çalıştır."
        )

    data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    required_columns = FEATURES + [
        TARGET,
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Eksik sütunlar: {missing}\n"
            "Önce feature_engineering.py ve "
            "target_generation.py dosyalarını çalıştır."
        )

    # Veri türlerini doğrula
    for column in FEATURES + [
        TARGET,
        "Close",
        "Future_Close",
        "Future_Return_20d",
    ]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    # Hedef yalnızca 0 veya 1 olmalı
    data = data[data[TARGET].isin([0, 1])].copy()

    data = data.dropna(
        subset=FEATURES + [
            TARGET,
            "Close",
            "Future_Close",
            "Future_Return_20d",
        ]
    )

    data = data.sort_values("Date").reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    if len(data) < 200:
        raise ValueError(
            f"Yeterli veri yok. Kullanılabilir örnek: {len(data)}"
        )

    # --------------------------------------------------
    # KRONOLOJİK EĞİTİM / TEST AYRIMI
    # --------------------------------------------------

    split = int(len(data) * (1 - TEST_RATIO))

    # Hedefler 20 işlem günlük olduğu için eğitim ve test
    # arasındaki hedef pencerelerinin çakışmasını azalt.
    train_end = split - HORIZON
    test_start = split

    if train_end <= 0 or test_start >= len(data):
        raise ValueError("Eğitim/test ayrımı yapılamadı.")

    X = data[FEATURES]
    y = data[TARGET]

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    test_data = data.iloc[test_start:].copy()

    if y_train.nunique() < 2:
        raise ValueError(
            "Eğitim verisinde iki hedef sınıfı da bulunmalı."
        )

    print("=" * 60)
    print("FINANCE AI - MODEL EĞİTİMİ")
    print("=" * 60)

    print(f"Toplam örnek: {len(data)}")
    print(f"Eğitim örneği: {len(X_train)}")
    print(f"Test örneği: {len(X_test)}")

    print(
        "Eğitim dönemi:",
        data["Date"].iloc[0].date(),
        "-",
        data["Date"].iloc[train_end - 1].date(),
    )

    print(
        "Test dönemi:",
        data["Date"].iloc[test_start].date(),
        "-",
        data["Date"].iloc[-1].date(),
    )

    print("\nEğitim sınıf dağılımı:")
    print(y_train.value_counts().sort_index())

    print("\nTest sınıf dağılımı:")
    print(y_test.value_counts().sort_index())

    # --------------------------------------------------
    # BASELINE: HER ZAMAN YÜKSELİŞ
    # --------------------------------------------------

    baseline_predictions = [1] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    baseline_balanced_accuracy = balanced_accuracy_score(
        y_test, baseline_predictions
    )

    print("\n" + "=" * 60)
    print("BASELINE: HER ZAMAN YÜKSELİŞ")
    print("=" * 60)

    print(f"Accuracy: {baseline_accuracy:.3f}")
    print(
        "Balanced accuracy: "
        f"{baseline_balanced_accuracy:.3f}"
    )

    # --------------------------------------------------
    # MODELLER
    # --------------------------------------------------

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    # --------------------------------------------------
    # EĞİTİM VE TEST
    # --------------------------------------------------

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)

        balanced_accuracy = balanced_accuracy_score(
            y_test, predictions
        )

        print(f"Accuracy: {accuracy:.3f}")
        print(
            "Balanced accuracy: "
            f"{balanced_accuracy:.3f}"
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

        print("\nTahmin sınıf dağılımı:")

        print(
            pd.Series(predictions)
            .value_counts()
            .sort_index()
        )

        print("\nYükseliş olasılığı özeti:")

        print(
            pd.Series(probabilities)
            .describe()
            .round(4)
        )

        # --------------------------------------------------
        # MODELİ KAYDET
        # --------------------------------------------------

        model_path = MODEL_DIR / f"{name}.joblib"

        joblib.dump(
            {
                "model": model,
                "features": FEATURES,
                "horizon": HORIZON,
            },
            model_path,
        )

        print(f"\nModel kaydedildi: {model_path}")

        # --------------------------------------------------
        # TEST TAHMİNLERİNİ KAYDET
        # --------------------------------------------------

        prediction_data = pd.DataFrame({
            "Date": test_data["Date"].values,
            "Actual_Target": y_test.values,
            "Predicted_Target": predictions,
            "Predicted_Up_Probability": probabilities,
            "Close": test_data["Close"].values,
            "Future_Close": test_data["Future_Close"].values,
            "Future_Return_20d": (
                test_data["Future_Return_20d"].values
            ),
        })

        prediction_path = (
            DATA_DIR / f"{name}_test_predictions.csv"
        )

        prediction_data.to_csv(
            prediction_path,
            index=False,
        )

        print(
            f"Test tahminleri kaydedildi: {prediction_path}"
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "Baseline Accuracy": baseline_accuracy,
            "Baseline Balanced Accuracy": (
                baseline_balanced_accuracy
            ),
        })

    # --------------------------------------------------
    # MODEL KARŞILAŞTIRMASI
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("MODEL KARŞILAŞTIRMASI")
    print("=" * 60)

    print(
        results_df.sort_values(
            "Balanced Accuracy",
            ascending=False,
        ).to_string(index=False)
    )

    results_path = DATA_DIR / "model_comparison.csv"

    results_df.to_csv(results_path, index=False)

    print(f"\nKarşılaştırma kaydedildi: {results_path}")

    print(
        "\nUYARI: Test performansı gelecekteki getiriyi "
        "garanti etmez. Bu modeller tek başına yatırım "
        "kararı vermek için yeterli değildir."
    )


if __name__ == "__main__":
    train_models()
