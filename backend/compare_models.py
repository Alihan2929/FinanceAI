
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "TUPRS_IS_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "model_comparison_new.csv"

HORIZON = 20
TEST_SIZE = 0.20

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


# =========================================================
# VERİYİ YÜKLE
# =========================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["Date", "Target"]

    missing = [
        col for col in required_columns
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

    for col in FEATURES + ["Target"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df.replace([float("inf"), float("-inf")], float("nan"))
        .dropna(subset=required_columns)
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    df = df[df["Target"].isin([0, 1])].copy()
    df["Target"] = df["Target"].astype(int)

    if len(df) < 200:
        raise ValueError(
            "Karşılaştırma için yeterli veri bulunamadı."
        )

    return df


# =========================================================
# MODEL METRİKLERİ
# =========================================================

def evaluate_model(name, y_true, y_pred):

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "Precision (pozitif sınıf)": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall (pozitif sınıf)": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 (pozitif sınıf)": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


# =========================================================
# KARŞILAŞTIRMA
# =========================================================

def compare_models():

    df = load_data()

    # Kronolojik test ayrımı.
    # Hedef 20 işlem gününü kapsadığı için sınırda
    # 20 satırlık boşluk bırakıyoruz.
    split_index = int(len(df) * (1 - TEST_SIZE))
    train_end = split_index - HORIZON
    test_start = split_index

    if train_end <= 0 or test_start >= len(df):
        raise ValueError(
            "Eğitim/test ayrımı yapılamadı."
        )

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["Target"]

    X_test = test_df[FEATURES]
    y_test = test_df["Target"]

    if y_train.nunique() < 2:
        raise ValueError(
            "Eğitim verisinde iki sınıf da bulunmalı."
        )

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_forest.fit(X_train, y_train)

    rf_pred = random_forest.predict(X_test)

    # -----------------------------------------------------
    # LOJİSTİK REGRESYON
    # -----------------------------------------------------

    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    logistic_regression.fit(X_train, y_train)

    lr_pred = logistic_regression.predict(X_test)

    # -----------------------------------------------------
    # METRİKLER
    # -----------------------------------------------------

    results = [
        evaluate_model(
            "Random Forest",
            y_test,
            rf_pred,
        ),
        evaluate_model(
            "Lojistik Regresyon",
            y_test,
            lr_pred,
        ),
    ]

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------
    # KONSOL RAPORU
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINANCEAI - MODEL KARŞILAŞTIRMASI")
    print("=" * 60)

    print(f"Toplam veri: {len(df)}")
    print(f"Eğitim örneği: {len(train_df)}")
    print(f"Test örneği: {len(test_df)}")

    print(
        f"Eğitim dönemi: "
        f"{train_df['Date'].min().date()} - "
        f"{train_df['Date'].max().date()}"
    )

    print(
        f"Test dönemi: "
        f"{test_df['Date'].min().date()} - "
        f"{test_df['Date'].max().date()}"
    )

    print("\n--- TEST SINIF DAĞILIMI ---")
    print(
        y_test.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .rename("Yüzde")
    )

    print("\n--- MODEL METRİKLERİ ---")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\n--- RANDOM FOREST RAPORU ---")
    print(
        classification_report(
            y_test,
            rf_pred,
            labels=[0, 1],
            target_names=[
                "Getiri pozitif değil",
                "Getiri pozitif",
            ],
            zero_division=0,
        )
    )

    print("--- LOJİSTİK REGRESYON RAPORU ---")
    print(
        classification_report(
            y_test,
            lr_pred,
            labels=[0, 1],
            target_names=[
                "Getiri pozitif değil",
                "Getiri pozitif",
            ],
            zero_division=0,
        )
    )

    print("--- RANDOM FOREST KARIŞIKLIK MATRİSİ ---")
    print(
        confusion_matrix(
            y_test,
            rf_pred,
            labels=[0, 1],
        )
    )

    print("--- LOJİSTİK REGRESYON KARIŞIKLIK MATRİSİ ---")
    print(
        confusion_matrix(
            y_test,
            lr_pred,
            labels=[0, 1],
        )
    )

    # -----------------------------------------------------
    # TEST TAHMİNLERİNİ KAYDET
    # -----------------------------------------------------

    predictions = pd.DataFrame({
        "Date": test_df["Date"].to_numpy(),
        "Actual_Target": y_test.to_numpy(),
        "RandomForest_Prediction": rf_pred,
        "LogisticRegression_Prediction": lr_pred,
        "RandomForest_Probability_Up": (
            random_forest.predict_proba(X_test)[
                :, list(random_forest.classes_).index(1)
            ]
            if 1 in random_forest.classes_
            else 0.0
        ),
        "LogisticRegression_Probability_Up": (
            logistic_regression.predict_proba(X_test)[
                :, list(
                    logistic_regression.named_steps["model"].classes_
                ).index(1)
            ]
            if 1 in logistic_regression.named_steps["model"].classes_
            else 0.0
        ),
    })

    predictions.to_csv(
        BASE_DIR / "data" / "model_predictions_comparison.csv",
        index=False,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nMetrikler kaydedildi: {OUTPUT_PATH}")
    print(
        "Tahminler kaydedildi: "
        f"{BASE_DIR / 'data' / 'model_predictions_comparison.csv'}"
    )

    # Sadece metriklere göre ön değerlendirme.
    best_model = results_df.loc[
        results_df["Balanced Accuracy"].idxmax(),
        "Model",
    ]

    print(
        "\nBalanced accuracy bakımından bu testte önde olan: "
        f"{best_model}"
    )

    print(
        "Bu sonuç tek bir test dönemine dayanır; "
        "gelecekte daha iyi performans garantisi değildir."
    )


if __name__ == "__main__":
    compare_models()
