
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "TUPRS_IS_dataset.csv"

OUTPUT_PATH = (
    BASE_DIR / "data" / "feature_importance.csv"
)

HORIZON = 20
TEST_RATIO = 0.20

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
# VERİYİ YÜKLE VE TEMİZLE
# =========================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + ["Date", "Target"]

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

    for column in FEATURES + ["Target"]:
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
    df = df.reset_index(drop=True)

    return df


# =========================================================
# ÖZELLİK ÖNEMİ
# =========================================================

def run_analysis():

    df = load_data()

    if len(df) < 200:
        raise ValueError(
            "Analiz için yeterli veri bulunamadı."
        )

    # Kronolojik eğitim/test ayrımı.
    split_index = int(len(df) * (1 - TEST_RATIO))
    train_end = split_index - HORIZON
    test_start = split_index

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Eğitim/test ayrımı yapılamadı."
        )

    X_train = train_df[FEATURES]
    y_train = train_df["Target"]

    X_test = test_df[FEATURES]
    y_test = test_df["Target"]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Eğitim ve test verisinde iki sınıf da bulunmalı."
        )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    baseline_score = balanced_accuracy_score(
        y_test,
        predictions,
    )

    # 1. Yöntem: Random Forest'ın kendi özellik önemleri.
    impurity_importance = model.feature_importances_

    # 2. Yöntem: Testte bir özelliği karıştırınca
    # balanced accuracy ne kadar değişiyor?
    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )

    results = pd.DataFrame({
        "Feature": FEATURES,
        "RF_Importance": impurity_importance,
        "Permutation_Mean": permutation.importances_mean,
        "Permutation_Std": permutation.importances_std,
    })

    results = results.sort_values(
        "Permutation_Mean",
        ascending=False,
    ).reset_index(drop=True)

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINANCEAI - ÖZELLİK ÖNEM ANALİZİ")
    print("=" * 65)

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

    print(f"Eğitim örneği: {len(train_df)}")
    print(f"Test örneği: {len(test_df)}")

    print(
        f"\nBaşlangıç balanced accuracy: "
        f"{baseline_score:.3f}"
    )

    print("\n--- ÖZELLİKLER (Permutation önemine göre) ---")

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSonuçlar kaydedildi: {OUTPUT_PATH}")

    print(
        "\nYorumlama:"
        "\n- Pozitif Permutation_Mean: özellik karıştırılınca "
        "başarı düşmüş; özellik faydalı olabilir."
        "\n- Sıfıra yakın değer: ölçülen katkı sınırlı olabilir."
        "\n- Negatif değer: karıştırma test başarısını artırmış; "
        "özellik gürültü taşıyor olabilir."
        "\n- Yüksek standart sapma, önem ölçümünün kararsız "
        "olabileceğini gösterir."
        "\n- Birbiriyle ilişkili özellikler önemlerini paylaşabilir."
    )

    print(
        "\nUYARI: Bu önem değerleri tek bir test döneminden "
        "hesaplanır. Özellik seçimi için bu test dönemini "
        "kullanıp aynı sonuç üzerinde başarı iddiasında "
        "bulunmayacağız."
    )


if __name__ == "__main__":
    run_analysis()
