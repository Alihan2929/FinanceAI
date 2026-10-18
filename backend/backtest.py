
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "TUPRS_IS_dataset.csv"
RESULTS_DIR = BASE_DIR / "data"

COMPARISON_PATH = RESULTS_DIR / "backtest_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "backtest_model_predictions.csv"

INITIAL_CAPITAL = 100_000.0
HORIZON = 20

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40

# Tek yönlü varsayımsal işlem maliyeti: %0,10
TRANSACTION_COST = 0.001

TRAIN_RATIO = 0.60
REFIT_INTERVAL = 20
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


# =========================================================
# MODELLER
# =========================================================

def create_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
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

        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    }


def get_positive_probabilities(model, X):
    """Target=1 sınıfına ait olasılıkları döndürür."""

    probabilities = model.predict_proba(X)

    if hasattr(model, "named_steps"):
        classes = list(
            model.named_steps["logisticregression"].classes_
        )
    else:
        classes = list(model.classes_)

    if 1 not in classes:
        return np.zeros(len(X), dtype=float)

    return probabilities[:, classes.index(1)]


# =========================================================
# VERİYİ YÜKLE VE KONTROL ET
# =========================================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [
        "Date",
        "Open",
        "Close",
        "Target",
    ]

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

    numeric_columns = FEATURES + [
        "Open",
        "Close",
        "Target",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = (
        df.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    df = df[
        (df["Open"] > 0)
        & (df["Close"] > 0)
        & (df["Target"].isin([0, 1]))
    ].copy()

    df["Target"] = df["Target"].astype(int)

    if len(df) <= HORIZON + 100:
        raise ValueError(
            "Backtest için yeterli veri yok."
        )

    # Hedefi henüz oluşmamış son 20 satırı çıkar.
    df = df.iloc[:-HORIZON].reset_index(drop=True)

    if df["Target"].nunique() < 2:
        raise ValueError(
            "Target sütununda iki sınıf da bulunmalı."
        )

    print(f"Veri başlangıcı: {df['Date'].min().date()}")
    print(f"Veri bitişi: {df['Date'].max().date()}")
    print(f"Kullanılabilir satır: {len(df)}")

    return df


# =========================================================
# WALK-FORWARD TAHMİNLERİ
# =========================================================

def generate_walk_forward_predictions(df, model_name):
    """
    Zaman sırasını koruyarak tahmin üretir.
    Her eğitim aşamasında, hedef ufku kadar boşluk bırakır.
    """

    n = len(df)
    test_start = int(n * TRAIN_RATIO)

    if test_start <= HORIZON:
        raise ValueError(
            "Başlangıç eğitim dönemi çok küçük."
        )

    probabilities = np.full(n, np.nan)
    fit_count = 0

    # Test öncesindeki son kapanış için de tahmin üret.
    for block_start in range(
        test_start - 1,
        n,
        REFIT_INTERVAL,
    ):
        block_end = min(
            block_start + REFIT_INTERVAL,
            n,
        )

        # Eğitim hedeflerinin test dönemine taşmaması için
        # HORIZON satırlık boşluk bırak.
        train_end = block_start - HORIZON

        if train_end <= 0:
            continue

        train_df = df.iloc[:train_end]

        if train_df["Target"].nunique() < 2:
            continue

        model = create_models()[model_name]

        model.fit(
            train_df[FEATURES],
            train_df["Target"],
        )

        block_df = df.iloc[block_start:block_end]

        probabilities[block_start:block_end] = (
            get_positive_probabilities(
                model,
                block_df[FEATURES],
            )
        )

        fit_count += 1

    result = df.copy()
    result["Probability_Up"] = probabilities

    result = (
        result.iloc[test_start - 1:]
        .copy()
        .reset_index(drop=True)
    )

    if result["Probability_Up"].isna().any():
        raise ValueError(
            f"{model_name}: Bazı satırlarda tahmin üretilemedi."
        )

    print(f"{model_name} yeniden eğitim sayısı: {fit_count}")

    return result, test_start


# =========================================================
# SİNYALLER VE POZİSYONLAR
# =========================================================

def create_positions(predictions):
    df = predictions.copy()

    df["Signal"] = "BEKLE"

    df.loc[
        df["Probability_Up"] >= BUY_THRESHOLD,
        "Signal",
    ] = "AL"

    df.loc[
        df["Probability_Up"] <= SELL_THRESHOLD,
        "Signal",
    ] = "SAT"

    # AL: hisseye gir.
    # SAT: nakde geç.
    # BEKLE: mevcut pozisyonu koru.
    position = 0
    target_positions = []

    for signal in df["Signal"]:
        if signal == "AL":
            position = 1
        elif signal == "SAT":
            position = 0

        target_positions.append(position)

    df["Target_Position"] = target_positions

    # Sinyal kapanışta oluşur, ertesi işlem gününde uygulanır.
    df["Applied_Position"] = (
        df["Target_Position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return df


# =========================================================
# GETİRİLER VE SERMAYE
# =========================================================

def calculate_returns(test, previous_close):
    df = test.copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("Test dönemi boş.")

    df["Previous_Close"] = df["Close"].shift(1)
    df.loc[0, "Previous_Close"] = previous_close

    if (
        (df["Previous_Close"] <= 0).any()
        or (df["Open"] <= 0).any()
        or (df["Close"] <= 0).any()
    ):
        raise ValueError("Geçersiz fiyat verisi.")

    # Önceki kapanıştan bugünkü açılışa kadar getiri.
    df["Overnight_Return"] = (
        df["Open"] / df["Previous_Close"] - 1
    )

    # Bugünkü açılıştan kapanışa kadar getiri.
    df["Intraday_Return"] = (
        df["Close"] / df["Open"] - 1
    )

    # Bir önceki işlem gününün uygulanan pozisyonu.
    df["Previous_Position"] = (
        df["Applied_Position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # Test başlangıcında portföy nakittedir.
    df.loc[0, "Previous_Position"] = 0

    # Pozisyon değiştiğinde işlem maliyeti uygula.
    df["Position_Change"] = (
        df["Applied_Position"]
        - df["Previous_Position"]
    ).abs()

    df["Transaction_Cost"] = (
        df["Position_Change"] * TRANSACTION_COST
    )

    # Gece boyunca önceki pozisyon taşınır.
    overnight_factor = (
        1
        + df["Previous_Position"]
        * df["Overnight_Return"]
    )

    # Açılıştaki pozisyon değişiminin maliyeti.
    cost_factor = 1 - df["Transaction_Cost"]

    # Açılıştan kapanışa kadar yeni pozisyon uygulanır.
    intraday_factor = (
        1
        + df["Applied_Position"]
        * df["Intraday_Return"]
    )

    df["Strategy_Return"] = (
        overnight_factor
        * cost_factor
        * intraday_factor
        - 1
    )

    # -----------------------------------------------------
    # AL VE TUT
    # -----------------------------------------------------
    # İlk test günü açılış fiyatından alım yapılır.
    # Sonraki günler kapanıştan kapanışa hesaplanır.
    df["Buy_Hold_Return"] = (
        df["Close"] / df["Previous_Close"] - 1
    )

    df.loc[0, "Buy_Hold_Return"] = (
        (df.loc[0, "Close"] / df.loc[0, "Open"])
        * (1 - TRANSACTION_COST)
        - 1
    )

    # Nakit stratejisinde getiri sıfır kabul edilir.
    df["Cash_Return"] = 0.0

    df["Strategy_Equity"] = (
        INITIAL_CAPITAL
        * (1 + df["Strategy_Return"]).cumprod()
    )

    df["Buy_Hold_Equity"] = (
        INITIAL_CAPITAL
        * (1 + df["Buy_Hold_Return"]).cumprod()
    )

    df["Cash_Equity"] = INITIAL_CAPITAL

    return df


# =========================================================
# PERFORMANS METRİKLERİ
# =========================================================

def calculate_metrics(returns):
    returns = (
        pd.Series(returns, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    if returns.empty:
        raise ValueError(
            "Performans ölçümü için getiri verisi yok."
        )

    equity = (
        INITIAL_CAPITAL
        * (1 + returns).cumprod()
    )

    final_capital = float(equity.iloc[-1])

    total_return = (
        final_capital / INITIAL_CAPITAL - 1
    ) * 100

    max_drawdown = (
        equity / equity.cummax() - 1
    ).min() * 100

    daily_std = returns.std(ddof=1)

    if daily_std > 0 and np.isfinite(daily_std):
        sharpe = (
            returns.mean()
            / daily_std
            * np.sqrt(252)
        )
    else:
        sharpe = np.nan

    return {
        "Başlangıç Sermayesi (TL)": round(
            INITIAL_CAPITAL, 2
        ),
        "Son Sermaye (TL)": round(
            final_capital, 2
        ),
        "Toplam Getiri (%)": round(
            total_return, 2
        ),
        "Maksimum Düşüş (%)": round(
            float(max_drawdown), 2
        ),
        "Yıllıklandırılmış Sharpe": (
            round(float(sharpe), 2)
            if np.isfinite(sharpe)
            else np.nan
        ),
    }


# =========================================================
# ANA BACKTEST
# =========================================================

def run_backtest():
    print("=" * 70)
    print("FINANCEAI - ÜÇ MODEL WALK-FORWARD BACKTEST")
    print("=" * 70)

    df = load_data()

    print(f"Özellik sayısı: {len(FEATURES)}")
    print(f"Hedef ufku: {HORIZON} işlem günü")
    print(f"İlk sermaye: {INITIAL_CAPITAL:,.2f} TL")

    all_comparisons = []
    all_predictions = []

    models = create_models()

    for model_name in models:
        print("\n" + "-" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        predictions, test_start = (
            generate_walk_forward_predictions(
                df,
                model_name,
            )
        )

        predictions = create_positions(predictions)

        # İlk satır test öncesi kapanış için üretilen tahmindir.
        test = (
            predictions.iloc[1:]
            .copy()
            .reset_index(drop=True)
        )

        previous_close = float(
            df.iloc[test_start - 1]["Close"]
        )

        # Test öncesi kapanışta üretilen sinyal,
        # ilk test gününün açılışında uygulanır.
        test.loc[0, "Applied_Position"] = int(
            predictions.iloc[0]["Target_Position"]
        )

        test["Applied_Position"] = (
            test["Applied_Position"].astype(int)
        )

        test = calculate_returns(
            test,
            previous_close,
        )

        test["Target"] = (
            df.iloc[test_start:]["Target"].to_numpy()
        )

        test["Model"] = model_name

        y_true = test["Target"].astype(int)

        # Sınıflandırma değerlendirmesi için 0.50 eşiği.
        y_pred = (
            test["Probability_Up"] >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        print(
            f"Test başlangıcı: {test['Date'].min().date()}"
        )
        print(
            f"Test bitişi: {test['Date'].max().date()}"
        )
        print(f"Test satırı sayısı: {len(test)}")
        print(f"Accuracy: %{accuracy * 100:.2f}")
        print(
            "Balanced Accuracy: "
            f"%{balanced_accuracy * 100:.2f}"
        )

        print("\nSınıflandırma raporu:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "Pozitif getiri değil",
                    "Pozitif getiri",
                ],
                zero_division=0,
            )
        )

        print("\nKarışıklık matrisi:")
        print(
            confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            )
        )

        strategies = [
            ("Model Stratejisi", "Strategy_Return"),
            ("Al ve Tut", "Buy_Hold_Return"),
            ("Nakit", "Cash_Return"),
        ]

        for strategy_name, return_column in strategies:
            metrics = calculate_metrics(
                test[return_column]
            )

            metrics["Model"] = model_name
            metrics["Strateji"] = strategy_name
            metrics["Accuracy (%)"] = round(
                accuracy * 100, 2
            )
            metrics["Balanced Accuracy (%)"] = round(
                balanced_accuracy * 100, 2
            )

            all_comparisons.append(metrics)

        all_predictions.append(test)

    # -----------------------------------------------------
    # SONUÇLARI KAYDET
    # -----------------------------------------------------

    comparison = pd.DataFrame(all_comparisons)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("MODEL VE STRATEJİ KARŞILAŞTIRMASI")
    print("=" * 70)

    print(
        comparison.to_string(index=False)
    )

    print(
        f"\nKarşılaştırma dosyası: {COMPARISON_PATH}"
    )

    print(
        f"Tahmin detayları: {PREDICTIONS_PATH}"
    )

    print(
        "\nUYARI: Bu tarihsel bir simülasyondur. İşlem maliyeti "
        "varsayımsaldır; gerçek komisyon, fiyat kayması, likidite "
        "ve emir yürütme koşulları tam olarak modellenmemiştir. "
        "Geçmiş performans gelecekteki getiriyi garanti etmez."
    )


if __name__ == "__main__":
    run_backtest()
