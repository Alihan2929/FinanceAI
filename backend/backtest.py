
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "TUPRS_IS_dataset.csv"
RESULTS_PATH = BASE_DIR / "data" / "backtest_results.csv"
COMPARISON_PATH = BASE_DIR / "data" / "backtest_comparison.csv"

INITIAL_CAPITAL = 100_000.0

# Hedef değişken 20 işlem gününü kapsıyor.
HORIZON = 20

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40

# Tek yön işlem maliyeti: %0,10
TRANSACTION_COST = 0.001

# İlk %60 eğitim, kalan dönem test.
TRAIN_RATIO = 0.60

# Model her 20 test satırında bir yeniden eğitilir.
REFIT_INTERVAL = 20

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
# VERİYİ YÜKLEME
# =========================================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [
        "Date",
        "Open",
        "Close",
        "Target",
    ]

    missing = [
        column
        for column in required_columns
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
        df.dropna(subset=required_columns)
        .sort_values("Date")
        .drop_duplicates(
            subset="Date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    df = df[
        (df["Open"] > 0)
        & (df["Close"] > 0)
        & (df["Target"].isin([0, 1]))
    ].reset_index(drop=True)

    df["Target"] = df["Target"].astype(int)

    if len(df) <= HORIZON + 100:
        raise ValueError(
            "Backtest için yeterli veri yok."
        )

    # Future_Close = Close.shift(-HORIZON) ile oluşturulan
    # hedefin son HORIZON satırında gelecek kapanışı yoktur.
    # Bu satırlar testten çıkarılır.
    df = df.iloc[:-HORIZON].reset_index(drop=True)

    if df["Target"].nunique() < 2:
        raise ValueError(
            "Target sütununda iki sınıf da bulunmalı."
        )

    return df


# =========================================================
# WALK-FORWARD TAHMİNLERİ
# =========================================================

def generate_walk_forward_predictions(df):
    n = len(df)

    test_start = int(n * TRAIN_RATIO)

    if test_start <= HORIZON:
        raise ValueError(
            "Başlangıç eğitim dönemi çok küçük."
        )

    probabilities = np.full(n, np.nan)
    fit_count = 0

    # İlk test gününden önceki kapanışta üretilen tahmin,
    # ilk test gününün açılış pozisyonunu belirler.
    for block_start in range(
        test_start - 1,
        n,
        REFIT_INTERVAL,
    ):
        block_end = min(
            block_start + REFIT_INTERVAL,
            n,
        )

        # Hedef 20 günlük olduğu için eğitim etiketlerinin
        # gelecek penceresi tahmin bloğuna taşmamalı.
        train_end = block_start - HORIZON

        if train_end <= 0:
            continue

        train_df = df.iloc[:train_end]

        if train_df["Target"].nunique() < 2:
            continue

        model = create_model()

        model.fit(
            train_df[FEATURES],
            train_df["Target"],
        )

        block_df = df.iloc[
            block_start:block_end
        ]

        classes = list(model.classes_)

        if 1 not in classes:
            block_probabilities = np.zeros(
                len(block_df)
            )
        else:
            positive_index = classes.index(1)

            block_probabilities = (
                model.predict_proba(
                    block_df[FEATURES]
                )[:, positive_index]
            )

        probabilities[
            block_start:block_end
        ] = block_probabilities

        fit_count += 1

    result = df.copy()
    result["Probability_Up"] = probabilities

    # Test başlangıcından önceki satırı da tutuyoruz;
    # bu satırın sinyali ilk test gününün pozisyonunda kullanılır.
    result = result.iloc[
        test_start - 1:
    ].copy().reset_index(drop=True)

    if result["Probability_Up"].isna().any():
        raise ValueError(
            "Bazı satırlarda tahmin üretilemedi. "
            "Veriyi ve eğitim ayarlarını kontrol et."
        )

    print(
        f"Model yeniden eğitim sayısı: {fit_count}"
    )

    return result, test_start


# =========================================================
# SİNYAL VE POZİSYON
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

    # Kapanışta oluşan sinyal ertesi işlem gününde uygulanır.
    df["Applied_Position"] = (
        df["Target_Position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return df


# =========================================================
# GETİRİ VE SERMAYE HESABI
# =========================================================

def calculate_returns(test, previous_close):
    df = test.copy().reset_index(drop=True)

    df["Previous_Close"] = df["Close"].shift(1)

    if len(df) > 0:
        df.loc[0, "Previous_Close"] = previous_close

    df["Overnight_Return"] = (
        df["Open"] / df["Previous_Close"] - 1
    )

    df["Intraday_Return"] = (
        df["Close"] / df["Open"] - 1
    )

    df["Previous_Position"] = (
        df["Applied_Position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    df["Position_Change"] = (
        df["Applied_Position"]
        - df["Previous_Position"]
    ).abs()

    # İlk test gününde nakitte başladığımız varsayılır.
    if len(df) > 0:
        df.loc[0, "Previous_Position"] = 0

        df.loc[0, "Position_Change"] = (
            df.loc[0, "Applied_Position"]
        )

    df["Transaction_Cost"] = (
        df["Position_Change"] * TRANSACTION_COST
    )

    # Gece getirisi önceki pozisyona,
    # gün içi getiri yeni pozisyona uygulanır.
    overnight_factor = (
        1
        + df["Previous_Position"]
        * df["Overnight_Return"].fillna(0)
    )

    intraday_factor = (
        1
        + df["Applied_Position"]
        * df["Intraday_Return"].fillna(0)
    )

    strategy_factor = (
        overnight_factor
        * intraday_factor
        * (1 - df["Transaction_Cost"])
    )

    df["Strategy_Return"] = strategy_factor - 1

    # Al ve tut: aynı test döneminin önceki kapanışından itibaren.
    df["Buy_Hold_Return"] = (
        df["Close"] / df["Previous_Close"] - 1
    )

    df["Buy_Hold_Return"] = (
        df["Buy_Hold_Return"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # İlk gün alım maliyeti.
    if len(df) > 0:
        df.loc[0, "Buy_Hold_Return"] = (
            (1 + df.loc[0, "Buy_Hold_Return"])
            * (1 - TRANSACTION_COST)
            - 1
        )

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
        pd.Series(returns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
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
        (equity / equity.cummax() - 1).min()
        * 100
    )

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
    print("=" * 60)
    print("FINANCEAI - WALK-FORWARD BACKTEST")
    print("=" * 60)

    df = load_data()

    print(f"Geçerli veri satırı: {len(df)}")
    print(f"Özellik sayısı: {len(FEATURES)}")
    print(f"Hedef ufku: {HORIZON} işlem günü")

    predictions, test_start = (
        generate_walk_forward_predictions(df)
    )

    predictions = create_positions(predictions)

    # İlk satır test öncesi sinyaldir.
    test = (
        predictions.iloc[1:]
        .copy()
        .reset_index(drop=True)
    )

    previous_close = float(
        df.iloc[test_start - 1]["Close"]
    )

    # İlk test gününün açılış pozisyonu, test öncesi
    # kapanışta oluşan sinyale göre belirlenir.
    initial_position = int(
        predictions.iloc[0]["Target_Position"]
    )

    test.loc[0, "Applied_Position"] = initial_position

    test = calculate_returns(
        test,
        previous_close,
    )

    # Gerçek etiketler yalnızca değerlendirme için kullanılır.
    test["Target"] = df.iloc[
        test_start:
    ]["Target"].to_numpy()

    y_true = test["Target"].astype(int)

    y_pred = (
        test["Probability_Up"] >= 0.5
    ).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    print(
        f"\nEğitim/test sınırı: "
        f"{df.iloc[test_start]['Date'].date()}"
    )
    print(f"Test başlangıcı: {test['Date'].min().date()}")
    print(f"Test bitişi: {test['Date'].max().date()}")
    print(f"Test satırı sayısı: {len(test)}")

    print("\n--- TAHMİN PERFORMANSI ---")
    print(f"Accuracy: %{accuracy * 100:.2f}")
    print(
        f"Balanced accuracy: "
        f"%{balanced_accuracy * 100:.2f}"
    )

    print("\n--- SINIFLANDIRMA RAPORU ---")
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

    print("\n--- KARIŞIKLIK MATRİSİ ---")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
    )

    comparison_rows = []

    for name, return_column in [
        ("Model Stratejisi", "Strategy_Return"),
        ("Al ve Tut", "Buy_Hold_Return"),
        ("Nakit", "Cash_Return"),
    ]:
        metrics = calculate_metrics(
            test[return_column]
        )

        metrics["Strateji"] = name
        comparison_rows.append(metrics)

    comparison = pd.DataFrame(comparison_rows)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n--- STRATEJİ KARŞILAŞTIRMASI ---")
    print(comparison.to_string(index=False))

    print(
        "\nPozisyon değişimi sayısı: "
        f"{int(test['Position_Change'].gt(0).sum())}"
    )

    print(f"\nDetaylı sonuç: {RESULTS_PATH}")
    print(f"Karşılaştırma: {COMPARISON_PATH}")

    print(
        "\nUYARI: Bu bir tarihsel simülasyondur. "
        "Fiyat kayması, likidite ve emir yürütme koşulları "
        "ayrıca modellenmemiştir. Geçmiş performans gelecekteki "
        "getiriyi garanti etmez."
    )


if __name__ == "__main__":
    run_backtest()
