
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# FINANCEAI - KRİPTO BACKTEST V2
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "crypto_raw"
DATA_DIR = BASE_DIR / "data" / "crypto_processed" / "walk_forward"

PREDICTIONS_PATH = DATA_DIR / "crypto_walk_forward_predictions.csv"

RESULTS_PATH = DATA_DIR / "crypto_backtest_results.csv"
SUMMARY_PATH = DATA_DIR / "crypto_backtest_summary.csv"
CURVES_PATH = DATA_DIR / "crypto_backtest_equity_curves.csv"

INITIAL_CAPITAL = 100_000.0

# Varsayımsal maliyetler; gerçek borsa ücretlerini kontrol et.
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

# 20 günlük pozitif getiri olasılığına dayalı sinyaller.
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45


# =========================================================
# VERİ OKUMA
# =========================================================

def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Tahmin dosyası bulunamadı: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["Date"],
    )

    required = [
        "Symbol",
        "Model",
        "Fold",
        "Date",
        "Predicted_Up_Probability",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik tahmin sütunları: {missing}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Fold"] = pd.to_numeric(df["Fold"], errors="coerce")
    df["Predicted_Up_Probability"] = pd.to_numeric(
        df["Predicted_Up_Probability"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Symbol",
            "Model",
            "Fold",
            "Date",
            "Predicted_Up_Probability",
        ]
    )

    df = df[
        df["Predicted_Up_Probability"].between(0, 1)
    ].copy()

    df["Fold"] = df["Fold"].astype(int)

    return df.sort_values(
        ["Symbol", "Model", "Fold", "Date"]
    ).reset_index(drop=True)


def load_prices(symbol):
    path = RAW_DIR / f"{symbol}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Fiyat dosyası bulunamadı: {path}")

    prices = pd.read_csv(path, parse_dates=["Date"])

    required = ["Date", "Open", "Close"]
    missing = [c for c in required if c not in prices.columns]

    if missing:
        raise ValueError(f"{symbol}: Eksik fiyat sütunları: {missing}")

    prices["Date"] = pd.to_datetime(
        prices["Date"]
    ).dt.normalize()

    for column in ["Open", "Close"]:
        prices[column] = pd.to_numeric(
            prices[column], errors="coerce"
        )

    prices = (
        prices.dropna(subset=["Date", "Open", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    prices = prices[
        (prices["Open"] > 0) & (prices["Close"] > 0)
    ].copy()

    # Sinyal günü t kapanışından sonra bilinir.
    # İşlem fiyatı t+1 gününün açılışıdır.
    prices["Execution_Open"] = prices["Open"].shift(-1)
    prices["Execution_Date"] = prices["Date"].shift(-1)

    # Ertesi günün kapanışı, portföy değerlemesi için kullanılır.
    prices["Execution_Close"] = prices["Close"].shift(-1)

    return prices[
        ["Date", "Execution_Date", "Execution_Open", "Execution_Close"]
    ]


# =========================================================
# STRATEJİ VE AL-TUT
# =========================================================

def simulate_group(predictions, prices):
    df = predictions.merge(
        prices,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    df = df.dropna(
        subset=[
            "Execution_Date",
            "Execution_Open",
            "Execution_Close",
        ]
    ).copy()

    df = df.sort_values("Execution_Date").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Backtest için yeterli test günü bulunamadı.")

    if df["Execution_Date"].duplicated().any():
        raise ValueError("Test döneminde yinelenen işlem tarihi var.")

    if (df["Execution_Open"] <= 0).any() or (
        df["Execution_Close"] <= 0
    ).any():
        raise ValueError("Geçersiz işlem fiyatı bulundu.")

    cash = INITIAL_CAPITAL
    quantity = 0.0
    in_position = False

    trades = 0
    buys = 0
    sells = 0
    curve = []

    for _, row in df.iterrows():
        probability = float(row["Predicted_Up_Probability"])
        execution_open = float(row["Execution_Open"])
        execution_close = float(row["Execution_Close"])
        execution_date = row["Execution_Date"]

        # Sinyal önceki günün kapanışında oluştu.
        # İşlem bu günün açılışında uygulanıyor.
        if not in_position and probability >= BUY_THRESHOLD:
            effective_buy = execution_open * (1 + SLIPPAGE)

            quantity = cash / (
                effective_buy * (1 + TRANSACTION_COST)
            )

            cash -= quantity * effective_buy * (
                1 + TRANSACTION_COST
            )

            in_position = True
            trades += 1
            buys += 1

        elif in_position and probability <= SELL_THRESHOLD:
            effective_sell = execution_open * (1 - SLIPPAGE)

            cash += quantity * effective_sell * (
                1 - TRANSACTION_COST
            )

            quantity = 0.0
            in_position = False
            trades += 1
            sells += 1

        # Gün sonu portföy değeri.
        equity = cash + quantity * execution_close

        curve.append({
            "Date": execution_date,
            "Equity": equity,
            "Position": int(in_position),
            "Probability": probability,
            "Execution_Open": execution_open,
            "Close": execution_close,
        })

    curve = pd.DataFrame(curve)

    # Son gün açık pozisyonu kapat.
    # Son gün kapanışı, son test gününün gerçekleşen kapanışıdır.
    if in_position:
        last_close = float(curve.iloc[-1]["Close"])
        effective_sell = last_close * (1 - SLIPPAGE)

        cash += quantity * effective_sell * (
            1 - TRANSACTION_COST
        )

        quantity = 0.0
        curve.loc[curve.index[-1], "Equity"] = cash
        curve.loc[curve.index[-1], "Position"] = 0

    curve["Equity"] = curve["Equity"].astype(float)

    # Başlangıç sermayesini de eğriye ekle.
    first_date = pd.to_datetime(curve["Date"].iloc[0])

    initial_row = pd.DataFrame([{
        "Date": first_date - pd.Timedelta(days=1),
        "Equity": INITIAL_CAPITAL,
        "Position": 0,
        "Probability": np.nan,
        "Execution_Open": np.nan,
        "Close": np.nan,
    }])

    curve = pd.concat(
        [initial_row, curve],
        ignore_index=True,
    )

    equity = curve["Equity"]
    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min() * 100

    daily_returns = equity.pct_change().dropna()

    # Günlük kripto verisi için 365 varsayımı.
    if len(daily_returns) >= 2 and daily_returns.std(ddof=1) > 0:
        sharpe = (
            daily_returns.mean() / daily_returns.std(ddof=1)
        ) * np.sqrt(365)
    else:
        sharpe = np.nan

    # Aynı dönem için al-tut: ilk uygulanabilir açılışta al,
    # son test gününün kapanışında sat.
    first_open = float(df.iloc[0]["Execution_Open"])
    last_close = float(df.iloc[-1]["Execution_Close"])

    effective_buy = first_open * (1 + SLIPPAGE)
    buy_quantity = INITIAL_CAPITAL / (
        effective_buy * (1 + TRANSACTION_COST)
    )

    effective_sell = last_close * (1 - SLIPPAGE)

    buy_hold_final = buy_quantity * effective_sell * (
        1 - TRANSACTION_COST
    )

    buy_hold_return = (
        buy_hold_final / INITIAL_CAPITAL - 1
    ) * 100

    return {
        "Initial_Capital": INITIAL_CAPITAL,
        "Final_Equity": float(equity.iloc[-1]),
        "Total_Return_Pct": float(total_return),
        "Max_Drawdown_Pct": float(max_drawdown),
        "Sharpe_Ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Trades": trades,
        "Buy_Count": buys,
        "Sell_Count": sells,
        "Test_Days": len(df),
        "Start_Date": str(df["Execution_Date"].iloc[0].date()),
        "End_Date": str(df["Execution_Date"].iloc[-1].date()),
        "BuyHold_Final_Equity": float(buy_hold_final),
        "BuyHold_Total_Return_Pct": float(buy_hold_return),
        "Excess_Return_Pct": float(total_return - buy_hold_return),
    }, curve


# =========================================================
# TÜM KRİPTOLAR VE MODELLER
# =========================================================

def main():
    predictions = load_predictions()

    all_results = []
    all_curves = []
    failures = []

    groups = predictions.groupby(
        ["Symbol", "Model", "Fold"],
        sort=True,
    )

    total_groups = len(groups)

    for index, ((symbol, model_name, fold), group) in enumerate(
        groups, start=1
    ):
        print(
            f"[{index}/{total_groups}] "
            f"{symbol} | {model_name} | Fold {fold}"
        )

        try:
            prices = load_prices(symbol)

            metrics, curve = simulate_group(group, prices)

            all_results.append({
                "Symbol": symbol,
                "Model": model_name,
                "Fold": fold,
                **metrics,
                "Transaction_Cost": TRANSACTION_COST,
                "Slippage": SLIPPAGE,
                "Buy_Threshold": BUY_THRESHOLD,
                "Sell_Threshold": SELL_THRESHOLD,
            })

            curve["Symbol"] = symbol
            curve["Model"] = model_name
            curve["Fold"] = fold
            all_curves.append(curve)

        except Exception as error:
            print(f"HATA: {symbol}, fold {fold}: {error}")

            failures.append({
                "Symbol": symbol,
                "Model": model_name,
                "Fold": fold,
                "Error": str(error),
            })

    if not all_results:
        raise RuntimeError("Backtest sonucu üretilemedi.")

    results = pd.DataFrame(all_results)

    results.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    curves = pd.concat(all_curves, ignore_index=True)

    curves.to_csv(
        CURVES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary_columns = [
        "Total_Return_Pct",
        "Max_Drawdown_Pct",
        "Sharpe_Ratio",
        "Trades",
        "BuyHold_Total_Return_Pct",
        "Excess_Return_Pct",
    ]

    summary = (
        results.groupby(["Symbol", "Model"])[summary_columns]
        .mean()
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("KRİPTO BACKTEST V2 - GENEL ORTALAMALAR")
    print("=" * 70)

    print(
        results.groupby("Model")[summary_columns]
        .mean()
        .round(4)
        .to_string()
    )

    print(f"\nBaşarılı test grubu: {len(results)}")
    print(f"Başarısız test grubu: {len(failures)}")
    print(f"Detaylı sonuç: {RESULTS_PATH}")
    print(f"Hisse bazlı özet: {SUMMARY_PATH}")
    print(f"Özsermaye eğrileri: {CURVES_PATH}")

    if failures:
        print("\nHatalar:")
        print(pd.DataFrame(failures).to_string(index=False))

    print(
        "\nUYARI: Geçmiş simülasyon gerçek piyasa gerçekleşmesini "
        "garanti etmez. Ücret ve kayma varsayımları örnektir."
    )


if __name__ == "__main__":
    main()
