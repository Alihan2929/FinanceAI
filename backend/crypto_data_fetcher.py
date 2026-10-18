
from pathlib import Path
import time

import pandas as pd
import requests


# =========================================================
# FINANCEAI - KRİPTO VERİ TOPLAMA
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "crypto_raw"

BASE_URL = "https://data-api.binance.vision"
INTERVAL = "1d"
YEARS_OF_HISTORY = 5
LIMIT = 1000
REQUEST_TIMEOUT = 20

CRYPTO_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "UNIUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "ARBUSDT",
    "OPUSDT",
]

COLUMNS = [
    "Open_Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close_Time",
    "Quote_Volume",
    "Trades",
    "Taker_Buy_Base_Volume",
    "Taker_Buy_Quote_Volume",
    "Ignore",
]


def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "FinanceAI/1.0"
    })
    return session


def get_available_symbols(session):
    """Spot piyasasında işlem gören USDT paritelerini kontrol et."""
    url = f"{BASE_URL}/api/v3/exchangeInfo"

    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    return {
        item["symbol"]
        for item in data.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("isSpotTradingAllowed", True)
    }


def fetch_klines(session, symbol, start_time):
    """Tüm geçmiş günlük mumları sayfalar hâlinde indir."""
    url = f"{BASE_URL}/api/v3/klines"

    rows = []
    current_start = start_time

    while True:
        response = session.get(
            url,
            params={
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": current_start,
                "limit": LIMIT,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        rows.extend(batch)

        # Sonraki istek son mumun açılış zamanından ilerler.
        next_start = int(batch[-1][0]) + 1

        if next_start <= current_start:
            raise RuntimeError(
                f"{symbol}: İndirme ilerleyemedi."
            )

        current_start = next_start

        if len(batch) < LIMIT:
            break

        time.sleep(0.15)

    return rows


def download_crypto(session, symbol, start_time):
    rows = fetch_klines(
        session=session,
        symbol=symbol,
        start_time=start_time,
    )

    if not rows:
        raise ValueError("API veri döndürmedi.")

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Binance zamanları milisaniye cinsindedir.
    df["Date"] = pd.to_datetime(
        df["Open_Time"],
        unit="ms",
        utc=True,
    ).dt.tz_localize(None)

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote_Volume",
        "Trades",
        "Taker_Buy_Base_Volume",
        "Taker_Buy_Quote_Volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    df = df.dropna(
        subset=["Date", "Open", "High", "Low", "Close", "Volume"]
    )

    # Tamamlanmamış son günlük mumu veri setine ekleme.
    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    df = df[df["Date"] < today_utc.tz_localize(None)]

    if df.empty:
        raise ValueError("Temizleme sonrası veri kalmadı.")

    # Her kriptonun dosyası ayrı tutulur.
    output_path = OUTPUT_DIR / f"{symbol}.csv"

    # Model hattında kolay kullanım için standart sütunları başa al.
    output_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote_Volume",
        "Trades",
        "Taker_Buy_Base_Volume",
        "Taker_Buy_Quote_Volume",
        "Open_Time",
        "Close_Time",
    ]

    df[output_columns].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"OK | {symbol} | {len(df)} mum | "
        f"{df['Date'].iloc[0].date()} - "
        f"{df['Date'].iloc[-1].date()}"
    )

    return {
        "Symbol": symbol,
        "Rows": len(df),
        "First_Date": str(df["Date"].iloc[0].date()),
        "Last_Date": str(df["Date"].iloc[-1].date()),
        "File": str(output_path),
        "Status": "OK",
        "Error": "",
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = get_session()

    # Yaklaşık beş yıllık geçmiş.
    start_date = (
        pd.Timestamp.now(tz="UTC")
        - pd.DateOffset(years=YEARS_OF_HISTORY)
    )
    start_time = int(start_date.timestamp() * 1000)

    print("=" * 65)
    print("FINANCEAI - KRİPTO VERİ TOPLAMA")
    print("=" * 65)
    print(f"Parite sayısı: {len(CRYPTO_SYMBOLS)}")
    print(f"Mum aralığı: {INTERVAL}")
    print(f"Çıktı klasörü: {OUTPUT_DIR}")
    print()

    try:
        available_symbols = get_available_symbols(session)
    except Exception as error:
        raise RuntimeError(
            "Binance piyasa bilgisi alınamadı. "
            "İnternet bağlantısını ve API erişimini kontrol et."
        ) from error

    results = []

    for index, symbol in enumerate(CRYPTO_SYMBOLS, start=1):
        print(f"[{index}/{len(CRYPTO_SYMBOLS)}] {symbol}")

        if symbol not in available_symbols:
            message = "USDT spot paritesi bulunamadı veya işlemde değil."
            print(f"ATLANDI | {symbol} | {message}")

            results.append({
                "Symbol": symbol,
                "Rows": 0,
                "First_Date": "",
                "Last_Date": "",
                "File": "",
                "Status": "SKIPPED",
                "Error": message,
            })
            continue

        try:
            result = download_crypto(
                session=session,
                symbol=symbol,
                start_time=start_time,
            )
            results.append(result)

        except Exception as error:
            print(f"HATA | {symbol} | {error}")

            results.append({
                "Symbol": symbol,
                "Rows": 0,
                "First_Date": "",
                "Last_Date": "",
                "File": "",
                "Status": "FAILED",
                "Error": str(error),
            })

        time.sleep(0.2)

    summary = pd.DataFrame(results)

    summary_path = OUTPUT_DIR / "download_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 65)
    print("İNDİRME ÖZETİ")
    print("=" * 65)
    print(summary.to_string(index=False))
    print()
    print(f"Başarılı: {(summary['Status'] == 'OK').sum()}")
    print(f"Atlanan: {(summary['Status'] == 'SKIPPED').sum()}")
    print(f"Başarısız: {(summary['Status'] == 'FAILED').sum()}")
    print(f"Özet dosyası: {summary_path}")


if __name__ == "__main__":
    main()
