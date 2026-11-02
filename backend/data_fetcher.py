
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


# =========================================================
# AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PERIOD = "5y"

# Son verinin kaç takvim günü geride olması uyarı versin?
# Hafta sonu ve resmî tatiller için tolerans bırakıyoruz.
MAX_DATA_AGE_DAYS = 7

# Yahoo Finance geçici hata, boş yanıt veya rate limit (çok fazla istek)
# döndürebilir. Bu yüzden indirmeyi birkaç kez, artan bekleme süresiyle
# tekrar deneriz ve hisseler arasında kısa bir ara veririz.
DOWNLOAD_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 3.0
DELAY_BETWEEN_TICKERS_SECONDS = 1.0

BIST100_TICKERS = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKSA.IS",
    "AKSEN.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS",
    "ASTOR.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS",
    "CCOLA.IS", "CIMSA.IS", "DOAS.IS", "DOHOL.IS",
    "ECILC.IS", "EGEEN.IS", "EKGYO.IS", "ENERY.IS",
    "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS",
    "GARAN.IS", "GESAN.IS", "GUBRF.IS", "HALKB.IS",
    "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS",
    "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
    "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS",
    "SKBNK.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS",
    "TTKOM.IS", "TTRAK.IS", "TUPRS.IS", "ULKER.IS",
    "VAKBN.IS", "VESTL.IS", "YKBNK.IS",
]


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================

def get_file_path(ticker: str) -> Path:
    """Ham fiyat dosyasının yolunu döndürür."""
    safe_ticker = ticker.replace(".", "_")
    return DATA_DIR / f"{safe_ticker}.csv"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo Finance verisini standart tarih ve sütun formatına getirir."""

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # yfinance bazı sürümlerde MultiIndex sütunlar döndürebilir.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            next(
                (
                    str(level)
                    for level in reversed(column)
                    if str(level) in {
                        "Open", "High", "Low", "Close",
                        "Volume", "Dividends",
                        "Stock Splits", "Capital Gains",
                    }
                ),
                str(column[-1]),
            )
            for column in df.columns
        ]

    # Tarih indeksini normal bir sütuna dönüştür.
    df.index = pd.to_datetime(df.index, errors="coerce")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"
    df = df.reset_index()

    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    ).dt.normalize()

    df = df.dropna(subset=["Date"])

    # Sayısal sütunları doğrula.
    for column in df.columns:
        if column != "Date":
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Kapanış fiyatı bulunmayan satırlar kullanılamaz.
    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    df = (
        df.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    return df


def merge_price_data(
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    """Eski ve yeni fiyat verilerini tarihe göre birleştirir."""

    frames = []

    if old_data is not None and not old_data.empty:
        frames.append(old_data)

    if new_data is not None and not new_data.empty:
        frames.append(new_data)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return normalize_dataframe(
        merged.set_index("Date")
    )


def get_latest_date(df: pd.DataFrame):
    if df.empty:
        return None

    return pd.Timestamp(df["Date"].max()).date()


def check_data_freshness(df: pd.DataFrame, ticker: str) -> bool:
    """Son fiyat tarihinin ne kadar eski olduğunu raporlar."""

    latest_date = get_latest_date(df)

    if latest_date is None:
        print(f"UYARI: {ticker} için son veri tarihi bulunamadı.")
        return False

    today = datetime.now().date()
    age_days = (today - latest_date).days

    print(f"Son veri tarihi: {latest_date}")
    print(f"Bugüne göre takvim günü farkı: {age_days}")

    if age_days > MAX_DATA_AGE_DAYS:
        print(
            f"UYARI: {ticker} verisi eski görünüyor. "
            "Piyasa tatili, veri sağlayıcı gecikmesi veya indirme "
            "sorunu olup olmadığını kontrol et."
        )
        return False

    return True


# =========================================================
# YAHOO FINANCE İNDİRME (YENİDEN DENEMELİ)
# =========================================================

def download_history(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    retries: int = DOWNLOAD_RETRIES,
) -> pd.DataFrame:
    """
    Yahoo Finance'ten günlük fiyatları indirir.

    Boş yanıt veya hata durumunda artan bekleme süresiyle yeniden dener.
    Başarısız olursa gerçek hata nedeniyle birlikte boş DataFrame döndürür.
    """

    last_reason = "bilinmeyen neden"

    for attempt in range(1, retries + 1):
        try:
            downloaded = yf.Ticker(ticker).history(
                period=period,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

            data = normalize_dataframe(downloaded)

            if not data.empty:
                return data

            last_reason = (
                "Yahoo Finance boş veri döndürdü (rate limit, "
                "geçersiz sembol veya geçici sorun olabilir)"
            )

        except Exception as error:
            last_reason = f"{type(error).__name__}: {error}"

        if attempt < retries:
            wait_seconds = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

            print(
                f"{ticker}: deneme {attempt}/{retries} başarısız "
                f"({last_reason}). {wait_seconds:.0f} sn sonra "
                "tekrar denenecek."
            )

            time.sleep(wait_seconds)

    print(
        f"HATA: {ticker} {retries} denemede indirilemedi - {last_reason}"
    )

    return pd.DataFrame()


# =========================================================
# TEK HİSSE VERİSİ GÜNCELLEME
# =========================================================

def fetch_stock_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
):
    """
    Yahoo Finance'ten geçmiş fiyatları alır,
    mevcut ham verilerle birleştirir ve kaydeder.

    NOT: Bu işlem model özelliklerini veya hedef değişkenleri
    içeren *_dataset.csv dosyalarını yeniden oluşturmaz.
    """

    file_path = get_file_path(ticker)

    try:
        print(f"\nVeri güncelleniyor: {ticker}")

        new_data = download_history(ticker, period)

        if new_data.empty:
            # İndirme başarısız: mevcut dosya olduğu gibi korunur, ancak
            # hisse "başarılı" sayılmaz; böylece rapor gerçeği gösterir.
            print(
                f"{ticker} için yeni veri alınamadı. "
                "Mevcut dosya (varsa) korunuyor."
            )

            return None

        # Önceden indirilmiş ham verileri oku.
        old_data = pd.DataFrame()

        if file_path.exists():
            try:
                old_data = pd.read_csv(
                    file_path,
                    parse_dates=["Date"],
                )

                old_data = normalize_dataframe(
                    old_data.set_index("Date")
                )

            except Exception as error:
                print(
                    f"Eski dosya okunamadı ({ticker}): {error}"
                )
                print("İndirilen yeni veriler kullanılacak.")

        # Eski kayıtları koru, yeni kayıtları ekle.
        merged = merge_price_data(old_data, new_data)

        if merged.empty:
            print(f"HATA: {ticker} için kaydedilebilir veri yok.")
            return None

        # Önce geçici dosyaya yaz; yazma başarılı olursa değiştir.
        temporary_path = file_path.with_suffix(".tmp")

        merged.to_csv(
            temporary_path,
            index=False,
            date_format="%Y-%m-%d",
        )

        temporary_path.replace(file_path)

        print(f"Kaydedildi: {file_path}")
        print(f"Toplam kayıt: {len(merged)}")
        print(f"İndirilen kayıt: {len(new_data)}")

        is_fresh = check_data_freshness(merged, ticker)

        print(
            "Güncellik kontrolü:",
            "OK" if is_fresh else "KONTROL GEREKLİ",
        )

        return merged

    except Exception as error:
        print(f"{ticker} güncellenirken hata: {error}")
        print("Mevcut dosya varsa korunmuştur.")
        return None


# =========================================================
# BİRDEN FAZLA HİSSE GÜNCELLEME
# =========================================================

def fetch_multiple_stocks(
    tickers: list[str],
    period: str = DEFAULT_PERIOD,
):
    successful = []
    failed = []
    stale = []

    total = len(tickers)

    for index, ticker in enumerate(tickers, start=1):
        print("\n" + "=" * 55)
        print(f"[{index}/{total}] {ticker}")
        print("=" * 55)

        data = fetch_stock_data(ticker, period)

        if data is None or data.empty:
            failed.append(ticker)
        else:
            successful.append(ticker)

            # Güncellik mesajı fetch_stock_data içinde zaten yazdırıldı;
            # burada yalnızca rapor için sessizce kontrol edilir.
            latest_date = get_latest_date(data)

            if (
                latest_date is None
                or (datetime.now().date() - latest_date).days
                > MAX_DATA_AGE_DAYS
            ):
                stale.append(ticker)

        # Yahoo Finance'in rate limit uygulamaması için kısa ara.
        if index < total:
            time.sleep(DELAY_BETWEEN_TICKERS_SECONDS)

    print("\n" + "=" * 55)
    print("VERİ GÜNCELLEME RAPORU")
    print("=" * 55)
    print(f"Toplam hisse: {total}")
    print(f"Başarılı: {len(successful)}")
    print(f"Başarısız: {len(failed)}")
    print(f"Eski veri uyarısı: {len(stale)}")
    print(f"Veri klasörü: {DATA_DIR}")

    if failed:
        print("\nİndirilemeyen hisseler:")
        for ticker in failed:
            print(f"- {ticker}")

    if stale:
        print("\nSon verisi eski görünen hisseler:")
        for ticker in stale:
            print(f"- {ticker}")

    return successful, failed


# =========================================================
# PROGRAMI ÇALIŞTIR
# =========================================================

if __name__ == "__main__":
    fetch_multiple_stocks(
        BIST100_TICKERS,
        period=DEFAULT_PERIOD,
    )
