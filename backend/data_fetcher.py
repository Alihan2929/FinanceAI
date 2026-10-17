import yfinance as yf
from pathlib import Path

# Proje içindeki veri klasörünü belirle
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_stock_data(ticker: str, period: str = "5y"):
    """Hissenin geçmiş fiyat verilerini indirir."""

    stock = yf.Ticker(ticker)
    data = stock.history(period=period, auto_adjust=True)

    if data.empty:
        print(f"{ticker} için veri bulunamadı.")
        return

    # Gereksiz saat dilimi bilgisini kaldır
    data.index = data.index.tz_localize(None)

    file_path = DATA_DIR / f"{ticker.replace('.', '_')}.csv"
    data.to_csv(file_path)

    print(f"Veriler kaydedildi: {file_path}")
    print(data.tail())


if __name__ == "__main__":
    fetch_stock_data("TUPRS.IS")