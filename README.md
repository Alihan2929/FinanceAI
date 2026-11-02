# FinanceAI

BIST hisseleri, ABD borsası hisseleri ve kripto paralar için makine öğrenmesi tabanlı **20 işlem günlük yön tahmini** yapan bir web uygulaması. Arka uçta FastAPI, ön yüzde React (Vite) kullanılır.

> **Uyarı:** Bu proje eğitim amaçlıdır. Üretilen tahminler yatırım tavsiyesi değildir ve geçmiş performans gelecekteki getiriyi garanti etmez.

## Özellikler

- **BIST100** hisseleri (ör. `THYAO.IS`, `AKBNK.IS`)
- **ABD hisseleri** (NASDAQ ve NYSE; ör. `AAPL`, `MSFT`, `TTWO`, `JPM`)
- **Kripto paralar** (Binance verisiyle)
- Her varlık için üç model: Logistic Regression, Random Forest, XGBoost
- Fiyat geçmişi grafikleri, tahmin sonucu ve model performans özetleri
- Teknik göstergelerden üretilen özellikler: getiriler, SMA/EMA oranları, RSI, MACD, Bollinger Bands, hacim değişimleri, OBV, volatilite, ATR

## Proje yapısı

```
FinanceAI/
├── backend/
│   ├── app/main.py              # FastAPI uygulaması (tüm endpoint'ler)
│   ├── data_fetcher.py          # Yahoo Finance'ten fiyat indirme (BIST100 listesi burada)
│   ├── us_tickers.py            # ABD hisse listesi (NASDAQ / NYSE)
│   ├── feature_engineering.py   # Teknik özelliklerin hesaplanması
│   ├── target_generation.py     # 20 günlük hedef değişkeni
│   ├── refresh_data.py          # BIST veri güncelleme hattı
│   ├── refresh_us_data.py       # ABD veri güncelleme hattı
│   ├── train_model.py           # BIST model eğitimi
│   ├── train_us_model.py        # ABD model eğitimi
│   ├── crypto_*.py              # Kripto veri, özellik, eğitim ve backtest
│   ├── data/                    # Ham veri, özellik ve veri setleri (CSV)
│   └── models/                  # Eğitilmiş modeller (.joblib); ABD için models/us
├── frontend/                    # React + Vite arayüzü
└── requirements.txt
```

## Kurulum

### Gereksinimler

- Python 3.11+
- Node.js 18+

### Arka uç

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

cd backend
uvicorn app.main:app --reload
```

API varsayılan olarak `http://127.0.0.1:8000` adresinde çalışır. Etkileşimli dokümantasyon: `http://127.0.0.1:8000/docs`

### Ön yüz

```bash
cd frontend
npm install
npm run dev
```

Ön yüz API adresini `frontend/src/App.jsx` içindeki `API_URL` sabitinden okur (varsayılan `http://127.0.0.1:8000`).

## Veri ve model hattı

Komutlar `backend/` klasöründen çalıştırılır.

**BIST:**

```bash
python refresh_data.py     # veri indir, özellikleri ve hedefleri üret
python train_model.py      # modelleri eğit
```

**ABD hisseleri:**

```bash
python refresh_us_data.py  # veri indir, özellikleri ve hedefleri üret
python train_us_model.py   # modelleri eğit
```

Veri indirme Yahoo Finance üzerinden yapılır. Geçici hata veya rate limit durumunda her hisse 3 kez, artan bekleme süresiyle yeniden denenir; başarısız hisseler raporda ayrıca listelenir ve mevcut dosyaları korunur.

### Yeni ABD hissesi ekleme

1. `backend/us_tickers.py` içindeki `NASDAQ_TICKERS` veya `NYSE_TICKERS` listesine sembolü ekle.
2. `python refresh_us_data.py` ile verisini indir.
3. `python train_us_model.py` ile modellerini eğit.

`/us/stocks` yalnızca modeli eğitilmiş hisseleri listeler; 3. adım yapılmadan hisse arayüzde görünmez.

## API özeti

| Endpoint | Açıklama |
| --- | --- |
| `GET /health` | Servis durumu |
| `GET /stocks` | BIST hisse listesi |
| `GET /stocks/{symbol}/history` | BIST fiyat geçmişi |
| `GET /predict/{symbol}` | BIST tahmini |
| `GET /us/stocks` | ABD hisse listesi |
| `GET /us/stocks/{symbol}/history` | ABD fiyat geçmişi |
| `GET /us/predict/{symbol}?model=random_forest` | ABD tahmini (`random_forest`, `logistic_regression`, `xgboost`) |
| `GET /crypto` | Kripto listesi |
| `GET /crypto/{symbol}/history` | Kripto fiyat geçmişi |
| `GET /crypto/predict/{symbol}` | Kripto tahmini |
| `GET /crypto/live/{symbol}` | Anlık kripto fiyatı |
| `GET /models` | Bir varlık için kullanılabilir modeller |
| `GET /model-performance` | Model performans özeti |

## Teknolojiler

- **Arka uç:** Python, FastAPI, pandas, NumPy, scikit-learn, XGBoost, yfinance, joblib
- **Ön yüz:** React, Vite, axios

## Lisans

Henüz bir lisans belirtilmemiştir.
