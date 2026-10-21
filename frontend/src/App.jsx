import { useCallback, useEffect, useMemo, useState } from "react";

import axios from "axios";

import {

  Activity,

  ArrowUpRight,

  BarChart3,

  CheckCircle2,

  CircleDollarSign,

  Cpu,

  Layers,

  LoaderCircle,

  RefreshCw,

  ShieldCheck,

  TrendingUp,

  Wifi,

  WifiOff,

} from "lucide-react";

import {

  Area,

  AreaChart,

  CartesianGrid,

  ResponsiveContainer,

  Tooltip,

  XAxis,

  YAxis,

} from "recharts";

import "./App.css";



const API_URL = "https://financeai-60j8.onrender.com";

const LIVE_REFRESH_MS = 5000;



const CRYPTO_FALLBACK = [

  "BTCUSDT",

  "ETHUSDT",

  "BNBUSDT",

  "SOLUSDT",

  "XRPUSDT",

  "ADAUSDT",

  "DOGEUSDT",

  "AVAXUSDT",

  "DOTUSDT",

  "LINKUSDT",

  "TRXUSDT",

  "LTCUSDT",

  "BCHUSDT",

  "ATOMUSDT",

  "NEARUSDT",

  "APTUSDT",

  "ETCUSDT",

  "FILUSDT",

  "ARBUSDT",

  "OPUSDT",

];



function formatPrice(value, currency = "TL") {

  const number = Number(value);



  if (value == null || value === "" || !Number.isFinite(number)) {

    return "—";

  }



  return `${number.toLocaleString("tr-TR", {

    minimumFractionDigits: 2,

    maximumFractionDigits: 4,

  })} ${currency}`;

}



function formatProbability(value) {

  const number = Number(value);



  if (value == null || value === "" || !Number.isFinite(number)) {

    return "—";

  }



  return `%${(number * 100).toLocaleString("tr-TR", {

    minimumFractionDigits: 1,

    maximumFractionDigits: 1,

  })}`;

}



function formatNumber(value, digits = 2) {

  const number = Number(value);



  if (value == null || value === "" || !Number.isFinite(number)) {

    return "—";

  }



  return number.toLocaleString("tr-TR", {

    minimumFractionDigits: digits,

    maximumFractionDigits: digits,

  });

}



function formatDateTime(value) {

  if (!value) return "—";



  const date = new Date(value);



  if (Number.isNaN(date.getTime())) {

    return String(value);

  }



  return date.toLocaleString("tr-TR", {

    day: "2-digit",

    month: "2-digit",

    year: "numeric",

    hour: "2-digit",

    minute: "2-digit",

    second: "2-digit",

  });

}



function normalizeSymbol(value) {

  return String(value || "")

    .trim()

    .toUpperCase()

    .replace(/\.IS$/i, "")

    .replace(/[^A-Z0-9]/g, "");

}



function App() {

  const [market, setMarket] = useState("stocks");

  const [stocks, setStocks] = useState([]);

  const [cryptos, setCryptos] = useState(

    CRYPTO_FALLBACK.map((symbol) => ({ symbol, name: symbol }))

  );



  const [selectedSymbol, setSelectedSymbol] = useState("TUPRS");

  const [prediction, setPrediction] = useState(null);
  const [selectedModel, setSelectedModel] = useState("random_forest");
  const [availableModels, setAvailableModels] = useState([
    { value: "random_forest", label: "Random Forest" },
  ]);
  const [modelsLoading, setModelsLoading] = useState(false);

  const [priceHistory, setPriceHistory] = useState([]);

  const [performance, setPerformance] = useState(null);



  const [livePrice, setLivePrice] = useState(null);

  const [livePriceTimestamp, setLivePriceTimestamp] = useState(null);

  const [livePriceSource, setLivePriceSource] = useState("");

  const [livePriceError, setLivePriceError] = useState("");

  const [livePriceLoading, setLivePriceLoading] = useState(false);



  const [apiOnline, setApiOnline] = useState(false);

  const [loading, setLoading] = useState(false);

  const [historyLoading, setHistoryLoading] = useState(false);

  const [performanceLoading, setPerformanceLoading] = useState(false);



  const [error, setError] = useState("");

  const [historyError, setHistoryError] = useState("");

  const [listError, setListError] = useState("");



  const assets = market === "stocks" ? stocks : cryptos;

  const currency = market === "stocks" ? "TL" : "USDT";



  const lastDate =

    prediction?.date ||

    priceHistory[priceHistory.length - 1]?.date ||

    "—";



  const displayedPrice =

    market === "crypto" && livePrice != null

      ? livePrice

      : prediction?.close ?? priceHistory[priceHistory.length - 1]?.close;



  const loadLists = useCallback(async () => {

    setListError("");



    const [stockResult, cryptoResult] = await Promise.allSettled([

      axios.get(`${API_URL}/stocks`, { timeout: 15000 }),

      axios.get(`${API_URL}/crypto`, { timeout: 15000 }),

    ]);



    let connected = false;



    if (stockResult.status === "fulfilled") {

      const body = stockResult.value.data;

      const raw = body?.stocks ?? body?.data ?? body;



      if (Array.isArray(raw)) {

        const normalized = raw

          .map((item) => {

            const symbol = normalizeSymbol(

              item.symbol || item.ticker || ""

            );



            return {

              ...item,

              symbol,

              name: item.name || symbol,

            };

          })

          .filter((item) => item.symbol);



        if (normalized.length > 0) {

          setStocks(normalized);

          connected = true;

        }

      }

    }



    if (cryptoResult.status === "fulfilled") {

      const body = cryptoResult.value.data;

      const raw =

        body?.cryptos ??

        body?.crypto ??

        body?.symbols ??

        body?.data ??

        body;



      if (Array.isArray(raw)) {

        const normalized = raw

          .map((item) => {

            const symbol =

              typeof item === "string"

                ? item

                : item.symbol || item.ticker || item.id || "";



            const normalizedSymbol = String(symbol)

              .trim()

              .toUpperCase()

              .replace("/", "");



            return {

              ...(typeof item === "object" ? item : {}),

              symbol: normalizedSymbol,

              name:

                typeof item === "object"

                  ? item.name || normalizedSymbol

                  : normalizedSymbol,

            };

          })

          .filter((item) => item.symbol);



        if (normalized.length > 0) {

          setCryptos(normalized);

          connected = true;

        }

      }

    }



    setApiOnline(connected);



    if (!connected) {

      setListError(

        "Varlık listesi alınamadı. Backend bağlantısını ve endpoint'leri kontrol et."

      );

    }

  }, []);



  const loadPrediction = useCallback(async (symbol, selectedMarket, modelName = selectedModel) => {

    if (!symbol || !modelName) return;



    setLoading(true);

    setError("");



    try {

      const endpoint =

        selectedMarket === "stocks"

          ? `${API_URL}/predict/${encodeURIComponent(symbol)}`

          : `${API_URL}/crypto/predict/${encodeURIComponent(symbol)}`;



      const response = await axios.get(endpoint, {
        params: { model: modelName },
        timeout: 20000,
      });



      setPrediction(response.data);

      setApiOnline(true);

    } catch (err) {

      setPrediction(null);

      setError(

        err.response?.data?.detail ||

          "Tahmin alınamadı. Model dosyasını ve backend endpoint'ini kontrol et."

      );

      setApiOnline(false);

    } finally {

      setLoading(false);

    }

  }, [selectedModel]);



  const loadAvailableModels = useCallback(async (symbol, selectedMarket) => {
    if (!symbol) return;

    setModelsLoading(true);
    setError("");
    setPrediction(null);

    try {
      const response = await axios.get(`${API_URL}/models`, {
        params: { symbol, market: selectedMarket },
        timeout: 10000,
      });

      const models = Array.isArray(response.data?.models)
        ? response.data.models.filter(
            (item) => item && typeof item.value === "string" && typeof item.label === "string"
          )
        : [];

      setAvailableModels(models);
      setSelectedModel((currentModel) => {
        if (models.some((item) => item.value === currentModel)) return currentModel;
        return response.data?.default || models[0]?.value || "";
      });

      if (!models.length) {
        setError(
          selectedMarket === "crypto"
            ? `${symbol} için eğitilmiş kripto modeli bulunamadı. Mevcut üç model BIST için eğitildiyse, model seçmek için BIST Pay Piyasası'na geç.`
            : `${symbol} için eğitilmiş model dosyası bulunamadı. Model dosyalarının backend/models klasöründe olduğunu kontrol et.`
        );
      }
    } catch (err) {
      setAvailableModels([]);
      setSelectedModel("");
      setError(
        err.response?.status === 404
          ? "Model listesi endpoint'i bulunamadı. backend/app/main.py dosyasını güncelle ve FastAPI'yi yeniden başlat."
          : err.response?.data?.detail || "Kullanılabilir modeller alınamadı. Backend bağlantısını kontrol et."
      );
    } finally {
      setModelsLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async (symbol, selectedMarket) => {

    if (!symbol) return;



    setHistoryLoading(true);

    setHistoryError("");



    try {

      const endpoint =

        selectedMarket === "stocks"

          ? `${API_URL}/stocks/${encodeURIComponent(symbol)}/history`

          : `${API_URL}/crypto/${encodeURIComponent(symbol)}/history`;



      const response = await axios.get(endpoint, {

        params: { limit: 120 },

        timeout: 15000,

      });



      const raw =

        response.data?.prices ??

        response.data?.history ??

        response.data?.data;



      if (!Array.isArray(raw)) {

        throw new Error("Fiyat geçmişi beklenen formatta değil.");

      }



      const normalized = raw

        .map((item) => ({

          date: String(item.date ?? item.Date ?? ""),

          close: Number(item.close ?? item.Close),

        }))

        .filter(

          (item) => item.date && Number.isFinite(item.close)

        )

        .sort((a, b) => a.date.localeCompare(b.date));



      setPriceHistory(normalized);



      if (!normalized.length) {

        setHistoryError("Fiyat geçmişi bulunamadı.");

      }

    } catch (err) {

      setPriceHistory([]);

      setHistoryError(

        err.response?.data?.detail ||

          err.message ||

          "Fiyat geçmişi alınamadı."

      );

    } finally {

      setHistoryLoading(false);

    }

  }, []);



  const loadLivePrice = useCallback(async (symbol) => {

    if (!symbol) return;



    setLivePriceLoading(true);

    setLivePriceError("");



    try {

      const response = await axios.get(

        `${API_URL}/crypto/live/${encodeURIComponent(symbol)}`,

        {

          timeout: 10000,

          headers: {

            "Cache-Control": "no-cache",

          },

          params: {

            _t: Date.now(),

          },

        }

      );



      const body = response.data;



      const price = Number(

        body?.price ??

          body?.lastPrice ??

          body?.last_price ??

          body?.close

      );



      if (!Number.isFinite(price) || price <= 0) {

        throw new Error("Canlı fiyat yanıtı geçerli bir fiyat içermiyor.");

      }



      setLivePrice(price);

      setLivePriceTimestamp(

        body?.timestamp || body?.time || new Date().toISOString()

      );

      setLivePriceSource(body?.source || "Binance");

      setApiOnline(true);

    } catch (err) {

      setLivePriceError(

        err.response?.data?.detail ||

          err.message ||

          "Canlı kripto fiyatı alınamadı."

      );

    } finally {

      setLivePriceLoading(false);

    }

  }, []);



  const loadPerformance = useCallback(async () => {

    setPerformanceLoading(true);



    try {

      const response = await axios.get(`${API_URL}/model-performance`, {

        timeout: 15000,

      });



      setPerformance(response.data);

    } catch {

      setPerformance(null);

    } finally {

      setPerformanceLoading(false);

    }

  }, []);



  useEffect(() => {

    loadLists();

    loadPerformance();



    axios

      .get(`${API_URL}/health`, { timeout: 5000 })

      .then(() => setApiOnline(true))

      .catch(() => setApiOnline(false));

  }, [loadLists, loadPerformance]);



  useEffect(() => {

    if (!assets.length) return;



    if (!assets.some((item) => item.symbol === selectedSymbol)) {

      setSelectedSymbol(assets[0].symbol);

    }

  }, [assets, selectedSymbol]);



  useEffect(() => {
    if (!selectedSymbol) return;
    loadAvailableModels(selectedSymbol, market);
  }, [selectedSymbol, market, loadAvailableModels]);

  useEffect(() => {

    if (!selectedSymbol || !selectedModel) return;



    loadPrediction(selectedSymbol, market, selectedModel);

    loadHistory(selectedSymbol, market);



    if (market === "crypto") {

      setLivePrice(null);

      setLivePriceTimestamp(null);

      setLivePriceSource("");

      setLivePriceError("");

      loadLivePrice(selectedSymbol);

    } else {

      setLivePrice(null);

      setLivePriceTimestamp(null);

      setLivePriceSource("");

      setLivePriceError("");

    }

  }, [

    selectedSymbol,

    market,

    loadPrediction,

    loadHistory,

    loadLivePrice,
    selectedModel,

  ]);



  useEffect(() => {

    if (market !== "crypto" || !selectedSymbol) return;



    const intervalId = window.setInterval(() => {

      loadLivePrice(selectedSymbol);

    }, LIVE_REFRESH_MS);



    return () => window.clearInterval(intervalId);

  }, [market, selectedSymbol, loadLivePrice]);



  const refreshAll = useCallback(() => {

    loadLists();

    if (selectedModel) loadPrediction(selectedSymbol, market, selectedModel);

    loadHistory(selectedSymbol, market);

    loadPerformance();



    if (market === "crypto") {

      loadLivePrice(selectedSymbol);

    }

  }, [

    loadLists,

    loadPrediction,

    loadHistory,

    loadPerformance,

    loadLivePrice,

    selectedSymbol,

    market,
    selectedModel,

  ]);



  const busy = loading || historyLoading || performanceLoading;

  const positiveProbability = prediction?.probability_positive_return;

  const negativeProbability = prediction?.probability_non_positive_return;

  const latestClose = displayedPrice;



  const performanceData = useMemo(() => {
    if (!performance) return null;

    // Bazı backend yanıtları doğrudan { columns, rows } biçimindedir.
    if (Array.isArray(performance.rows) || Array.isArray(performance.results)) {
      return performance;
    }

    const candidates = market === "stocks"
      ? [
          performance.stock_model_comparison,
          performance.stocks_model_comparison,
          performance.model_comparison,
          performance.stock_walk_forward,
          performance.walk_forward,
          performance.stocks,
        ]
      : [
          performance.crypto_model_comparison,
          performance.crypto_walk_forward,
          performance.crypto_backtest,
          performance.crypto_walk_forward_summary,
          performance.crypto_backtest_summary,
          performance.cryptos,
        ];

    return candidates.find((item) =>
      Array.isArray(item) ||
      Array.isArray(item?.rows) ||
      Array.isArray(item?.results) ||
      Array.isArray(item?.data)
    ) ?? performance;
  }, [performance, market]);

  const indicators = prediction?.technical_indicators;



  return (

    <div className="app-shell">

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">

            <TrendingUp size={22} strokeWidth={2.5} />

          </div>

          <div>

            <h2>FinanceAI</h2>

            <span>Intelligence Deck</span>

          </div>

        </div>



        <div className="nav-label">KONTROL MERKEZİ</div>



        <nav className="nav-group">

          <a href="#overview" className="nav-item active">

            <BarChart3 size={18} />

            <span>Genel Bakış</span>

          </a>



          <a href="#analysis" className="nav-item">

            <Activity size={18} />

            <span>Piyasa Analizi</span>

          </a>



          <a href="#indicators" className="nav-item">

            <Layers size={18} />

            <span>Teknik Göstergeler</span>

          </a>



          <a href="#performance" className="nav-item">

            <ShieldCheck size={18} />

            <span>Model Performansı</span>

          </a>

        </nav>



        <div className="sidebar-bottom">

          <div className="system-pill">

            <div className={`status-dot ${apiOnline ? "active" : ""}`} />

            <span>{apiOnline ? "Core API Bağlı" : "Bağlantı Kesik"}</span>

          </div>

          <p>FinanceAI v2.4 · Quant Edition</p>

        </div>

      </aside>



      <main className="main-content" id="overview">

        <header className="topbar">

          <div>

            <div className="eyebrow">

              <span className="dot" /> QUANTITATIVE ANALYTICS

            </div>

            <h1>Piyasa Terminali</h1>

          </div>



          <div className="topbar-right">

            <div className={`status-badge ${apiOnline ? "online" : "offline"}`}>

              {apiOnline ? <Wifi size={14} /> : <WifiOff size={14} />}

              <span>{apiOnline ? "Online" : "Offline"}</span>

            </div>



            <button

              className="refresh-button"

              onClick={refreshAll}

              disabled={busy}

              aria-label="Verileri yenile"

              title="Tüm verileri yenile"

            >

              <RefreshCw size={16} className={busy ? "spin" : ""} />

            </button>

          </div>

        </header>



        <section className="welcome">

          <div className="welcome-content">

            <span className="welcome-tag">MAKİNE ÖĞRENMESİ DESTEKLİ</span>

            <h2>Akıllı Piyasa Öngörüleri</h2>

            <p>

              BIST ve kripto varlıkları için gerçek zamanlı volatilite, teknik sinyal ve yapay zekâ model projeksiyonları.

            </p>

          </div>

          <div className="welcome-icon">

            <Cpu size={36} strokeWidth={1.8} />

          </div>

        </section>



        <section className="stock-toolbar" id="analysis">

          <div className="toolbar-info">

            <div className="section-label">VARLIK YÖNETİMİ</div>

            <p>

              İşlem gören <strong>{assets.length}</strong> {market === "stocks" ? "hisse" : "kripto"} listeleniyor

            </p>

          </div>



          <div className="stock-actions">

            <div className="select-wrapper">

              <select

                value={market}

                onChange={(event) => {

                  const nextMarket = event.target.value;

                  const nextAssets =

                    nextMarket === "stocks" ? stocks : cryptos;



                  setMarket(nextMarket);

                  setSelectedSymbol(nextAssets[0]?.symbol || "");

                  setPrediction(null);

                  setPriceHistory([]);

                  setLivePrice(null);

                  setLivePriceTimestamp(null);

                  setLivePriceSource("");

                  setLivePriceError("");

                  setError("");

                  setHistoryError("");

                }}

                aria-label="Piyasa seçimi"

              >

                <option value="stocks">BIST Pay Piyasası</option>

                <option value="crypto">Kripto Varlıklar</option>

              </select>

            </div>



            <div className="select-wrapper symbol-select">

              <select

                value={selectedSymbol}

                onChange={(event) => setSelectedSymbol(event.target.value)}

                aria-label="Varlık seçimi"

                disabled={!assets.length}

              >

                {assets.map((item) => (

                  <option key={item.symbol} value={item.symbol}>

                    {item.symbol} {item.name && item.name !== item.symbol ? `· ${item.name}` : ""}

                  </option>

                ))}

              </select>

            </div>



            <button

              className="primary-button"

              onClick={refreshAll}

              disabled={busy || !selectedSymbol}

            >

              {loading ? (

                <>

                  <LoaderCircle size={15} className="spin" />

                  <span>Analiz Ediliyor</span>

                </>

              ) : (

                <>

                  <span>Analiz Et</span>

                  <ArrowUpRight size={15} />

                </>

              )}

            </button>

          </div>

        </section>



        {listError && <div className="error-banner">{listError}</div>}

        {error && <div className="error-banner">Tahmin Hatası: {error}</div>}

        {historyError && <div className="error-banner">Grafik Hatası: {historyError}</div>}

        {market === "crypto" && livePriceError && (

          <div className="error-banner">Canlı Fiyat Uyarısı: {livePriceError}</div>

        )}



        <section className="metrics-grid">

          <StatCard

            label={market === "crypto" ? "Spot Canlı Fiyat" : "Son Kapanış"}

            value={formatPrice(latestClose, currency)}

            detail={

              market === "crypto"

                ? livePrice != null

                  ? `Son akış: ${formatDateTime(livePriceTimestamp)}`

                  : livePriceLoading

                    ? "Fiyat akışı bekleniyor..."

                    : "Canlı veri hazır değil"

                : `İşlem günü: ${lastDate}`

            }

            icon={<CircleDollarSign size={20} />}

            accent={market === "crypto" && livePrice != null}

          />



          <StatCard

            label="Pozitif Getiri Olasılığı"

            value={formatProbability(positiveProbability)}

            detail="AI Projeksiyonu"

            icon={<Activity size={20} />}

            accent

          />



          <StatCard

            label="Tahmin Ufku"

            value={`${prediction?.horizon_trading_days ?? 20} Gün`}

            detail="İşlem periyodu"

            icon={<TrendingUp size={20} />}

          />



          <article className="metric-card model-selector-card">
            <div className="metric-heading">
              <span>Kullanılan Model</span>
              <div className="metric-icon-wrap"><ShieldCheck size={20} /></div>
            </div>
            <select
              className="model-select"
              style={{
                width: "100%",
                marginTop: 8,
                padding: "10px 12px",
                borderRadius: 9,
                border: "1px solid rgba(148, 163, 184, 0.28)",
                background: "#0b1220",
                color: "#f8fafc",
                fontWeight: 700,
                outline: "none",
              }}
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
              disabled={modelsLoading || !availableModels.length}
              aria-label="Tahmin modeli seçimi"
            >
              {availableModels.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
            <div className="metric-note">
              {modelsLoading
                ? "Modeller yükleniyor..."
                : prediction?.model || (availableModels.length
                    ? "Model seçip tahmini güncelle"
                    : market === "crypto"
                      ? "Bu varlık için eğitilmiş model yok"
                      : "Eğitilmiş model dosyası bulunamadı")}
            </div>
          </article>

        </section>



        {market === "crypto" && (

          <section className="panel live-price-panel">

            <div className="panel-header">

              <div>

                <h3>Canlı Piyasa Akışı</h3>

                <p>{selectedSymbol} · Gerçek Zamanlı Spot Borsa Fiyatı</p>

              </div>



              <div className="live-pill">

                <span

                  className={`pulse-dot ${

                    livePrice != null ? "pulse-active" : ""

                  }`}

                />

                <span>

                  {livePriceLoading

                    ? "Akış güncelleniyor..."

                    : livePrice != null

                      ? "Canlı fiyat sorgusu aktif"

                      : "Beklemede"}

                </span>

              </div>

            </div>



            <div className="live-price-content">

              <div className="live-price-value">

                {formatPrice(livePrice, "USDT")}

              </div>



              <div className="live-price-details">

                <div className="data-row">

                  <span>Likidite Kaynağı</span>

                  <strong>{livePriceSource || "Binance Spot"}</strong>

                </div>



                <div className="data-row">

                  <span>Son Paket Zamanı</span>

                  <strong>{formatDateTime(livePriceTimestamp)}</strong>

                </div>



                <div className="data-row">

                  <span>Polling Frekansı</span>

                  <strong>5000 ms</strong>

                </div>

              </div>

            </div>

          </section>

        )}



        <section className="content-grid">

          <div className="panel chart-panel">

            <div className="panel-header">

              <div>

                <h3>Fiyat Trend Grafiği</h3>

                <p>{selectedSymbol} · Günlük Kapanış Fiyat Dinamiği</p>

              </div>

              <span className="tag">D1 GÜNLÜK</span>

            </div>



            <div className="chart-area">

              {historyLoading ? (

                <div className="empty-chart">

                  <LoaderCircle size={32} className="spin accent-icon" />

                  <strong>Geçmiş Fiyatlar Yükleniyor</strong>

                </div>

              ) : priceHistory.length ? (

                <ResponsiveContainer width="100%" height="100%">

                  <AreaChart

                    data={priceHistory}

                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}

                  >

                    <defs>

                      <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">

                        <stop offset="0%" stopColor="#35d49a" stopOpacity={0.3} />

                        <stop offset="95%" stopColor="#35d49a" stopOpacity={0.0} />

                      </linearGradient>

                    </defs>



                    <CartesianGrid

                      stroke="rgba(255, 255, 255, 0.05)"

                      strokeDasharray="4 4"

                      vertical={false}

                    />



                    <XAxis

                      dataKey="date"

                      tickFormatter={(value) => String(value).slice(5)}

                      stroke="#64748b"

                      tick={{ fontSize: 11, fill: "#64748b" }}

                      axisLine={false}

                      tickLine={false}

                      minTickGap={30}

                    />



                    <YAxis

                      domain={["auto", "auto"]}

                      stroke="#64748b"

                      tick={{ fontSize: 11, fill: "#64748b" }}

                      axisLine={false}

                      tickLine={false}

                      width={65}

                    />



                    <Tooltip

                      contentStyle={{

                        background: "rgba(10, 16, 26, 0.95)",

                        border: "1px solid rgba(255, 255, 255, 0.1)",

                        borderRadius: 10,

                        boxShadow: "0 10px 25px rgba(0,0,0,0.5)",

                        color: "#f1f5f9",

                        backdropFilter: "blur(8px)",

                      }}

                      labelFormatter={(value) => `Tarih: ${value}`}

                      formatter={(value) => [

                        formatPrice(value, currency),

                        "Kapanış Değeri",

                      ]}

                    />



                    <Area

                      type="monotone"

                      dataKey="close"

                      stroke="#35d49a"

                      strokeWidth={2.5}

                      fill="url(#priceFill)"

                      dot={false}

                      activeDot={{

                        r: 5,

                        fill: "#35d49a",

                        stroke: "#06090e",

                        strokeWidth: 2,

                      }}

                    />

                  </AreaChart>

                </ResponsiveContainer>

              ) : (

                <div className="empty-chart">

                  <BarChart3 size={32} />

                  <strong>Fiyat grafiği çizilemedi</strong>

                  <p>Bu sembole ait veri dizini bulunamadı.</p>

                </div>

              )}

            </div>



            <div className="chart-footer">

              <span>{priceHistory.length} Kayıt Noktası</span>

              <span>

                {priceHistory.length

                  ? `${priceHistory[0].date} → ${

                      priceHistory[priceHistory.length - 1].date

                    }`

                  : "Dönem verisi yok"}

              </span>

            </div>

          </div>



          <div className="panel signal-panel">

            <div className="panel-header">

              <div>

                <h3>Model Çıkarımı</h3>

                <p>{selectedSymbol} Projeksiyonu</p>

              </div>

              <Activity size={18} className="accent-icon" />

            </div>



            <div
              className={`signal-box ${
                !loading && prediction?.signal
                  ? `signal-${prediction.signal.toLowerCase()}`
                  : ""
              }`}
            >

              <div className="signal-tag">PROJEKSİYON SİNYALİ</div>

              <strong className="signal-title">

                {loading ? "Hesaplanıyor..." : prediction?.signal || "YATAY / NÖTR"}

              </strong>

              {!loading && prediction?.signal_thresholds && (
                <small className="signal-thresholds">
                  AL ≥ {(prediction.signal_thresholds.buy * 100).toFixed(0)}% ·
                  SAT ≤ {(prediction.signal_thresholds.sell * 100).toFixed(0)}%
                  {" "}
                  (
                  {prediction.signal_thresholds.mode === "dynamic"
                    ? "dinamik"
                    : prediction.signal_thresholds.mode === "manual"
                    ? "manuel"
                    : "sabit"}
                  )
                </small>
              )}

              <p>

                Bu gösterge geçmiş testler üzerinden elde edilen istatistiki bir olasılıktır. Kesinlik içermez.

              </p>

            </div>



            <div className="probabilities">

              <ProbabilityBar

                label="Pozitif Trend Olasılığı"

                value={positiveProbability}

                color="#35d49a"

              />



              <ProbabilityBar

                label="Düşüş / Yatay Olasılık"

                value={negativeProbability}

                color="#60a5fa"

              />

            </div>



            <div className="last-updated">

              <div className="data-row">

                <span>Model Veri Tarihi</span>

                <strong>{lastDate}</strong>

              </div>



              {market === "crypto" && (

                <div className="data-row">

                  <span>Canlı Fiyat Akışı</span>

                  <strong>{formatDateTime(livePriceTimestamp)}</strong>

                </div>

              )}



              <div className="data-row">

                <span>Vade Ufku</span>

                <strong>{prediction?.horizon_trading_days ?? 20} İş Günü</strong>

              </div>

            </div>



            <div className="disclaimer">

              <ShieldCheck size={16} />

              <span>

                {prediction?.disclaimer ||

                  "Üretilen tahminler deneysel olup yatırım tavsiyesi niteliği taşımaz."}

              </span>

            </div>

          </div>

        </section>



        <section className="panel indicators-panel" id="indicators">

          <div className="panel-header">

            <div>

              <h3>Teknik İndikatörler</h3>

              <p>{selectedSymbol} · 14-50 periyotluk gösterge tablosu</p>

            </div>

            <Layers size={18} className="accent-icon" />

          </div>



          {indicators ? (

            <TechnicalIndicators indicators={indicators} />

          ) : (

            <div className="performance-empty">

              {loading

                ? "Göstergeler hesaplanıyor..."

                : "Teknik gösterge seti backend üzerinden sağlanamadı."}

            </div>

          )}

        </section>



        <section className="panel performance-panel" id="performance">

          <div className="panel-header">

            <div>

              <h3>Model Başarım Metrikleri</h3>

              <p>

                Hedef Varlık: <strong>{selectedSymbol || "—"}</strong>

              </p>

            </div>

            <ShieldCheck size={18} className="accent-icon" />

          </div>



          {performanceLoading ? (

            <div className="empty-chart">

              <LoaderCircle size={28} className="spin accent-icon" />

              <strong>Test sonuçları derleniyor...</strong>

            </div>

          ) : performanceData ? (

            <PerformanceTable

              data={performanceData}

              selectedSymbol={selectedSymbol}

              market={market}

            />

          ) : (

            <div className="performance-empty">

              {selectedSymbol} sembolü için kayıtlı backtest metriği bulunamadı.

            </div>

          )}

        </section>



        <footer className="app-footer">

          <span>FinanceAI Core Terminal · Research & Trading Engine</span>

          <div className="api-status-text">

            {apiOnline ? (

              <CheckCircle2 size={13} color="#35d49a" />

            ) : (

              <WifiOff size={13} color="#f87171" />

            )}

            <span>Sistem: {apiOnline ? "Bağlantı Kuruldu" : "Erişim Yok"}</span>

          </div>

        </footer>

      </main>

    </div>

  );

}



function StatCard({ label, value, detail, icon, accent = false }) {

  return (

    <article className="metric-card">

      <div className="metric-heading">

        <span>{label}</span>

        <div className="metric-icon-wrap">{icon}</div>

      </div>

      <strong className={accent ? "accent-value" : ""}>{value}</strong>

      <div className="metric-note">{detail}</div>

    </article>

  );

}



function ProbabilityBar({ label, value, color }) {

  const number = Number(value);

  const valid = value != null && value !== "" && Number.isFinite(number);

  const ratio = valid ? Math.max(0, Math.min(1, number)) : 0;



  return (

    <div className="probability-block">

      <div className="probability-row">

        <span>{label}</span>

        <strong>{valid ? formatProbability(number) : "—"}</strong>

      </div>



      <div className="progress-track">

        <div

          className="progress-value"

          style={{

            width: `${ratio * 100}%`,

            background: color,

            boxShadow: `0 0 10px ${color}66`,

          }}

        />

      </div>

    </div>

  );

}



function TechnicalIndicators({ indicators }) {

  const percent = (value) => {

    const number = Number(value);



    if (value == null || value === "" || !Number.isFinite(number)) {

      return "—";

    }



    return `${number.toLocaleString("tr-TR", {

      minimumFractionDigits: 2,

      maximumFractionDigits: 2,

    })}%`;

  };



  const rows = [

    { label: "RSI (14)", value: formatNumber(indicators.rsi), description: "Aşırı alım (>70) / aşırı satım (<30) eşiği." },

    { label: "MACD / Kapanış", value: percent(indicators.macd), description: "MACD çizgisinin fiyata normalize oranı." },

    { label: "MACD Sinyali", value: percent(indicators.macd_signal), description: "Sinyal çizgisinin kapanışa oranı." },

    { label: "MACD Histogram", value: percent(indicators.macd_histogram), description: "Momentum fark açılması." },

    { label: "EMA 12", value: formatNumber(indicators.ema_12, 4), description: "12 günlük hızlı hareketli ortalama." },

    { label: "EMA 26", value: formatNumber(indicators.ema_26, 4), description: "26 günlük yavaş hareketli ortalama." },

    { label: "EMA Farkı", value: percent(indicators.ema_cross), description: "Hızlı/Yavaş EMA spread farkı." },

    { label: "SMA 5", value: formatNumber(indicators.sma_5, 4), description: "Kısa vade 5 dönem basit ortalama." },

    { label: "SMA 20", value: formatNumber(indicators.sma_20, 4), description: "Orta vade 20 dönem basit ortalama." },

    { label: "SMA 50", value: formatNumber(indicators.sma_50, 4), description: "Ana trend 50 dönem basit ortalama." },

    { label: "SMA 20 / Fiyat", value: formatNumber(indicators.sma_20_ratio, 4), description: "Fiyatın 20 günlük ortalamaya mesafesi." },

    { label: "SMA 50 / Fiyat", value: formatNumber(indicators.sma_50_ratio, 4), description: "Fiyatın ana trende mesafesi." },

    { label: "Bollinger Konumu", value: formatNumber(indicators.bb_position, 3), description: "Bant içi normalize fiyat konumu [0-1]." },

    { label: "Bollinger Genişliği", value: percent(indicators.bb_width), description: "Kanal oynaklığı ve daralma metriği." },

    { label: "Hacim Değişimi", value: percent(indicators.volume_change), description: "Önceki bara göre işlem hacmi farkı." },

    { label: "Hacim / Ort.", value: formatNumber(indicators.volume_ratio, 3), description: "Hacmin 20 günlük ortalamaya oranı." },

    { label: "OBV Değişimi", value: formatNumber(indicators.obv_change, 4), description: "Denge işlem hacmi yön farkı." },

    { label: "20G Volatilite", value: percent(indicators.volatility_20d), description: "20 günlük standart getiri sapması." },

    { label: "ATR / Kapanış", value: percent(indicators.atr_ratio), description: "Yüzdesel ortalama gerçek aralık." },

  ];



  return (

    <div className="indicator-grid">

      {rows.map((row) => (

        <article className="indicator-card" key={row.label}>

          <span>{row.label}</span>

          <strong>{row.value}</strong>

          <small>{row.description}</small>

        </article>

      ))}

    </div>

  );

}



function PerformanceTable({ data, selectedSymbol, market }) {
  const source = useMemo(() => {
    if (!data) return [];

    if (Array.isArray(data)) return data;

    const possibleRows = [
      data.rows,
      data.results,
      data.data,
      data.models,
      data.records,
    ];

    const found = possibleRows.find(Array.isArray);
    return found || [];
  }, [data]);

  const normalizeKey = (value) =>
    String(value ?? "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");

  const getValue = (row, ...keys) => {
    if (!row || typeof row !== "object") return undefined;

    for (const key of keys) {
      if (row[key] !== undefined && row[key] !== null && row[key] !== "") {
        return row[key];
      }
    }

    const wanted = keys.map(normalizeKey);
    const actualKey = Object.keys(row).find((key) =>
      wanted.includes(normalizeKey(key))
    );

    return actualKey ? row[actualKey] : undefined;
  };

  const symbolOf = (row) =>
    getValue(row, "Symbol", "symbol", "Ticker", "ticker", "Asset", "asset");

  const hasSymbolColumn = source.some((row) => symbolOf(row) != null);

  const matchingRows = hasSymbolColumn
    ? source.filter(
        (row) =>
          normalizeSymbol(symbolOf(row)) === normalizeSymbol(selectedSymbol)
      )
    : source;

  const rows = matchingRows;

  const formatMetric = (value) => {
    if (value == null || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);

    // Metrikler 0-1 aralığındaysa yüzdeye çevir; zaten yüzdeyse olduğu gibi bırak.
    const formatted = number >= 0 && number <= 1 ? number * 100 : number;
    return `${formatted.toLocaleString("tr-TR", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 2,
    })}%`;
  };

  const metricValue = (row, ...keys) => {
    const value = getValue(row, ...keys);
    return formatMetric(value);
  };

  if (!rows.length) {
    return (
      <div className="performance-empty">
        <h4>
          {hasSymbolColumn
            ? `${selectedSymbol} için performans kaydı bulunamadı`
            : "Performans kaydı bulunamadı"}
        </h4>
        <p>
          {hasSymbolColumn
            ? `API raporunda ${selectedSymbol} sembolüne ait satır bulunamadı. Başka bir sembolün sonuçları gösterilmiyor.`
            : "Backend yanıtında okunabilir satır verisi bulunamadı. /model-performance yanıtının rows alanını kontrol et."}
        </p>
      </div>
    );
  }

  return (
    <div className="performance-content">
      <div className="performance-section-heading">
        <div>
          <h4>{hasSymbolColumn ? selectedSymbol : "Model karşılaştırması"}</h4>
          <p>
            {market === "stocks" ? "BIST hissesi" : "Kripto varlık"} · Test sonuçları
          </p>
        </div>
        <span className="performance-badge">{rows.length} KAYIT</span>
      </div>

      <div className="table-scroll">
        <table className="performance-table">
          <thead>
            <tr>
              <th>Varlık</th>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Balanced Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>ROC AUC</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const symbol = symbolOf(row) || selectedSymbol || "—";
              const model = getValue(row, "Model", "model", "Model_Name", "model_name") || "—";
              return (
                <tr key={`${symbol}-${model}-${index}`}>
                  <td>{symbol}</td>
                  <td>{String(model).replaceAll("_", " ")}</td>
                  <td>{metricValue(row, "Accuracy", "accuracy", "Accuracy_mean")}</td>
                  <td>{metricValue(row, "Balanced_Accuracy", "Balanced Accuracy", "balanced_accuracy", "Balanced_Accuracy_mean")}</td>
                  <td>{metricValue(row, "Precision", "precision", "Precision_mean")}</td>
                  <td>{metricValue(row, "Recall", "recall", "Recall_mean")}</td>
                  <td>{metricValue(row, "F1", "f1", "F1_Score", "F1_mean")}</td>
                  <td>{metricValue(row, "ROC_AUC", "ROC AUC", "roc_auc", "ROC_AUC_mean")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="performance-note">
        * Bu değerler geçmiş test verilerinin metrikleridir; gelecekteki getiriyi garanti etmez.
      </p>
    </div>
  );
}

export default App;