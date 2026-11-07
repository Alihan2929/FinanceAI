import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Activity,
  Bitcoin,
  CircleDollarSign,
  Cpu,
  Landmark,
  Layers,
  LineChart as LineChartIcon,
  LoaderCircle,
  Moon,
  RefreshCw,
  Search,
  ShieldCheck,
  Sun,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const LIVE_REFRESH_MS = 5000;
const CRYPTO_FALLBACK = [
  "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
  "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
  "TRXUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT",
  "APTUSDT", "ETCUSDT", "FILUSDT", "ARBUSDT", "OPUSDT",
];

const MARKETS = [
  { key: "stocks", tab: "BIST", label: "Borsa İstanbul", icon: Landmark },
  { key: "us", tab: "ABD", label: "ABD Piyasaları", icon: CircleDollarSign },
  { key: "crypto", tab: "Kripto", label: "Kripto Piyasası", icon: Bitcoin },
];

const RANGES = [
  { days: 30, label: "30G" },
  { days: 60, label: "60G" },
  { days: 120, label: "120G" },
];

const CHART_COLORS = {
  light: { up: "#059669", down: "#dc2626", grid: "#eceff3", axis: "#9ca3af", sma: "#f59e0b" },
  dark: { up: "#34d399", down: "#f87171", grid: "#1b2330", axis: "#5b6676", sma: "#fbbf24" },
};

/* ───────── formatters ───────── */
function formatPrice(value) {
  const number = Number(value);
  if (value == null || value === "" || !Number.isFinite(number)) return "—";
  return number.toLocaleString("tr-TR", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

function formatProbability(value) {
  const number = Number(value);
  if (value == null || value === "" || !Number.isFinite(number)) return "—";
  return `%${(number * 100).toLocaleString("tr-TR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}`;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (value == null || value === "" || !Number.isFinite(number)) return "—";
  return number.toLocaleString("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatSigned(value, digits = 2) {
  const number = Number(value);
  if (value == null || !Number.isFinite(number)) return "—";
  return number.toLocaleString("tr-TR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
    signDisplay: "exceptZero",
  });
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("tr-TR", {
    day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function normalizeSymbol(value) {
  return String(value || "").trim().toUpperCase().replace(/\.IS$/i, "").replace(/[^A-Z0-9]/g, "");
}

function signalTone(signal) {
  const s = String(signal || "").toUpperCase();
  if (/(BUY|AL\b|LONG|BULL|POSITIVE|UP)/.test(s)) return "up";
  if (/(SELL|SAT\b|SHORT|BEAR|NEGATIVE|DOWN)/.test(s)) return "down";
  return "flat";
}

const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

/* ───────── small presentational pieces ───────── */
function Gauge({ value }) {
  const pct = clamp(Number(value) || 0, 0, 1);
  const tone = pct > 0.5 ? "up" : pct < 0.5 ? "down" : "flat";
  return (
    <svg viewBox="0 0 120 74" className="gauge" role="img" aria-label="Yükseliş olasılığı göstergesi">
      <path d="M10 62 A50 50 0 0 1 110 62" pathLength="100" className="gauge-track" />
      {pct > 0 && (
        <path
          d="M10 62 A50 50 0 0 1 110 62"
          pathLength="100"
          className={`gauge-fill ${tone}`}
          strokeDasharray={`${pct * 100} 100`}
        />
      )}
      <line x1="60" y1="5" x2="60" y2="19" className="gauge-tick" />
      <text x="60" y="54" textAnchor="middle" className="gauge-value">
        {formatProbability(value)}
      </text>
      <text x="60" y="68" textAnchor="middle" className="gauge-label">
        YÜKSELİŞ OLASILIĞI
      </text>
    </svg>
  );
}

function ChartTooltip({ active, payload, label, smaColor, lineColor }) {
  if (!active || !payload?.length) return null;
  const close = payload.find((p) => p.dataKey === "close");
  const sma = payload.find((p) => p.dataKey === "sma20");
  return (
    <div className="tt">
      <div className="tt-date">{label}</div>
      <div className="tt-row">
        <i style={{ background: lineColor }} />
        <span>Kapanış</span>
        <b>{formatPrice(close?.value)}</b>
      </div>
      {sma?.value != null && (
        <div className="tt-row">
          <i style={{ background: smaColor }} />
          <span>SMA 20</span>
          <b>{formatPrice(sma.value)}</b>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, tone }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <b className={tone || ""}>{value}</b>
    </div>
  );
}

/* ───────── main ───────── */
function App() {
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("fai-theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      /* ignore */
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const [market, setMarket] = useState("stocks");
  const [query, setQuery] = useState("");
  const [range, setRange] = useState(120);
  const [showSma, setShowSma] = useState(true);
  const [stocks, setStocks] = useState([]);
  const [usStocks, setUsStocks] = useState([]);
  const [cryptos, setCryptos] = useState(CRYPTO_FALLBACK.map((symbol) => ({ symbol, name: symbol })));
  const [selectedSymbol, setSelectedSymbol] = useState("TUPRS");
  const [prediction, setPrediction] = useState(null);
  const [selectedModel, setSelectedModel] = useState("random_forest");
  const [availableModels, setAvailableModels] = useState([{ value: "random_forest", label: "Random Forest" }]);
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

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("fai-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const assets = market === "stocks" ? stocks : market === "us" ? usStocks : cryptos;
  const currency = market === "stocks" ? "TL" : market === "us" ? "USD" : "USDT";
  const marketMeta = MARKETS.find((m) => m.key === market);
  const selectedAsset = assets.find((item) => item.symbol === selectedSymbol);

  const lastDate = prediction?.date || priceHistory[priceHistory.length - 1]?.date || "—";
  const displayedPrice = market === "crypto" && livePrice != null
    ? livePrice
    : prediction?.close ?? priceHistory[priceHistory.length - 1]?.close;

  /* ───── data loading (unchanged logic) ───── */
  const loadLists = useCallback(async () => {
    setListError("");
    const [stockResult, usStockResult, cryptoResult] = await Promise.allSettled([
      axios.get(`${API_URL}/stocks`, { timeout: 15000 }),
      axios.get(`${API_URL}/us/stocks`, { timeout: 15000 }),
      axios.get(`${API_URL}/crypto`, { timeout: 15000 }),
    ]);

    let connected = false;

    if (stockResult.status === "fulfilled") {
      const body = stockResult.value.data;
      const raw = body?.stocks ?? body?.data ?? body;
      if (Array.isArray(raw)) {
        const normalized = raw
          .map((item) => ({ ...item, symbol: normalizeSymbol(item.symbol || item.ticker || ""), name: item.name || item.symbol }))
          .filter((item) => item.symbol);
        if (normalized.length > 0) { setStocks(normalized); connected = true; }
      }
    }

    if (usStockResult.status === "fulfilled") {
      const body = usStockResult.value.data;
      const raw = body?.stocks ?? body?.data ?? body;
      if (Array.isArray(raw)) {
        const normalized = raw
          .map((item) => ({ ...item, symbol: String(item.symbol || item.ticker || "").trim().toUpperCase(), name: item.name || item.symbol }))
          .filter((item) => item.symbol);
        setUsStocks(normalized);
        if (normalized.length > 0) connected = true;
      }
    }

    if (cryptoResult.status === "fulfilled") {
      const body = cryptoResult.value.data;
      const raw = body?.cryptos ?? body?.crypto ?? body?.symbols ?? body?.data ?? body;
      if (Array.isArray(raw)) {
        const normalized = raw
          .map((item) => {
            const sym = typeof item === "string" ? item : item.symbol || item.ticker || item.id || "";
            const normSym = String(sym).trim().toUpperCase().replace("/", "");
            return { ...(typeof item === "object" ? item : {}), symbol: normSym, name: typeof item === "object" ? item.name || normSym : normSym };
          })
          .filter((item) => item.symbol);
        if (normalized.length > 0) { setCryptos(normalized); connected = true; }
      }
    }

    setApiOnline(connected);
    if (!connected) setListError("Varlık listesi alınamadı.");
  }, []);

  const loadPrediction = useCallback(async (symbol, selectedMarket, modelName = selectedModel) => {
    if (!symbol || !modelName) return;
    setLoading(true);
    setError("");
    try {
      const endpoint = selectedMarket === "stocks"
        ? `${API_URL}/predict/${encodeURIComponent(symbol)}`
        : selectedMarket === "us"
        ? `${API_URL}/us/predict/${encodeURIComponent(symbol)}`
        : `${API_URL}/crypto/predict/${encodeURIComponent(symbol)}`;

      const response = await axios.get(endpoint, { params: { model: modelName }, timeout: 20000 });
      setPrediction(response.data);
      setApiOnline(true);
    } catch (err) {
      setPrediction(null);
      setError(err.response?.data?.detail || "Tahmin alınamadı.");
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
      const response = await axios.get(`${API_URL}/models`, { params: { symbol, market: selectedMarket }, timeout: 10000 });
      const models = Array.isArray(response.data?.models)
        ? response.data.models.filter((item) => item && typeof item.value === "string" && typeof item.label === "string")
        : [];
      setAvailableModels(models);
      setSelectedModel((currentModel) => {
        if (models.some((item) => item.value === currentModel)) return currentModel;
        return response.data?.default || models[0]?.value || "";
      });
      if (!models.length) setError(`${symbol} için eğitilmiş model bulunamadı.`);
    } catch (err) {
      setAvailableModels([]);
      setSelectedModel("");
      setError(err.response?.data?.detail || "Modeller alınamadı.");
    } finally {
      setModelsLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async (symbol, selectedMarket) => {
    if (!symbol) return;
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const endpoint = selectedMarket === "stocks"
        ? `${API_URL}/stocks/${encodeURIComponent(symbol)}/history`
        : selectedMarket === "us"
        ? `${API_URL}/us/stocks/${encodeURIComponent(symbol)}/history`
        : `${API_URL}/crypto/${encodeURIComponent(symbol)}/history`;

      const response = await axios.get(endpoint, { params: { limit: 120 }, timeout: 15000 });
      const raw = response.data?.prices ?? response.data?.history ?? response.data?.data;
      if (!Array.isArray(raw)) throw new Error("Format hatası.");

      const normalized = raw
        .map((item) => ({ date: String(item.date ?? item.Date ?? ""), close: Number(item.close ?? item.Close) }))
        .filter((item) => item.date && Number.isFinite(item.close))
        .sort((a, b) => a.date.localeCompare(b.date));

      setPriceHistory(normalized);
      if (!normalized.length) setHistoryError("Geçmiş veri yok.");
    } catch (err) {
      setPriceHistory([]);
      setHistoryError(err.response?.data?.detail || err.message || "Geçmiş alınamadı.");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const loadLivePrice = useCallback(async (symbol) => {
    if (!symbol) return;
    setLivePriceLoading(true);
    setLivePriceError("");
    try {
      const response = await axios.get(`${API_URL}/crypto/live/${encodeURIComponent(symbol)}`, {
        timeout: 10000,
        headers: { "Cache-Control": "no-cache" },
        params: { _t: Date.now() },
      });
      const body = response.data;
      const price = Number(body?.price ?? body?.lastPrice ?? body?.last_price ?? body?.close);

      if (!Number.isFinite(price) || price <= 0) throw new Error("Fiyat hatası.");

      setLivePrice(price);
      setLivePriceTimestamp(body?.timestamp || body?.time || new Date().toISOString());
      setLivePriceSource(body?.source || "Binance");
      setApiOnline(true);
    } catch (err) {
      setLivePriceError(err.response?.data?.detail || err.message || "Canlı fiyat hatası.");
    } finally {
      setLivePriceLoading(false);
    }
  }, []);

  const loadPerformance = useCallback(async (symbol = selectedSymbol, selectedMarket = market, modelName = selectedModel) => {
    if (!symbol) return;
    setPerformanceLoading(true);
    try {
      const url = selectedMarket === "us"
        ? `${API_URL}/us/model-performance/${encodeURIComponent(symbol)}`
        : `${API_URL}/model-performance`;
      const response = await axios.get(url, {
        params: selectedMarket === "us" ? { model: modelName || "random_forest" } : {},
        timeout: 30000,
      });
      setPerformance(response.data);
    } catch (err) {
      setPerformance({ rows: [], message: err.response?.data?.detail || "Performans alınamadı." });
    } finally {
      setPerformanceLoading(false);
    }
  }, [selectedSymbol, market, selectedModel]);

  useEffect(() => {
    loadLists();
    loadPerformance();
    axios.get(`${API_URL}/health`, { timeout: 5000 })
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
    loadPerformance(selectedSymbol, market, selectedModel);
  }, [selectedSymbol, market, selectedModel, loadAvailableModels, loadPerformance]);

  useEffect(() => {
    if (!selectedSymbol || !selectedModel) return;
    loadPrediction(selectedSymbol, market, selectedModel);
    loadHistory(selectedSymbol, market);

    setLivePrice(null);
    setLivePriceTimestamp(null);
    setLivePriceSource("");
    setLivePriceError("");
    if (market === "crypto") loadLivePrice(selectedSymbol);
  }, [selectedSymbol, market, loadPrediction, loadHistory, loadLivePrice, selectedModel]);

  useEffect(() => {
    if (market !== "crypto" || !selectedSymbol) return;
    const intervalId = window.setInterval(() => loadLivePrice(selectedSymbol), LIVE_REFRESH_MS);
    return () => window.clearInterval(intervalId);
  }, [market, selectedSymbol, loadLivePrice]);

  useEffect(() => {
    if (!selectedSymbol || !selectedModel) return;
    const predictionRefreshMs = market === "crypto" ? 60_000 : 5 * 60_000;
    const intervalId = window.setInterval(() => {
      loadPrediction(selectedSymbol, market, selectedModel);
      loadHistory(selectedSymbol, market);
    }, predictionRefreshMs);
    return () => window.clearInterval(intervalId);
  }, [market, selectedSymbol, selectedModel, loadPrediction, loadHistory]);

  const refreshAll = useCallback(() => {
    loadLists();
    if (selectedModel) loadPrediction(selectedSymbol, market, selectedModel);
    loadHistory(selectedSymbol, market);
    loadPerformance();
    if (market === "crypto") loadLivePrice(selectedSymbol);
  }, [loadLists, loadPrediction, loadHistory, loadPerformance, loadLivePrice, selectedSymbol, market, selectedModel]);

  const changeMarket = (nextMarket) => {
    if (nextMarket === market) return;
    const nextAssets = nextMarket === "stocks" ? stocks : nextMarket === "us" ? usStocks : cryptos;
    setMarket(nextMarket);
    setSelectedSymbol(nextAssets[0]?.symbol || "");
    setQuery("");
    setPrediction(null);
    setPriceHistory([]);
    setLivePrice(null);
    setError("");
  };

  /* ───── derived view data ───── */
  const filteredAssets = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return assets;
    return assets.filter(
      (item) => item.symbol.toLowerCase().includes(q) || String(item.name || "").toLowerCase().includes(q)
    );
  }, [assets, query]);

  const historyWithSma = useMemo(
    () =>
      priceHistory.map((point, i, arr) => {
        if (i < 19) return { ...point, sma20: null };
        let sum = 0;
        for (let j = i - 19; j <= i; j += 1) sum += arr[j].close;
        return { ...point, sma20: sum / 20 };
      }),
    [priceHistory]
  );

  const chartData = useMemo(() => historyWithSma.slice(-range), [historyWithSma, range]);

  const stats = useMemo(() => {
    if (!chartData.length) return null;
    const closes = chartData.map((d) => d.close);
    const first = closes[0];
    const last = closes[closes.length - 1];
    return {
      ret: first ? (last / first - 1) * 100 : null,
      high: Math.max(...closes),
      low: Math.min(...closes),
      avg: closes.reduce((a, b) => a + b, 0) / closes.length,
    };
  }, [chartData]);

  const lastClose = priceHistory[priceHistory.length - 1]?.close;
  const prevClose = priceHistory[priceHistory.length - 2]?.close;
  const crypto = market === "crypto" && livePrice != null;
  const changeBase = crypto ? lastClose : prevClose;
  const changeLabel = crypto ? "Son kapanışa göre" : "Önceki kapanışa göre";
  const priceNumber = Number(displayedPrice);
  const changeAbs = Number.isFinite(priceNumber) && changeBase ? priceNumber - changeBase : null;
  const changePct = changeAbs != null && changeBase ? (changeAbs / changeBase) * 100 : null;
  const changeTone = changeAbs == null || changeAbs === 0 ? "flat" : changeAbs > 0 ? "up" : "down";

  const colors = CHART_COLORS[theme];
  const lineColor = stats?.ret != null && stats.ret < 0 ? colors.down : colors.up;

  const busy = loading || historyLoading || performanceLoading;
  const positiveProbability = prediction?.probability_positive_return;
  const posNum = clamp(Number(positiveProbability) || 0, 0, 1);
  const negRaw = Number(prediction?.probability_non_positive_return);
  const negNum = Number.isFinite(negRaw) ? clamp(negRaw, 0, 1) : prediction ? 1 - posNum : 0;
  const tone = signalTone(prediction?.signal);
  const selectedModelLabel = availableModels.find((m) => m.value === selectedModel)?.label || selectedModel || "—";

  const performanceData = useMemo(() => {
    if (!performance) return null;
    if (market === "us") return performance;
    if (Array.isArray(performance.rows) || Array.isArray(performance.results)) return performance;

    const candidates = market === "stocks"
      ? [performance.stock_model_comparison, performance.stocks_model_comparison, performance.model_comparison, performance.stock_walk_forward, performance.walk_forward, performance.stocks]
      : [performance.crypto_model_comparison, performance.crypto_walk_forward, performance.crypto_backtest, performance.crypto_walk_forward_summary, performance.crypto_backtest_summary, performance.cryptos];

    return candidates.find((item) => Array.isArray(item) || Array.isArray(item?.rows) || Array.isArray(item?.results) || Array.isArray(item?.data)) ?? performance;
  }, [performance, market]);

  const indicators = prediction?.technical_indicators;

  const rsi = Number(indicators?.rsi);
  const rsiValid = indicators && Number.isFinite(rsi);
  const rsiState = !rsiValid ? "" : rsi >= 70 ? "Aşırı alım bölgesi" : rsi <= 30 ? "Aşırı satım bölgesi" : "Nötr bölge";

  const trendChips = useMemo(() => {
    if (!indicators) return [];
    const n = (v) => (v == null || v === "" ? NaN : Number(v));
    const chips = [];
    const sma20 = n(indicators.sma_20);
    const sma50 = n(indicators.sma_50);
    const ema12 = n(indicators.ema_12);
    const ema26 = n(indicators.ema_26);
    const macd = n(indicators.macd);
    if (Number.isFinite(priceNumber) && Number.isFinite(sma20))
      chips.push({ up: priceNumber >= sma20, text: priceNumber >= sma20 ? "Fiyat SMA 20 üzerinde" : "Fiyat SMA 20 altında" });
    if (Number.isFinite(sma20) && Number.isFinite(sma50))
      chips.push({ up: sma20 >= sma50, text: sma20 >= sma50 ? "SMA 20 > SMA 50" : "SMA 20 < SMA 50" });
    if (Number.isFinite(ema12) && Number.isFinite(ema26))
      chips.push({ up: ema12 >= ema26, text: ema12 >= ema26 ? "EMA 12 > EMA 26" : "EMA 12 < EMA 26" });
    if (Number.isFinite(macd))
      chips.push({ up: macd >= 0, text: macd >= 0 ? "MACD pozitif" : "MACD negatif" });
    return chips;
  }, [indicators, priceNumber]);

  return (
    <div className="app">
      {/* ═════════ SIDEBAR ═════════ */}
      <aside className="side">
        <div className="brand">
          <div className="brand-mark"><LineChartIcon size={17} /></div>
          <div>
            <div className="brand-name">Finance<b>AI</b></div>
            <div className="brand-sub">Tahmin Terminali</div>
          </div>
        </div>

        <div className="seg" role="tablist">
          {MARKETS.map(({ key, tab }) => (
            <button
              key={key}
              role="tab"
              aria-selected={market === key}
              className={market === key ? "on" : ""}
              onClick={() => changeMarket(key)}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="search">
          <Search size={14} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Sembol ara…"
            spellCheck={false}
          />
        </div>

        <div className="list-head">
          <span>VARLIK</span>
          <span>{filteredAssets.length}</span>
        </div>

        <ul className="asset-list">
          {filteredAssets.map((item) => (
            <li key={item.symbol}>
              <button
                className={`asset ${item.symbol === selectedSymbol ? "on" : ""}`}
                onClick={() => setSelectedSymbol(item.symbol)}
              >
                <span className="asset-sym">{item.symbol}</span>
                {item.name && item.name !== item.symbol && <span className="asset-name">{item.name}</span>}
              </button>
            </li>
          ))}
          {!filteredAssets.length && <li className="asset-empty">Sonuç bulunamadı</li>}
        </ul>

        <div className="side-foot">
          <span className={`dot ${apiOnline ? "on" : ""}`} />
          <div>
            <div className="foot-title">{apiOnline ? "API bağlı" : "API çevrimdışı"}</div>
            <div className="foot-sub">127.0.0.1:8000</div>
          </div>
        </div>
      </aside>

      {/* ═════════ MAIN ═════════ */}
      <main className="main">
        {/* ── Header ── */}
        <header className="head">
          <div className="head-id">
            <div className="head-row">
              <h1>{selectedSymbol || "—"}</h1>
              <span className="chip">{marketMeta.tab}</span>
              <span className={`chip ${crypto ? "live" : ""}`}>{crypto ? "CANLI" : "GÜNLÜK"}</span>
            </div>
            <p>
              {selectedAsset?.name && selectedAsset.name !== selectedSymbol ? selectedAsset.name : marketMeta.label}
              {" · "}
              {currency}
            </p>
          </div>

          <div className="head-price">
            <div className="price">{formatPrice(displayedPrice)}</div>
            <div className="change-row">
              <span className={`change ${changeTone}`}>
                {changeTone === "up" ? <TrendingUp size={14} /> : changeTone === "down" ? <TrendingDown size={14} /> : null}
                {changeAbs == null ? "—" : `${formatSigned(changeAbs, 2)} (${formatSigned(changePct, 2)}%)`}
              </span>
              <span className="muted">{changeLabel}</span>
            </div>
          </div>

          <div className="head-actions">
            <div className="field">
              <label htmlFor="model">Model</label>
              <select
                id="model"
                className="select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={modelsLoading || !availableModels.length}
              >
                {availableModels.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
                ))}
              </select>
            </div>
            <button className="btn primary" onClick={refreshAll} disabled={busy}>
              <RefreshCw size={14} className={busy ? "spin" : ""} />
              Yenile
            </button>
            <button
              className="btn icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Temayı değiştir"
              title="Temayı değiştir"
            >
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>

        <div className="meta-line">
          {crypto
            ? <>Kaynak: <b>{livePriceSource}</b> · Son paket: {formatDateTime(livePriceTimestamp)}</>
            : <>Veri tarihi: <b>{lastDate}</b></>}
          {market === "crypto" && !crypto && (livePriceLoading ? " · Canlı fiyat alınıyor…" : " · Canlı akış bekleniyor")}
        </div>

        {listError && <div className="alert">{listError}</div>}
        {error && <div className="alert">{error}</div>}
        {historyError && !priceHistory.length && <div className="alert">{historyError}</div>}
        {market === "crypto" && livePriceError && <div className="alert">{livePriceError}</div>}

        <div className="grid">
          {/* ── Chart ── */}
          <section className="card span-8">
            <div className="card-head">
              <span className="card-title"><LineChartIcon size={14} /> Fiyat Grafiği</span>
              <div className="tools">
                <button className={`pill ${showSma ? "on" : ""}`} onClick={() => setShowSma((v) => !v)}>
                  <i style={{ background: colors.sma }} /> SMA 20
                </button>
                <div className="seg small">
                  {RANGES.map((r) => (
                    <button key={r.days} className={range === r.days ? "on" : ""} onClick={() => setRange(r.days)}>
                      {r.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="chart-box">
              {historyLoading ? (
                <div className="empty"><LoaderCircle size={22} className="spin" /></div>
              ) : chartData.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="fillArea" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={lineColor} stopOpacity={0.22} />
                        <stop offset="100%" stopColor={lineColor} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={colors.grid} vertical={false} />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(v) => String(v).slice(5)}
                      stroke={colors.grid}
                      tick={{ fontSize: 11, fill: colors.axis }}
                      axisLine={false}
                      tickLine={false}
                      minTickGap={36}
                    />
                    <YAxis
                      domain={["auto", "auto"]}
                      orientation="right"
                      tick={{ fontSize: 11, fill: colors.axis }}
                      axisLine={false}
                      tickLine={false}
                      width={62}
                      tickFormatter={(v) => formatNumber(v, Math.abs(v) < 10 ? 2 : 0)}
                    />
                    <Tooltip
                      cursor={{ stroke: colors.axis, strokeDasharray: "3 3" }}
                      content={<ChartTooltip smaColor={colors.sma} lineColor={lineColor} />}
                    />
                    <Area type="monotone" dataKey="close" stroke={lineColor} strokeWidth={2} fill="url(#fillArea)" dot={false} />
                    {showSma && (
                      <Line type="monotone" dataKey="sma20" stroke={colors.sma} strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls={false} />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty">Veri yok</div>
              )}
            </div>

            <div className="stats">
              <Stat
                label={`${range}G Getiri`}
                value={stats?.ret != null ? `${formatSigned(stats.ret, 2)}%` : "—"}
                tone={stats?.ret != null ? (stats.ret > 0 ? "up" : stats.ret < 0 ? "down" : "") : ""}
              />
              <Stat label="En Yüksek" value={stats ? formatPrice(stats.high) : "—"} />
              <Stat label="En Düşük" value={stats ? formatPrice(stats.low) : "—"} />
              <Stat label="Ortalama" value={stats ? formatPrice(stats.avg) : "—"} />
            </div>
          </section>

          {/* ── Signal ── */}
          <section className="card span-4">
            <div className="card-head">
              <span className="card-title"><Activity size={14} /> Model Sinyali</span>
              <span className="chip">{selectedModelLabel}</span>
            </div>

            <div className="gauge-wrap">
              <Gauge value={positiveProbability} />
              <div className={`signal ${tone}`}>{loading ? "…" : prediction?.signal || "NÖTR"}</div>
            </div>

            <div className="split">
              <div className="split-bar">
                <div className="split-up" style={{ width: `${posNum * 100}%` }} />
                <div className="split-down" style={{ width: `${negNum * 100}%` }} />
              </div>
              <div className="split-legend">
                <span><i className="up" /> Yükseliş <b>{formatProbability(positiveProbability)}</b></span>
                <span><i className="down" /> Düşüş / yatay <b>{prediction ? formatProbability(negNum) : "—"}</b></span>
              </div>
            </div>

            <div className="kv">
              <div><span><Cpu size={13} /> Aktif model</span><b>{selectedModelLabel}</b></div>
              <div><span>Kullanılabilir model</span><b>{availableModels.length}</b></div>
            </div>
          </section>

          {/* ── Indicators ── */}
          <section className="card span-5">
            <div className="card-head">
              <span className="card-title"><Layers size={14} /> Teknik Göstergeler</span>
            </div>

            {indicators ? (
              <>
                <div className="rsi">
                  <div className="rsi-top">
                    <span>RSI (14)</span>
                    <b>{formatNumber(indicators.rsi)}</b>
                  </div>
                  <div className="rsi-track">
                    <span className="rsi-marker" style={{ left: `${rsiValid ? clamp(rsi, 0, 100) : 0}%` }} />
                  </div>
                  <div className="rsi-scale">
                    <span>0</span><span>30</span><span>70</span><span>100</span>
                  </div>
                  {rsiState && <div className="rsi-state">{rsiState}</div>}
                </div>

                <div className="metrics">
                  <div><span>EMA 12</span><b>{formatNumber(indicators.ema_12)}</b></div>
                  <div><span>EMA 26</span><b>{formatNumber(indicators.ema_26)}</b></div>
                  <div><span>SMA 20</span><b>{formatNumber(indicators.sma_20)}</b></div>
                  <div><span>SMA 50</span><b>{formatNumber(indicators.sma_50)}</b></div>
                  <div><span>MACD %</span><b>{formatNumber(indicators.macd)}%</b></div>
                  <div><span>Bollinger konumu</span><b>{formatNumber(indicators.bb_position)}</b></div>
                  <div className="wide"><span>Volatilite (20G)</span><b>{formatNumber(indicators.volatility_20d)}%</b></div>
                </div>

                {trendChips.length > 0 && (
                  <div className="chips">
                    {trendChips.map((c) => (
                      <span key={c.text} className={`tchip ${c.up ? "up" : "down"}`}>{c.text}</span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="empty">Hesaplanıyor…</div>
            )}
          </section>

          {/* ── Performance ── */}
          <section className="card span-7">
            <div className="card-head">
              <span className="card-title"><ShieldCheck size={14} /> Model Test Başarımı</span>
              <span className="chip">{selectedSymbol}</span>
            </div>
            {performanceData ? (
              <PerformanceTable data={performanceData} selectedSymbol={selectedSymbol} />
            ) : (
              <div className="empty">Veriler yükleniyor…</div>
            )}
          </section>
        </div>

        <p className="disclaimer">
          Bu ekrandaki tahminler istatistiksel model çıktılarıdır; yatırım tavsiyesi niteliği taşımaz.
        </p>
      </main>
    </div>
  );
}

/* ═════════ Performance table ═════════ */
function PerformanceTable({ data, selectedSymbol }) {
  const source = useMemo(() => {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    return [data.rows, data.results, data.data, data.models].find(Array.isArray) || [];
  }, [data]);

  const getValue = (row, ...keys) => {
    if (!row || typeof row !== "object") return undefined;
    for (const key of keys) {
      if (row[key] !== undefined && row[key] !== null && row[key] !== "") return row[key];
    }
    return undefined;
  };

  const symbolOf = (row) => getValue(row, "Symbol", "symbol", "Ticker", "ticker");
  const hasSymbolColumn = source.some((row) => symbolOf(row) != null);
  const rows = hasSymbolColumn
    ? source.filter((row) => normalizeSymbol(symbolOf(row)) === normalizeSymbol(selectedSymbol))
    : source;

  const toPercent = (val) => {
    if (val == null || val === "") return null;
    const num = Number(val);
    if (!Number.isFinite(num)) return null;
    return num >= 0 && num <= 1 ? num * 100 : num;
  };

  const metric = (val) => {
    if (val == null || val === "") return { text: "—", pct: null };
    const pct = toPercent(val);
    if (pct == null) return { text: String(val), pct: null };
    return { text: `%${pct.toFixed(1)}`, pct };
  };

  const bestIndex = useMemo(() => {
    let best = -1;
    let bestScore = -Infinity;
    rows.forEach((row, i) => {
      const score = toPercent(getValue(row, "F1", "f1"));
      if (score != null && score > bestScore) {
        bestScore = score;
        best = i;
      }
    });
    return rows.length > 1 ? best : -1;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows]);

  if (!rows.length) return <div className="empty">Kayıt bulunamadı</div>;

  const columns = [
    ["Accuracy", ["Accuracy", "accuracy"]],
    ["Balanced", ["Balanced_Accuracy", "balanced_accuracy"]],
    ["Precision", ["Precision", "precision"]],
    ["Recall", ["Recall", "recall"]],
    ["F1", ["F1", "f1"]],
  ];

  return (
    <div className="table-scroll">
      <table className="table">
        <thead>
          <tr>
            <th>Model</th>
            {columns.map(([label]) => (
              <th key={label}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className={index === bestIndex ? "best" : ""}>
              <td>
                {String(getValue(row, "Model", "model") || "—").replace(/_/g, " ")}
                {index === bestIndex && <em>EN İYİ F1</em>}
              </td>
              {columns.map(([label, keys]) => {
                const m = metric(getValue(row, ...keys));
                return (
                  <td key={label}>
                    <span>{m.text}</span>
                    {m.pct != null && (
                      <span className="mini"><i style={{ width: `${clamp(m.pct, 0, 100)}%` }} /></span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;