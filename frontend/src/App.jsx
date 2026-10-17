
import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleDollarSign,
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

const API_URL = "http://127.0.0.1:8000";

const STOCKS = [{ symbol: "TUPRS", name: "Tüpraş" }];

function formatPrice(value) {
  const number = Number(value);
  if (value == null || value === "" || !Number.isFinite(number)) return "—";

  return `${number.toLocaleString("tr-TR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} TL`;
}

function formatProbability(value) {
  const number = Number(value);
  if (value == null || value === "" || !Number.isFinite(number)) return "—";

  return `%${(number * 100).toLocaleString("tr-TR", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })}`;
}

function formatNumber(value, digits = 3) {
  const number = Number(value);
  if (value == null || value === "" || !Number.isFinite(number)) return "—";

  return number.toLocaleString("tr-TR", {
    maximumFractionDigits: digits,
  });
}

function formatPercentPoints(value) {
  const number = Number(value);
  if (value == null || value === "" || !Number.isFinite(number)) return "—";

  return `${number.toLocaleString("tr-TR", {
    maximumFractionDigits: 2,
  })}%`;
}

function normalizeKey(value) {
  return String(value)
    .toLocaleLowerCase("tr-TR")
    .replace(/[\s_%()-]/g, "")
    .replace(/[ıİ]/g, "i");
}

function findValue(object, candidates) {
  if (!object || typeof object !== "object") return null;

  const key = Object.keys(object).find((item) =>
    candidates.some((candidate) => normalizeKey(item) === normalizeKey(candidate))
  );

  return key ? object[key] : null;
}

function getRows(value) {
  if (Array.isArray(value)) return value;

  if (value && typeof value === "object") {
    if (Array.isArray(value.rows)) return value.rows;
    if (Array.isArray(value.data)) return value.data;
    if (Array.isArray(value.results)) return value.results;
    if (Array.isArray(value.records)) return value.records;
    return [value];
  }

  return [];
}

function getMetricAverage(rows, names) {
  const values = rows
    .map((row) => findValue(row, names))
    .filter((value) => value !== null && value !== "")
    .map(Number)
    .filter(Number.isFinite);

  if (!values.length) return null;

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function App() {
  const [selectedStock, setSelectedStock] = useState("TUPRS");
  const [prediction, setPrediction] = useState(null);
  const [priceHistory, setPriceHistory] = useState([]);
  const [performance, setPerformance] = useState(null);

  const [apiOnline, setApiOnline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [performanceLoading, setPerformanceLoading] = useState(false);

  const [error, setError] = useState("");
  const [historyError, setHistoryError] = useState("");
  const [performanceError, setPerformanceError] = useState("");

  const checkHealth = useCallback(async () => {
    try {
      await axios.get(`${API_URL}/health`, { timeout: 5000 });
      setApiOnline(true);
      return true;
    } catch {
      setApiOnline(false);
      return false;
    }
  }, []);

  const loadPrediction = useCallback(async (symbol) => {
    setLoading(true);
    setError("");

    try {
      const response = await axios.get(`${API_URL}/predict/${symbol}`, {
        timeout: 15000,
      });

      setPrediction(response.data);
      setApiOnline(true);
    } catch (err) {
      setPrediction(null);
      setError(
        err.response?.data?.detail ||
          "Tahmin alınamadı. Backend'i ve model dosyasını kontrol et."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async (symbol) => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await axios.get(
        `${API_URL}/stocks/${symbol}/history`,
        {
          params: { limit: 120 },
          timeout: 10000,
        }
      );

      const items = response.data?.prices;

      if (!Array.isArray(items)) {
        throw new Error("API fiyat geçmişini beklenen biçimde döndürmedi.");
      }

      const normalized = items
        .map((item) => ({
          date: String(item.date ?? ""),
          close: Number(item.close),
        }))
        .filter(
          (item) => item.date && Number.isFinite(item.close)
        )
        .sort((a, b) => a.date.localeCompare(b.date));

      setPriceHistory(normalized);

      if (!normalized.length) {
        setHistoryError("Gösterilecek fiyat geçmişi bulunamadı.");
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

  const loadPerformance = useCallback(async () => {
    setPerformanceLoading(true);
    setPerformanceError("");

    try {
      const response = await axios.get(`${API_URL}/model-performance`, {
        timeout: 10000,
      });

      setPerformance(response.data);
    } catch (err) {
      setPerformance(null);
      setPerformanceError(
        err.response?.data?.detail ||
          "Model performansı alınamadı."
      );
    } finally {
      setPerformanceLoading(false);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([
      checkHealth(),
      loadPrediction(selectedStock),
      loadHistory(selectedStock),
      loadPerformance(),
    ]);
  }, [
    checkHealth,
    loadPrediction,
    loadHistory,
    loadPerformance,
    selectedStock,
  ]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const positiveProbability =
    prediction?.probability_positive_return;

  const negativeProbability =
    prediction?.probability_non_positive_return;

  const signal = prediction?.signal || "Tahmin bekleniyor";

  const latestClose =
    prediction?.close ??
    priceHistory[priceHistory.length - 1]?.close;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <TrendingUp size={24} />
          </div>

          <div>
            <h2>FinanceAI</h2>
            <span>Market Intelligence</span>
          </div>
        </div>

        <div className="nav-label">MENÜ</div>

        <a href="#overview" className="nav-item active">
          <BarChart3 size={18} />
          <span>Genel Bakış</span>
        </a>

        <a href="#analysis" className="nav-item">
          <Activity size={18} />
          <span>Hisse Analizi</span>
        </a>

        <a href="#performance" className="nav-item">
          <ShieldCheck size={18} />
          <span>Model Performansı</span>
        </a>

        <div className="sidebar-bottom">
          <ShieldCheck size={17} />
          <div>
            <strong>FinanceAI</strong>
            <span>Deneysel analiz platformu</span>
          </div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div>
            <div className="eyebrow">PİYASA ANALİZ PLATFORMU</div>
            <h1>Genel Bakış</h1>
          </div>

          <div className="topbar-right">
            <div className={`status ${apiOnline ? "online" : "offline"}`}>
              {apiOnline ? <Wifi size={15} /> : <WifiOff size={15} />}
              <span>
                {apiOnline ? "API bağlantısı aktif" : "API bağlantısı yok"}
              </span>
            </div>

            <button
              className="refresh-button"
              onClick={refreshAll}
              disabled={loading || historyLoading || performanceLoading}
              aria-label="Verileri yenile"
              title="Verileri yenile"
            >
              <RefreshCw
                size={17}
                className={
                  loading || historyLoading || performanceLoading
                    ? "spin"
                    : ""
                }
              />
            </button>
          </div>
        </header>

        <section className="welcome">
          <div>
            <div className="eyebrow">YAPAY ZEKÂ DESTEKLİ ANALİZ</div>
            <h2>Piyasayı verilerle keşfet.</h2>
            <p>
              BIST hisseleri için makine öğrenmesi tabanlı deneysel analiz.
            </p>
          </div>

          <div className="welcome-icon">
            <TrendingUp size={42} />
          </div>
        </section>

        <section className="stock-toolbar" id="analysis">
          <div>
            <div className="section-label">ANALİZ EDİLEN HİSSE</div>
            <p>Hisse seçerek model tahminini güncelleyebilirsin.</p>
          </div>

          <div className="stock-actions">
            <select
              value={selectedStock}
              onChange={(event) => setSelectedStock(event.target.value)}
              aria-label="Hisse seçimi"
            >
              {STOCKS.map((stock) => (
                <option key={stock.symbol} value={stock.symbol}>
                  {stock.symbol} · {stock.name}
                </option>
              ))}
            </select>

            <button
              className="primary-button"
              onClick={refreshAll}
              disabled={loading || historyLoading || performanceLoading}
            >
              {loading ? "Analiz ediliyor..." : "Analiz et"}
            </button>
          </div>
        </section>

        {error && <div className="error-banner">Tahmin: {error}</div>}
        {historyError && (
          <div className="error-banner">Grafik: {historyError}</div>
        )}

        <section className="metrics-grid">
          <StatCard
            label="Son kapanış"
            value={formatPrice(latestClose)}
            detail={`Veri tarihi: ${prediction?.date || "—"}`}
            icon={<CircleDollarSign size={19} />}
          />

          <StatCard
            label="Pozitif getiri olasılığı"
            value={formatProbability(positiveProbability)}
            detail="Model tahmini"
            icon={<Activity size={19} />}
            accent
          />

          <StatCard
            label="Tahmin ufku"
            value={`${prediction?.horizon_trading_days ?? 20} gün`}
            detail="İşlem günü"
            icon={<TrendingUp size={19} />}
          />

          <StatCard
            label="Model"
            value={prediction?.model || "Random Forest"}
            detail="Deneysel sınıflandırıcı"
            icon={<ShieldCheck size={19} />}
          />
        </section>

        <section className="content-grid">
          <div className="panel">
            <div className="panel-header">
              <div>
                <h3>Fiyat grafiği</h3>
                <p>{selectedStock} · Geçmiş kapanış fiyatları</p>
              </div>
              <span className="tag">GÜNLÜK</span>
            </div>

            <div className="chart-area">
              {historyLoading ? (
                <div className="empty-chart">
                  <LoaderCircle size={27} className="spin" />
                  <strong>Fiyat geçmişi yükleniyor</strong>
                </div>
              ) : priceHistory.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={priceHistory}
                    margin={{ top: 10, right: 12, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#35d49a" stopOpacity={0.25} />
                        <stop offset="100%" stopColor="#35d49a" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>

                    <CartesianGrid stroke="#263440" strokeDasharray="3 3" />

                    <XAxis
                      dataKey="date"
                      tickFormatter={(value) => value.slice(5)}
                      stroke="#8494a4"
                      tick={{ fontSize: 10 }}
                      minTickGap={35}
                    />

                    <YAxis
                      domain={["auto", "auto"]}
                      stroke="#8494a4"
                      tick={{ fontSize: 10 }}
                      width={45}
                    />

                    <Tooltip
                      contentStyle={{
                        background: "#111a23",
                        border: "1px solid #30404e",
                        borderRadius: 8,
                        color: "#edf3f8",
                      }}
                      labelFormatter={(value) => `Tarih: ${value}`}
                      formatter={(value) => [formatPrice(value), "Kapanış"]}
                    />

                    <Area
                      type="monotone"
                      dataKey="close"
                      stroke="#35d49a"
                      strokeWidth={2}
                      fill="url(#priceFill)"
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-chart">
                  <BarChart3 size={30} />
                  <strong>Fiyat grafiği gösterilemiyor</strong>
                  <p>
                    Geçmiş fiyat endpoint'ini ve backend bağlantısını kontrol et.
                  </p>
                </div>
              )}
            </div>

            <div className="chart-footer">
              <span>{priceHistory.length} fiyat kaydı</span>
              <span>
                {priceHistory.length
                  ? `${priceHistory[0].date} — ${priceHistory[priceHistory.length - 1].date}`
                  : "Tarih aralığı yok"}
              </span>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <h3>Model değerlendirmesi</h3>
                <p>Seçili hisse için son tahmin</p>
              </div>
              <Activity size={19} className="green-icon" />
            </div>

            <div className="signal-box">
              <div className="eyebrow">MODEL SİNYALİ</div>
              <strong>{loading ? "Tahmin hesaplanıyor..." : signal}</strong>
              <p>
                Bu sinyal deneysel bir model tahminidir; kesin fiyat yönü veya
                getiri garantisi değildir.
              </p>
            </div>

            <ProbabilityBar
              label="Pozitif getiri olasılığı"
              value={positiveProbability}
              color="#35d49a"
            />

            <ProbabilityBar
              label="Pozitif olmayan getiri olasılığı"
              value={negativeProbability}
              color="#88a9ff"
            />

            <div className="last-updated">
              <div className="data-row">
                <span>Veri tarihi</span>
                <strong>{prediction?.date || "—"}</strong>
              </div>
              <div className="data-row">
                <span>Tahmin ufku</span>
                <strong>
                  {prediction?.horizon_trading_days ?? 20} işlem günü
                </strong>
              </div>
            </div>

            <div className="disclaimer">
              <ShieldCheck size={16} />
              <span>
                {prediction?.disclaimer ||
                  "Bu çıktı deneysel bir model tahminidir, yatırım tavsiyesi değildir."}
              </span>
            </div>
          </div>
        </section>

        <section className="panel performance-panel" id="performance">
          <div className="panel-header">
            <div>
              <h3>Model performansı</h3>
              <p>Walk-forward değerlendirmesi ve backtest karşılaştırması</p>
            </div>
            <ShieldCheck size={19} className="green-icon" />
          </div>

          {performanceLoading ? (
            <div className="empty-chart">
              <LoaderCircle size={28} className="spin" />
              <strong>Performans verileri yükleniyor</strong>
            </div>
          ) : performanceError ? (
            <div className="error-banner">{performanceError}</div>
          ) : performance ? (
            <PerformanceData data={performance} />
          ) : (
            <div className="empty-chart">
              <BarChart3 size={28} />
              <strong>Performans verisi bulunamadı</strong>
            </div>
          )}
        </section>

        <footer className="app-footer">
          <span>FinanceAI · Eğitim ve araştırma projesi</span>
          <span>
            {apiOnline ? <CheckCircle2 size={14} /> : <WifiOff size={14} />}
            API: {apiOnline ? "Bağlı" : "Bağlantı yok"}
          </span>
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
        {icon}
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
          style={{ width: `${ratio * 100}%`, background: color }}
        />
      </div>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <article className="performance-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function PerformanceData({ data }) {
  const walkForward = data?.walk_forward;
  const backtest = data?.backtest_comparison;

  const modelRows = getRows(walkForward?.rows ?? walkForward);
  const backtestRows = getRows(backtest?.rows ?? backtest);

  const randomForestRows = modelRows.filter((row) =>
    String(findValue(row, ["Model"]) ?? "")
      .toLocaleLowerCase("tr-TR")
      .includes("random forest")
  );

  const logisticRows = modelRows.filter((row) =>
    String(findValue(row, ["Model"]) ?? "")
      .toLocaleLowerCase("tr-TR")
      .includes("logistic regression")
  );

  const buildModelMetrics = (rows) => ({
    accuracy: getMetricAverage(rows, ["Ortalama Accuracy", "Accuracy", "Mean Accuracy"]),
    balanced: getMetricAverage(rows, ["Ortalama Balanced Accuracy", "Balanced Accuracy", "Mean Balanced Accuracy"]),
    precision: getMetricAverage(rows, ["Ortalama Precision", "Precision", "Mean Precision"]),
    recall: getMetricAverage(rows, ["Ortalama Recall", "Recall", "Mean Recall"]),
    f1: getMetricAverage(rows, ["Ortalama F1", "F1", "F1 Score", "Mean F1"]),
  });

  const rf = buildModelMetrics(randomForestRows);
  const logistic = buildModelMetrics(logisticRows);

  const formatMetricPercent = (value) =>
    value == null
      ? "—"
      : formatPercentPoints(value * 100);

  const strategyName = (row, index) =>
    String(findValue(row, ["Strateji", "Strategy"]) ?? `Strateji ${index + 1}`);

  const backtestIsUseful = backtestRows.some(
    (row) =>
      findValue(row, ["Son Sermaye (TL)", "Son Sermaye", "Final Equity"]) !== null
  );

  return (
    <div className="performance-content">
      <section className="performance-block">
        <div className="performance-section-heading">
          <div>
            <h4>Walk-forward sonuçları</h4>
            <p>Zaman sıralı test katlarının model metrikleri.</p>
          </div>
          <span className="performance-badge">MODEL</span>
        </div>

        {randomForestRows.length ? (
          <>
            <div className="performance-model-title">
              <strong>Random Forest</strong>
              <span>{randomForestRows.length} kayıt</span>
            </div>
            <div className="performance-metrics">
              <MetricCard label="Doğruluk" value={formatMetricPercent(rf.accuracy)} />
              <MetricCard label="Dengeli doğruluk" value={formatMetricPercent(rf.balanced)} />
              <MetricCard label="Precision" value={formatMetricPercent(rf.precision)} />
              <MetricCard label="Recall" value={formatMetricPercent(rf.recall)} />
              <MetricCard label="F1 skoru" value={formatNumber(rf.f1)} />
            </div>
          </>
        ) : (
          <div className="performance-empty">
            Random Forest metrikleri bulunamadı.
          </div>
        )}

        {logisticRows.length > 0 && (
          <div className="comparison-model">
            <h5>Logistic Regression</h5>
            <div className="performance-metrics">
              <MetricCard label="Doğruluk" value={formatMetricPercent(logistic.accuracy)} />
              <MetricCard label="Dengeli doğruluk" value={formatMetricPercent(logistic.balanced)} />
              <MetricCard label="Precision" value={formatMetricPercent(logistic.precision)} />
              <MetricCard label="Recall" value={formatMetricPercent(logistic.recall)} />
              <MetricCard label="F1 skoru" value={formatNumber(logistic.f1)} />
            </div>
          </div>
        )}

        {!randomForestRows.length && !logisticRows.length && (
          <details className="raw-data">
            <summary>Walk-forward API verisini görüntüle</summary>
            <pre>{JSON.stringify(walkForward, null, 2)}</pre>
          </details>
        )}
      </section>

      <section className="performance-block">
        <div className="performance-section-heading">
          <div>
            <h4>Backtest karşılaştırması</h4>
            <p>Stratejilerin sermaye ve risk sonuçları.</p>
          </div>
          <span className="performance-badge">BACKTEST</span>
        </div>

        {backtestIsUseful ? (
          <div className="backtest-table-wrap">
            <table className="backtest-table">
              <thead>
                <tr>
                  <th>Strateji</th>
                  <th>Son sermaye</th>
                  <th>Toplam getiri</th>
                  <th>Maks. düşüş</th>
                  <th>Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {backtestRows.map((row, index) => {
                  const name = strategyName(row, index);
                  const capital = findValue(row, ["Son Sermaye (TL)", "Son Sermaye", "Final Equity"]);
                  const totalReturn = findValue(row, ["Toplam Getiri (%)", "Toplam Getiri", "Total Return"]);
                  const drawdown = findValue(row, ["Maksimum Düşüş (%)", "Maksimum Düşüş", "Max Drawdown"]);
                  const sharpe = findValue(row, ["Yıllıklandırılmış Sharpe", "Sharpe", "Sharpe Ratio"]);

                  const returnNumber = Number(totalReturn);

                  return (
                    <tr key={`${name}-${index}`}>
                      <td>{name}</td>
                      <td>{formatPrice(capital)}</td>
                      <td
                        className={
                          Number.isFinite(returnNumber)
                            ? returnNumber >= 0
                              ? "positive-number"
                              : "negative-number"
                            : ""
                        }
                      >
                        {formatPercentPoints(totalReturn)}
                      </td>
                      <td>{formatPercentPoints(drawdown)}</td>
                      <td>{formatNumber(sharpe)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="performance-empty">
            Backtest verileri beklenen alanlarda bulunamadı.
          </div>
        )}

        {!backtestIsUseful && (
          <details className="raw-data">
            <summary>Backtest API verisini görüntüle</summary>
            <pre>{JSON.stringify(backtest, null, 2)}</pre>
          </details>
        )}
      </section>
    </div>
  );
}

export default App;
