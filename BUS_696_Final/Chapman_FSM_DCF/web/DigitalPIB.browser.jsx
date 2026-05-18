const { useState, useEffect, useCallback } = React;

// ─── Constants ────────────────────────────────────────────────────────────────
const DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"];

const SECTIONS = [
  { key: "snapshot", label: "Snapshot", icon: "◈" },
  { key: "income", label: "Income Stmt", icon: "▤" },
  { key: "balance", label: "Balance Sheet", icon: "⊞" },
  { key: "cashflow", label: "Cash Flow", icon: "◎" },
  { key: "valuation", label: "Valuation", icon: "◇" },
  { key: "estimates", label: "Estimates", icon: "⟁" },
  { key: "news", label: "Recent News", icon: "◉" },
];

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

  :root {
    --bg:        #0a0e1a;
    --surface:   #0f1628;
    --border:    #1e2d4a;
    --border2:   #243558;
    --accent:    #00d4ff;
    --accent2:   #0099cc;
    --green:     #00e896;
    --red:       #ff4d6d;
    --amber:     #ffb547;
    --text:      #e8eef8;
    --muted:     #6b7fa3;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.5;
  }

  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    background: var(--bg);
  }

  /* ── Top Bar ── */
  .topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    z-index: 10;
  }

  .topbar-logo {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .topbar-logo span { color: var(--muted); font-weight: 300; }

  .ticker-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    flex: 1;
  }

  .ticker-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s;
    user-select: none;
  }

  .ticker-pill:hover { border-color: var(--accent); color: var(--accent); }
  .ticker-pill.active { background: var(--accent); color: #000; border-color: var(--accent); }
  .ticker-pill.loading { opacity: 0.5; }

  .ticker-pill .remove {
    font-size: 10px;
    color: var(--muted);
    line-height: 1;
    padding: 0 1px;
  }
  .ticker-pill:hover .remove { color: var(--red); }
  .ticker-pill.active .remove { color: rgba(0,0,0,0.5); }

  .add-ticker {
    display: flex;
    gap: 6px;
  }

  .ticker-input {
    width: 80px;
    padding: 4px 8px;
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: 3px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    outline: none;
    transition: border-color 0.15s;
  }
  .ticker-input:focus { border-color: var(--accent); }
  .ticker-input::placeholder { color: var(--muted); font-weight: 300; text-transform: none; }

  .btn-add {
    padding: 4px 12px;
    background: transparent;
    border: 1px solid var(--accent2);
    border-radius: 3px;
    color: var(--accent);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-add:hover { background: var(--accent); color: #000; }

  /* ── Layout ── */
  .main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  .sidebar {
    width: 130px;
    flex-shrink: 0;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 12px 0;
    gap: 2px;
  }

  .sidebar-label {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0 14px 8px;
  }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    cursor: pointer;
    border-left: 2px solid transparent;
    transition: all 0.12s;
    font-size: 12px;
    color: var(--muted);
  }
  .nav-item:hover { color: var(--text); background: rgba(255,255,255,0.03); }
  .nav-item.active { color: var(--accent); border-left-color: var(--accent); background: rgba(0,212,255,0.05); }

  .nav-icon {
    font-size: 11px;
    width: 14px;
    text-align: center;
    flex-shrink: 0;
  }

  .nav-label { font-family: var(--sans); font-size: 11px; font-weight: 500; }

  /* ── Content ── */
  .content {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    scrollbar-width: thin;
    scrollbar-color: var(--border2) transparent;
  }

  .content-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }

  .content-title {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .ticker-badge {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 600;
    color: var(--accent);
  }

  .company-name {
    font-size: 13px;
    color: var(--text);
    font-weight: 300;
  }

  /* ── Loading / Error ── */
  .state-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    gap: 12px;
    color: var(--muted);
  }

  .spinner {
    width: 28px;
    height: 28px;
    border: 2px solid var(--border2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .loading-text {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.05em;
  }

  .error-text {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--red);
    text-align: center;
    max-width: 400px;
    line-height: 1.6;
  }

  /* ── Snapshot cards ── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
  }

  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 14px;
    transition: border-color 0.15s;
  }
  .kpi-card:hover { border-color: var(--border2); }

  .kpi-label {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .kpi-value {
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
  }

  .kpi-sub {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
  }

  .up { color: var(--green); }
  .down { color: var(--red); }
  .neutral { color: var(--amber); }

  /* ── Data Table ── */
  .data-section {
    margin-bottom: 24px;
  }

  .section-title {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 11px;
  }

  .data-table th {
    text-align: right;
    padding: 6px 10px;
    background: var(--surface);
    color: var(--muted);
    font-weight: 500;
    font-size: 10px;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }
  .data-table th:first-child { text-align: left; }

  .data-table td {
    padding: 6px 10px;
    text-align: right;
    border-bottom: 1px solid rgba(30,45,74,0.5);
    color: var(--text);
  }
  .data-table td:first-child {
    text-align: left;
    color: var(--muted);
    font-weight: 400;
  }

  .data-table tr:hover td { background: rgba(255,255,255,0.02); }
  .data-table tr.bold td { font-weight: 600; color: var(--accent); }
  .data-table tr.subtotal td { color: var(--text); font-weight: 500; border-top: 1px solid var(--border2); }

  /* ── News items ── */
  .news-list { display: flex; flex-direction: column; gap: 10px; }

  .news-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 14px;
    transition: border-color 0.15s;
  }
  .news-item:hover { border-color: var(--border2); }

  .news-headline {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 4px;
    line-height: 1.4;
  }

  .news-meta {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    display: flex;
    gap: 12px;
  }

  .news-sentiment {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 2px;
    letter-spacing: 0.08em;
  }
  .news-sentiment.positive { background: rgba(0,232,150,0.15); color: var(--green); }
  .news-sentiment.negative { background: rgba(255,77,109,0.15); color: var(--red); }
  .news-sentiment.neutral  { background: rgba(255,181,71,0.12); color: var(--amber); }

  /* ── Prose blocks (for narrative data) ── */
  .prose-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 16px;
    font-size: 12px;
    line-height: 1.7;
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    white-space: pre-wrap;
  }

  /* ── No ticker selected ── */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
    color: var(--muted);
  }

  .empty-icon {
    font-size: 40px;
    opacity: 0.3;
  }

  .empty-title {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
  }

  .empty-sub { font-size: 12px; font-weight: 300; }

  /* ── Refresh button ── */
  .refresh-btn {
    padding: 3px 10px;
    background: transparent;
    border: 1px solid var(--border2);
    border-radius: 3px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 10px;
    cursor: pointer;
    transition: all 0.15s;
    margin-left: auto;
  }
  .refresh-btn:hover { border-color: var(--accent); color: var(--accent); }

  .timestamp {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    margin-left: 8px;
  }

  /* ── Scrollbar ── */
  .content::-webkit-scrollbar { width: 4px; }
  .content::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
`;

// ─── API Call ─────────────────────────────────────────────────────────────────
// ─── API Integration (Financial Modeling Prep) ───────────────────────────────
// ─── Cloud Infrastructure (Google Cloud Platform) ───────────────────────────
const CLOUD_FUNCTION_URL = window.CLOUD_FUNCTION_URL;

function formatCurrency(val) {
  if (val === null || val === undefined || isNaN(val)) return "N/A";
  if (Math.abs(val) >= 1e12) return `$${(val / 1e12).toFixed(2)}T`;
  if (Math.abs(val) >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
  if (Math.abs(val) >= 1e6) return `$${(val / 1e6).toFixed(2)}M`;
  return `$${val.toLocaleString()}`;
}

async function fetchStockData(ticker, section) {
  try {
    const res = await fetch(CLOUD_FUNCTION_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, section })
    });

    if (!res.ok) throw new Error(`Cloud Function Error: ${res.status}`);
    const data = await res.json();

    // ─── Data Mapping Logic ───
    switch (section) {
      case "snapshot": {
        // Handle logic where Cloud Function might return combined profile/quote
        const p = Array.isArray(data) ? data[0] : (data.profile?.[0] || data);
        const q = Array.isArray(data) ? data[0] : (data.quote?.[0] || data);
        return {
          companyName: p.companyName || q.name,
          ticker: ticker,
          exchange: p.exchangeShortName || q.exchange,
          price: `$${q.price?.toFixed(2) || "0.00"}`,
          change: `${q.change >= 0 ? "+" : ""}${q.change?.toFixed(2)} (${q.changesPercentage?.toFixed(2)}%)`,
          changeDirection: q.change > 0 ? "up" : q.change < 0 ? "down" : "neutral",
          marketCap: formatCurrency(q.marketCap),
          peRatio: q.pe ? `${q.pe.toFixed(1)}x` : "N/A",
          evEbitda: "N/A",
          eps: `$${q.eps?.toFixed(2) || "0.00"}`,
          dividendYield: p.lastDiv ? `${((p.lastDiv / q.price) * 100).toFixed(2)}%` : "0.0%",
          week52High: `$${q.yearHigh?.toFixed(2)}`,
          week52Low: `$${q.yearLow?.toFixed(2)}`,
          avgVolume: formatCurrency(q.avgVolume),
          beta: p.beta?.toFixed(2) || "N/A",
          sector: p.sector,
          description: p.description?.split(". ").slice(0, 2).join(". ") + "."
        };
      }

      case "income":
      case "balance":
      case "cashflow": {
        // These sections use the standard DataTable format
        if (!Array.isArray(data)) return data; // Already mapped by backend?
        const rows = section === "income" ? [
          { label: "Revenue", values: data.map(d => formatCurrency(d.revenue)), style: "bold" },
          { label: "Gross Profit", values: data.map(d => formatCurrency(d.grossProfit)), style: "subtotal" },
          { label: "Net Income", values: data.map(d => formatCurrency(d.netIncome)), style: "bold" },
          { label: "EPS (Diluted)", values: data.map(d => `$${d.epsdiluted?.toFixed(2)}`), style: "normal" }
        ] : section === "balance" ? [
          { label: "Total Assets", values: data.map(d => formatCurrency(d.totalAssets)), style: "bold" },
          { label: "Total Liabilities", values: data.map(d => formatCurrency(d.totalLiabilities)), style: "bold" },
          { label: "Total Equity", values: data.map(d => formatCurrency(d.totalStockholdersEquity)), style: "bold" }
        ] : [
          { label: "Cash From Operations", values: data.map(d => formatCurrency(d.netCashProvidedByOperatingActivities)), style: "bold" },
          { label: "Free Cash Flow", values: data.map(d => formatCurrency(d.freeCashFlow)), style: "bold" }
        ];

        return {
          years: data.map(d => d.date?.split("-")[0] || "N/A"),
          rows: rows
        };
      }

      case "news": {
        if (!Array.isArray(data)) return data;
        return {
          items: data.map(n => ({
            headline: n.title,
            source: n.site,
            date: new Date(n.publishedDate).toLocaleDateString(),
            sentiment: "neutral",
            summary: n.text
          }))
        };
      }

      default: return data;
    }
  } catch (err) {
    console.error("GCP Fetch Error:", err);
    throw err;
  }
}



// ─── Sub-components ───────────────────────────────────────────────────────────
function KPICard({ label, value, sub, colorClass }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value ${colorClass || ""}`}>{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function DataTable({ years, rows }) {
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Metric</th>
          {years.map(y => <th key={y}>{y}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i} className={row.style !== "normal" ? row.style : ""}>
            <td>{row.label}</td>
            {row.values.map((v, j) => <td key={j}>{v}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SnapshotView({ data }) {
  if (!data) return null;
  const isUp = data.changeDirection === "up";
  const isDown = data.changeDirection === "down";
  return (
    <div>
      <div className="data-section">
        <div className="kpi-grid">
          <KPICard label="PRICE" value={data.price}
            sub={data.change} colorClass={isUp ? "up" : isDown ? "down" : "neutral"} />
          <KPICard label="MARKET CAP" value={data.marketCap} />
          <KPICard label="P/E RATIO" value={data.peRatio} />
          <KPICard label="EV/EBITDA" value={data.evEbitda} />
          <KPICard label="EPS (TTM)" value={data.eps} />
          <KPICard label="DIV YIELD" value={data.dividendYield} />
          <KPICard label="52W HIGH" value={data.week52High} />
          <KPICard label="52W LOW" value={data.week52Low} />
          <KPICard label="AVG VOLUME" value={data.avgVolume} />
          <KPICard label="BETA" value={data.beta} />
        </div>
      </div>
      <div className="data-section">
        <div className="section-title">COMPANY OVERVIEW</div>
        <div className="prose-block">{data.description}{"\n\n"}Sector: {data.sector}   |   Exchange: {data.exchange}</div>
      </div>
    </div>
  );
}

function ValuationView({ data }) {
  if (!data) return null;
  const { currentMultiples, historicalMultiples, analystTargets, dcfSummary } = data;
  return (
    <div>
      <div className="data-section">
        <div className="section-title">CURRENT TRADING MULTIPLES</div>
        <table className="data-table">
          <thead><tr><th>Metric</th><th>Value</th><th>vs. Peers</th></tr></thead>
          <tbody>
            {currentMultiples.map((m, i) => (
              <tr key={i}>
                <td>{m.metric}</td>
                <td>{m.value}</td>
                <td style={{ color: m.vsPeers?.includes("premium") ? "var(--amber)" : "var(--muted)" }}>{m.vsPeers}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="data-section">
        <div className="section-title">HISTORICAL MULTIPLES</div>
        <table className="data-table">
          <thead><tr><th>Period</th>{historicalMultiples.years.map(y => <th key={y}>{y}</th>)}</tr></thead>
          <tbody>
            <tr><td>P/E</td>{historicalMultiples.pe.map((v, i) => <td key={i}>{v}</td>)}</tr>
            <tr><td>EV/EBITDA</td>{historicalMultiples.evEbitda.map((v, i) => <td key={i}>{v}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <div className="data-section">
        <div className="section-title">ANALYST PRICE TARGETS</div>
        <div className="kpi-grid">
          <KPICard label="CONSENSUS" value={analystTargets.consensus} sub={`Upside: ${analystTargets.upside}`} colorClass="up" />
          <KPICard label="HIGH TARGET" value={analystTargets.high} />
          <KPICard label="LOW TARGET" value={analystTargets.low} />
          <KPICard label="BUYS / HOLDS / SELLS" value={`${analystTargets.buys} / ${analystTargets.holds} / ${analystTargets.sells}`} />
        </div>
      </div>
      <div className="data-section">
        <div className="section-title">DCF COMMENTARY</div>
        <div className="prose-block">{dcfSummary}</div>
      </div>
    </div>
  );
}

function EstimatesView({ data }) {
  if (!data) return null;
  return (
    <div>
      <div className="data-section">
        <div className="section-title">CONSENSUS ESTIMATES — {data.numAnalysts} ANALYSTS</div>
        <DataTable years={data.years} rows={data.rows} />
      </div>
    </div>
  );
}

function NewsView({ data }) {
  if (!data) return null;
  return (
    <div className="news-list">
      {data.items.map((item, i) => (
        <div className="news-item" key={i}>
          <div className="news-headline">{item.headline}</div>
          <div style={{ marginBottom: 6 }}>
            <span className={`news-sentiment ${item.sentiment}`}>{item.sentiment.toUpperCase()}</span>
          </div>
          <div style={{ fontSize: 12, color: "var(--muted)", fontFamily: "var(--sans)", fontWeight: 300, lineHeight: 1.6 }}>{item.summary}</div>
          <div className="news-meta" style={{ marginTop: 8 }}>
            <span>{item.source}</span>
            <span>{item.date}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
function DigitalPIB() {
  const [tickers, setTickers] = useState(DEFAULT_TICKERS);
  const [activeTicker, setActiveTicker] = useState(null);
  const [activeSection, setActiveSection] = useState("snapshot");
  const [inputVal, setInputVal] = useState("");
  const [cache, setCache] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [timestamp, setTimestamp] = useState(null);

  const cacheKey = activeTicker && activeSection ? `${activeTicker}__${activeSection}` : null;
  const currentData = cacheKey ? cache[cacheKey] : null;

  const load = useCallback(async (ticker, section, force = false) => {
    const key = `${ticker}__${section}`;
    if (!force && cache[key]) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchStockData(ticker, section);
      setCache(prev => ({ ...prev, [key]: data }));
      setTimestamp(new Date().toLocaleTimeString());
    } catch (e) {
      setError(`Failed to load ${section} data for ${ticker}: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [cache]);

  const saveToCloud = async () => {
    if (!activeTicker || !currentData) return;
    setLoading(true);
    try {
      const res = await fetch(CLOUD_FUNCTION_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: activeTicker,
          section: activeSection,
          action: 'store',
          content: currentData
        })
      });
      const result = await res.json();
      alert(`Successfully saved to GCS: financial-model/${result.path}`);
    } catch (e) {
      alert(`Cloud Save Failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTicker && activeSection) load(activeTicker, activeSection);
  }, [activeTicker, activeSection]);

  const addTicker = () => {
    const t = inputVal.trim().toUpperCase();
    if (t && !tickers.includes(t)) {
      setTickers(prev => [...prev, t]);
      setActiveTicker(t);
      setActiveSection("snapshot");
    }
    setInputVal("");
  };

  const removeTicker = (t, e) => {
    e.stopPropagation();
    setTickers(prev => prev.filter(x => x !== t));
    if (activeTicker === t) setActiveTicker(null);
  };

  const renderContent = () => {
    if (!activeTicker) return (
      <div className="empty-state">
        <div className="empty-icon">◈</div>
        <div className="empty-title">SELECT A TICKER</div>
        <div className="empty-sub">Choose a stock above or add a new one to begin</div>
      </div>
    );

    if (loading) return (
      <div className="state-box">
        <div className="spinner" />
        <div className="loading-text">Fetching live data for {activeTicker}…</div>
      </div>
    );

    if (error) return (
      <div className="state-box">
        <div className="error-text">{error}</div>
      </div>
    );

    if (!currentData) return (
      <div className="state-box">
        <div className="loading-text">No data loaded yet.</div>
      </div>
    );

    switch (activeSection) {
      case "snapshot": return <SnapshotView data={currentData} />;
      case "income": return <div className="data-section"><DataTable years={currentData.years} rows={currentData.rows} /></div>;
      case "balance": return <div className="data-section"><DataTable years={currentData.years} rows={currentData.rows} /></div>;
      case "cashflow": return <div className="data-section"><DataTable years={currentData.years} rows={currentData.rows} /></div>;
      case "valuation": return <ValuationView data={currentData} />;
      case "estimates": return <EstimatesView data={currentData} />;
      case "news": return <NewsView data={currentData} />;
      default: return null;
    }
  };

  const activeLabel = SECTIONS.find(s => s.key === activeSection)?.label;

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        {/* Top Bar */}
        <div className="topbar">
          <div className="topbar-logo">DIGITAL PIB <span>/ Public Information Book</span></div>
          <div className="ticker-pills">
            {tickers.map(t => (
              <div
                key={t}
                className={`ticker-pill ${activeTicker === t ? "active" : ""} ${loading && activeTicker === t ? "loading" : ""}`}
                onClick={() => { setActiveTicker(t); setActiveSection("snapshot"); }}
              >
                {t}
                <span className="remove" onClick={e => removeTicker(t, e)}>✕</span>
              </div>
            ))}
          </div>
          <div className="add-ticker">
            <input
              className="ticker-input"
              value={inputVal}
              onChange={e => setInputVal(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && addTicker()}
              placeholder="Add ticker…"
              maxLength={6}
            />
            <button className="btn-add" onClick={addTicker}>+ ADD</button>
          </div>
        </div>

        <div className="main">
          {/* Sidebar */}
          <div className="sidebar">
            <div className="sidebar-label">Sections</div>
            {SECTIONS.map(s => (
              <div
                key={s.key}
                className={`nav-item ${activeSection === s.key ? "active" : ""}`}
                onClick={() => { setActiveSection(s.key); if (activeTicker) load(activeTicker, s.key); }}
              >
                <span className="nav-icon">{s.icon}</span>
                <span className="nav-label">{s.label}</span>
              </div>
            ))}
          </div>

          {/* Content */}
          <div className="content">
            {activeTicker && (
              <div className="content-header">
                <div className="content-title">{activeLabel}</div>
                <div className="ticker-badge">{activeTicker}</div>
                {currentData && <div className="company-name">{currentData.companyName || ""}</div>}
                {timestamp && <div className="timestamp">as of {timestamp}</div>}
                {activeTicker && !loading && (
                  <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
                    <button className="refresh-btn" style={{ marginLeft: 0 }} onClick={saveToCloud}>
                      ☁ Save to Cloud
                    </button>
                    <button className="refresh-btn" style={{ marginLeft: 0 }} onClick={() => load(activeTicker, activeSection, true)}>
                      ↻ Refresh
                    </button>
                  </div>
                )}
              </div>
            )}
            {renderContent()}
          </div>
        </div>
      </div>
    </>
  );
}


const rootEl = document.getElementById('root');
const root = ReactDOM.createRoot(rootEl);
root.render(<DigitalPIB />);

