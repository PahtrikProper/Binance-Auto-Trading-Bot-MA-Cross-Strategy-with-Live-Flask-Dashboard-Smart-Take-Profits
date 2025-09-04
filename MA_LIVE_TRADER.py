from flask import Flask, render_template_string, request, redirect, url_for, jsonify
import threading, requests, json, time, math, hmac, hashlib, traceback, os
from urllib.parse import urlencode
import websocket

# ======================= USER CONFIGURATION ========================
SYMBOL = "ADAUSDT"
INTERVAL = "3m"
API_KEY = "insert your binance api key here"
API_SECRET = "insert the api secret key here"
FEE_RATE_TRADE = 0.001
SLIPPAGE_PCT = 0.05
TP_LEVELS = [0.70, 0.40, 0.22]
TP_ARM_BUFFER = 0.10
TP_EXIT_BUFFER = 0.01
MIN_BUY_CONFIRM_TICKS = 1
SELL_CONFIRM_TICKS = 3
TICK_THROTTLE_SECS = 15

MIN_FAST_LEN = 2
MAX_SLOW_LEN = 200

BACKTEST_HOURS = 24
CHART_HIST_LIMIT = 2000
WS_RECONNECT_SECS = 10

BASE_URL = "https://api.binance.com"
PUBLIC_TIMEOUT = 20
SIGNED_TIMEOUT = 20

INTERVAL_TO_SECONDS = {
    "1s": 1, "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000
}
CANDLE_SECONDS = INTERVAL_TO_SECONDS.get(INTERVAL, 180)
INTERVAL_MS = CANDLE_SECONDS * 1000

BASE_ASSET = SYMBOL[:-4] if SYMBOL.endswith("USDT") else SYMBOL.split("USDT")[0]
QUOTE_ASSET = "USDT"

# ========== STATE & LOGGING ==========
state = {
    "live_running": False,
    "stop_signal": False,
    "live_log": [],
    "last_trade": None,
    "best_params": None,
    "backtest_metrics": None,
    "markers": [],
    "trades": [],
}

app = Flask(__name__)

# ========== MODERN HTML DASHBOARD ==========
HTML = """
<!doctype html>
<html>
<head>
    <title>{{ symbol }} EMA Tick Trader</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
      body { background: #16181c; color: #e5e7eb; font-family: Arial,sans-serif; margin:0;}
      .container { max-width: 1100px; margin: 28px auto; padding: 24px; background: #1b1e25; border-radius: 12px; box-shadow: 0 2px 18px #0004;}
      h1 { color: #48fa83; }
      h2 { color: #45caff; }
      .status { font-weight: bold; }
      .row { display: flex; gap: 40px; }
      .col { flex:1;}
      button, input[type=number] { font-size:1.1em; border-radius:7px; border:1px ADAid #333;}
      button { background:#45caff; color:#131313; padding: 8px 20px;}
      button:disabled { background:#5c7087; color:#ddd;}
      pre { font-size:1em; background: #232838; color: #89e2ff; border-radius:8px; padding:7px 13px;}
      .logs { background:#181b24; color:#bbf7d0; max-height:180px; overflow:auto; border-radius:7px; margin-bottom: 8px;}
      .statbox { background: #13181f; border-radius:8px; padding:7px 11px; margin:7px 0; font-size:1.05em;}
      .chart-wrap { background: #13181f; border-radius:12px; padding:8px; margin-bottom:18px;}
    </style>
    <script>
    let chart, priceSeries, fastSeries, slowSeries;
    function initChart() {
      chart = LightweightCharts.createChart(document.getElementById('chart'), {
        width: 1040, height: 420,
        layout: { background: { color: '#1b1e25' }, textColor: '#e5e7eb' },
        grid: { vertLines: { color: '#23293a' }, horzLines: { color: '#23293a' } },
        rightPriceScale: { borderColor: '#2b3245' },
        timeScale: { borderColor: '#2b3245', timeVisible:true, secondsVisible:true }
      });
      priceSeries = chart.addLineSeries({ color:'deepskyblue', lineWidth:2 });
      fastSeries = chart.addLineSeries({ color:'orange', lineWidth:2 });
      slowSeries = chart.addLineSeries({ color:'yellow', lineWidth:2 });
    }
    function updateChart() {
      fetch('/chartdata').then(r=>r.json()).then(d=>{
        priceSeries.setData(d.prices);
        fastSeries.setData(d.fast);
        slowSeries.setData(d.slow);
        priceSeries.setMarkers(d.markers);
      });
    }
    function poll() {
      fetch('/status').then(r=>r.json()).then(d=>{
        document.getElementById('live_status').innerHTML = d.live_status;
        document.getElementById('live_log').innerText = d.live_log.join('\\n');
        document.getElementById('best_result').innerHTML = d.best_result;
        document.getElementById('ema_settings').innerHTML = d.ema_settings;
        document.getElementById('trades').innerHTML = d.trades;
        document.getElementById('backtest_metrics').innerHTML = d.backtest_metrics;
      });
      updateChart();
      setTimeout(poll, 4000);
    }
    window.onload = function() { initChart(); poll(); }
    </script>
</head>
<body>
<div class="container">
  <h1>{{ symbol }} Tick EMA Dashboard</h1>
  <div class="chart-wrap">
    <div id="chart" style="width:1040px; height:420px;"></div>
  </div>
  <div class="row">
    <div class="col">
      <h2>Backtest</h2>
      <form method="post" action="/start_backtest">
        <button type="submit">Start Backtest</button>
      </form>
      <div class="statbox" id="best_result"></div>
      <div class="statbox" id="backtest_metrics"></div>
    </div>
    <div class="col">
      <h2>Live Trading</h2>
      <form method="post" action="/start_live"><button type="submit" {% if live_running %}disabled{% endif %}>Start Live Trader</button></form>
      <form method="post" action="/stop_live"><button type="submit" style="background:#ff5c5c;color:white;" {% if not live_running %}disabled{% endif %}>Stop Live Trader</button></form>
      <div class="status" id="live_status"></div>
      <pre class="logs" id="live_log"></pre>
    </div>
    <div class="col">
      <h2>Best Params</h2>
      <div class="statbox" id="ema_settings"></div>
      <h2>Last Trades</h2>
      <div id="trades" class="logs"></div>
    </div>
  </div>
</div>
</body>
</html>
"""
# ====================== BINANCE & STRATEGY HELPERS ======================
def public_get(path, params=None):
    r = requests.get(BASE_URL + path, params=params, timeout=PUBLIC_TIMEOUT)
    r.raise_for_status()
    return r.json()

def signed_headers():
    if not API_KEY: raise RuntimeError("BINANCE_API_KEY not set")
    return {"X-MBX-APIKEY": API_KEY}

def signed_request(method, path, params=None):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing API keys")
    if params is None: params = {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    query = urlencode(params, doseq=True)
    signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    url = BASE_URL + path
    if method == "GET":
        r = requests.get(url, headers=signed_headers(), params=params, timeout=SIGNED_TIMEOUT)
    elif method == "POST":
        r = requests.post(url, headers=signed_headers(), params=params, timeout=SIGNED_TIMEOUT)
    else:
        raise ValueError("Unsupported method")
    r.raise_for_status()
    return r.json()

def fetch_klines_recent(symbol, interval, limit=CHART_HIST_LIMIT):
    return public_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})

def to_price_time(candles):
    prices, times = [], []
    for c in candles:
        times.append(c[0] // 1000)
        prices.append(float(c[4]))
    return prices, times

def prefix_sums(arr):
    ps, s = [0.0]*len(arr), 0.0
    for i, v in enumerate(arr):
        s += v
        ps[i] = s
    return ps

def mean_at(ps, i, win):
    if win == 0: return 0.0
    j = i - win
    total = ps[i] if j < 0 else ps[i] - ps[j]
    return total / win

def calc_ma_series(prices, window):
    if len(prices) < window or window == 0: return [None]*len(prices)
    ps = prefix_sums(prices)
    out = [None]*(window-1)
    for i in range(window-1, len(prices)):
        out.append(mean_at(ps, i, window))
    return out

# --- Binance Symbol Filters & Balance Helpers ---
_filters_cache = None
def get_symbol_filters(symbol):
    global _filters_cache
    if _filters_cache is None:
        info = public_get("/api/v3/exchangeInfo")
        _filters_cache = {s["symbol"]: s for s in info.get("symbols", [])}
    s = _filters_cache.get(symbol)
    if not s:
        raise RuntimeError(f"Symbol {symbol} not found in exchangeInfo")
    lot = next((f for f in s["filters"] if f["filterType"] == "LOT_SIZE"), None)
    notional = next((f for f in s["filters"] if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")), None)
    step_size = float(lot["stepSize"]) if lot else 0.000001
    min_qty = float(lot["minQty"]) if lot else 0.0
    min_notional = float(notional["minNotional"]) if notional and "minNotional" in notional else 0.0
    step_str = f"{step_size:.16f}".rstrip("0")
    qty_precision = len(step_str.split(".")[1]) if "." in step_str else 0
    return step_size, min_qty, min_notional, qty_precision

def get_balances():
    acct = signed_request("GET", "/api/v3/account")
    bals = {b["asset"]: (float(b["free"]), float(b["locked"])) for b in acct["balances"]}
    base_free = bals.get(BASE_ASSET, (0.0, 0.0))[0]
    quote_free = bals.get(QUOTE_ASSET, (0.0, 0.0))[0]
    return base_free, quote_free

def buy(quote_balance, price, fee, step_size, min_qty, min_notional, qty_precision):
    raw_qty = quote_balance / (price * (1 + fee))
    qty = math.floor(raw_qty / step_size) * step_size
    qty = math.floor(qty * (10 ** qty_precision)) / (10 ** qty_precision)
    notional = qty * price
    if qty < min_qty or notional < min_notional:
        state['live_log'].append(f"[Buy] Skipped: qty={qty:.5f} notional={notional:.3f} (min_qty={min_qty}, min_notional={min_notional})")
        return None
    try:
        res = signed_request("POST", "/api/v3/order",
                             {"symbol": SYMBOL, "side": "BUY", "type": "MARKET", "quantity": f"{qty:.{qty_precision}f}"})
        state['live_log'].append(f"[Buy] Market order placed: qty={qty:.4f} @ {price:.4f}")
        return qty
    except Exception as e:
        state['live_log'].append(f"[Buy] Order failed: {e}")
        return None

def sell(base_balance, price, step_size, min_qty, min_notional, qty_precision):
    qty = math.floor(base_balance / step_size) * step_size
    qty = math.floor(qty * (10 ** qty_precision)) / (10 ** qty_precision)
    notional = qty * price
    if qty < min_qty or notional < min_notional:
        state['live_log'].append(f"[Sell] Skipped: qty={qty:.5f} notional={notional:.3f} (min_qty={min_qty}, min_notional={min_notional})")
        return None
    try:
        res = signed_request("POST", "/api/v3/order",
                             {"symbol": SYMBOL, "side": "SELL", "type": "MARKET", "quantity": f"{qty:.{qty_precision}f}"})
        state['live_log'].append(f"[Sell] Market order placed: qty={qty:.4f} @ {price:.4f}")
        return qty
    except Exception as e:
        state['live_log'].append(f"[Sell] Order failed: {e}")
        return None

# --- Backtest (slim version, with best MA scan) ---
def grid_search_lengths(prices, times, min_len=MIN_FAST_LEN, max_len=MAX_SLOW_LEN):
    best = None
    best_metrics = (0.0, 0.0)
    best_trades = 0
    n = len(prices)
    if n < max_len + 5:
        max_len = max(10, min(max_len, n - 5))
    for fast in range(min_len, max_len):
        for slow in range(fast + 1, max_len + 1):
            trades, winrate, pnl = backtest_for_lengths(prices, times, fast, slow)
            if trades == 0: continue
            score = (winrate, pnl)
            if score > best_metrics:
                best_metrics = score
                best = (fast, slow)
                best_trades = trades
    if best is None:
        best = (MIN_FAST_LEN, MAX_SLOW_LEN)
        best_trades, wr, pnl = backtest_for_lengths(prices, times, *best)
        best_metrics = (wr, pnl)
    return best, {"trades": best_trades, "winrate": best_metrics[0], "pnl": best_metrics[1]}

def backtest_for_lengths(prices, times, fast_len, slow_len):
    n = len(prices)
    if n < slow_len + 2:
        return 0, 0.0, 0.0
    ps = prefix_sums(prices)
    state_ = "flat"
    last_buy = None
    peak = None
    max_gain_pct = 0.0
    downtrend_started = False
    level_armed = {lvl: False for lvl in TP_LEVELS}
    wins, total, total_pnl = 0, 0, 0.0
    ma_fast = [None]*(fast_len-1) + [mean_at(ps, i, fast_len) for i in range(fast_len-1, n)]
    ma_slow = [None]*(slow_len-1) + [mean_at(ps, i, slow_len) for i in range(slow_len-1, n)]
    buy_confirm = 0
    for i in range(slow_len, n):
        if i - 1 < slow_len - 1: continue
        slow_prev = ma_slow[i-1]; slow_curr = ma_slow[i]
        fast_prev = ma_fast[i-1] if i-1 >= fast_len-1 else None
        fast_curr = ma_fast[i] if i >= fast_len-1 else None
        close_prev = prices[i-1]; close_curr = prices[i]
        buy_signal = False
        if state_ == "flat":
            if close_prev <= slow_prev and close_curr > slow_curr and fast_prev is not None and fast_curr is not None and fast_curr > fast_prev:
                buy_signal = True
            if buy_signal:
                buy_confirm += 1
            else:
                buy_confirm = 0
            if buy_confirm >= MIN_BUY_CONFIRM_TICKS:
                state_ = "long"
                last_buy = close_curr
                peak = close_curr
                max_gain_pct = 0.0
                downtrend_started = False
                level_armed = {lvl: False for lvl in TP_LEVELS}
                buy_confirm = 0
                continue
        if state_ == "long" and last_buy is not None:
            if close_curr > (peak or close_curr): peak = close_curr
            gross_gain = (close_curr - last_buy) / last_buy * 100.0
            if gross_gain > max_gain_pct: max_gain_pct = gross_gain
            if close_curr < peak: downtrend_started = True
            for lvl in TP_LEVELS:
                if not level_armed[lvl] and max_gain_pct >= (lvl + TP_ARM_BUFFER):
                    level_armed[lvl] = True
            sell_cross = False
            if fast_prev is not None and fast_curr is not None and close_prev >= fast_prev and close_curr < fast_curr:
                sell_cross = True
            if close_prev >= slow_prev and close_curr < slow_curr:
                sell_cross = True
            sell_retrace = False
            if downtrend_started:
                for lvl in TP_LEVELS:
                    if level_armed[lvl] and gross_gain <= (lvl - TP_EXIT_BUFFER):
                        sell_retrace = True; break
            if sell_cross or sell_retrace:
                pnl = (close_curr - last_buy) / last_buy * 100.0
                total_pnl += pnl
                total += 1
                if pnl > 0: wins += 1
                state_ = "flat"
                last_buy = None
                peak = None
                max_gain_pct = 0.0
                downtrend_started = False
                level_armed = {lvl: False for lvl in TP_LEVELS}
    winrate = (wins / total * 100.0) if total > 0 else 0.0
    return total, winrate, total_pnl

# --- TickStrategy class (live, with dashboard sync) ---
class TickStrategy(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ws = None
        self._stop_event = threading.Event()
        self.live_prices = []
        self.live_times = []
        self.fast_ma = []
        self.slow_ma = []
        self.state = "flat"
        self.last_buy_price = None
        self.peak_since_buy = None
        self.max_gain_pct = 0.0
        self.downtrend_started = False
        self.level_armed = {lvl: False for lvl in TP_LEVELS}
        self.buy_confirm = 0
        self.buy_in_progress = False
        self.last_tick_proc_time = 0
        self.markers = []
        self.trades = []

    def stop(self):
        self._stop_event.set()
        try:
            if self.ws: self.ws.close()
        except: pass

    def run(self):
        candles = fetch_klines_recent(SYMBOL, INTERVAL, CHART_HIST_LIMIT)
        prices, times = to_price_time(candles)
        self.live_prices = prices[-CHART_HIST_LIMIT:].copy()
        self.live_times = times[-CHART_HIST_LIMIT:].copy()
        self.fast_ma = calc_ma_series(self.live_prices, MIN_FAST_LEN)
        self.slow_ma = calc_ma_series(self.live_prices, MAX_SLOW_LEN)
        while not self._stop_event.is_set():
            try:
                self._run_ws()
            except Exception as e:
                state['live_log'].append(f"[TickStrategy] Exception: {e}")
                traceback.print_exc()
                time.sleep(WS_RECONNECT_SECS)

    def _run_ws(self):
        ws_url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_close=lambda ws, *a: state['live_log'].append("[TickStrategy] WebSocket closed."),
            on_error=lambda ws, err: state['live_log'].append(f"[TickStrategy] WebSocket error: {err}")
        )
        self.ws.run_forever()

    def _update_ma(self):
        self.fast_ma = calc_ma_series(self.live_prices, MIN_FAST_LEN)
        self.slow_ma = calc_ma_series(self.live_prices, MAX_SLOW_LEN)
        if len(self.fast_ma) > CHART_HIST_LIMIT:
            self.fast_ma = self.fast_ma[-CHART_HIST_LIMIT:]
        if len(self.slow_ma) > CHART_HIST_LIMIT:
            self.slow_ma = self.slow_ma[-CHART_HIST_LIMIT:]

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            price = float(msg['p'])
            ts = int(msg['T']) // 1000

            if ts < self.last_tick_proc_time + TICK_THROTTLE_SECS:
                return
            self.last_tick_proc_time = ts

            if self.live_times and ts <= self.live_times[-1]:
                return
            self.live_prices.append(price)
            self.live_times.append(ts)
            if len(self.live_prices) > CHART_HIST_LIMIT:
                self.live_prices.pop(0)
                self.live_times.pop(0)
            self._update_ma()
            idx = len(self.live_prices) - 1
            fast = self.fast_ma[idx]
            slow = self.slow_ma[idx]
            fast_prev = self.fast_ma[idx-1] if idx > 0 else None
            slow_prev = self.slow_ma[idx-1] if idx > 0 else None
            price_prev = self.live_prices[idx-1] if idx > 0 else None

            # === BUY LOGIC ===
            if self.state == "flat":
                buy_signal = False
                if price_prev is not None and fast_prev is not None and slow_prev is not None:
                    if price_prev <= slow_prev and price > slow and fast > fast_prev:
                        buy_signal = True
                if buy_signal:
                    self.buy_confirm += 1
                else:
                    self.buy_confirm = 0
                if self.buy_confirm >= MIN_BUY_CONFIRM_TICKS and not self.buy_in_progress:
                    state['live_log'].append(f"[TickStrategy] BUY confirmed at {price:.6f} (time={ts})")
                    self._handle_buy(price, ts)
                    self.buy_in_progress = True
                    self.buy_confirm = 0

            # === SELL LOGIC ===
            if self.state == "long" and self.last_buy_price is not None:
                if price > (self.peak_since_buy or price): self.peak_since_buy = price
                gross_gain = (price - self.last_buy_price) / self.last_buy_price * 100.0
                if gross_gain > self.max_gain_pct: self.max_gain_pct = gross_gain
                if price < self.peak_since_buy: self.downtrend_started = True
                for lvl in TP_LEVELS:
                    if not self.level_armed[lvl] and self.max_gain_pct >= (lvl + TP_ARM_BUFFER):
                        self.level_armed[lvl] = True
                sell_cross = False
                if fast_prev is not None and fast is not None and price_prev is not None:
                    if price_prev >= fast_prev and price < fast:
                        sell_cross = True
                if slow_prev is not None and slow is not None and price_prev is not None:
                    if price_prev >= slow_prev and price < slow:
                        sell_cross = True
                sell_retrace = False
                if self.downtrend_started:
                    for lvl in TP_LEVELS:
                        if self.level_armed[lvl] and gross_gain <= (lvl - TP_EXIT_BUFFER):
                            sell_retrace = True; break
                if sell_cross or sell_retrace:
                    state['live_log'].append(f"[TickStrategy] SELL confirmed at {price:.6f} (time={ts})")
                    self._handle_sell(price, ts)
        except Exception as e:
            state['live_log'].append(f"[TickStrategy] on_message error: {e}")
            traceback.print_exc()

    def _handle_buy(self, price, ts):
        try:
            step_size, min_qty, min_notional, qty_precision = get_symbol_filters(SYMBOL)
            _, quote_free = get_balances()
            qty = buy(quote_free, price, FEE_RATE_TRADE, step_size, min_qty, min_notional, qty_precision)
            if qty:
                self.state = "long"
                self.last_buy_price = price
                self.peak_since_buy = price
                self.max_gain_pct = 0.0
                self.downtrend_started = False
                self.level_armed = {lvl: False for lvl in TP_LEVELS}
                state['trades'].append(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] BUY @ {price:.4f} qty={qty:.4f}")
                self.markers.append({
                    "time": ts,
                    "position": "below",
                    "color": "lime",
                    "shape": "arrowUp",
                    "text": f"BUY {price:.4f}"
                })
                if len(self.markers) > CHART_HIST_LIMIT:
                    self.markers = self.markers[-CHART_HIST_LIMIT:]
            self.buy_in_progress = False
        except Exception as e:
            state['live_log'].append(f"[TickStrategy] BUY EXCEPTION: {e}")
            traceback.print_exc()
            self.buy_in_progress = False

    def _handle_sell(self, price, ts):
        try:
            step_size, min_qty, min_notional, qty_precision = get_symbol_filters(SYMBOL)
            base_free, _ = get_balances()
            qty = sell(base_free, price, step_size, min_qty, min_notional, qty_precision)
            if qty:
                state['trades'].append(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] SELL @ {price:.4f} qty={qty:.4f}")
                self.state = "flat"
                self.last_buy_price = None
                self.peak_since_buy = None
                self.max_gain_pct = 0.0
                self.downtrend_started = False
                self.level_armed = {lvl: False for lvl in TP_LEVELS}
                self.buy_confirm = 0
                self.buy_in_progress = False
                self.markers.append({
                    "time": ts,
                    "position": "above",
                    "color": "red",
                    "shape": "arrowDown",
                    "text": f"SELL {price:.4f}"
                })
                if len(self.markers) > CHART_HIST_LIMIT:
                    self.markers = self.markers[-CHART_HIST_LIMIT:]
        except Exception as e:
            state['live_log'].append(f"[TickStrategy] SELL EXCEPTION: {e}")
            traceback.print_exc()
# =================== FLASK ROUTES AND ENDPOINTS ===================

@app.route("/")
def index():
    return render_template_string(
        HTML,
        symbol=SYMBOL,
        live_running=state["live_running"]
    )

@app.route("/chartdata")
def chartdata():
    # Get latest candles, mas, and markers for chart
    try:
        candles = fetch_klines_recent(SYMBOL, INTERVAL, CHART_HIST_LIMIT)
        prices, times = to_price_time(candles)
        price_line = [{"time": t, "value": p} for t, p in zip(times, prices)]

        fast_ma = calc_ma_series(prices, MIN_FAST_LEN)
        fast_line = [
            {"time": times[i], "value": v} for i, v in enumerate(fast_ma) if v is not None
        ]
        slow_ma = calc_ma_series(prices, MAX_SLOW_LEN)
        slow_line = [
            {"time": times[i], "value": v} for i, v in enumerate(slow_ma) if v is not None
        ]
        # Live markers (from running strat)
        strat = globals().get("strat_thread", None)
        markers = strat.markers if strat and hasattr(strat, "markers") else []
        return jsonify({
            "prices": price_line,
            "fast": fast_line,
            "slow": slow_line,
            "markers": markers,
        })
    except Exception as e:
        return jsonify({
            "prices": [],
            "fast": [],
            "slow": [],
            "markers": [],
            "error": str(e)
        })

@app.route("/status")
def status():
    # Web UI: Show all key stats/logs
    best = state.get("best_params")
    metrics = state.get("backtest_metrics")
    strat = globals().get("strat_thread", None)

    ema_str = ""
    best_str = ""
    bt_metrics = ""
    if best:
        ema_str = f"Fast: <b>{best[0]}</b> / Slow: <b>{best[1]}</b>"
    if metrics:
        best_str = f"Trades: {metrics.get('trades',0)} | Win: {metrics.get('winrate',0):.2f}% | PnL: {metrics.get('pnl',0):.2f}%"
        bt_metrics = f"<b>Backtest PnL:</b> {metrics.get('pnl',0):.2f}% &nbsp; <b>Winrate:</b> {metrics.get('winrate',0):.2f}% &nbsp; <b>Trades:</b> {metrics.get('trades',0)}"
    trades_str = "<br>".join(state.get("trades", [])[-8:]) if state.get("trades") else "None"
    return jsonify({
        "live_status": "RUNNING" if state["live_running"] else "IDLE",
        "live_log": state["live_log"][-16:],
        "ema_settings": ema_str,
        "best_result": best_str,
        "backtest_metrics": bt_metrics,
        "trades": trades_str,
    })

@app.route("/start_backtest", methods=["POST"])
def start_backtest():
    def worker():
        try:
            state["live_log"].append("[Backtest] Starting...")
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - BACKTEST_HOURS * 60 * 60 * 1000
            candles = fetch_klines_recent(SYMBOL, INTERVAL, CHART_HIST_LIMIT)
            prices, times = to_price_time(candles)
            best, metrics = grid_search_lengths(prices, times, MIN_FAST_LEN, MAX_SLOW_LEN)
            state["best_params"] = best
            state["backtest_metrics"] = metrics
            state["live_log"].append(f"[Backtest] DONE. Best: fast={best[0]}, slow={best[1]} | PnL={metrics['pnl']:.2f}% | Win={metrics['winrate']:.2f}%")
        except Exception as ex:
            state["live_log"].append(f"[Backtest] ERROR: {ex}")
            traceback.print_exc()
    threading.Thread(target=worker, daemon=True).start()
    return redirect(url_for('index'))

@app.route("/start_live", methods=["POST"])
def start_live():
    if not state["live_running"]:
        state["live_running"] = True
        # Start/restart TickStrategy background thread
        global strat_thread
        try:
            if "strat_thread" in globals() and strat_thread.is_alive():
                strat_thread.stop()
                time.sleep(2)
        except Exception: pass
        strat_thread = TickStrategy()
        strat_thread.start()
        state["live_log"].append("[Live] Trader started.")
    return redirect(url_for('index'))

@app.route("/stop_live", methods=["POST"])
def stop_live():
    state["stop_signal"] = True
    state["live_running"] = False
    state["live_log"].append("[Live] Trader stopping...")
    try:
        global strat_thread
        if "strat_thread" in globals() and strat_thread.is_alive():
            strat_thread.stop()
    except Exception: pass
    return redirect(url_for('index'))

# ================ RUN FLASK APP ================
if __name__ == "__main__":
    # Run any auto-backtest for best params at startup
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - BACKTEST_HOURS * 60 * 60 * 1000
        candles = fetch_klines_recent(SYMBOL, INTERVAL, CHART_HIST_LIMIT)
        prices, times = to_price_time(candles)
        best, metrics = grid_search_lengths(prices, times, MIN_FAST_LEN, MAX_SLOW_LEN)
        state["best_params"] = best
        state["backtest_metrics"] = metrics
        print("[AUTO BACKTEST] Best fast/slow:", best)
        print("[AUTO BACKTEST] Trades:", metrics['trades'], "Winrate:", metrics['winrate'], "PnL:", metrics['pnl'])
    except Exception as ex:
        print("[AUTO BACKTEST] EXCEPTION:", ex)
        traceback.print_exc()
    # Start web server
    print(f"Dashboard running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
