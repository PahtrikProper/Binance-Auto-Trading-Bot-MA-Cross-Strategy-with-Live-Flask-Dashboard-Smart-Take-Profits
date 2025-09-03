from flask import Flask, request
import threading, requests, json, time, math, hmac, hashlib, traceback
from urllib.parse import urlencode
import websocket

app = Flask(__name__)

# ======================= USER CONFIGURATION ========================
# --- Binance/API/Pair ---
SYMBOL = "SOLUSDT"
INTERVAL = "3m"
API_KEY = "insert your binance key here"
API_SECRET = "insert your secret binance key here"

# --- Strategy/Trading Parameters ---
FEE_RATE_TRADE = 0.001
SLIPPAGE_PCT = 0.05
TP_LEVELS = [0.70, 0.40, 0.22]
TP_ARM_BUFFER = 0.10
TP_EXIT_BUFFER = 0.01
MIN_BUY_CONFIRM_TICKS = 1
SELL_CONFIRM_TICKS = 1
TICK_THROTTLE_SECS = 15

# --- MA Parameters ---
MIN_FAST_LEN = 11
MAX_SLOW_LEN = 200

# --- Backtest/Chart ---
BACKTEST_HOURS = 17
CHART_HIST_LIMIT = 200
WS_RECONNECT_SECS = 10

# =================== END USER CONFIGURATION ========================

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

def fetch_klines_range(symbol, interval, start_ms, end_ms, per_req=1000):
    out, cur = [], start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": per_req,
            "startTime": cur,
            "endTime": end_ms
        }
        batch = public_get("/api/v3/klines", params)
        if not batch: break
        out.extend(batch)
        last_open = batch[-1][0]
        next_start = last_open + INTERVAL_MS
        if next_start <= cur: next_start = cur + INTERVAL_MS
        cur = next_start
    return out

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

def net_gain_pct(buy_price, sell_price, fee_pct=FEE_RATE_TRADE, slip_pct=SLIPPAGE_PCT):
    eff_sell = sell_price * (1 - (fee_pct + slip_pct)/100.0)
    eff_buy  = buy_price  * (1 + fee_pct/100.0)
    return (eff_sell / eff_buy - 1.0) * 100.0

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
        print(f"Buy skipped: qty={qty} (min {min_qty}), notional={notional} (min {min_notional})")
        return None
    try:
        res = signed_request("POST", "/api/v3/order",
                             {"symbol": SYMBOL, "side": "BUY", "type": "MARKET", "quantity": f"{qty:.{qty_precision}f}"})
        print(f"BUY ORDER PLACED: qty={qty}, price≈{price} (orderId={res.get('orderId')})")
        return qty
    except Exception as e:
        print(f"Buy order failed: {e}")
        return None
    
    
def grid_search_lengths(prices, times, min_len=MIN_FAST_LEN, max_len=MAX_SLOW_LEN):
    best = None
    best_metrics = (0.0, 0.0)
    best_trades = 0
    n = len(prices)
    if n < max_len + 5:
        max_len = max(10, min(max_len, n - 5))
    print(f"[BACKTEST] fast={min_len}..{max_len-1}, slow=fast+1..{max_len}")
    start_t = time.time()
    for fast in range(min_len, max_len):
        if (fast - min_len) % 20 == 0:
            elapsed = time.time() - start_t
            print(f"[BACKTEST] Progress fast={fast}/{max_len-1} ({elapsed:.1f}s)")
        for slow in range(fast + 1, max_len + 1):
            trades, winrate, pnl = backtest_for_lengths(prices, times, fast, slow)
            if trades == 0: continue
            score = (winrate, pnl)
            if score > best_metrics:
                best_metrics = score
                best = (fast, slow)
                best_trades = trades
    elapsed = time.time() - start_t
    if best is None:
        best = (MIN_FAST_LEN, MAX_SLOW_LEN)
        best_trades, wr, pnl = backtest_for_lengths(prices, times, *best)
        best_metrics = (wr, pnl)
        print(f"[BACKTEST] No trading combos found, fallback to {best}")
    print(f"[BACKTEST] DONE in {elapsed:.2f}s → fast={best[0]}, slow={best[1]}, trades={best_trades}, winrate={best_metrics[0]:.2f}%, pnl={best_metrics[1]:.2f}%")
    return best, {"trades": best_trades, "winrate": best_metrics[0], "pnl": best_metrics[1]}

def sell(base_balance, price, step_size, min_qty, min_notional, qty_precision):
    qty = math.floor(base_balance / step_size) * step_size
    qty = math.floor(qty * (10 ** qty_precision)) / (10 ** qty_precision)
    notional = qty * price
    if qty < min_qty or notional < min_notional:
        print(f"Sell skipped: qty={qty} (min {min_qty}), notional={notional} (min {min_notional})")
        return None
    try:
        res = signed_request("POST", "/api/v3/order",
                             {"symbol": SYMBOL, "side": "SELL", "type": "MARKET", "quantity": f"{qty:.{qty_precision}f}"})
        print(f"SELL ORDER PLACED: qty={qty}, price≈{price} (orderId={res.get('orderId')})")
        return qty
    except Exception as e:
        print(f"Sell order failed: {e}")
        return None

# --- TickStrategy class with buy/sell markers for the chart ---
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
        self.markers = []  # <--- marker list for chart

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
                print(f"[TICK STRATEGY] Exception: {e}", flush=True)
                traceback.print_exc()
                time.sleep(WS_RECONNECT_SECS)

    def _run_ws(self):
        ws_url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_close=lambda ws, *a: print("[TICK STRATEGY] WebSocket closed.", flush=True),
            on_error=lambda ws, err: print(f"[TICK STRATEGY] WebSocket error: {err}", flush=True)
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
            # === BUY LOGIC: Require MIN_BUY_CONFIRM_TICKS consecutive ticks
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
                    print(f"[TICK STRATEGY] {MIN_BUY_CONFIRM_TICKS}-tick BUY confirmed at {price:.6f} (time={ts})", flush=True)
                    self._handle_buy(price, ts)
                    self.buy_in_progress = True
                    self.buy_confirm = 0
            # === SELL/TP LOGIC: SELL_CONFIRM_TICKS (always 1)
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
                    print(f"[TICK STRATEGY] SELL confirmed at {price:.6f} (time={ts})", flush=True)
                    self._handle_sell(price, ts)
        except Exception as e:
            print(f"[TICK STRATEGY] on_message error: {e}", flush=True)
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
                print(f"[TICK STRATEGY] BUY PLACED at {price:.6f}", flush=True)
                self.markers.append({
                    "time": ts,
                    "position": "belowBar",
                    "color": "lime",
                    "shape": "arrowUp",
                    "text": f"BUY {price:.4f}"
                })
                # --- LIVE MARKER PATCH ---
                if len(self.markers) > CHART_HIST_LIMIT:
                    self.markers = self.markers[-CHART_HIST_LIMIT:]
            else:
                print(f"[TICK STRATEGY] BUY skipped (qty/filters)", flush=True)
            self.buy_in_progress = False
        except Exception as e:
            print(f"[TICK STRATEGY] BUY EXCEPTION: {e}", flush=True)
            traceback.print_exc()
            self.buy_in_progress = False

    def _handle_sell(self, price, ts):
        try:
            step_size, min_qty, min_notional, qty_precision = get_symbol_filters(SYMBOL)
            base_free, _ = get_balances()
            qty = sell(base_free, price, step_size, min_qty, min_notional, qty_precision)
            if qty:
                print(f"[TICK STRATEGY] SELL PLACED at {price:.6f}", flush=True)
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
                    "position": "aboveBar",
                    "color": "red",
                    "shape": "arrowDown",
                    "text": f"SELL {price:.4f}"
                })
                # --- LIVE MARKER PATCH ---
                if len(self.markers) > CHART_HIST_LIMIT:
                    self.markers = self.markers[-CHART_HIST_LIMIT:]
            else:
                print(f"[TICK STRATEGY] SELL skipped (qty/filters)", flush=True)
        except Exception as e:
            print(f"[TICK STRATEGY] SELL EXCEPTION: {e}", flush=True)
            traceback.print_exc()

def backtest_for_lengths(prices, times, fast_len, slow_len):
    n = len(prices)
    if n < slow_len + 2:
        return 0, 0.0, 0.0
    ps = prefix_sums(prices)
    state = "flat"
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
        if state == "flat":
            if close_prev <= slow_prev and close_curr > slow_curr and fast_prev is not None and fast_curr is not None and fast_curr > fast_prev:
                buy_signal = True
            if buy_signal:
                buy_confirm += 1
            else:
                buy_confirm = 0
            if buy_confirm >= MIN_BUY_CONFIRM_TICKS:
                state = "long"
                last_buy = close_curr
                peak = close_curr
                max_gain_pct = 0.0
                downtrend_started = False
                level_armed = {lvl: False for lvl in TP_LEVELS}
                buy_confirm = 0
                continue
        if state == "long" and last_buy is not None:
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
                state = "flat"
                last_buy = None
                peak = None
                max_gain_pct = 0.0
                downtrend_started = False
                level_armed = {lvl: False for lvl in TP_LEVELS}
    winrate = (wins / total * 100.0) if total > 0 else 0.0
    return total, winrate, total_pnl

@app.route("/")
def index():
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{SYMBOL} • TICK STRAT LIVE</title>
  <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    body {{ background: #0f1115; color: #e5e7eb; margin: 0; font-family: ui-sans-serif, system-ui, -apple-system; }}
    .wrap {{ max-width: 1160px; margin: 18px auto; padding: 0 14px; }}
    .badge span {{ display:inline-block; padding:6px 10px; background:#161a22; border:1px solid #232837; border-radius:10px; margin:4px; }}
    #container {{ width: 100%; height: 640px; margin-top: 14px; }}
    .muted {{ color:#9aa4b2 }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="badge">
    <span>Symbol: <b>{SYMBOL}</b></span>
    <span>Interval: <b>{INTERVAL}</b></span>
    <span>Trading: <b>LIVE TICK</b></span>
  </div>
  <div class="muted">Signals use live ticks (min {TICK_THROTTLE_SECS}s per tick). BUY: {MIN_BUY_CONFIRM_TICKS}-tick confirm ({MIN_BUY_CONFIRM_TICKS*TICK_THROTTLE_SECS}s+), SELL: {SELL_CONFIRM_TICKS}-tick confirm. All chart logic matches bot.</div>
  <div id="container"></div>
  <div>
    <button onclick="runBacktest()">Run Backtest ({BACKTEST_HOURS}h)</button>
    <pre id="bt"></pre>
  </div>
</div>
<script>
(() => {{
  const SYMBOL = "{SYMBOL}";
  const chart = LightweightCharts.createChart(document.getElementById('container'), {{
    width: document.getElementById('container').clientWidth,
    height: 640,
    layout: {{ background: {{ color: '#0f1115' }}, textColor: '#e5e7eb' }},
    grid: {{ vertLines: {{ color: '#1b202b' }}, horzLines: {{ color: '#1b202b' }} }},
    rightPriceScale: {{ borderColor: '#232837' }},
    timeScale: {{ borderColor: '#232837', timeVisible: true, secondsVisible: true }}
  }});
  let priceData = [];
  let fastMAData = [];
  let slowMAData = [];
  let markers = [];
  const priceSeries = chart.addLineSeries({{ lineWidth: 2, color: 'deepskyblue' }});
  const fastSeries  = chart.addLineSeries({{ lineWidth: 2, color: 'orange' }});
  const slowSeries  = chart.addLineSeries({{ lineWidth: 2, color: 'yellow' }});
  function rollingMeanEndingAt(data, idx, win) {{
    if (idx < win - 1) return null;
    let sum = 0; for (let i = idx - win + 1; i <= idx; i++) sum += data[i].value;
    return sum / win;
  }}
  fetch('/seed').then(r => r.json()).then(seed => {{
    priceData = seed.priceData;
    fastMAData = seed.fastMAData;
    slowMAData = seed.slowMAData;
    markers = seed.markers || [];
    priceSeries.setData(priceData);
    fastSeries.setData(fastMAData);
    slowSeries.setData(slowMAData);
    priceSeries.setMarkers(markers);
    chart.timeScale().fitContent();
  }});
  const ws = new WebSocket("wss://stream.binance.com:9443/ws/" + SYMBOL.toLowerCase() + "@trade");
  ws.onmessage = (event) => {{
    const msg = JSON.parse(event.data);
    const price = parseFloat(msg.p);
    const ts = Math.floor(msg.T / 1000);
    if (!priceData.length) return;
    let lastPoint = priceData[priceData.length - 1];
    if (ts <= lastPoint.time) return;
    if (ts < lastPoint.time + {TICK_THROTTLE_SECS}) return;
    priceData.push({{time: ts, value: price}});
    if (priceData.length > {CHART_HIST_LIMIT}) priceData.shift();
    const i = priceData.length - 1;
    const fastVal = rollingMeanEndingAt(priceData, i, {MIN_FAST_LEN});
    if (fastVal !== null) {{
      fastMAData.push({{ time: ts, value: fastVal }});
      if (fastMAData.length > {CHART_HIST_LIMIT}) fastMAData.shift();
      fastSeries.setData(fastMAData);
    }}
    const slowVal = rollingMeanEndingAt(priceData, i, {MAX_SLOW_LEN});
    if (slowVal !== null) {{
      slowMAData.push({{ time: ts, value: slowVal }});
      if (slowMAData.length > {CHART_HIST_LIMIT}) slowMAData.shift();
      slowSeries.setData(slowMAData);
    }}
    priceSeries.setData(priceData);
    // --- LIVE MARKER PATCH: markers now update live via polling, not here!
  }};
  // --- LIVE MARKER PATCH START ---
  let lastMarkerCount = 0;
  function pollMarkers() {{
      fetch('/markers')
        .then(r => r.json())
        .then(data => {{
          if (data.markers && data.markers.length !== lastMarkerCount) {{
            priceSeries.setMarkers(data.markers);
            lastMarkerCount = data.markers.length;
          }}
        }})
        .catch(() => {{}});
      setTimeout(pollMarkers, 3000);
  }}
  pollMarkers();
  // --- LIVE MARKER PATCH END ---
  window.runBacktest = function() {{
    document.getElementById('bt').innerText = 'Running...';
    fetch('/backtest').then(r => r.json()).then(bt => {{
      document.getElementById('bt').innerText =
        'Best fast/slow: ' + bt.best[0] + '/' + bt.best[1] + '\\n' +
        'Trades: ' + bt.metrics.trades + '\\n' +
        'Winrate: ' + bt.metrics.winrate.toFixed(2) + '%\\n' +
        'PnL: ' + bt.metrics.pnl.toFixed(2) + '%';
    }}).catch(e => {{
      document.getElementById('bt').innerText = 'Error: ' + e;
    }});
  }}
}})();
</script>
</body>
</html>
"""

@app.route("/seed")
def seed():
    candles = fetch_klines_recent(SYMBOL, INTERVAL, CHART_HIST_LIMIT)
    prices, times = to_price_time(candles)
    priceData = [{"time": t, "value": p} for t, p in zip(times, prices)]
    fastMAData = []
    slowMAData = []
    for i in range(len(prices)):
        fv = None
        sv = None
        if i >= MIN_FAST_LEN-1:
            fv = sum(prices[i-MIN_FAST_LEN+1:i+1])/MIN_FAST_LEN
        if i >= MAX_SLOW_LEN-1:
            sv = sum(prices[i-MAX_SLOW_LEN+1:i+1])/MAX_SLOW_LEN
        if fv is not None:
            fastMAData.append({"time": times[i], "value": fv})
        if sv is not None:
            slowMAData.append({"time": times[i], "value": sv})
    # markers from running strategy, if exists
    markers = getattr(globals().get("strat_thread", None), "markers", [])
    return json.dumps({
        "priceData": priceData,
        "fastMAData": fastMAData,
        "slowMAData": slowMAData,
        "markers": markers,
    })

@app.route("/markers")  # --- LIVE MARKER PATCH ---
def markers_api():
    markers = getattr(globals().get("strat_thread", None), "markers", [])
    return json.dumps({"markers": markers})

@app.route("/backtest")
def backtest_api():
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - BACKTEST_HOURS * 60 * 60 * 1000
    candles = fetch_klines_range(SYMBOL, INTERVAL, start_ms, now_ms)
    prices, times = to_price_time(candles)
    best, metrics = grid_search_lengths(prices, times, MIN_FAST_LEN, MAX_SLOW_LEN)
    return json.dumps({
        "best": best,
        "metrics": metrics
    })

if __name__ == "__main__":
    print(f"Trading mode: LIVE, tick-based, buy={MIN_BUY_CONFIRM_TICKS} tick confirm, sell={SELL_CONFIRM_TICKS} tick, all logic matches chart lines. Tick throttle={TICK_THROTTLE_SECS}s", flush=True)
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - BACKTEST_HOURS * 60 * 60 * 1000
        bt_candles = fetch_klines_range(SYMBOL, INTERVAL, start_ms, now_ms)
        bt_prices, bt_times = to_price_time(bt_candles)
        best, metrics = grid_search_lengths(bt_prices, bt_times, MIN_FAST_LEN, MAX_SLOW_LEN)
        print("[AUTO BACKTEST] Best fast/slow:", best)
        print("[AUTO BACKTEST] Trades:", metrics['trades'], "Winrate:", metrics['winrate'], "PnL:", metrics['pnl'])
    except Exception as ex:
        print("[AUTO BACKTEST] EXCEPTION:", ex)
        traceback.print_exc()
    strat_thread = TickStrategy()
    strat_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=True)
