from flask import Flask, request
import requests, json, time, math, hmac, hashlib, traceback
from urllib.parse import urlencode

app = Flask(__name__)

# ========== USER CONFIG ==========
SYMBOL = "SOLUSDT"
INTERVAL = "3m"
FEE_RATE_TRADE = 0.001
FEE_RATE_PCT = FEE_RATE_TRADE * 100.0
SLIPPAGE_PCT = 0.05
MIN_NET_PROFIT = -999
TP_LEVELS = [0.70, 0.40, 0.22]
TP_ARM_BUFFER = 0.1
TP_EXIT_BUFFER = 0.01

API_KEY = "INSERT YOUR BINANCE API KEY HERE"
API_SECRET = "INSERT YOUR BINANCE API SECRET HERE"

# ========== CONSTANTS ==========
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

# ========== HTTP HELPERS ==========
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

# ========== MARKET DATA ==========
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

def fetch_klines_recent(symbol, interval, limit=500):
    return public_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})

def to_price_time(candles):
    prices, times = [], []
    for c in candles:
        times.append(c[0] // 1000)
        prices.append(float(c[4]))
    return prices, times

# ========== MA HELPERS ==========
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

# ========== PROFIT HELPERS ==========
def net_gain_pct(buy_price, sell_price, fee_pct=FEE_RATE_PCT, slip_pct=SLIPPAGE_PCT):
    eff_sell = sell_price * (1 - (fee_pct + slip_pct)/100.0)
    eff_buy  = buy_price  * (1 + fee_pct/100.0)
    return (eff_sell / eff_buy - 1.0) * 100.0

# ========== BACKTEST (ON CLOSE ONLY, AND FAST MA TREND) ==========
def backtest_for_lengths(prices, times, fast_len, slow_len):
    n = len(prices)
    if n < slow_len + 2 or fast_len == 0 or slow_len == 0:
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
    for i in range(slow_len, n):
        if i - 1 < slow_len - 1: continue
        slow_prev = ma_slow[i-1]; slow_curr = ma_slow[i]
        fast_prev = ma_fast[i-1] if i-1 >= fast_len-1 else None
        fast_curr = ma_fast[i] if i >= fast_len-1 else None
        close_prev = prices[i-1]; close_curr = prices[i]
        if state == "flat" and close_prev <= slow_prev and close_curr > slow_curr \
            and fast_prev is not None and fast_curr is not None and fast_curr > fast_prev:
            state = "long"
            last_buy = close_curr
            peak = close_curr
            max_gain_pct = 0.0
            downtrend_started = False
            level_armed = {lvl: False for lvl in TP_LEVELS}
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

def grid_search_lengths(prices, times, min_len=2, max_len=300):
    best = None
    best_metrics = (0.0, 0.0)
    best_trades = 0
    n = len(prices)
    if n < max_len + 5:
        max_len = max(10, min(max_len, n - 5))
    print(f"[BACKTEST] START fast={min_len}..{max_len-1}, slow=fast+1..{max_len}", flush=True)
    start_t = time.time()
    for fast in range(min_len, max_len):
        if (fast - min_len) % 20 == 0:
            elapsed = time.time() - start_t
            print(f"[BACKTEST] Progress fast={fast}/{max_len-1} ({elapsed:.1f}s)", flush=True)
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
        best = (2, 200)
        best_trades, wr, pnl = backtest_for_lengths(prices, times, *best)
        best_metrics = (wr, pnl)
        print(f"[BACKTEST] No trading combos found, fallback to {best}", flush=True)
    print(f"[BACKTEST] DONE in {elapsed:.2f}s → fast={best[0]}, slow={best[1]}, "
          f"trades={best_trades}, winrate={best_metrics[0]:.2f}%, pnl={best_metrics[1]:.2f}%", flush=True)
    return best, {"trades": best_trades, "winrate": best_metrics[0], "pnl": best_metrics[1]}

# ========== FILTERS & BALANCES ==========
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

# ========== ORDER PLACEMENT (ALWAYS LIVE) ==========
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

# ========== SERVER-SIDE STATE ==========
last_candle_action_time = None
last_signal_side = None
current_fast_len = None
current_slow_len = None

# ========== WEB ROUTE ==========
@app.route("/")
def index():
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 12 * 60 * 60 * 1000  # 12 hours back
    bt_candles = fetch_klines_range(SYMBOL, INTERVAL, start_ms, now_ms)
    bt_prices, bt_times = to_price_time(bt_candles)
    (fast_len, slow_len), metrics = grid_search_lengths(bt_prices, bt_times, 21, 180)
    disp_candles = fetch_klines_recent(SYMBOL, INTERVAL, limit=500)
    prices, times = to_price_time(disp_candles)
    ma_fast = calc_ma_series(prices, fast_len)
    ma_slow = calc_ma_series(prices, slow_len)
    price_points = [{"time": t, "value": v} for t, v in zip(times, prices)]
    fast_points  = [{"time": t, "value": v} for t, v in zip(times, ma_fast) if v is not None]
    slow_points  = [{"time": t, "value": v} for t, v in zip(times, ma_slow) if v is not None]
    markers = []
    state = "flat"
    last_label = "sell"
    last_buy_price = None
    peak_since_buy = None
    max_gain_pct = 0.0
    downtrend_started = False
    level_armed = {lvl: False for lvl in TP_LEVELS}
    start_i = max(slow_len, 1)
    for i in range(start_i, len(prices)):
        if i - 1 < slow_len - 1: continue
        slow_prev = ma_slow[i-1]; slow_curr = ma_slow[i]
        fast_prev = ma_fast[i-1] if i-1 >= fast_len-1 else None
        fast_curr = ma_fast[i] if i >= fast_len-1 else None
        if slow_prev is None or slow_curr is None: continue
        close_prev = prices[i-1]; close_curr = prices[i]
        if state == "flat" and close_prev <= slow_prev and close_curr > slow_curr \
           and fast_prev is not None and fast_curr is not None and fast_curr > fast_prev:
            last_buy_price = close_curr
            peak_since_buy = close_curr
            max_gain_pct = 0.0
            downtrend_started = False
            level_armed = {lvl: False for lvl in TP_LEVELS}
            markers.append({
                "time": times[i], "position": "belowBar", "color": "lime",
                "shape": "arrowUp", "text": f"BUY {close_curr:.4f} (fast={fast_len}, slow={slow_len})"
            })
            state = "long"; last_label = "buy"; continue
        if state == "long" and last_buy_price is not None:
            if close_curr > (peak_since_buy or close_curr): peak_since_buy = close_curr
            gross_gain = (close_curr - last_buy_price) / last_buy_price * 100.0
            if gross_gain > max_gain_pct: max_gain_pct = gross_gain
            if close_curr < peak_since_buy: downtrend_started = True
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
            if (sell_cross or sell_retrace) and last_label != "sell":
                pct = ((close_curr - last_buy_price) / last_buy_price) * 100.0
                tag = "SELL" if sell_cross else "TP-RETRACE SELL"
                markers.append({
                    "time": times[i], "position": "aboveBar", "color": "red",
                    "shape": "arrowDown", "text": f"{tag} {close_curr:.4f} ({pct:.2f}%)"
                })
                state = "flat"; last_label = "sell"
                last_buy_price = None; peak_since_buy = None
                max_gain_pct = 0.0; downtrend_started = False
                level_armed = {lvl: False for lvl in TP_LEVELS}
    last_state_js = "long" if markers and markers[-1]['text'].startswith("BUY") else "flat"
    last_label_js = "buy" if last_state_js == "long" else "sell"
    try:
        step_size, min_qty, min_notional, qty_precision = get_symbol_filters(SYMBOL)
    except Exception as e:
        print(f"[FILTERS] Failed: {e}")
        step_size = 0.000001; min_qty = 0.0; min_notional = 0.0; qty_precision = 6
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{SYMBOL} • LIVE (fast={fast_len}, slow={slow_len}) • Trades={metrics['trades']} • Win={metrics['winrate']:.1f}% • PnL={metrics['pnl']:.2f}%</title>
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
    <span>Fast MA: <b>{fast_len}</b></span>
    <span>Slow MA: <b>{slow_len}</b></span>
    <span>Backtest — Trades: <b>{metrics['trades']}</b></span>
    <span>Winrate: <b>{metrics['winrate']:.1f}%</b></span>
    <span>PnL: <b>{metrics['pnl']:.2f}%</b></span>
    <span>Trading: <b>LIVE</b></span>
  </div>
  <div class="muted">Backtested 12 hours (on close). Signals are always on candle close — no delay.</div>
  <div id="container"></div>
</div>
<script>
(() => {{
  const CANDLE_SECONDS = {CANDLE_SECONDS};
  const SYMBOL = "{SYMBOL}";
  const TP_LEVELS = {json.dumps(TP_LEVELS)};
  const FAST = {fast_len};
  const SLOW = {slow_len};
  const priceData = {json.dumps(price_points)};
  const fastMAData = {json.dumps(fast_points)};
  const slowMAData = {json.dumps(slow_points)};
  const markers = {json.dumps(markers)};
  let lastState = "{last_state_js}";
  let lastLabel = "{last_label_js}";
  let lastBuyPrice = null;
  let peakSinceBuy = null;
  let maxGainPct = 0.0;
  let downtrendStarted = false;
  let levelArmed = Object.fromEntries(TP_LEVELS.map(l => [l, false]));
  function rollingMeanEndingAt(data, idx, win) {{
    if (idx < win - 1) return null;
    let sum = 0; for (let i = idx - win + 1; i <= idx; i++) sum += data[i].value;
    return sum / win;
  }}
  const chart = LightweightCharts.createChart(document.getElementById('container'), {{
    width: document.getElementById('container').clientWidth,
    height: 640,
    layout: {{ background: {{ color: '#0f1115' }}, textColor: '#e5e7eb' }},
    grid: {{ vertLines: {{ color: '#1b202b' }}, horzLines: {{ color: '#1b202b' }} }},
    rightPriceScale: {{ borderColor: '#232837' }},
    timeScale: {{ borderColor: '#232837', timeVisible: true, secondsVisible: true }}
  }});
  const priceSeries = chart.addLineSeries({{ lineWidth: 2, color: 'deepskyblue' }});
  const fastSeries  = chart.addLineSeries({{ lineWidth: 2, color: 'orange' }});
  const slowSeries  = chart.addLineSeries({{ lineWidth: 2, color: 'yellow' }});
  priceSeries.setData(priceData);
  fastSeries.setData(fastMAData);
  slowSeries.setData(slowMAData);
  priceSeries.setMarkers(markers);
  chart.timeScale().fitContent();
  // Live update
  const ws = new WebSocket("wss://stream.binance.com:9443/ws/" + SYMBOL.toLowerCase() + "@trade");
  ws.onmessage = (event) => {{
    const msg = JSON.parse(event.data);
    const price = parseFloat(msg.p);
    const ts = Math.floor(msg.T / 1000);
    if (!priceData.length) return;
    let lastPoint = priceData[priceData.length - 1];
    let nextCandleTime = lastPoint.time + CANDLE_SECONDS;
    if (ts < nextCandleTime) {{
      lastPoint.value = price;
      priceSeries.setData(priceData);
      const i = priceData.length - 1;
      const fastVal = rollingMeanEndingAt(priceData, i, FAST);
      if (fastVal !== null) {{
        if (fastMAData.length && fastMAData[fastMAData.length-1].time === lastPoint.time) fastMAData[fastMAData.length-1].value = fastVal;
        else fastMAData.push({{ time: lastPoint.time, value: fastVal }});
        fastSeries.setData(fastMAData);
      }}
      const slowVal = rollingMeanEndingAt(priceData, i, SLOW);
      if (slowVal !== null) {{
        if (slowMAData.length && slowMAData[slowMAData.length-1].time === lastPoint.time) slowMAData[slowMAData.length-1].value = slowVal;
        else slowMAData.push({{ time: lastPoint.time, value: slowVal }});
        slowSeries.setData(slowMAData);
      }}
    }} else {{
      const closedIdx = priceData.length - 1;
      const prevIdx = closedIdx - 1;
      const close_prev = priceData[prevIdx].value;
      const close_curr = priceData[closedIdx].value;
      const fast_prev = rollingMeanEndingAt(priceData, prevIdx, FAST);
      const fast_curr = rollingMeanEndingAt(priceData, closedIdx, FAST);
      const slow_prev = rollingMeanEndingAt(priceData, prevIdx, SLOW);
      const slow_curr = rollingMeanEndingAt(priceData, closedIdx, SLOW);
      if (lastState === "flat" && close_prev <= slow_prev && close_curr > slow_curr
          && fast_prev !== null && fast_curr !== null && fast_curr > fast_prev) {{
        markers.push({{
          time: priceData[closedIdx].time, position: "belowBar", color: "lime", shape: "arrowUp",
          text: "BUY " + close_curr.toFixed(4)
        }});
        priceSeries.setMarkers(markers);
        lastState = "long"; lastLabel = "buy"; lastBuyPrice = close_curr;
        peakSinceBuy = close_curr; maxGainPct = 0.0; downtrendStarted = false;
        levelArmed = Object.fromEntries(TP_LEVELS.map(l => [l, false]));
        fetch("/signal", {{
          method: "POST", headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ side: "BUY", price: close_curr, candle_time: priceData[closedIdx].time }})
        }});
      }}
      if (lastState === "long" && lastBuyPrice) {{
        if (close_curr > (peakSinceBuy ?? close_curr)) peakSinceBuy = close_curr;
        const grossGain = ((close_curr - lastBuyPrice) / lastBuyPrice) * 100.0;
        if (grossGain > maxGainPct) maxGainPct = grossGain;
        if (close_curr < peakSinceBuy) downtrendStarted = true;
        for (const lvl of TP_LEVELS) {{
          if (!levelArmed[lvl] && maxGainPct >= (lvl + {TP_ARM_BUFFER})) levelArmed[lvl] = true;
        }}
        let sellCross = false;
        if (fast_prev !== null && fast_curr !== null && close_prev >= fast_prev && close_curr < fast_curr) sellCross = true;
        if (slow_prev !== null && slow_curr !== null && close_prev >= slow_prev && close_curr < slow_curr) sellCross = true;
        let sellRetrace = false;
        if (downtrendStarted) {{
          for (const lvl of TP_LEVELS) {{
            if (levelArmed[lvl] && grossGain <= (lvl - {TP_EXIT_BUFFER})) {{ sellRetrace = true; break; }}
          }}
        }}
        if ((sellCross || sellRetrace) && lastLabel !== "sell") {{
          const text = (sellCross ? "SELL " : "TP-RETRACE SELL ") + close_curr.toFixed(4) + " (" + grossGain.toFixed(2) + "%)";
          markers.push({{ time: priceData[closedIdx].time, position: "aboveBar", color: "red", shape: "arrowDown", text }});
          priceSeries.setMarkers(markers);
          lastState = "flat"; lastLabel = "sell";
          lastBuyPrice = null; peakSinceBuy = null; maxGainPct = 0.0; downtrendStarted = false;
          levelArmed = Object.fromEntries(TP_LEVELS.map(l => [l, false]));
          fetch("/signal", {{
            method: "POST", headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ side: "SELL", price: close_curr, candle_time: priceData[closedIdx].time }})
          }});
        }}
      }}
      priceData.push({{time: nextCandleTime, value: price}});
      priceSeries.setData(priceData);
    }}
  }};
}})();
</script>
</body>
</html>
"""

# ========== SIGNAL ENDPOINT ==========
@app.post("/signal")
def signal():
    global last_candle_action_time, last_signal_side, current_fast_len, current_slow_len
    data = request.get_json(force=True, silent=True) or {}
    side = (data or {}).get("side")
    candle_time = int((data or {}).get("candle_time", 0))
    try:
        price = float((data or {}).get("price", 0.0))
    except:
        price = 0.0
    if side not in ("BUY", "SELL") or price <= 0 or candle_time <= 0:
        return ("bad request", 400)
    if last_candle_action_time == candle_time:
        return json.dumps({"ok": True, "skipped": "duplicate_candle"}), 200, {"Content-Type": "application/json"}
    last_candle_action_time = candle_time
    step_size, min_qty, min_notional, qty_precision = get_symbol_filters(SYMBOL)
    try:
        base_free, quote_free = get_balances()
    except Exception as e:
        print(f"[BALANCES] Failed: {e}")
        base_free, quote_free = 0.0, 0.0
    if side == "BUY":
        qty = buy(quote_free, price, FEE_RATE_TRADE, step_size, min_qty, min_notional, qty_precision)
        if qty: last_signal_side = "BUY"
        resp = {"ok": qty is not None, "qty": qty}
    else:
        qty = sell(base_free, price, step_size, min_qty, min_notional, qty_precision)
        if qty: last_signal_side = "SELL"
        resp = {"ok": qty is not None, "qty": qty}
    return json.dumps(resp), 200, {"Content-Type": "application/json"}

# ========== ON-DEMAND 12-HOUR BACKTEST API ==========
@app.get("/backtest")
def backtest_route():
    try:
        print("[/backtest] INITIATED", flush=True)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 12 * 60 * 60 * 1000
        bt_candles = fetch_klines_range(SYMBOL, INTERVAL, start_ms, now_ms)
        if not bt_candles or len(bt_candles) < 10:
            print("[/backtest] ERROR: Not enough candles fetched", flush=True)
            return json.dumps({"error": "Not enough candles for backtest"}), 400, {"Content-Type": "application/json"}
        bt_prices, bt_times = to_price_time(bt_candles)
        print(f"[/backtest] Candles: {len(bt_prices)}", flush=True)
        (fast_len, slow_len), metrics = grid_search_lengths(bt_prices, bt_times, 21, 180)
        print(f"[/backtest] COMPLETED", flush=True)
        return json.dumps({
            "fast": fast_len,
            "slow": slow_len,
            "trades": metrics["trades"],
            "winrate": metrics["winrate"],
            "pnl": metrics["pnl"]
        }), 200, {"Content-Type": "application/json"}
    except Exception as ex:
        print("[/backtest] EXCEPTION:", ex, flush=True)
        traceback.print_exc()
        return json.dumps({"error": str(ex)}), 500, {"Content-Type": "application/json"}

if __name__ == "__main__":
    print("Trading mode: LIVE, 3m candles, strict on-close cross logic. (Always sells on crossdown)", flush=True)
    # --- AUTO BACKTEST on STARTUP ---
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 12 * 60 * 60 * 1000
        bt_candles = fetch_klines_range(SYMBOL, INTERVAL, start_ms, now_ms)
        if not bt_candles or len(bt_candles) < 10:
            print("[AUTO BACKTEST] ERROR: Not enough candles fetched", flush=True)
        else:
            bt_prices, bt_times = to_price_time(bt_candles)
            print(f"[AUTO BACKTEST] Candles: {len(bt_prices)}", flush=True)
            (fast_len, slow_len), metrics = grid_search_lengths(bt_prices, bt_times, 21, 180)
            print(f"[AUTO BACKTEST] DONE. fast={fast_len}, slow={slow_len}, trades={metrics['trades']}, winrate={metrics['winrate']:.2f}%, pnl={metrics['pnl']:.2f}%", flush=True)
    except Exception as ex:
        print("[AUTO BACKTEST] EXCEPTION:", ex, flush=True)
        traceback.print_exc()
    app.run(host="0.0.0.0", port=5000, debug=True)
