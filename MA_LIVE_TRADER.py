from flask import Flask, render_template_string, request, redirect, url_for, jsonify
import threading, time, random, pandas as pd, numpy as np, ccxt, sqlite3, os

app = Flask(__name__)

# === USER CONFIG ===
SYMBOL = 'SOL/USDT'
TIMEFRAME = '15m'
DATA_LIMIT = 500
TRADE_SIZE = 0.05

API_KEY = "insert binance api key here"
API_SECRET = "insert binance secret key here"

# --- ENTRY/EXIT MA RANGES ---
BUY_FAST_RANGE = (2, 40)
BUY_SLOW_RANGE = (150, 350)
SELL_FAST_RANGE = (2, 25)
SELL_MID_RANGE = (26, 70)
SELL_SLOW_RANGE = (71, 250)

# --- Default MA values ---
BUY_MA1_LEN = 2
BUY_MA2_LEN = 14
SELL_MA1_LEN = 6
SELL_MA2_LEN = 22
SELL_MA3_LEN = 100

MAX_TESTS = 10000
SELL_CROSS_CONFIRM_TICKS = 2

# --- TP Strategy ---
TP_LEVELS = [0.70, 0.40, 0.22]
TP_ARM_BUFFER = 0.10
TP_EXIT_BUFFER = 0.01

# === LOCAL SQLITE DATABASE ===
DB_FILE = "backtest.db"
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    entry_time TEXT,
    entry REAL,
    exit_time TEXT,
    exit REAL,
    profit REAL
)
""")
c.execute("""
CREATE TABLE markers (
    id INTEGER PRIMARY KEY,
    time INTEGER,
    position TEXT,
    color TEXT,
    shape TEXT,
    text TEXT
)
""")
conn.commit()

# === STATE ===
state = {
    "backtest_running": False,
    "backtest_progress": 0,
    "backtest_total": 0,
    "backtest_log": [],
    "best_result": None,
    "ema_settings": None,
    "live_running": False,
    "live_log": [],
    "stop_signal": False,
}

# === HTML DASHBOARD ===
HTML = """
<!doctype html>
<html>
<head>
    <title>EMA Cross Web Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
      body { background: #16181c; color: #e5e7eb; font-family: Arial,sans-serif; margin:0;}
      .container { max-width: 1100px; margin: 28px auto; padding: 24px; background: #1b1e25; border-radius: 12px; box-shadow: 0 2px 18px #0004;}
      h1 { color: #48fa83; }
      h2 { color: #45caff; }
      .status { font-weight: bold; }
      .row { display: flex; gap: 40px; }
      .col { flex:1;}
      button, input[type=number] { font-size:1.1em; border-radius:7px; border:1px solid #333;}
      button { background:#45caff; color:#131313; padding: 8px 20px;}
      button:disabled { background:#5c7087; color:#ddd;}
      pre { font-size:1em; background: #232838; color:#89e2ff; border-radius:8px; padding:7px 13px;}
      .logs { background:#181b24; color:#bbf7d0; max-height:180px; overflow:auto; border-radius:7px; margin-bottom: 8px;}
      .statbox { background: #13181f; border-radius:8px; padding:7px 11px; margin:7px 0; font-size:1.05em;}
      .chart-wrap { background: #13181f; border-radius:12px; padding:8px; margin-bottom:18px;}
    </style>
    <script>
    let chart, priceSeries, emaSeries=[];
    function initChart() {
      chart = LightweightCharts.createChart(document.getElementById('chart'), {
        width: 1040, height: 420,
        layout: { background: { color: '#1b1e25' }, textColor: '#e5e7eb' },
        grid: { vertLines: { color: '#23293a' }, horzLines: { color: '#23293a' } },
        rightPriceScale: { borderColor: '#2b3245' },
        timeScale: { borderColor: '#2b3245', timeVisible:true, secondsVisible:false }
      });
      priceSeries = chart.addLineSeries({ color:'deepskyblue', lineWidth:2 });
      emaSeries = [
        chart.addLineSeries({ color:'lime', lineWidth:2 }),      // Buy Fast
        chart.addLineSeries({ color:'orange', lineWidth:2 }),    // Buy Slow
        chart.addLineSeries({ color:'#ff1aff', lineWidth:2 }),   // Sell Fast
        chart.addLineSeries({ color:'#f4d35e', lineWidth:2 }),   // Sell Mid
        chart.addLineSeries({ color:'#95ffce', lineWidth:2 }),   // Sell Slow
      ];
    }
    function updateChart() {
      fetch('/chartdata').then(r=>r.json()).then(d=>{
        priceSeries.setData(d.prices);
        for(let i=0;i<5;++i) emaSeries[i].setData(d.emas[i]);
        priceSeries.setMarkers(d.markers);
      });
    }
    function poll() {
      fetch('/status').then(r=>r.json()).then(d=>{
        document.getElementById('backtest_status').innerHTML = d.backtest_status;
        document.getElementById('backtest_log').innerText = d.backtest_log.join('\\n');
        document.getElementById('live_status').innerHTML = d.live_status;
        document.getElementById('live_log').innerText = d.live_log.join('\\n');
        document.getElementById('ema_settings').innerHTML = d.ema_settings;
        document.getElementById('best_result').innerHTML = d.best_result;
        document.getElementById('trades').innerHTML = d.trades;
      });
      updateChart();
      setTimeout(poll, 4000);
    }
    window.onload = function() { initChart(); poll(); }
    </script>
</head>
<body>
<div class="container">
  <h1>EMA Cross Web Dashboard</h1>
  <div class="chart-wrap">
    <div id="chart" style="width:1040px; height:420px;"></div>
  </div>
  <div class="row">
    <div class="col">
      <h2>Backtest</h2>
      <form method="post" action="/start_backtest">
        <label>Max tests: <input type="number" name="num_tests" value="60" min="1" max="10000" style="width:80px;" {% if backtest_running %}disabled{% endif %}></label>
        <button type="submit" {% if backtest_running %}disabled{% endif %}>Start Backtest</button>
      </form>
      <div class="status" id="backtest_status"></div>
      <pre class="logs" id="backtest_log"></pre>
      <div class="statbox" id="best_result"></div>
    </div>
    <div class="col">
      <h2>Live Trading</h2>
      <form method="post" action="/start_live"><button type="submit" {% if live_running %}disabled{% endif %}>Start Live Trader</button></form>
      <form method="post" action="/stop_live"><button type="submit" style="background:#ff5c5c;color:white;" {% if not live_running %}disabled{% endif %}>Stop Live Trader</button></form>
      <div class="status" id="live_status"></div>
      <pre class="logs" id="live_log"></pre>
    </div>
    <div class="col">
      <h2>Best MAs</h2>
      <div class="statbox" id="ema_settings"></div>
      <h2>Last Trades</h2>
      <div id="trades" class="logs"></div>
    </div>
  </div>
</div>
</body>
</html>
"""
# === CORE HELPERS ===
def fetch_ohlcv(symbol, tf, limit, exchange):
    df = pd.DataFrame(
        exchange.fetch_ohlcv(symbol, tf, limit=limit),
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# ================= BACKTEST (Memory-Safe, SQLite SAFE) =================
def backtest(
    df, 
    buy_fast=BUY_MA1_LEN, buy_slow=BUY_MA2_LEN,
    sell_fast=SELL_MA1_LEN, sell_mid=SELL_MA2_LEN, sell_slow=SELL_MA3_LEN,
    plot_markers=False  # NEW PARAMETER
):
    slowest = max(buy_fast, buy_slow, sell_fast, sell_mid, sell_slow)
    if len(df) < slowest + 10:
        return {
            'buy_fast': buy_fast, 'buy_slow': buy_slow,
            'sell_fast': sell_fast, 'sell_mid': sell_mid, 'sell_slow': sell_slow,
            'total_profit': 0,
            'num_trades': 0,
            'win_rate': 0
        }

    alpha_bf = 2 / (buy_fast + 1)
    alpha_bs = 2 / (buy_slow + 1)
    alpha_sf = 2 / (sell_fast + 1)
    alpha_sm = 2 / (sell_mid + 1)
    alpha_ss = 2 / (sell_slow + 1)

    ema_bf = df['close'].iloc[0]
    ema_bs = df['close'].iloc[0]
    ema_sf = df['close'].iloc[0]
    ema_sm = df['close'].iloc[0]
    ema_ss = df['close'].iloc[0]

    in_trade = False
    entry_price = None
    peak = None
    max_gain_pct = 0.0
    downtrend_started = False
    tp_armed = {lvl: False for lvl in TP_LEVELS}
    cross_down_count = 0

    total_profit = 0.0
    num_trades = 0
    win_trades = 0

    for idx, row in df.iterrows():
        price = row['close']
        ts = row['timestamp']

        ema_bf = alpha_bf * price + (1 - alpha_bf) * ema_bf
        ema_bs = alpha_bs * price + (1 - alpha_bs) * ema_bs
        ema_sf = alpha_sf * price + (1 - alpha_sf) * ema_sf
        ema_sm = alpha_sm * price + (1 - alpha_sm) * ema_sm
        ema_ss = alpha_ss * price + (1 - alpha_ss) * ema_ss

        buy_cross_up = ema_bf > ema_bs
        buy_slow_trending_up = price > ema_bs
        buy_signal = buy_cross_up and buy_slow_trending_up

        sell_cross_down_mid = ema_sf < ema_sm
        sell_cross_down_slow = ema_sf < ema_ss
        sell_signal = sell_cross_down_mid or sell_cross_down_slow
        cross_down = ema_sf < ema_ss

        if not in_trade and buy_signal:
            in_trade = True
            entry_price = price
            peak = price
            tp_armed = {lvl: False for lvl in TP_LEVELS}
            cross_down_count = 0
            if plot_markers:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO markers (time, position, color, shape, text) VALUES (?, ?, ?, ?, ?)",
                        (int(ts.timestamp()), "below", "lime", "arrowUp", "BUY")
                    )
                    cursor.close()
        elif in_trade:
            peak = max(peak, price)
            gross_gain = (price - entry_price) / entry_price * 100.0
            for lvl in TP_LEVELS:
                if not tp_armed[lvl] and gross_gain >= lvl + TP_ARM_BUFFER:
                    tp_armed[lvl] = True
            retrace_sell = any(tp_armed[lvl] and gross_gain <= lvl - TP_EXIT_BUFFER for lvl in TP_LEVELS)

            if cross_down:
                cross_down_count += 1
            else:
                cross_down_count = 0

            if sell_signal or retrace_sell or cross_down_count >= SELL_CROSS_CONFIRM_TICKS:
                profit = price - entry_price
                total_profit += profit
                num_trades += 1
                if profit > 0:
                    win_trades += 1
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO trades (entry_time, entry, exit_time, exit, profit) VALUES (?, ?, ?, ?, ?)",
                        (str(ts), entry_price, str(ts), price, profit)
                    )
                    if plot_markers:
                        cursor.execute(
                            "INSERT INTO markers (time, position, color, shape, text) VALUES (?, ?, ?, ?, ?)",
                            (int(ts.timestamp()), "above", "red", "arrowDown", "SELL")
                        )
                    cursor.close()
                in_trade = False
                peak = None
                max_gain_pct = 0.0
                downtrend_started = False
                tp_armed = {lvl: False for lvl in TP_LEVELS}
                cross_down_count = 0

    win_rate = win_trades / num_trades if num_trades > 0 else 0
    return {
        'buy_fast': buy_fast,
        'buy_slow': buy_slow,
        'sell_fast': sell_fast,
        'sell_mid': sell_mid,
        'sell_slow': sell_slow,
        'total_profit': total_profit,
        'num_trades': num_trades,
        'win_rate': win_rate
    }

# =============== RANDOMISED SEARCH (NEW) ===============
def random_search(df, max_tests=60):
    state["backtest_log"].clear()
    best_result = None
    best_combo = None
    tested = set()
    for idx in range(max_tests):
        buy_fast = random.randint(BUY_FAST_RANGE[0], BUY_FAST_RANGE[1])
        buy_slow = random.randint(BUY_SLOW_RANGE[0], BUY_SLOW_RANGE[1])
        if buy_fast >= buy_slow:
            buy_fast, buy_slow = sorted([buy_fast, buy_slow])

        sell_fast = random.randint(SELL_FAST_RANGE[0], SELL_FAST_RANGE[1])
        sell_mid = random.randint(SELL_MID_RANGE[0], SELL_MID_RANGE[1])
        sell_slow = random.randint(SELL_SLOW_RANGE[0], SELL_SLOW_RANGE[1])
        arr = sorted([sell_fast, sell_mid, sell_slow])
        sell_fast, sell_mid, sell_slow = arr[0], arr[1], arr[2]
        combo_key = (buy_fast, buy_slow, sell_fast, sell_mid, sell_slow)
        if combo_key in tested:
            continue
        tested.add(combo_key)

        result = backtest(df, buy_fast, buy_slow, sell_fast, sell_mid, sell_slow)
        state["backtest_progress"] = idx + 1
        state["backtest_total"] = max_tests
        state["backtest_log"].append(
            f"Test {idx+1}/{max_tests}: BF={buy_fast} BS={buy_slow} | SF={sell_fast} SM={sell_mid} SS={sell_slow} | "
            f"Profit={result['total_profit']:.2f} | Trades={result['num_trades']}"
        )
        if best_result is None or result['total_profit'] > best_result['total_profit']:
            best_result = result
            best_combo = (buy_fast, buy_slow, sell_fast, sell_mid, sell_slow)

    return best_combo, best_result

# ================= CHART DATA LOADER =================
def load_chart_data():
    try:
        exchange = ccxt.binance()
        df = fetch_ohlcv(SYMBOL, TIMEFRAME, DATA_LIMIT, exchange)
        prices = [{"time": int(ts.timestamp()), "value": close} for ts, close in zip(df['timestamp'], df['close'])]

        # Only compute EMAs for best combination
        if state.get("ema_settings"):
            settings = state["ema_settings"]
        else:
            settings = [BUY_MA1_LEN, BUY_MA2_LEN, SELL_MA1_LEN, SELL_MA2_LEN, SELL_MA3_LEN]

        ema_lines = [
            ema(df['close'], settings[0]),
            ema(df['close'], settings[1]),
            ema(df['close'], settings[2]),
            ema(df['close'], settings[3]),
            ema(df['close'], settings[4]),
        ]
        emas = []
        for line in ema_lines:
            emas.append([{"time": int(ts.timestamp()), "value": float(v) if not pd.isna(v) else None}
                         for ts, v in zip(df['timestamp'], line)])

        # Load markers from DB
        c.execute("SELECT time, position, color, shape, text FROM markers ORDER BY time ASC")
        markers = [{"time": row[0], "position": row[1], "color": row[2], "shape": row[3], "text": row[4]} for row in c.fetchall()]

        return prices, emas, markers
    except Exception as e:
        return [], [[{}],[{}],[{}],[{}],[{}]], []
# =================== FLASK ROUTES ===================
@app.route("/")
def index():
    return render_template_string(
        HTML,
        backtest_running=state["backtest_running"],
        live_running=state["live_running"]
    )

@app.route("/chartdata")
def chartdata():
    prices, emas, markers = load_chart_data()
    return jsonify({"prices": prices, "emas": emas, "markers": markers})

@app.route("/status")
def status():
    def fmt_trades():
        c.execute("SELECT entry_time, entry, exit_time, exit, profit FROM trades ORDER BY id DESC LIMIT 8")
        rows = c.fetchall()
        if not rows:
            return "None"
        lines = [f"[Entry] {r[0]} @ {r[1]:.3f} → [Exit] {r[2]} @ {r[3]:.3f} | P/L: {r[4]:.3f}" for r in rows]
        return "\n".join(lines)

    best = state.get("best_result", None)
    ema_str = ""
    best_str = ""
    if best:
        ema_str = f"Buy: <b>{best.get('buy_fast','')} / {best.get('buy_slow','')}</b> | " \
                  f"Sell: <b>{best.get('sell_fast','')} / {best.get('sell_mid','')} / {best.get('sell_slow','')}</b>"
        best_str = f"Profit: <b>{best.get('total_profit',0):.2f}</b> | Trades: {best.get('num_trades',0)} | Win: {best.get('win_rate',0):.2%}"

    return jsonify({
        "backtest_status": ("RUNNING" if state["backtest_running"] else "IDLE") + f" ({state['backtest_progress']}/{state['backtest_total']})",
        "backtest_log": state["backtest_log"][-16:],
        "best_result": best_str,
        "ema_settings": ema_str,
        "live_status": "RUNNING" if state["live_running"] else "IDLE",
        "live_log": state["live_log"][-16:],
        "trades": fmt_trades(),
    })

# =================== START BACKTEST ===================
@app.route("/start_backtest", methods=["POST"])
def start_backtest():
    if state["backtest_running"]:
        return redirect(url_for('index'))

    try:
        num_tests = int(request.form.get("num_tests", 60))
        num_tests = max(1, min(num_tests, MAX_TESTS))
    except Exception:
        num_tests = 60

    def worker():
        try:
            state["backtest_running"] = True
            state["backtest_progress"] = 0
            state["backtest_total"] = 0
            state["backtest_log"].clear()

            # Clear DB for new backtest (using global cursor: main thread)
            c.execute("DELETE FROM trades")
            c.execute("DELETE FROM markers")
            conn.commit()

            exchange = ccxt.binance()
            df = fetch_ohlcv(SYMBOL, TIMEFRAME, DATA_LIMIT, exchange)
            best_combo, best_result = random_search(df, num_tests)

            if best_result is None:
                state["backtest_log"].append("[ERROR] No valid MA parameter combinations!")
                state["backtest_running"] = False
                return

            state["best_result"] = best_result
            state["ema_settings"] = [best_combo[0], best_combo[1], best_combo[2], best_combo[3], best_combo[4]]

        except Exception as ex:
            state["backtest_log"].append(f"[ERROR] {ex}")
        finally:
            state["backtest_running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return redirect(url_for('index'))

# =================== LIVE TRADER ===================
def run_live_trader():
    state["live_running"] = True
    state["stop_signal"] = False
    state["live_log"].append("[Live] Trader started.")

    ema_params = state.get("ema_settings") or [BUY_MA1_LEN, BUY_MA2_LEN, SELL_MA1_LEN, SELL_MA2_LEN, SELL_MA3_LEN]
    buy_fast, buy_slow, sell_fast, sell_mid, sell_slow = ema_params

    try:
        exchange = ccxt.binance({'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True})
        in_trade = False
        entry_price = None
        peak = None
        max_gain_pct = 0.0
        downtrend_started = False
        tp_armed = {lvl: False for lvl in TP_LEVELS}
        cross_down_count = 0

        while state["live_running"] and not state["stop_signal"]:
            df = fetch_ohlcv(SYMBOL, TIMEFRAME, DATA_LIMIT, exchange)
            i = len(df) - 1
            if i < 2:
                time.sleep(60)
                continue

            price = df['close'][i]
            ts = df['timestamp'][i]

            # ENTRY LOGIC
            ema_buy_fast = ema(df['close'], buy_fast)
            ema_buy_slow = ema(df['close'], buy_slow)
            buy_cross_up = (ema_buy_fast[i-1] < ema_buy_slow[i-1]) and (ema_buy_fast[i] > ema_buy_slow[i])
            buy_slow_trending_up = ema_buy_slow[i] > ema_buy_slow[i-1]
            buy_signal = buy_cross_up and buy_slow_trending_up

            # EXIT LOGIC
            ema_sell_fast = ema(df['close'], sell_fast)
            ema_sell_mid = ema(df['close'], sell_mid)
            ema_sell_slow = ema(df['close'], sell_slow)
            sell_cross_down_mid = (ema_sell_fast[i-1] > ema_sell_mid[i-1]) and (ema_sell_fast[i] < ema_sell_mid[i])
            sell_cross_down_slow = (ema_sell_fast[i-1] > ema_sell_slow[i-1]) and (ema_sell_fast[i] < ema_sell_slow[i])
            sell_signal = sell_cross_down_mid or sell_cross_down_slow
            cross_down = ema_sell_fast[i] < ema_sell_slow[i]

            # --- TRADE MANAGEMENT ---
            if not in_trade and buy_signal:
                in_trade = True
                entry_price = price
                peak = price
                max_gain_pct = 0.0
                downtrend_started = False
                tp_armed = {lvl: False for lvl in TP_LEVELS}
                cross_down_count = 0
                state["live_log"].append(f"[{ts}] BUY @ {price:.3f}")

            elif in_trade:
                if price > (peak or price):
                    peak = price
                gross_gain = (price - entry_price)/entry_price*100.0
                if gross_gain > max_gain_pct:
                    max_gain_pct = gross_gain
                if price < peak:
                    downtrend_started = True

                # TP arming
                for lvl in TP_LEVELS:
                    if not tp_armed[lvl] and max_gain_pct >= (lvl + TP_ARM_BUFFER):
                        tp_armed[lvl] = True

                retrace_sell = False
                if downtrend_started:
                    for lvl in TP_LEVELS:
                        if tp_armed[lvl] and gross_gain <= (lvl - TP_EXIT_BUFFER):
                            retrace_sell = True
                            break

                if cross_down:
                    cross_down_count += 1
                else:
                    cross_down_count = 0

                if retrace_sell or (sell_signal and any(tp_armed.values())) or (cross_down_count >= SELL_CROSS_CONFIRM_TICKS):
                    profit = price - entry_price
                    state["live_log"].append(f"[{ts}] SELL @ {price:.3f} PnL: {profit:.3f}")
                    in_trade = False
                    peak = None
                    max_gain_pct = 0.0
                    downtrend_started = False
                    tp_armed = {lvl: False for lvl in TP_LEVELS}
                    cross_down_count = 0

            # Keep last 100 logs
            state["live_log"] = state["live_log"][-100:]
            time.sleep(60)  # Wait before next candle

    except Exception as ex:
        state["live_log"].append(f"[ERROR] {ex}")
    finally:
        state["live_log"].append("[Live] Trader stopped.")
        state["live_running"] = False

# =================== START/STOP LIVE ===================
@app.route("/start_live", methods=["POST"])
def start_live():
    if not state["live_running"]:
        threading.Thread(target=run_live_trader, daemon=True).start()
    return redirect(url_for('index'))

@app.route("/stop_live", methods=["POST"])
def stop_live():
    state["stop_signal"] = True
    state["live_log"].append("[Live] Stopping trader...")
    return redirect(url_for('index'))

# =================== RUN APP ===================
if __name__ == "__main__":
    print("Starting EMA Cross web dashboard at http://127.0.0.1:5000 ...")
    app.run(debug=False, port=5000)
