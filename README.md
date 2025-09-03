# Binance-Auto-Trading-Bot-MA-Cross-Strategy-with-Live-Flask-Dashboard-Smart-Take-Profits

---

## **GitHub Repo Title**

```
Binance Auto-Trading Bot: MA Cross Strategy with Live Flask Dashboard & Smart Take-Profits
```

---

## **README.md – Strategy Details Section**

---

### 📈 **Strategy Details**

This bot implements a robust moving average (MA) cross trading strategy with a focus on strict, actionable signals, live charting, and intelligent take-profit handling.
It is designed for **Binance spot trading**, and includes a real-time Flask dashboard with streaming updates and historical backtesting.

---

#### **Key Features**

* **MA Cross Logic:**
  Enters a trade ("long") **when the price crosses above a slow MA** and the fast MA is trending up, confirmed on candle close.
* **Smart Take-Profit (TP):**
  Multiple trailing TP levels are "armed" only after a sufficient gain. The bot dynamically tracks the peak price after entry and exits on a retrace below each armed TP, or on crossdown.
* **Strict On-Close Entries and Exits:**
  All signals are evaluated **only at candle close** (no pre-close lookahead), mirroring proper trading discipline and ensuring backtests match live results.
* **No “double buy/sell” false signals:**
  Buys and sells are debounced to never repeat within a candle.
* **Backtesting Built-In:**
  On startup and on every page load, the bot runs a full parameter search for best fast/slow MA lengths, reporting winrate and PnL over the last 12 hours.
* **Live Dashboard:**
  See all price, MA, and signal markers in real time, with chart updates on every tick and candle.

---

#### **Trade Signal Summary**

* **Buy ("Long") Signal:**

  * The **price crosses above the slow MA** (from below to above) on candle close.
  * The **fast MA is trending up** (current fast MA > previous fast MA).
* **Sell Signal:**

  * If in a trade, sell immediately if **price crosses below either the slow MA or fast MA** on candle close.
  * OR: If a trailing take-profit has been armed and price retraces back below that TP by a set buffer amount.

---

#### **Risk and Execution**

* **Market Orders Only:**
  All trades are executed as market orders for maximum fill reliability.
* **Fee and Slippage Awareness:**
  Net profit calculations always include both exchange fees and a configurable slippage factor.
* **Lot size and min notional checks** on every order to prevent Binance API errors.
* **Configurable trade asset, time interval, TP levels, and backtest range.**

---

#### **Backtest/Optimization**

* **Every run/backtest:**

  * Searches a wide range of fast and slow MA lengths for optimal results over the last 12 hours.
  * Shows best settings, winrate, trade count, and total PnL.

---

#### **Disclaimer**

> **This bot is for educational and research purposes only. Trading cryptocurrencies is highly risky.
> Always use with caution and never risk more than you can afford to lose.
> You are solely responsible for your use of this software.**

---
