# Binance-Auto-Trading-Bot-MA-Cross-Strategy-with-Live-Flask-Dashboard-Smart-Take-Profits
# 📈 Tick EMA Cross Bot: Strategy and Implementation

This bot implements a **robust moving average (MA) cross strategy** for Binance spot markets, with live web dashboard, streaming chart, intelligent multi-level take-profits, and instant backtesting of the best parameters.
<img width="1178" height="832" alt="image" src="https://github.com/user-attachments/assets/c8cc59ab-ea11-4517-9da8-a4cd049c7f8d" />

---

## Key Features

- **MA Cross Logic**: Enters a trade (“long”) only when price crosses above the slow MA on candle close *and* the fast MA is trending up (fast MA > previous fast MA).
- **Strict On-Close Signal Handling**: All buy/sell signals are evaluated only after a completed candle (no lookahead, no “mid-candle” cheats).  
- **No Double Signals**: Buys and sells are strictly debounced and only processed once per completed candle.
- **Multiple Trailing Take-Profit Levels**:
    - Each TP level is “armed” only after price rises past that TP+buffer.
    - If price retraces below any armed TP level (minus a small exit buffer), the bot exits immediately.
- **MA Cross Exit**: If price crosses below either the fast MA or slow MA on close, bot exits the trade immediately, regardless of TP status.
- **Live Trading Dashboard**: Real-time streaming chart with price, both MAs, buy/sell markers, and live trade logs.
- **Binance Market Order Execution**: All trades use market orders for speed/reliability, with strict lot size & notional checks.
- **Fee/Slippage-Aware**: All net profits, backtests, and trade calcs include both exchange fees and a user-defined slippage allowance.
- **Parameter Backtesting**: At every run (and on demand from the dashboard), the bot scans a range of MA lengths and reports the best winrate, PnL, and trade count over the last 12 hours.

---

## Trade Signal Summary

**Buy ("Long") Signal**:
- Price crosses from below to above the slow MA on candle close.
- Fast MA is rising (fast MA now > fast MA previous).
- (Debounced: Only 1 buy per candle.)

**Sell Signal (Exit Trade)**:
- If in a trade, sell immediately if price crosses below either fast MA or slow MA on candle close.
- OR: If any trailing TP level is armed, and price retraces below that TP minus a buffer, exit immediately.

---

## Risk and Execution
- **Market Orders Only**: No limit orders; instant execution to avoid missed trades.
- **Fee and Slippage Included**: All calculations account for actual exchange fee and user-configurable slippage.
- **Strict Binance API Checks**: Order size and minimum notional always validated before sending to API.

---

## Backtesting and Optimization
- **Parameter Grid Search**: Scans all valid combinations of fast/slow MA lengths in range, reporting winrate, trade count, and PnL.
- **Latest 12 Hours**: Backtest range is set to the last 12 hours of price data for optimal recent-fit.
- **Immediate Results**: Best parameters and backtest summary are shown instantly in the dashboard.

---

## Disclaimer

> This bot is for educational and research purposes only.  
> Crypto trading is highly risky and can result in significant financial loss.  
> Use with extreme caution and **never risk more than you can afford to lose**.  
> You are solely responsible for your use of this software.

---
