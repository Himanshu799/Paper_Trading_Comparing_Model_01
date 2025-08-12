#!/usr/bin/env python3
import os
import time
from time import perf_counter
from typing import Tuple, List
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from alpaca_trade_api.rest import REST, TimeFrame, APIError


# ── CONFIG ───────────────────────────────────────────────────────────────
TICKERS = os.environ.get("TICKERS", "AAPL,JPM,AMZN,TSLA,MSFT").split(",")

# Must match training env:
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 10))       # env.window_size
MIN_CASH_BUFFER = float(os.environ.get("MIN_CASH_BUFFER", 0.05))  # env.min_cash_buffer
TRANSACTION_COST = float(os.environ.get("TRANSACTION_COST", 0.001))  # used only for sizing conservatism
MODEL_PATH = os.environ.get("MODEL_PATH", "ppo_multistock_rl.zip")

SLEEP_INTERVAL = int(os.environ.get("SLEEP_INTERVAL", 60))  # seconds between loops

# Alpaca creds
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Timeframe + feed (match training data cadence)
# ALPACA_TIMEFRAME: "minute" or "day"
ALPACA_TIMEFRAME = os.environ.get("ALPACA_TIMEFRAME", "day").lower()
ALPACA_FEED      = os.environ.get("ALPACA_FEED", "iex")  # "iex" (free) or "sip" (paid)

# Lookback to ensure enough bars for WINDOW_SIZE even at market open
HIST_LOOKBACK_MINUTES = int(os.environ.get("HIST_LOOKBACK_MINUTES", 24 * 60))
HIST_LOOKBACK_DAYS    = int(os.environ.get("HIST_LOOKBACK_DAYS", 365))

# Safety buffer for history slicing
HIST_N = max(3 * WINDOW_SIZE, 300)

# Fractional trading via notional orders
USE_NOTIONAL_ORDERS = True


# ── RL AGENT ─────────────────────────────────────────────────────────────
def _const(v):
    return lambda _progress: v

_model_path_stem = MODEL_PATH[:-4] if MODEL_PATH.endswith(".zip") else MODEL_PATH
agent = PPO.load(
    _model_path_stem,
    custom_objects={
        "lr_schedule": _const(3e-4),
        "clip_range": _const(0.2),
    },
)


# ── ALPACA CLIENT ────────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


# ── HELPERS (mirror env logic) ──────────────────────────────────────────
def _tf_and_start():
    if ALPACA_TIMEFRAME == "minute":
        return TimeFrame.Minute, datetime.now(timezone.utc) - timedelta(minutes=HIST_LOOKBACK_MINUTES)
    return TimeFrame.Day, datetime.now(timezone.utc) - timedelta(days=HIST_LOOKBACK_DAYS)

def get_recent_bars(sym: str, need: int) -> pd.DataFrame:
    """
    Return at least `need` bars sorted by timestamp ascending.
    Columns: timestamp, open, high, low, close, volume (lowercase).
    """
    tf, start_dt = _tf_and_start()
    try:
        bars = api.get_bars(
            symbol=sym,
            timeframe=tf,
            start=start_dt.isoformat(),
            feed=ALPACA_FEED,
            limit=None
        ).df
    except APIError as e:
        print(f"  ❌ get_bars error for {sym}: {e}")
        return pd.DataFrame()

    if bars is None or bars.empty:
        return pd.DataFrame()

    bars = bars.reset_index()
    bars.columns = [str(c).lower() for c in bars.columns]
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in bars.columns]
    bars = bars[cols].sort_values("timestamp")
    if len(bars) >= need:
        bars = bars.tail(need).copy()
    return bars

def build_price_window(bars_df: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    Normalized close window (length = window_size) like env._price_windows():
    window / window[0]
    """
    closes = bars_df["close"].to_numpy(dtype=np.float64)
    if len(closes) < window_size:
        raise ValueError(f"Need {window_size} closes, have {len(closes)}")
    win = closes[-window_size:]
    base = float(max(win[0], 1e-12))
    return (win / base).astype(np.float64)  # (window_size,)

def compose_observation(price_windows: List[np.ndarray],
                        cash_ratio: float,
                        shares_vec: np.ndarray) -> np.ndarray:
    """
    Obs = concat(price_windows) + [cash_ratio] + shares
    dtype float32, shape = (K*W + 1 + K,)
    """
    per_asset = np.concatenate(price_windows, axis=0).astype(np.float64)  # K*W
    obs_vec = np.concatenate([per_asset, np.array([cash_ratio], dtype=np.float64), shares_vec], axis=0)
    return obs_vec.astype(np.float32).reshape(1, -1)

def renorm_weights_clipped(w: np.ndarray) -> np.ndarray:
    """
    Clip to [0,1], renormalize to sum<=1 (match env.step()).
    """
    w = np.clip(np.asarray(w, dtype=np.float64).ravel(), 0.0, 1.0)
    s = float(w.sum())
    if s > 1.0:
        w = w / s
    return w


# ── MAIN LOOP ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Starting LIVE RL deploy (CEEMD/CNN-LSTM price-window env). Ctrl+C to exit.", flush=True)

    try:
        while True:
            loop_start = perf_counter()
            now = pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern")
            print(f"\n🔄 Loop start: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

            # 1) Market check (only for intraday)
            try:
                clock = api.get_clock()
            except Exception as e:
                print(f"  ❌ get_clock error: {e}")
                time.sleep(SLEEP_INTERVAL); continue

            if ALPACA_TIMEFRAME == "minute" and not clock.is_open:
                print(f"  ❌ Market closed, next open at {clock.next_open}")
                time.sleep(SLEEP_INTERVAL); continue
            else:
                print(f"  ⏱ Market status: {'OPEN' if clock.is_open else 'CLOSED'}; next close {clock.next_close}")

            # 2) Live account snapshot
            try:
                account = api.get_account()
                cash = float(account.cash)
                # live positions; qty may be fractional
                positions = {p.symbol: float(p.qty) for p in api.list_positions()}
            except Exception as e:
                print(f"  ❌ account/positions error: {e}")
                time.sleep(SLEEP_INTERVAL); continue

            # 3) Build observation exactly like env: [K*WINDOW] + [1 cash_ratio] + [K shares]
            price_windows = []
            latest_prices = []
            failed = []

            for sym in TICKERS:
                try:
                    bars = get_recent_bars(sym, need=max(HIST_N, WINDOW_SIZE))
                    if bars.empty or len(bars) < WINDOW_SIZE:
                        raise ValueError(f"Not enough bars ({len(bars)}/{WINDOW_SIZE})")
                    pw = build_price_window(bars, WINDOW_SIZE)        # (W,)
                    price_windows.append(pw)
                    latest_prices.append(float(bars["close"].iloc[-1]))
                except Exception as e:
                    failed.append(f"{sym}: {e}")
                    print(f"  ⚠️  {sym}: {e}")

            if failed:
                print("  ⚠️  Missing features for one or more tickers, skipping loop.")
                time.sleep(SLEEP_INTERVAL); continue

            latest_prices = np.array(latest_prices, dtype=np.float64)
            shares_vec = np.array([positions.get(sym, 0.0) for sym in TICKERS], dtype=np.float64)

            pos_val = float(np.sum(shares_vec * latest_prices))
            net_worth = cash + pos_val
            if net_worth <= 0:
                print("  ❌ Net worth non-positive, skipping.")
                time.sleep(SLEEP_INTERVAL); continue

            cash_ratio = float(cash / net_worth)
            obs = compose_observation(price_windows, cash_ratio, shares_vec)

            # 4) Policy -> target weights (long-only, sum ≤ 1). Match env behavior.
            raw_action, _ = agent.predict(obs, deterministic=True)
            weights = renorm_weights_clipped(raw_action.reshape(-1))

            # 5) Target allocations with minimum cash buffer (match env)
            port_val = net_worth
            investable_val = max(0.0, port_val * (1.0 - MIN_CASH_BUFFER))
            target_values = weights * investable_val
            current_values = shares_vec * latest_prices
            dollar_trades = target_values - current_values  # +buy $, -sell $

            # 6) Rebalance with market orders (fractional using notional). Conservative with fees.
            #    BUY: cap by available cash after buffer & fee; SELL: cap by current position value
            fee = TRANSACTION_COST
            for i, sym in enumerate(TICKERS):
                dv = float(dollar_trades[i])
                px = float(latest_prices[i])
                if abs(dv) < 1e-6:
                    continue

                if dv > 0:
                    # desired notional buy; include fee and enforce cash buffer
                    max_afford = max(0.0, cash - MIN_CASH_BUFFER * port_val)
                    # amount we can send such that cost_with_fee ≤ max_afford
                    max_notional = max_afford / (1.0 + fee)
                    notional = min(dv, max_notional)
                    if notional <= 0:
                        continue
                    try:
                        if USE_NOTIONAL_ORDERS:
                            api.submit_order(
                                symbol=sym, notional=round(notional, 2),
                                side="buy", type="market", time_in_force="day"
                            )
                            cash -= notional * (1.0 + fee)  # approximate until broker settles
                            print(f"  ✅ BUY  ${notional:,.2f} {sym} (market)")
                        else:
                            qty = notional / max(px, 1e-12)
                            api.submit_order(symbol=sym, qty=qty, side="buy", type="market", time_in_force="day")
                            cash -= qty * px * (1.0 + fee)
                            print(f"  ✅ BUY  {qty:.6f} {sym} @ ~{px:.2f}")
                    except APIError as e:
                        print(f"  ❌ BUY {sym}: {e}")

                else:
                    sell_needed = -dv
                    pos_sh = shares_vec[i]
                    if pos_sh <= 0:
                        continue
                    max_sell_value = pos_sh * px
                    notional = min(sell_needed, max_sell_value)
                    if notional <= 0:
                        continue
                    try:
                        if USE_NOTIONAL_ORDERS:
                            api.submit_order(
                                symbol=sym, notional=round(notional, 2),
                                side="sell", type="market", time_in_force="day"
                            )
                            cash += notional * (1.0 - fee)
                            print(f"  ✅ SELL ${notional:,.2f} {sym} (market)")
                        else:
                            qty = notional / max(px, 1e-12)
                            qty = min(qty, pos_sh)
                            api.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
                            cash += qty * px * (1.0 - fee)
                            print(f"  ✅ SELL {qty:.6f} {sym} @ ~{px:.2f}")
                    except APIError as e:
                        print(f"  ❌ SELL {sym}: {e}")

            # 7) Loop summary & sleep
            loop_time = perf_counter() - loop_start
            next_run = now + pd.Timedelta(seconds=SLEEP_INTERVAL)
            print(f"✅ Loop done in {loop_time:.2f}s. Next run ~ {next_run.strftime('%H:%M:%S %Z')}", flush=True)
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("🛑  Stopped by user", flush=True)
    except Exception as e:
        print(f"⚠️  Error in loop: {e}", flush=True)
        time.sleep(5)
