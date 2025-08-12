#!/usr/bin/env python3
import os
import time
from time import perf_counter
from typing import Tuple, List, Dict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from alpaca_trade_api.rest import REST, TimeFrame, APIError


# ========== CONFIG ==========
TICKERS = os.environ.get("TICKERS", "AAPL,JPM,AMZN,TSLA,MSFT").split(",")

# Live feature generation window (whatever your encoder expects)
# e.g., 60-256 recent bars — must match your encoder’s receptive field
ENCODER_LOOKBACK = int(os.environ.get("ENCODER_LOOKBACK", 128))

# PPO model path (trained with MultiStockEnvFromState)
MODEL_PATH = os.environ.get("MODEL_PATH", "ppo_from_cnnlstm_state.zip")

# Trading controls
MIN_CASH_BUFFER = float(os.environ.get("MIN_CASH_BUFFER", 0.02))   # keep 2% cash
MAX_POSITION_WEIGHT = float(os.environ.get("MAX_POSITION_WEIGHT", 1.0))
SLEEP_INTERVAL = int(os.environ.get("SLEEP_INTERVAL", 60))         # seconds between loops

# Alpaca
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_FEED       = os.environ.get("ALPACA_FEED", "iex")            # "iex" (free) or "sip" (paid)

# Bars timeframe: "minute" or "day" (use what you trained on)
ALPACA_TIMEFRAME = os.environ.get("ALPACA_TIMEFRAME", "minute").lower()

# History lookback to ensure we can always build ENCODER_LOOKBACK even near open
HIST_LOOKBACK_MINUTES = int(os.environ.get("HIST_LOOKBACK_MINUTES", 24 * 60))  # 1 day of minutes
HIST_LOOKBACK_DAYS    = int(os.environ.get("HIST_LOOKBACK_DAYS", 365))         # 1 year of dailies

# Fractional trading: we use NOTIONAL orders for buys and sells
# Requires Alpaca fractional trading to be enabled on the account.
USE_NOTIONAL_ORDERS = True


# ========== RL AGENT ==========
def _const(v): return lambda _progress: v

_model_path_stem = MODEL_PATH[:-4] if MODEL_PATH.endswith(".zip") else MODEL_PATH
agent = PPO.load(
    _model_path_stem,
    custom_objects={
        "lr_schedule": _const(2.5e-4),
        "clip_range": _const(0.2),
    },
)


# ========== ALPACA CLIENT ==========
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


# ========== MATH/ENV HELPERS (mirror training) ==========
def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def position_weights(shares: np.ndarray, prices: np.ndarray, cash: float) -> np.ndarray:
    pos_val = shares * prices
    total = cash + float(pos_val.sum())
    if total <= 0:
        return np.zeros_like(shares, dtype=np.float64)
    return pos_val / total


# ========== DATA FETCH ==========
def _tf_and_start():
    if ALPACA_TIMEFRAME == "day":
        return TimeFrame.Day, datetime.now(timezone.utc) - timedelta(days=HIST_LOOKBACK_DAYS)
    else:
        return TimeFrame.Minute, datetime.now(timezone.utc) - timedelta(minutes=HIST_LOOKBACK_MINUTES)

def get_recent_bars(sym: str, need: int) -> pd.DataFrame:
    """
    Return at least `need` bars sorted by timestamp ascending.
    Columns: timestamp, open, high, low, close, volume (lowercase)
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


# ========== STATE ENCODER HOOK (YOU MUST IMPLEMENT) ==========
# Replace this with your real CNN-LSTM encoder:
#   - Load your trained encoder once (global)
#   - Preprocess bars exactly like in training (scaling/normalization/CEEMD/etc.)
#   - Return a 1-D float64 vector of length D per ticker (the same D used in training)

_ENCODER_READY = False
def _init_encoder_once():
    global _ENCODER_READY
    if _ENCODER_READY:
        return
    # ===== TODO: load your encoder model/weights here =====
    #   Example (PyTorch):
    #       from state_encoder import build_encoder, preprocess
    #       global _ENC, _preprocess
    #       _ENC = build_encoder(checkpoint_path=os.environ["ENCODER_PATH"])
    #       _preprocess = preprocess
    #       _ENCODER_READY = True
    #
    #   Example (Keras):
    #       from state_encoder_keras import load_encoder, preprocess
    #       _ENC = load_encoder(os.environ["ENCODER_PATH"])
    #       _preprocess = preprocess
    #       _ENCODER_READY = True
    #
    # For now, fail loudly if not implemented to avoid silent mismatch.
    _ENCODER_READY = False

def compute_state_vector(sym: str, bars_df: pd.DataFrame) -> np.ndarray:
    """
    Input: recent bars for `sym` (at least ENCODER_LOOKBACK rows)
    Output: 1-D numpy array of length D (the SAME D as in training)
    """
    if not _ENCODER_READY:
        raise NotImplementedError(
            "State encoder not wired. Implement _init_encoder_once() and compute_state_vector() "
            "to produce the D-dim CNN-LSTM state per ticker from live bars."
        )
    # ===== TODO: run your real preprocessing + encoder forward pass =====
    # x = _preprocess(bars_df)     # shape [1, time, features] per your training
    # state_vec = _ENC(x)          # -> (1, D)
    # return np.asarray(state_vec[0], dtype=np.float64)
    return np.zeros(64, dtype=np.float64)  # placeholder; delete when wired


# ========== LIVE LOOP ==========
if __name__ == "__main__":
    print("▶️  Starting LIVE RL deploy (CNN-LSTM state). Ctrl+C to exit.", flush=True)

    # Try to init encoder (edit the function to actually load your model)
    try:
        _init_encoder_once()
    except Exception as e:
        print(f"⚠️  Encoder init warning: {e}")

    try:
        while True:
            loop_start = perf_counter()
            now = pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern")
            print(f"\n🔄 Loop start: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

            # 1) Market check
            try:
                clock = api.get_clock()
            except Exception as e:
                print(f"  ❌ get_clock error: {e}")
                time.sleep(SLEEP_INTERVAL); continue
            if not clock.is_open:
                print(f"  ❌ Market closed, next open at {clock.next_open}")
                time.sleep(SLEEP_INTERVAL); continue
            print(f"  ⏱ Market open, next close at {clock.next_close}")

            # 2) Account, cash, positions (live)
            try:
                account = api.get_account()
                cash = float(account.cash)
                positions = {p.symbol: float(p.qty) for p in api.list_positions()}  # qty can be fractional
            except Exception as e:
                print(f"  ❌ account/positions error: {e}")
                time.sleep(SLEEP_INTERVAL); continue

            # 3) Build per-ticker state and latest prices
            state_rows: List[np.ndarray] = []
            latest_prices: List[float] = []
            failed_syms: List[str] = []

            for sym in TICKERS:
                try:
                    bars = get_recent_bars(sym, need=max(ENCODER_LOOKBACK, 10))
                    if bars.empty or len(bars) < ENCODER_LOOKBACK:
                        raise ValueError(f"Not enough bars ({len(bars)}/{ENCODER_LOOKBACK})")
                    state_vec = compute_state_vector(sym, bars)     # (D,)
                    if state_vec.ndim != 1:
                        raise ValueError(f"State vector for {sym} must be 1-D, got {state_vec.shape}")
                    state_rows.append(np.asarray(state_vec, dtype=np.float64))
                    latest_prices.append(float(bars['close'].iloc[-1]))
                except Exception as e:
                    failed_syms.append(sym)
                    print(f"  ⚠️ {sym}: {e}")

            if failed_syms:
                print(f"  ⚠️ Missing states for: {', '.join(failed_syms)}; skipping loop.")
                time.sleep(SLEEP_INTERVAL); continue

            # 4) Compose observation [K*D, K weights, 1 cash_ratio]
            K = len(TICKERS)
            D = state_rows[0].shape[0]
            state_concat = np.concatenate(state_rows, axis=0)           # (K*D,)
            prices_vec = np.array(latest_prices, dtype=np.float64)      # (K,)

            # live shares from broker positions, aligned to TICKERS order
            shares_vec = np.array([positions.get(sym, 0.0) for sym in TICKERS], dtype=np.float64)

            port_val = float((shares_vec * prices_vec).sum() + cash)
            if port_val <= 0:
                print("  ❌ Non-positive portfolio value; skipping loop.")
                time.sleep(SLEEP_INTERVAL); continue

            w_vec = position_weights(shares_vec, prices_vec, cash)      # (K,)
            cash_ratio = np.array([cash / port_val], dtype=np.float64)  # (1,)

            obs_vec = np.concatenate([state_concat, w_vec, cash_ratio], axis=0).astype(np.float32)
            obs = obs_vec.reshape(1, -1)

            # 5) Policy → logits → softmax → (1 - cash buffer) → clip to MAX_POSITION_WEIGHT
            logits, _ = agent.predict(obs, deterministic=True)
            weights = softmax(logits) * (1.0 - MIN_CASH_BUFFER)
            weights = np.clip(weights, 0.0, MAX_POSITION_WEIGHT)

            # 6) Target dollar allocations and trade deltas ($)
            target_values = weights * port_val
            current_values = shares_vec * prices_vec
            dollar_trades = target_values - current_values  # positive = buy $, negative = sell $

            # 7) Place market orders (fractional via notional)
            #    Enforce cash buffer on buys; cap sells to existing position.
            for i, sym in enumerate(TICKERS):
                dv = float(dollar_trades[i])
                px = float(prices_vec[i])
                if abs(dv) < 1e-6:
                    continue

                if dv > 0:
                    # BUY notional, limited by available cash after buffer
                    # (Alpaca will reject if notional is too small — ignore silently)
                    max_afford = max(0.0, cash - MIN_CASH_BUFFER * port_val)
                    notional = min(dv, max_afford)
                    if notional <= 0:
                        continue
                    try:
                        if USE_NOTIONAL_ORDERS:
                            api.submit_order(symbol=sym, notional=round(notional, 2),
                                             side="buy", type="market", time_in_force="day")
                            cash -= notional  # approximate cash change; broker will settle exactly
                            print(f"  ✅ BUY  ${notional:,.2f} {sym} (market)")
                        else:
                            qty = notional / max(px, 1e-12)
                            api.submit_order(symbol=sym, qty=qty, side="buy",
                                             type="market", time_in_force="day")
                            cash -= qty * px
                            print(f"  ✅ BUY  {qty:.6f} {sym} @ ~{px:.2f}")
                    except APIError as e:
                        print(f"  ❌ BUY {sym}: {e}")

                else:
                    # SELL up to current position value (don’t go short)
                    pos_sh = shares_vec[i]
                    if pos_sh <= 0:
                        continue
                    sell_value_needed = -dv
                    max_sell_value = pos_sh * px
                    notional = min(sell_value_needed, max_sell_value)
                    if notional <= 0:
                        continue
                    try:
                        if USE_NOTIONAL_ORDERS:
                            api.submit_order(symbol=sym, notional=round(notional, 2),
                                             side="sell", type="market", time_in_force="day")
                            cash += notional  # approximate
                            print(f"  ✅ SELL ${notional:,.2f} {sym} (market)")
                        else:
                            qty = notional / max(px, 1e-12)
                            qty = min(qty, pos_sh)
                            api.submit_order(symbol=sym, qty=qty, side="sell",
                                             type="market", time_in_force="day")
                            cash += qty * px
                            print(f"  ✅ SELL {qty:.6f} {sym} @ ~{px:.2f}")
                    except APIError as e:
                        print(f"  ❌ SELL {sym}: {e}")

            # 8) Loop summary & sleep
            loop_time = perf_counter() - loop_start
            next_run = now + pd.Timedelta(seconds=SLEEP_INTERVAL)
            print(f"✅ Loop done in {loop_time:.2f}s. Next run ~ {next_run.strftime('%H:%M:%S %Z')}", flush=True)
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("🛑  Stopped by user", flush=True)
    except Exception as e:
        print(f"⚠️  Error in loop: {e}", flush=True)
        time.sleep(5)
