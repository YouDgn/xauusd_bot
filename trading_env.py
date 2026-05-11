# ============================================================
# trading_env.py — Environnement Gymnasium REFACTORISÉ
# Version : Expert Institutionnel
# ============================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict
from risk_manager import FeatureEngineer
import config

logger = logging.getLogger(__name__)


def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    if len(ts) < max_lag + 1:
        return 0.5
    try:
        lags = range(2, max_lag)
        tau  = np.array([np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags])
        tau[tau == 0] = 1e-10
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def detect_order_blocks(high, low, close, lookback=10):
    if len(close) < lookback + 2:
        return 0.0, 0.0
    atr = np.mean(high[-lookback:] - low[-lookback:]) + 1e-8
    current = close[-1]
    bull_ob = bear_ob = current
    for i in range(len(close) - lookback, len(close) - 1):
        move = close[i+1] - close[i]
        if move > atr * 1.5 and close[i] < close[i-1]:
            bull_ob = low[i]
        if move < -atr * 1.5 and close[i] > close[i-1]:
            bear_ob = high[i]
    dist_bull = np.clip((current - bull_ob) / atr, -3, 3)
    dist_bear = np.clip((bear_ob - current) / atr, -3, 3)
    return float(dist_bull), float(dist_bear)


def detect_fvg(high, low, lookback=5):
    if len(high) < lookback + 3:
        return 0.0
    score = 0.0
    for i in range(len(high) - lookback - 2, len(high) - 2):
        if low[i+2] > high[i]:
            score += 1.0
        elif high[i+2] < low[i]:
            score -= 1.0
    return float(np.clip(score / lookback, -1, 1))


def find_pivot_sl(high, low, direction, lookback=20):
    if len(high) < lookback:
        return 0.0
    h = high[-lookback:]
    l = low[-lookback:]
    return float(np.min(l)) if direction == 1 else float(np.max(h))


class XAUUSDTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    HOLD  = 0
    BUY   = 1
    SELL  = 2
    CLOSE = 3

    # Friction marché réelle
    SPREAD_PIPS     = 30.0
    SLIPPAGE_PIPS   = 5.0
    COMMISSION_PIPS = 5.0
    PIP_VALUE       = 0.1
    FRICTION        = (30.0 + 5.0 + 5.0) * 0.1  # = 4.0$

    # Reward
    W_SORTINO      = 0.3   # Léger signal régularité
    W_HWM_PENALTY  = 0.3   # Léger drawdown
    W_CLOSE        = 2.0   # Fort signal sur les clôtures
    W_HOLD_PENALTY = 0.005  # 0.02→0.005 : Très faible, laisse liberté à l'IA

    # Verrous risque
    MAX_TRADES_PER_DAY = 3
    BARS_PER_DAY       = 96
    BREAKEVEN_RR       = 1.0

    def __init__(self, df, lookback=config.LOOKBACK_BARS, sentiment_score=0.0):
        super().__init__()

        self.df            = df.copy()
        self.lookback      = lookback
        self.ext_sentiment = sentiment_score
        self.macro_vector  = np.zeros(18, dtype=np.float32)

        self.fe           = FeatureEngineer()
        self.df_feat      = self.fe.compute_features(self.df)
        self.feature_cols = [c for c in self.fe.get_feature_columns() if c in self.df_feat.columns]

        self.n_smc      = 6
        self.n_features = len(self.feature_cols) + self.n_smc + 4

        self._feat_np  = self.df_feat[self.feature_cols].values.astype(np.float32)
        self._close_np = self.df_feat["Close"].values.astype(np.float32)
        self._high_np  = self.df_feat["High"].values.astype(np.float32)
        self._low_np   = self.df_feat["Low"].values.astype(np.float32)
        if "atr_14" in self.df_feat.columns:
            self._atr_np = self.df_feat["atr_14"].values.astype(np.float32)
        else:
            self._atr_np = np.full(len(self.df_feat), float(self.df_feat["Close"].std()), dtype=np.float32)

        self._smc_np = self._precompute_smc()

        obs_size = self.lookback * self.n_features + 18
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.reset()

    def _precompute_smc(self):
        n   = len(self._close_np)
        smc = np.zeros((n, self.n_smc), dtype=np.float32)
        win = min(30, self.lookback)
        for i in range(win, n):
            h = self._high_np[max(0, i-win):i+1]
            l = self._low_np[max(0, i-win):i+1]
            c = self._close_np[max(0, i-win):i+1]
            atr = float(self._atr_np[i]) + 1e-8

            ob_bull, ob_bear = detect_order_blocks(h, l, c, min(10, len(c)-1))
            fvg   = detect_fvg(h, l, min(5, max(1, len(h)-3)))
            hurst = hurst_exponent(c, max_lag=min(20, len(c)//2))

            if len(c) >= 50:
                # EMA rapide sans pandas (alpha = 2/(span+1))
                def fast_ema(arr, span):
                    alpha = 2.0 / (span + 1)
                    e = arr[0]
                    for x in arr[1:]:
                        e = alpha * x + (1 - alpha) * e
                    return e
                ema20 = fast_ema(c, 20)
                ema50 = fast_ema(c, 50)
                ema_trend = float(np.clip((ema20 - ema50) / atr, -3, 3))
            else:
                ema_trend = 0.0

            bb_squeeze = float(np.clip(np.std(c[-20:]) / atr, 0, 3)) if len(c) >= 20 else 1.0

            smc[i] = [ob_bull, ob_bear, fvg, hurst - 0.5, ema_trend, bb_squeeze]
        return smc

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step   = self.lookback + 200
        self.position       = 0
        self.entry_price    = 0.0
        self.entry_step     = 0
        self.sl_price       = 0.0
        self.tp_price       = 0.0
        self.breakeven_done = False
        self.initial_equity = 10_000.0
        self.equity         = self.initial_equity
        self.max_equity     = self.initial_equity
        self.total_trades   = 0
        self.winning_trades = 0
        self.sl_hits        = 0
        self.tp_hits        = 0
        self.episode_pnl    = 0.0
        self.trades_today   = 0
        self.day_start_step = self.current_step
        self._ret_buf  = np.zeros(50, dtype=np.float32)  # Buffer circulaire
        self._ret_idx  = 0
        self._ret_full = False
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        prev_equity   = self.equity
        reward        = 0.0
        trade_info    = ""
        current_price = self._get_close_price(self.current_step)
        atr           = self._get_atr(self.current_step)

        if self.current_step - self.day_start_step >= self.BARS_PER_DAY:
            self.trades_today   = 0
            self.day_start_step = self.current_step

        # ── Filtre de Session (22h-08h = HOLD forcé) ──────────
        # Le spread XAUUSD est 3-5x plus large hors session London/NY
        bar_hour = 0
        if hasattr(self.df_feat.index, 'hour'):
            try:
                bar_hour = self.df_feat.index[min(self.current_step, len(self.df_feat)-1)].hour
            except Exception:
                bar_hour = 12  # fallback = heure de trading
        dead_zone = (bar_hour >= 22) or (bar_hour < 8)
        if dead_zone and action in (self.BUY, self.SELL):
            action = self.HOLD  # Force HOLD hors session

        if action == self.BUY and self.position == 0:
            if self.trades_today < self.MAX_TRADES_PER_DAY:
                h = self._high_np[max(0, self.current_step-20):self.current_step+1]
                l = self._low_np[max(0, self.current_step-20):self.current_step+1]
                pivot_sl = find_pivot_sl(h, l, 1, min(20, len(h)))
                sl_dist  = max(current_price - pivot_sl, atr * 1.0)
                tp_dist  = sl_dist * config.TAKE_PROFIT_ATR_MULT / config.STOP_LOSS_ATR_MULT
                self.position     = 1
                self.entry_price  = current_price
                self.entry_step   = self.current_step
                self.sl_price     = current_price - sl_dist
                self.tp_price     = current_price + tp_dist
                self.breakeven_done = False
                self.trades_today += 1
                reward    -= self.FRICTION / self.initial_equity
                trade_info = f"BUY@{current_price:.1f} SL={self.sl_price:.1f} TP={self.tp_price:.1f}"

        elif action == self.SELL and self.position == 0:
            if self.trades_today < self.MAX_TRADES_PER_DAY:
                h = self._high_np[max(0, self.current_step-20):self.current_step+1]
                l = self._low_np[max(0, self.current_step-20):self.current_step+1]
                pivot_sl = find_pivot_sl(h, l, -1, min(20, len(h)))
                sl_dist  = max(pivot_sl - current_price, atr * 1.0)
                tp_dist  = sl_dist * config.TAKE_PROFIT_ATR_MULT / config.STOP_LOSS_ATR_MULT
                self.position     = -1
                self.entry_price  = current_price
                self.entry_step   = self.current_step
                self.sl_price     = current_price + sl_dist
                self.tp_price     = current_price - tp_dist
                self.breakeven_done = False
                self.trades_today += 1
                reward    -= self.FRICTION / self.initial_equity
                trade_info = f"SELL@{current_price:.1f} SL={self.sl_price:.1f} TP={self.tp_price:.1f}"

        elif action == self.CLOSE and self.position != 0:
            pnl, r = self._close_position(current_price)
            reward          += r - self.FRICTION / self.initial_equity
            self.equity     += pnl
            self.episode_pnl+= pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            trade_info = f"CLOSE@{current_price:.1f} PnL={pnl:.2f}"

        # Breakeven à RR 1:1
        if self.position != 0 and not self.breakeven_done:
            rr = self._current_rr(current_price)
            if rr >= self.BREAKEVEN_RR:
                self.sl_price       = self.entry_price
                self.breakeven_done = True

        # SL / TP
        if self.position != 0 and action != self.CLOSE:
            hi = self._get_high_price(self.current_step)
            lo = self._get_low_price(self.current_step)
            sl_hit = (self.position == 1 and lo <= self.sl_price) or \
                     (self.position == -1 and hi >= self.sl_price)
            tp_hit = (self.position == 1 and hi >= self.tp_price) or \
                     (self.position == -1 and lo <= self.tp_price)

            if tp_hit:
                pnl, r = self._close_position(self.tp_price)
                reward += r - self.FRICTION / self.initial_equity
                self.equity += pnl; self.episode_pnl += pnl
                self.total_trades += 1; self.winning_trades += 1; self.tp_hits += 1
                trade_info = f"TP@{self.tp_price:.1f} PnL={pnl:.2f}"
            elif sl_hit:
                pnl, r = self._close_position(self.sl_price)
                reward += r - self.FRICTION / self.initial_equity
                self.equity += pnl; self.episode_pnl += pnl
                self.total_trades += 1; self.sl_hits += 1
                trade_info = f"SL@{self.sl_price:.1f} PnL={pnl:.2f}"

        # Reward ajustée au risque
        step_return = (self.equity - prev_equity) / self.initial_equity
        self._ret_buf[self._ret_idx] = step_return
        self._ret_idx = (self._ret_idx + 1) % 50
        if self._ret_idx == 0:
            self._ret_full = True

        if self.position != 0:
            reward += self.W_SORTINO * self._sortino_step()
            unr     = self._compute_unrealized_pnl(current_price)
            reward += float(np.clip(unr / self.initial_equity * 10, -1, 1))
        else:
            reward -= self.W_HOLD_PENALTY

        # Pénalité HWM drawdown exponentielle
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        dd = (self.max_equity - self.equity) / (self.max_equity + 1e-8)
        if dd > 0.05:
            reward -= self.W_HWM_PENALTY * dd * 5  # linéaire = 10x plus rapide que exp

        self.current_step += 1
        max_step  = len(self.df_feat) - 1
        terminated = self.current_step >= max_step
        self.done  = terminated

        if terminated and self.position != 0:
            fp  = self._get_close_price(min(self.current_step, max_step))
            pnl, _ = self._close_position(fp)
            self.equity += pnl

        info = {
            "equity": self.equity, "position": self.position,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "episode_pnl": self.episode_pnl,
            "sl_hits": self.sl_hits, "tp_hits": self.tp_hits,
            "trades_today": self.trades_today, "trade_info": trade_info,
        }
        # Clipping global de la reward (évite gradient explosion)
        reward = float(np.clip(reward, -10.0, 10.0))
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        end   = self.current_step
        start = max(0, end - self.lookback)
        window = self._feat_np[start:end]
        if len(window) < self.lookback:
            pad    = np.zeros((self.lookback - len(window), len(self.feature_cols)), dtype=np.float32)
            window = np.vstack([pad, window])
        mean = window.mean(axis=0); std = window.std(axis=0); std[std < 1e-8] = 1e-8
        window = (window - mean) / std

        smc_idx = min(max(0, end), len(self._smc_np)-1)
        smc_window = np.empty((self.lookback, self.n_smc), dtype=np.float32)
        smc_window[:] = self._smc_np[smc_idx]

        current_price = float(self._close_np[min(end, len(self._close_np)-1)])
        unr  = self._compute_unrealized_pnl(current_price) / (self.initial_equity + 1e-8)
        dd   = (self.max_equity - self.equity) / (self.max_equity + 1e-8)

        extra = np.empty((self.lookback, 4), dtype=np.float32)
        extra[:, 0] = float(self.position)
        extra[:, 1] = float(self.ext_sentiment)
        extra[:, 2] = float(np.clip(unr, -1, 1))
        extra[:, 3] = float(np.clip(-dd, -1, 0))

        return np.concatenate([
            np.hstack([window, smc_window, extra]).flatten(),
            self.macro_vector
        ]).astype(np.float32)

    def _sortino_step(self):
        if not self._ret_full and self._ret_idx < 5:
            return 0.0
        r   = self._ret_buf  # déjà numpy, pas de conversion
        mu  = r.mean()
        neg = r[r < 0]
        if len(neg) == 0:
            return float(np.clip(mu * 100, 0, 1))
        return float(np.clip(mu / (np.std(neg) + 1e-8), -1, 1))

    def _current_rr(self, price):
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        sl_dist = abs(self.entry_price - self.sl_price) + 1e-8
        return self.position * (price - self.entry_price) / sl_dist

    def _get_close_price(self, step): return float(self._close_np[min(step, len(self._close_np)-1)])
    def _get_high_price(self, step):  return float(self._high_np[min(step, len(self._high_np)-1)])
    def _get_low_price(self, step):   return float(self._low_np[min(step, len(self._low_np)-1)])
    def _get_atr(self, step):         return float(self._atr_np[min(step, len(self._atr_np)-1)])

    def _compute_unrealized_pnl(self, current_price):
        if self.position == 0: return 0.0
        return self.position * (current_price - self.entry_price) * self.PIP_VALUE

    def _close_position(self, close_price):
        pnl      = self._compute_unrealized_pnl(close_price)
        pnl_norm = pnl / self.initial_equity
        reward   = self.W_CLOSE * (1.0 + np.clip(pnl_norm * 50, 0, 2.0)) if pnl > 0 \
                   else -self.W_CLOSE * (1.0 + np.clip(abs(pnl_norm) * 50, 0, 2.0))
        self.position = 0; self.entry_price = 0.0; self.breakeven_done = False
        return pnl, float(reward)

    def update_sentiment(self, score):
        self.ext_sentiment = float(np.clip(score, -1, 1))

    def update_macro(self, macro_vector):
        if macro_vector is not None and len(macro_vector) == 18:
            self.macro_vector = macro_vector.astype(np.float32)

    def render(self):
        pos = {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.position]
        print(f"Step={self.current_step} | {pos} | Equity={self.equity:.2f} | PnL={self.episode_pnl:.2f} | Trades={self.total_trades} ({self.trades_today}/day)")