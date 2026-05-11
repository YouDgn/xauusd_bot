# ============================================================
# risk_manager.py — Gestion du Risque & Money Management
# ============================================================

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import config

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Calcule les tailles de lots, SL/TP, et vérifie les règles de risque.
    Tout ce qui protège le capital.
    """

    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.start_balance: float = 0.0
        self.daily_high_equity: float = 0.0
        self._init_day()

    def _init_day(self):
        """Initialise les métriques de début de journée."""
        stats = self.mt5.get_account_stats()
        self.start_balance       = stats.get("balance", 10000.0)
        self.daily_high_equity   = self.start_balance
        logger.info(f"📊 Balance de départ journée : {self.start_balance:.2f}")

    # ── Calculs de Lot ─────────────────────────────────────────

    def calculate_lot_size(
        self,
        stop_loss_pips: float,
        symbol_info:    Dict,
        equity:         Optional[float] = None
    ) -> float:
        """
        Lot sizing :
        - Si lot manuel défini via dashboard → utilise ce lot
        - Sinon → 1% du compte réel MT5
        """
        lot_min = config.LOT_MIN
        lot_max = config.LOT_MAX

        # Lot manuel (défini via dashboard ou config)
        use_manual = getattr(self, '_use_manual', False) or getattr(config, 'USE_MANUAL_LOT', False)
        if use_manual:
            lot = getattr(self, '_manual_lot', None) or getattr(config, 'MANUAL_LOT_SIZE', 0.05)
            lot = float(lot)
            lot = max(lot_min, min(lot_max, lot))
            logger.info(f"Lot manuel : {lot:.2f}")
            return lot

        # Equity réelle MT5
        if equity is None or equity <= 0:
            stats  = self.mt5.get_account_stats()
            equity = stats.get("equity", 10000)

        RISK_PCT    = 0.01
        risk_amount = equity * RISK_PCT

        # Valeur d'un pip par lot pour XAUUSD
        pip_value_per_lot = (
            symbol_info.get("trade_contract_size", 100) *
            symbol_info.get("point", 0.01)
        )

        if stop_loss_pips <= 0 or pip_value_per_lot <= 0:
            logger.warning("SL ou pip_value invalide, lot minimum utilisé.")
            return lot_min

        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Arrondir au step du broker
        vol_step = symbol_info.get("volume_step", 0.01)
        lot_size = round(lot_size / vol_step) * vol_step
        lot_size = max(
            symbol_info.get("volume_min", lot_min),
            min(symbol_info.get("volume_max", lot_max), lot_size)
        )

        logger.info(
            f"Lot auto : {lot_size:.2f} | "
            f"Capital={equity:.0f}$ | Risk=1% | "
            f"Risque={risk_amount:.2f}$ | SL={stop_loss_pips:.1f} pips"
        )
        return lot_size

    def calculate_sl_tp(
        self,
        action: str,
        entry:  float,
        atr:    float,
        point:  float,
    ) -> Tuple[float, float]:
        """
        Calcule SL et TP basés sur l'ATR.
        Lit STOP_LOSS_ATR_MULT et TAKE_PROFIT_ATR_MULT depuis config
        en temps réel → les changements R:R du dashboard sont immédiats.
        """
        # Lecture config en temps réel
        sl_mult = config.STOP_LOSS_ATR_MULT    # ex: 1.5
        tp_mult = config.TAKE_PROFIT_ATR_MULT  # ex: 3.0 (RR 1:2)

        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        if action == "BUY":
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance

        rr = tp_mult / sl_mult
        logger.info(
            f"SL/TP | {action} @ {entry:.2f} | "
            f"SL={sl:.2f} TP={tp:.2f} | "
            f"ATR={atr:.2f} | RR=1:{rr:.1f}"
        )
        return round(sl, 2), round(tp, 2)

    def sl_to_pips(self, entry: float, sl: float, point: float) -> float:
        """Convertit la distance SL en pips."""
        return abs(entry - sl) / point

    # ── Vérificateurs de Règles ────────────────────────────────

    def check_daily_profit_target(self) -> bool:
        """
        Retourne True si l'objectif de gain journalier est atteint.
        → Le bot doit s'arrêter.
        """
        stats     = self.mt5.get_account_stats()
        equity    = stats.get("equity", self.start_balance)
        pnl_pct   = (equity - self.start_balance) / self.start_balance

        if pnl_pct >= config.DAILY_PROFIT_TARGET:
            logger.warning(
                f"🎯 Objectif journalier atteint ! "
                f"+{pnl_pct*100:.2f}% ≥ {config.DAILY_PROFIT_TARGET*100:.2f}%"
            )
            return True
        return False

    def check_kill_switch(self) -> bool:
        """Obsolète — protection gérée dans live_bot.py directement."""
        return False  # Désactivé — live_bot utilise sa propre protection

    def _check_kill_switch_legacy(self) -> bool:
        """
        Retourne True si la perte journalière dépasse le seuil.
        → Kill switch : fermer tout et arrêter.
        """
        # Garde-fou : start_balance doit être initialisé correctement
        if self.start_balance <= 0:
            logger.warning("Kill switch ignoré : start_balance non initialisé")
            return False

        stats   = self.mt5.get_account_stats()
        equity  = stats.get("equity", self.start_balance)
        pnl_pct = (equity - self.start_balance) / self.start_balance

        # Ignorer si la différence est inférieure à 1$ (bruit)
        if abs(equity - self.start_balance) < 1.0:
            return False

        if pnl_pct <= -config.DAILY_MAX_LOSS:
            logger.critical(
                f"KILL SWITCH ACTIVE ! "
                f"{pnl_pct*100:.2f}% <= -{config.DAILY_MAX_LOSS*100:.2f}%"
            )
            return True
        return False

    def check_daily_profit_target(self) -> bool:
        """Retourne True si l objectif journalier est atteint."""
        if self.start_balance <= 0:
            return False

        stats   = self.mt5.get_account_stats()
        equity  = stats.get("equity", self.start_balance)
        pnl_pct = (equity - self.start_balance) / self.start_balance

        if pnl_pct >= config.DAILY_PROFIT_TARGET:
            logger.warning(
                f"Objectif journalier atteint ! "
                f"+{pnl_pct*100:.2f}% >= {config.DAILY_PROFIT_TARGET*100:.2f}%"
            )
            return True
        return False

    def check_max_trades(self) -> bool:
        """Retourne True si le nombre maximum de trades simultanés est atteint."""
        positions = self.mt5.get_open_positions()
        if len(positions) >= config.MAX_OPEN_TRADES:
            logger.debug(f"Max trades atteint ({len(positions)}/{config.MAX_OPEN_TRADES})")
            return True
        return False

    def update_daily_high(self):
        """Met à jour le plus haut equity de la journée (pour le drawdown)."""
        stats  = self.mt5.get_account_stats()
        equity = stats.get("equity", self.daily_high_equity)
        if equity > self.daily_high_equity:
            self.daily_high_equity = equity

    def get_current_drawdown(self) -> float:
        """Retourne le drawdown courant depuis le plus haut de la journée."""
        stats  = self.mt5.get_account_stats()
        equity = stats.get("equity", self.daily_high_equity)
        if self.daily_high_equity > 0:
            return (self.daily_high_equity - equity) / self.daily_high_equity
        return 0.0

    # ── Statistiques de Performance ────────────────────────────

    def get_session_stats(self) -> Dict:
        """Retourne un dictionnaire de statistiques de session."""
        stats    = self.mt5.get_account_stats()
        equity   = stats.get("equity", self.start_balance)
        pnl_abs  = equity - self.start_balance
        pnl_pct  = pnl_abs / self.start_balance if self.start_balance > 0 else 0.0
        drawdown = self.get_current_drawdown()

        return {
            "start_balance":  self.start_balance,
            "current_equity": equity,
            "pnl_abs":        round(pnl_abs, 2),
            "pnl_pct":        round(pnl_pct * 100, 3),
            "daily_high":     self.daily_high_equity,
            "drawdown_pct":   round(drawdown * 100, 3),
            "open_trades":    len(self.mt5.get_open_positions()),
            "profit_target_pct": config.DAILY_PROFIT_TARGET * 100,
            "kill_switch_pct":   config.DAILY_MAX_LOSS * 100,
        }


class FeatureEngineer:
    """
    Calcule les indicateurs techniques utilisés comme features pour l'IA.
    Utilise la bibliothèque `ta` (compatible Python 3.10).
    """

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule un ensemble complet d'indicateurs techniques sur le DataFrame OHLCV.
        Retourne un DataFrame enrichi pour le réseau de neurones.
        """
        import ta as ta_lib

        df = df.copy()

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # ── Trend ──────────────────────────────────────────────
        df["ema_8"]   = close.ewm(span=8,   adjust=False).mean()
        df["ema_21"]  = close.ewm(span=21,  adjust=False).mean()
        df["ema_50"]  = close.ewm(span=50,  adjust=False).mean()
        df["sma_200"] = close.rolling(200).mean()

        # ── Momentum ───────────────────────────────────────────
        df["rsi_14"] = ta_lib.momentum.rsi(close, window=14)

        macd_ind         = ta_lib.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        df["macd"]       = macd_ind.macd()
        df["macd_signal"]= macd_ind.macd_signal()
        df["macd_hist"]  = macd_ind.macd_diff()

        # ── Volatilité ─────────────────────────────────────────
        df["atr_14"] = ta_lib.volatility.average_true_range(high, low, close, window=14)

        bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pct"]   = bb.bollinger_pband()   # (close - lower) / (upper - lower)

        # ── Volume ─────────────────────────────────────────────
        df["volume_sma"]   = volume.rolling(20).mean()
        df["volume_ratio"] = volume / (df["volume_sma"] + 1e-9)

        # ── Stochastique ───────────────────────────────────────
        stoch = ta_lib.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # ── Retours ────────────────────────────────────────────
        df["return_1"]  = close.pct_change(1)
        df["return_5"]  = close.pct_change(5)
        df["return_20"] = close.pct_change(20)

        # ── Price Position ─────────────────────────────────────
        df["close_vs_ema21"]  = (close - df["ema_21"])  / (df["ema_21"]  + 1e-9)
        df["close_vs_sma200"] = (close - df["sma_200"]) / (df["sma_200"] + 1e-9)

        # ── High/Low Ratio ─────────────────────────────────────
        df["hl_ratio"] = (high - low) / (close + 1e-9)

        df.dropna(inplace=True)
        return df

    @staticmethod
    def get_feature_columns() -> list:
        """Retourne la liste des colonnes features techniques utilisées par l'IA."""
        return [
            "return_1", "return_5", "return_20",
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "atr_14",
            "bb_pct",
            "stoch_k", "stoch_d",
            "volume_ratio",
            "close_vs_ema21", "close_vs_sma200",
            "hl_ratio",
        ]

    @staticmethod
    def get_macro_feature_size() -> int:
        """Taille du vecteur macro (DXY + taux + sessions)."""
        return 18  # Voir macro_features.py get_feature_vector()

    @staticmethod
    def get_atr(df: pd.DataFrame) -> float:
        """Retourne l'ATR(14) de la dernière barre."""
        if "atr_14" in df.columns and not df["atr_14"].empty:
            return float(df["atr_14"].iloc[-1])
        return float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])

    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        """Normalisation Z-score par colonne (en-ligne pour le live)."""
        mean = features.mean(axis=0)
        std  = features.std(axis=0) + 1e-8
        return (features - mean) / std