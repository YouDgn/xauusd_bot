# ============================================================
# macro_features.py — Features Macro : DXY, Taux 10 ans, Sessions
# ============================================================
# Données récupérées via yfinance (Yahoo Finance) — gratuit, sans API key
# Mise à jour automatique toutes les heures en live
# ============================================================

import logging
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import pytz

logger = logging.getLogger(__name__)


class MacroFeaturesModule:
    """
    Récupère et met à jour les données macro en temps réel :

    1. DXY  (Dollar Index)       — Corrélation -0.85 avec l'or
    2. US10Y (Taux 10 ans USA)   — Taux réels vs or
    3. Session de trading        — Londres/NY/Asie/Hors-session
    4. Jour de la semaine        — Patterns hebdomadaires
    """

    def __init__(self):
        self._data: Dict = {
            # DXY
            "dxy_price":        100.0,   # Prix actuel DXY
            "dxy_return_1d":    0.0,     # Variation journalière DXY
            "dxy_return_5d":    0.0,     # Variation 5 jours DXY
            "dxy_vs_sma20":     0.0,     # DXY au-dessus/dessous SMA20
            "dxy_rsi":          50.0,    # RSI du DXY

            # Taux 10 ans US
            "us10y_rate":       4.0,     # Taux en %
            "us10y_change_1d":  0.0,     # Variation journalière en bps
            "us10y_change_5d":  0.0,     # Variation 5 jours
            "real_rate_proxy":  0.0,     # Taux 10 ans - inflation proxy

            # Session de trading
            "session_asia":     0,       # 1 si session Asie active
            "session_london":   0,       # 1 si session Londres active
            "session_newyork":  0,       # 1 si session New York active
            "session_overlap":  0,       # 1 si chevauchement Londres/NY
            "hour_sin":         0.0,     # Heure encodée cycliquement (sin)
            "hour_cos":         1.0,     # Heure encodée cycliquement (cos)

            # Jour de la semaine
            "day_monday":       0,
            "day_tuesday":      0,
            "day_wednesday":    0,       # Souvent FOMC
            "day_thursday":     0,
            "day_friday":       0,       # NFP + clôture positions
            "day_sin":          0.0,     # Jour encodé cycliquement
            "day_cos":          1.0,

            # Métadonnées
            "last_update":      None,
            "data_available":   False,
        }
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── API Publique ───────────────────────────────────────────

    def start(self):
        """Lance la mise à jour en arrière-plan (toutes les heures)."""
        self._running = True
        # Première mise à jour synchrone
        self._update_all()
        # Puis thread en arrière-plan
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("MacroFeatures : module démarré.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_features(self) -> Dict:
        """Retourne toutes les features macro (thread-safe)."""
        with self._lock:
            # Toujours mettre à jour les features temps-réel (session/heure)
            self._update_time_features_inplace()
            return dict(self._data)

    def get_feature_vector(self) -> np.ndarray:
        """
        Retourne un vecteur numpy normalisé des features macro.
        Utilisé directement comme input supplémentaire du réseau IA.
        """
        d = self.get_features()
        vector = np.array([
            # DXY (normalisé autour de 0)
            np.clip(d["dxy_return_1d"] * 100, -3, 3),      # % var jour
            np.clip(d["dxy_return_5d"] * 100, -5, 5),      # % var 5j
            np.clip(d["dxy_vs_sma20"] * 100, -5, 5),       # vs SMA20
            np.clip((d["dxy_rsi"] - 50) / 50, -1, 1),      # RSI centré

            # Taux 10 ans (normalisé)
            np.clip(d["us10y_rate"] / 10, 0, 1),            # Niveau absolu
            np.clip(d["us10y_change_1d"] / 20, -1, 1),     # Variation bps/j
            np.clip(d["us10y_change_5d"] / 50, -1, 1),     # Variation bps/5j
            np.clip(d["real_rate_proxy"] / 5, -1, 1),       # Taux réels proxy

            # Sessions (binaires)
            float(d["session_asia"]),
            float(d["session_london"]),
            float(d["session_newyork"]),
            float(d["session_overlap"]),

            # Encodage temporel cyclique
            d["hour_sin"],
            d["hour_cos"],
            d["day_sin"],
            d["day_cos"],

            # Jours spéciaux
            float(d["day_wednesday"]),   # FOMC souvent mercredi
            float(d["day_friday"]),      # NFP + clôture
        ], dtype=np.float32)

        return np.clip(vector, -3, 3)

    def get_feature_names(self) -> list:
        """Noms des features dans l'ordre du vecteur."""
        return [
            "dxy_return_1d", "dxy_return_5d", "dxy_vs_sma20", "dxy_rsi",
            "us10y_rate", "us10y_change_1d", "us10y_change_5d", "real_rate_proxy",
            "session_asia", "session_london", "session_newyork", "session_overlap",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "day_wednesday", "day_friday",
        ]

    # ── Mises à jour ───────────────────────────────────────────

    def _update_loop(self):
        """Met à jour les données toutes les heures."""
        while self._running:
            time.sleep(3600)  # 1 heure
            try:
                self._update_all()
            except Exception as e:
                logger.error(f"MacroFeatures update error: {e}")

    def _update_all(self):
        """Récupère DXY + Taux 10 ans via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance non installé. pip install yfinance")
            logger.warning("Utilisation des valeurs par défaut pour les features macro.")
            with self._lock:
                self._update_time_features_inplace()
            return

        dxy_ok   = self._fetch_dxy(yf)
        rates_ok = self._fetch_us10y(yf)

        with self._lock:
            self._update_time_features_inplace()
            self._data["data_available"] = dxy_ok or rates_ok
            self._data["last_update"]    = datetime.utcnow()

        logger.info(
            f"MacroFeatures mis a jour | "
            f"DXY={self._data['dxy_price']:.2f} | "
            f"US10Y={self._data['us10y_rate']:.2f}% | "
            f"Session={self._get_session_name()}"
        )

    def _fetch_dxy(self, yf) -> bool:
        """Récupère les données du Dollar Index (DX-Y.NYB)."""
        try:
            ticker = yf.Ticker("DX-Y.NYB")
            hist   = ticker.history(period="30d", interval="1d")

            if hist.empty or len(hist) < 5:
                logger.warning("DXY: données insuffisantes")
                return False

            closes = hist["Close"].values
            price  = float(closes[-1])
            sma20  = float(np.mean(closes[-20:])) if len(closes) >= 20 else price
            ret_1d = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0.0
            ret_5d = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0.0

            # RSI du DXY
            delta = np.diff(closes[-15:])
            gain  = np.mean(delta[delta > 0]) if any(delta > 0) else 0
            loss  = abs(np.mean(delta[delta < 0])) if any(delta < 0) else 1e-9
            rsi   = 100 - (100 / (1 + gain / loss))

            with self._lock:
                self._data.update({
                    "dxy_price":     price,
                    "dxy_return_1d": float(ret_1d),
                    "dxy_return_5d": float(ret_5d),
                    "dxy_vs_sma20":  float((price - sma20) / sma20),
                    "dxy_rsi":       float(rsi),
                })
            return True

        except Exception as e:
            logger.debug(f"DXY fetch error: {e}")
            return False

    def _fetch_us10y(self, yf) -> bool:
        """Récupère le taux des obligations US 10 ans (^TNX)."""
        try:
            ticker = yf.Ticker("^TNX")
            hist   = ticker.history(period="30d", interval="1d")

            if hist.empty or len(hist) < 2:
                logger.warning("US10Y: données insuffisantes")
                return False

            closes    = hist["Close"].values
            rate      = float(closes[-1])           # En %
            change_1d = float(closes[-1] - closes[-2]) * 100    # En bps
            change_5d = float(closes[-1] - closes[-5]) * 100 if len(closes) >= 5 else 0.0

            # Proxy taux réels = taux 10 ans - inflation estimée (CPI ~3%)
            inflation_proxy = 3.0
            real_rate = rate - inflation_proxy

            with self._lock:
                self._data.update({
                    "us10y_rate":      rate,
                    "us10y_change_1d": change_1d,
                    "us10y_change_5d": change_5d,
                    "real_rate_proxy": real_rate,
                })
            return True

        except Exception as e:
            logger.debug(f"US10Y fetch error: {e}")
            return False

    def _update_time_features_inplace(self):
        """Met à jour les features temporelles (appelé sans lock)."""
        now_utc = datetime.now(timezone.utc)
        hour    = now_utc.hour + now_utc.minute / 60.0
        weekday = now_utc.weekday()  # 0=lundi, 6=dimanche

        # ── Sessions de trading (heures UTC) ──────────────────
        # Asie    : 00:00 - 08:00 UTC
        # Londres : 08:00 - 17:00 UTC
        # New York: 13:00 - 22:00 UTC
        # Overlap : 13:00 - 17:00 UTC (Londres + NY)

        asia_open    = 0.0  <= hour < 8.0
        london_open  = 8.0  <= hour < 17.0
        ny_open      = 13.0 <= hour < 22.0
        overlap      = 13.0 <= hour < 17.0

        # Week-end = marchés fermés
        is_weekend = weekday >= 5
        if is_weekend:
            asia_open = london_open = ny_open = overlap = False

        # ── Encodage cyclique heure (sin/cos) ──────────────────
        hour_rad = (hour / 24.0) * 2 * np.pi
        hour_sin = float(np.sin(hour_rad))
        hour_cos = float(np.cos(hour_rad))

        # ── Encodage cyclique jour semaine ─────────────────────
        day_rad = (weekday / 7.0) * 2 * np.pi
        day_sin = float(np.sin(day_rad))
        day_cos = float(np.cos(day_rad))

        self._data.update({
            "session_asia":    int(asia_open),
            "session_london":  int(london_open),
            "session_newyork": int(ny_open),
            "session_overlap": int(overlap),
            "hour_sin":        hour_sin,
            "hour_cos":        hour_cos,
            "day_monday":    int(weekday == 0),
            "day_tuesday":   int(weekday == 1),
            "day_wednesday": int(weekday == 2),
            "day_thursday":  int(weekday == 3),
            "day_friday":    int(weekday == 4),
            "day_sin":       day_sin,
            "day_cos":       day_cos,
        })

    def _get_session_name(self) -> str:
        """Retourne le nom de la session active."""
        d = self._data
        if d["session_overlap"]:  return "OVERLAP London/NY"
        if d["session_newyork"]:  return "NEW YORK"
        if d["session_london"]:   return "LONDRES"
        if d["session_asia"]:     return "ASIE"
        return "HORS SESSION"

    def get_dashboard_string(self) -> str:
        """Résumé pour le dashboard."""
        d = self.get_features()
        dxy_arrow = "↑" if d["dxy_return_1d"] > 0 else "↓"
        us10y_arrow = "↑" if d["us10y_change_1d"] > 0 else "↓"
        return (
            f"DXY={d['dxy_price']:.2f}{dxy_arrow} | "
            f"US10Y={d['us10y_rate']:.2f}%{us10y_arrow} | "
            f"Session={self._get_session_name()}"
        )