# ============================================================
# csv_data_loader.py — Chargeur de Données Historiques CSV
# ============================================================
# Ce module charge le fichier xauusd_gold_history_10years.csv
# et l'intègre comme source de données alternative (ou enrichissement)
# pour l'entraînement du modèle IA.
#
# Utilisations :
#   1) Entraînement sans MT5 (offline)
#   2) Enrichir les données MT5 avec les annotations d'événements
#   3) Ajouter une feature "contexte historique" à l'agent RL
# ============================================================

import os
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Chemin par défaut du CSV (à la racine du projet)
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "xauusd_gold_history_10years.csv")

# Mapping des types d'événements vers un score numérique (pour l'IA)
EVENT_TYPE_SCORES: Dict[str, float] = {
    "GUERRE":           0.9,   # Très bullish pour l'or
    "CRISE":            0.75,
    "GEOPOLITIQUE":     0.65,
    "INFLATION":        0.55,
    "FED":              0.0,   # Neutre (dépend du contexte BULLISH/BEARISH)
    "MACRO":            0.0,
    "RECORD":           0.5,
    "BANQUES CENTRALES":0.6,
    "POLITIQUE":        0.2,
    "COMMERCE":         0.1,
    "FOREX":           -0.3,   # Souvent bearish (dollar fort)
    "":                 0.0,
}

IMPACT_MULTIPLIER: Dict[str, float] = {
    "BULLISH":  1.0,
    "BEARISH": -1.0,
    "NEUTRAL":  0.0,
    "":         0.0,
}


class GoldCSVLoader:
    """
    Charge et traite les données historiques XAUUSD depuis le CSV.
    
    Fonctionnalités :
    - Chargement et validation des données
    - Calcul de features techniques sur les données mensuelles
    - Encodage numérique des événements pour l'IA
    - Fusion avec des données MT5 haute fréquence
    - Rapport statistique des 10 ans de données
    """

    def __init__(self, csv_path: str = DEFAULT_CSV_PATH):
        self.csv_path  = csv_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df:     Optional[pd.DataFrame] = None

    # ── Chargement ─────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Charge le CSV et retourne un DataFrame traité."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"CSV introuvable : {self.csv_path}\n"
                f"Place le fichier 'xauusd_gold_history_10years.csv' dans le dossier 'data/'"
            )

        logger.info(f"📂 Chargement du CSV : {self.csv_path}")
        df = pd.read_csv(self.csv_path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Renommer pour compatibilité avec le reste du bot
        df.rename(columns={
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        }, inplace=True)

        self.df_raw = df.copy()

        # Traitement
        df = self._encode_events(df)
        df = self._add_technical_features(df)
        df = self._add_regime_labels(df)

        self.df = df
        logger.info(
            f"✅ {len(df)} enregistrements chargés | "
            f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}"
        )
        return df

    # ── Encodage des Événements ────────────────────────────────

    def _encode_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode les colonnes texte des événements en valeurs numériques."""
        df = df.copy()

        # Score du type d'événement
        df["event_type_score"] = df["event_type"].fillna("").map(
            lambda x: EVENT_TYPE_SCORES.get(x.strip(), 0.0)
        )

        # Multiplicateur d'impact
        df["impact_multiplier"] = df["impact"].fillna("").map(
            lambda x: IMPACT_MULTIPLIER.get(x.strip(), 0.0)
        )

        # Score événement final = type × impact
        df["event_score"] = df["event_type_score"] * df["impact_multiplier"].replace(0, 1)

        # Clamp entre -1 et 1
        df["event_score"] = df["event_score"].clip(-1.0, 1.0)

        # Flag : y a-t-il un événement majeur ?
        df["has_major_event"] = (df["event"].fillna("") != "").astype(int)

        # One-hot encoding simplifié des types
        major_types = ["GUERRE", "CRISE", "FED", "INFLATION", "GEOPOLITIQUE", "RECORD"]
        for etype in major_types:
            df[f"event_{etype.lower()}"] = (
                df["event_type"].fillna("").str.upper() == etype
            ).astype(int)

        return df

    # ── Features Techniques (sur données mensuelles) ───────────

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des indicateurs techniques adaptés aux données mensuelles."""
        df = df.copy()

        # Retours
        df["return_1m"]  = df["Close"].pct_change(1)
        df["return_3m"]  = df["Close"].pct_change(3)
        df["return_6m"]  = df["Close"].pct_change(6)
        df["return_12m"] = df["Close"].pct_change(12)

        # Moyennes mobiles
        df["sma_6m"]  = df["Close"].rolling(6).mean()
        df["sma_12m"] = df["Close"].rolling(12).mean()
        df["sma_24m"] = df["Close"].rolling(24).mean()

        # Volatilité
        df["volatility_3m"]  = df["return_1m"].rolling(3).std()
        df["volatility_12m"] = df["return_1m"].rolling(12).std()

        # RSI mensuel (14 périodes = ~14 mois)
        delta = df["Close"].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        df["rsi_14m"] = 100 - (100 / (1 + rs))

        # ATR mensuel
        df["hl_range"] = df["High"] - df["Low"]
        df["atr_6m"]   = df["hl_range"].rolling(6).mean()

        # Position vs SMA
        df["price_vs_sma12"] = (df["Close"] - df["sma_12m"]) / (df["sma_12m"] + 1e-9)
        df["price_vs_sma24"] = (df["Close"] - df["sma_24m"]) / (df["sma_24m"] + 1e-9)

        # Drawdown depuis ATH
        df["ath_rolling"] = df["High"].cummax()
        df["drawdown_from_ath"] = (df["ath_rolling"] - df["Close"]) / df["ath_rolling"]

        # Volume relatif
        df["volume_ratio"] = df["Volume"] / (df["Volume"].rolling(12).mean() + 1e-9)

        return df

    # ── Régimes de Marché ──────────────────────────────────────

    def _add_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Étiquette chaque période avec un régime de marché :
        0 = Bear, 1 = Sideways, 2 = Bull
        Utile pour l'analyse et l'entraînement conditionnel.
        """
        df = df.copy()
        sma6  = df["sma_6m"]
        sma12 = df["sma_12m"]
        ret6  = df["return_6m"]

        conditions = [
            (sma6 > sma12) & (ret6 > 0.05),   # Bull trend
            (sma6 < sma12) & (ret6 < -0.05),   # Bear trend
        ]
        choices = [2, 0]
        df["market_regime"] = np.select(conditions, choices, default=1)

        regime_labels = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}
        df["regime_label"] = df["market_regime"].map(regime_labels)

        return df

    # ── Accès aux Données ──────────────────────────────────────

    def get_event_context(self, target_date: datetime) -> Dict:
        """
        Retourne le contexte événementiel pour une date donnée.
        Utilisé par le live bot pour enrichir l'observation IA.
        """
        if self.df is None:
            return {"event_score": 0.0, "has_major_event": 0, "regime": 1}

        # Trouver la donnée mensuelle la plus proche
        idx = self.df.index.searchsorted(target_date, side="right") - 1
        if idx < 0:
            return {"event_score": 0.0, "has_major_event": 0, "regime": 1}

        row = self.df.iloc[idx]
        return {
            "event_score":     float(row.get("event_score", 0.0)),
            "has_major_event": int(row.get("has_major_event", 0)),
            "regime":          int(row.get("market_regime", 1)),
            "volatility_3m":   float(row.get("volatility_3m", 0.0)),
            "return_12m":      float(row.get("return_12m", 0.0)),
            "event_text":      str(row.get("event", "")),
        }

    def get_feature_columns(self) -> list:
        """Retourne les colonnes numériques utilisables comme features IA."""
        return [
            "return_1m", "return_3m", "return_6m", "return_12m",
            "volatility_3m", "volatility_12m",
            "rsi_14m", "atr_6m",
            "price_vs_sma12", "price_vs_sma24",
            "drawdown_from_ath", "volume_ratio",
            "event_score", "has_major_event",
            "event_crise", "event_guerre", "event_fed",
            "event_inflation", "event_geopolitique",
        ]

    def get_ohlcv(self) -> pd.DataFrame:
        """Retourne seulement les colonnes OHLCV propres."""
        if self.df is None:
            raise RuntimeError("Appelle d'abord .load()")
        return self.df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # ── Rapport Statistique ────────────────────────────────────

    def print_report(self):
        """Affiche un rapport complet des 10 années de données."""
        if self.df is None:
            print("⚠️ Charge d'abord les données avec .load()")
            return

        df = self.df

        print("\n" + "=" * 65)
        print("   📊 RAPPORT DONNÉES HISTORIQUES XAUUSD (10 ANS)")
        print("=" * 65)

        # Statistiques générales
        print(f"\n📅 Période       : {df.index[0]:%Y-%m} → {df.index[-1]:%Y-%m}")
        print(f"📈 Enregistrements: {len(df)} mois")
        print(f"💰 Prix min      : ${df['Low'].min():.2f}  ({df['Low'].idxmin().strftime('%Y-%m')})")
        print(f"💰 Prix max      : ${df['High'].max():.2f} ({df['High'].idxmax().strftime('%Y-%m')})")
        print(f"📊 Perf totale   : {(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:+.1f}%")

        # Régimes de marché
        regime_counts = df["regime_label"].value_counts()
        print(f"\n🏷️  Régimes de marché :")
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"   {regime:<10} : {count:3d} mois ({pct:.1f}%)")

        # Événements majeurs
        events = df[df["has_major_event"] == 1][["Close", "event", "event_type", "impact", "return_1m"]]
        print(f"\n📰 Événements majeurs ({len(events)}) :")
        print("-" * 65)
        for date, row in events.iterrows():
            ret_str = f"{row['return_1m']*100:+.1f}%" if pd.notna(row['return_1m']) else "N/A"
            print(
                f"  {date.strftime('%Y-%m')} | ${row['Close']:6.0f} | {ret_str:>7} | "
                f"[{row['event_type']:<15}] {row['event'][:40]}"
            )

        # Meilleures et pires périodes
        df_clean = df.dropna(subset=["return_1m"])
        top3    = df_clean.nlargest(3, "return_1m")[["Close", "return_1m", "event"]]
        worst3  = df_clean.nsmallest(3, "return_1m")[["Close", "return_1m", "event"]]

        print(f"\n🟢 Top 3 meilleures performances mensuelles :")
        for date, row in top3.iterrows():
            print(f"   {date.strftime('%Y-%m')} | {row['return_1m']*100:+.1f}% | {str(row['event'])[:45]}")

        print(f"\n🔴 Top 3 pires performances mensuelles :")
        for date, row in worst3.iterrows():
            print(f"   {date.strftime('%Y-%m')} | {row['return_1m']*100:+.1f}% | {str(row['event'])[:45]}")

        print("\n" + "=" * 65)

    # ── Fusion avec données MT5 haute fréquence ────────────────

    @staticmethod
    def enrich_mt5_data(df_mt5: pd.DataFrame, df_csv: pd.DataFrame) -> pd.DataFrame:
        """
        Fusionne les données MT5 (haute fréquence) avec le contexte
        mensuel du CSV (événements, régime de marché).
        
        Les features mensuelles sont propagées vers l'avant (forward fill)
        sur les barres intra-day.
        """
        monthly_features = [
            "event_score", "has_major_event", "market_regime",
            "volatility_12m", "return_12m", "drawdown_from_ath",
        ]
        # Garder seulement les colonnes existantes
        monthly_features = [c for c in monthly_features if c in df_csv.columns]

        if not monthly_features:
            logger.warning("Aucune feature mensuelle disponible pour l'enrichissement.")
            return df_mt5

        # Ré-indexer les données mensuelles sur le calendrier MT5
        df_monthly = df_csv[monthly_features].copy()
        df_monthly = df_monthly.reindex(
            df_mt5.index.union(df_monthly.index)
        ).ffill().reindex(df_mt5.index)

        # Fusionner
        result = df_mt5.copy()
        for col in monthly_features:
            result[f"macro_{col}"] = df_monthly[col].values

        logger.info(
            f"✅ Enrichissement MT5 : +{len(monthly_features)} features macro mensuelles"
        )
        return result


# ── Point d'entrée standalone ──────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    loader = GoldCSVLoader(
        os.path.join(os.path.dirname(__file__), "..", "data", "xauusd_gold_history_10years.csv")
    )
    df = loader.load()
    loader.print_report()

    print("\n📋 Aperçu des features IA :")
    feat_cols = [c for c in loader.get_feature_columns() if c in df.columns]
    print(df[feat_cols].tail(6).to_string())