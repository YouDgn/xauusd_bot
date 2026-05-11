# ============================================================
# config.py — Configuration Centrale du Bot XAUUSD
# ============================================================

import os
from dataclasses import dataclass, field
from typing import List

# ── Compte MT5 ──────────────────────────────────────────────
MT5_LOGIN    = 0    # Renseigné via l'écran de login du dashboard
MT5_PASSWORD = ""   # Ne jamais mettre ici
MT5_SERVER   = ""   # Ne jamais mettre ici
MANUAL_LOT_SIZE = 0.05
USE_MANUAL_LOT  = True
SYMBOL       = "XAUUSD"
MAGIC_NUMBER = 20240101    # Identifiant unique pour les ordres du bot

# ── Paramètres de l'Instrument ───────────────────────────────
POINT_VALUE  = 0.1         # Valeur d'un point pour XAUUSD (à vérifier dans MT5)
LOT_MIN      = 0.01
LOT_MAX      = 10.0
LOT_STEP     = 0.01
DEVIATION    = 20          # Slippage max en points

# ── Gestion du Risque ────────────────────────────────────────
# NOTE : Ces paramètres sont GÉRÉS AUTOMATIQUEMENT par l'IA (trading_env.py)
# Ne pas modifier — l'env RL les calcule dynamiquement
STOP_LOSS_ATR_MULT    = 8.0    # Extrême : SL très loin
TAKE_PROFIT_ATR_MULT  = 24.0   # Extrême : RR 1:3 (TP = 3× SL)
RISK_PER_TRADE_PCT    = 0.01   # 1% de risque par trade

# Protection compte (live_bot.py) — seul paramètre restant
DAILY_MAX_LOSS        = 0.05   # Arrêt si -5% sur la session
DAILY_PROFIT_TARGET   = 0.05   # Info seulement (géré par l'IA)

# ── Timeframe & Données ──────────────────────────────────────
TIMEFRAME         = "M15"    # Timeframe principal (M15 = 15 minutes)
LOOKBACK_BARS     = 100      # 50→100 : ATR plus stable et représentatif
TRAINING_YEARS    = 2        # 5→2 ans : données MT5 réelles suffisantes
TICK_INTERVAL_SEC = 1        # Intervalle de polling des ticks en live

# ── Réseau de Neurones ───────────────────────────────────────
HIDDEN_SIZE       = 128      # 256→128 : 2x plus rapide, qualité quasi identique
NUM_LAYERS        = 2        # 3→2 : moins de calcul
DROPOUT           = 0.1

# ── Hyperparamètres PPO ──────────────────────────────────────
PPO_LR            = 3e-4     # Revenir à 3e-4 (converge mieux)
PPO_GAMMA         = 0.99     # Discount factor
PPO_GAE_LAMBDA    = 0.95     # GAE lambda
PPO_CLIP_EPS      = 0.2      # Clipping epsilon
PPO_VALUE_COEF    = 0.5      # Coefficient de la value loss
PPO_ENTROPY_COEF  = 0.15     # 0.10→0.15 : Plus d'exploration pour découvrir meilleures stratégies
PPO_UPDATE_EPOCHS = 4        # bon équilibre vitesse/qualité
PPO_BATCH_SIZE    = 512      # bon pour CPU
PPO_ROLLOUT_STEPS = 8192     # 4096→8192 : encore moins d'overhead à 1000+ steps/s

# ── Entraînement ─────────────────────────────────────────────
TOTAL_TRAIN_STEPS  = 1_000_000
SAVE_EVERY_STEPS   = 25_000   # 50k→25k : évaluations 2x plus fréquentes
MODEL_PATH         = "models/ppo_xauusd.pt"
CHECKPOINT_DIR     = "models/checkpoints/"

# ── News & Sentiment ─────────────────────────────────────────
NEWS_UPDATE_INTERVAL = 300   # Rafraîchissement des news toutes les 5 min
SENTIMENT_WEIGHT     = 0.15  # Poids du sentiment dans la décision IA

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.forexfactory.com/ff_calendar_thisweek.xml",
    "https://www.investing.com/rss/news_25.rss",   # Gold news
    "https://www.kitco.com/rss/news.xml",
]

GOLD_KEYWORDS = [
    "gold", "xauusd", "bullion", "inflation", "fed", "federal reserve",
    "interest rate", "dollar", "usd", "treasury", "safe haven",
    "geopolitical", "war", "crisis", "recession", "cpi", "pce"
]

# ── Logging & Dashboard ──────────────────────────────────────
LOG_FILE         = "logs/bot_decisions.log"
LOG_LEVEL        = "INFO"
DASHBOARD_REFRESH = 2        # Secondes entre chaque refresh du dashboard

# ── Device Calcul (AMD DirectML) ────────────────────────────
USE_DIRECTML = True          # False = CPU fallback