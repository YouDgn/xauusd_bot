# ============================================================
# live_bot.py — Bot de Trading XAUUSD en Production
# ============================================================
# Lance avec : python live_bot.py
# Arrête avec : Ctrl+C (ferme proprement les positions)
# ============================================================

import sys
import os
import time
import signal
import logging
import threading
import numpy as np
from datetime import datetime, date
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from mt5_connector import MT5Connector
from ppo_agent import PPOAgent
from risk_manager import RiskManager, FeatureEngineer
from news_sentiment import NewsSentimentModule
from macro_features import MacroFeaturesModule
from dashboard import TradingDashboard
from web_dashboard import WebDashboardServer

# ── Logging ────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

import io as _io
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG pour voir toutes les décisions IA
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(
            stream=_io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        ),
    ]
)
logger = logging.getLogger("LIVE_BOT")


class DecisionLogger:
    """Enregistre chaque décision de l'IA dans un fichier texte structuré."""

    def __init__(self, path: str = config.LOG_FILE):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log_decision(
        self,
        action_name:  str,
        price:        float,
        probs:        List[float],
        sentiment:    float,
        account_stats: dict,
        reason:       str = "",
        lot_size:     float = 0.0,
        sl:           float = 0.0,
        tp:           float = 0.0,
    ):
        """Enregistre une décision de trading avec contexte complet."""
        entry = (
            f"\n{'='*70}\n"
            f"DÉCISION IA — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*70}\n"
            f"  Action        : {action_name}\n"
            f"  Prix XAUUSD   : {price:.2f}\n"
            f"  Sentiment     : {sentiment:+.4f}\n"
            f"  Probabilités  : "
            f"HOLD={probs[0]:.3f} BUY={probs[1]:.3f} "
            f"SELL={probs[2]:.3f} CLOSE={probs[3]:.3f}\n"
        )
        if action_name in ("BUY", "SELL"):
            entry += (
                f"  Lot Size      : {lot_size:.2f}\n"
                f"  Stop Loss     : {sl:.2f}\n"
                f"  Take Profit   : {tp:.2f}\n"
                f"  Risque €/$    : {account_stats.get('equity', 0) * config.RISK_PER_TRADE_PCT:.2f}\n"
            )
        entry += (
            f"  Balance       : {account_stats.get('balance', 0):.2f}\n"
            f"  Equity        : {account_stats.get('equity', 0):.2f}\n"
            f"  PnL Open      : {account_stats.get('profit', 0):.2f}\n"
        )
        if reason:
            entry += f"  Raison        : {reason}\n"

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry)


class XAUUSDBot:
    """
    Bot de trading XAUUSD autonome.
    Orchestre l'ensemble des modules : MT5, PPO, News, RiskManager, Dashboard.
    """

    def __init__(self):
        self.mt5       = MT5Connector()
        self.news_mod  = NewsSentimentModule()
        self.macro_mod = MacroFeaturesModule()
        self.dashboard = TradingDashboard()
        self.web_dash  = WebDashboardServer(port=8765, bot_ref=self)
        self.dec_log   = DecisionLogger()
        self.fe        = FeatureEngineer()

        self.agent:    Optional[PPOAgent]    = None
        self.risk_mgr: Optional[RiskManager] = None
        self.running   = False
        self._obs_size: int = 0
        self._run_event   = threading.Event()
        self._last_trade_time = 0.0   # Timestamp du dernier ordre envoyé
        self._trade_cooldown  = 15.0  # Minimum 15 secondes entre deux ordres
        self._position_open_time = None  # Timestamp quand la position a été ouverte
        self._min_hold_bars = 5  # Minimum 5 bars (75 min en M15) avant de fermer

        # Register signal handlers pour arrêt propre
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ── Démarrage ──────────────────────────────────────────────

    def start(self):
        """Initialise et lance le bot. Peut être rappelé après un stop."""
        try:
            if self._run_event.is_set():
                logger.warning("Bot deja en cours")
                return

            logger.info(">>> start() appelé")
            self.web_dash.broadcast_sync({"log_message": "Initialisation en cours..."})
            self._start_inner()

        except Exception as e:
            logger.exception(f"CRASH dans start() : {e}")
            self.web_dash.broadcast_sync({
                "log_message": f"ERREUR : {str(e)[:120]}",
                "status": "STOPPED"
            })
            self._run_event.clear()
            self.running = False

    def _start_inner(self):
        """Corps réel du démarrage."""
        print("\n" + "=" * 70)
        print("   XAUUSD AI TRADING BOT — DEMARRAGE")
        print("=" * 70 + "\n")

        # 1. Connexion MT5
        logger.info("Connexion à MetaTrader5...")
        if not self.mt5.connect():
            msg = "ERREUR: Impossible de se connecter a MT5. Lance MT5 et active Algo Trading."
            logger.error(msg)
            self.web_dash.broadcast_sync({"log_message": msg, "status": "STOPPED"})
            self._run_event.clear()
            self.running = False
            return

        # 2. Récupérer les données live initiales
        logger.info("Chargement des données récentes...")
        df = self.mt5.get_latest_bars(n_bars=config.LOOKBACK_BARS + 250)
        if df is None or len(df) < config.LOOKBACK_BARS:
            msg = "Donnees MT5 insuffisantes — verifie la connexion."
            logger.error(msg)
            self.web_dash.broadcast_sync({"log_message": msg, "status": "STOPPED"})
            self._run_event.clear()
            self.running = False
            return

        # 3. Calculer la taille d'observation (doit correspondre exactement au training)
        df_feat   = self.fe.compute_features(df)
        feat_cols = [c for c in self.fe.get_feature_columns() if c in df_feat.columns]
        n_raw_features = len(feat_cols) + 4   # + position/sentiment/pnl/dd
        macro_size     = 18                    # DXY + US10Y + Sessions (macro_features.py)
        self._obs_size = config.LOOKBACK_BARS * n_raw_features + macro_size

        logger.info(f"Taille observation live : {self._obs_size} "
                    f"(tech={config.LOOKBACK_BARS * n_raw_features} + macro={macro_size})")

        # 4. Charger l'agent PPO
        logger.info("Chargement du modèle IA...")
        self.agent = PPOAgent(obs_size=self._obs_size, n_actions=4)
        if not self.agent.load(config.MODEL_PATH):
            print("⚠️  ATTENTION: Aucun modèle trouvé. Lance d'abord : python train.py")
            print("   Le bot va tout de même démarrer en mode HOLD.")
            logger.warning("Modèle non trouvé — mode HOLD uniquement.")

        # 5. Gestionnaire de risque (init apres connexion MT5)
        self.risk_mgr = RiskManager(self.mt5)
        # Forcer re-lecture de la balance maintenant que MT5 est connecte
        self.risk_mgr._init_day()
        logger.info(f"Balance depart : {self.risk_mgr.start_balance:.2f}")

        # 6. Demarrer les modules asynchrones (réinit si déjà stoppés)
        logger.info("Demarrage du module news...")
        try:
            self.news_mod.force_update()
            self.news_mod.start()
        except Exception as e:
            logger.warning(f"News module: {e}")
            from news_sentiment import NewsSentimentModule
            self.news_mod = NewsSentimentModule()
            self.news_mod.force_update()
            self.news_mod.start()

        logger.info("Demarrage du module macro...")
        try:
            self.macro_mod.start()
        except Exception as e:
            logger.warning(f"Macro module: {e}")
            from macro_features import MacroFeaturesModule
            self.macro_mod = MacroFeaturesModule()
            self.macro_mod.start()

        logger.info("Tous les modules initialises. Trading demarre.")
        self.dashboard.add_log("Bot demarre")
        self.web_dash.broadcast_sync({"log_message": "Bot demarre", "status": "EN COURS"})
        self.running = True
        self._run_event.set()
        self._main_loop()

    # ── Boucle Principale ──────────────────────────────────────

    def _dashboard_update_loop(self):
        """Thread dédié : met à jour le dashboard web toutes les 500ms."""
        while self._run_event.is_set():
            try:
                tick          = self.mt5.get_tick()
                account_stats = self.mt5.get_account_stats()
                positions     = self.mt5.get_open_positions()

                if tick and account_stats:
                    mid_price     = (tick["bid"] + tick["ask"]) / 2
                    session_stats = self.risk_mgr.get_session_stats() if self.risk_mgr else {}
                    macro_data    = self.macro_mod.get_features()
                    sentiment, _  = self.news_mod.get_current_sentiment()
                    action_probs  = getattr(self, "_last_probs",  [0.25]*4)
                    action_name   = getattr(self, "_last_action", "HOLD")

                    # Sérialiser positions (datetime → str)
                    positions_safe = []
                    for p in positions:
                        ps = dict(p)
                        if "open_time" in ps:
                            ps["open_time"] = str(ps["open_time"])
                        positions_safe.append(ps)

                    self.web_dash.broadcast_sync({
                        "price":         round(float(mid_price), 2),
                        "bid":           round(float(tick["bid"]), 2),
                        "ask":           round(float(tick["ask"]), 2),
                        "spread":        round(float(tick.get("spread", 0)), 1),
                        "sentiment":     round(float(sentiment), 3),
                        "balance":       round(float(account_stats.get("balance", 0)), 2),
                        "equity":        round(float(account_stats.get("equity", 0)), 2),
                        "daily_pnl":     round(float(session_stats.get("pnl_abs", 0)), 2),
                        "daily_pnl_pct": round(float(session_stats.get("pnl_pct", 0)), 3),
                        "drawdown":      round(float(session_stats.get("drawdown_pct", 0)), 3),
                        "open_trades":   len(positions),
                        "positions":     positions_safe,
                        "ai_action":     str(action_name),
                        "ai_probs":      [round(float(x), 3) for x in action_probs],
                        "status":        "EN COURS",
                        "macro": {
                            "session_asia":    int(macro_data.get("session_asia", 0)),
                            "session_london":  int(macro_data.get("session_london", 0)),
                            "session_newyork": int(macro_data.get("session_newyork", 0)),
                            "dxy_price":       round(float(macro_data.get("dxy_price", 0)), 2),
                            "us10y_rate":      round(float(macro_data.get("us10y_rate", 0)), 2),
                        }
                    })
            except Exception as e:
                logger.debug(f"dashboard_update_loop error: {e}")

            # Mise à jour toutes les 500ms
            for _ in range(5):
                if not self._run_event.is_set(): break
                time.sleep(0.1)

    def _main_loop(self):
        """Boucle de trading principale."""
        last_bar_time = None
        symbol_info   = self.mt5.get_symbol_info()

        if symbol_info is None:
            logger.error(f"Symbole {config.SYMBOL} introuvable dans MT5.")
            return

        # Lancer le thread de mise à jour dashboard (500ms)
        self._last_probs  = [0.25, 0.25, 0.25, 0.25]
        self._last_action = "HOLD"
        dash_thread = threading.Thread(
            target=self._dashboard_update_loop,
            daemon=True,
            name="DashboardUpdateThread"
        )
        dash_thread.start()
        logger.info("Thread dashboard update demarre (500ms)")

        while self._run_event.is_set():
            try:
                # ── A. Récupérer données marché ────────────────
                # Vérif immédiate du flag d'arrêt
                if not self._run_event.is_set():
                    break

                tick = self.mt5.get_tick()
                if tick is None:
                    logger.warning("Tick MT5 None — en attente...")
                    self.web_dash.broadcast_sync({"log_message": "Tick MT5 indisponible — reconnexion..."})
                    for _ in range(int(config.TICK_INTERVAL_SEC * 10)):
                        if not self._run_event.is_set(): break
                        time.sleep(0.1)
                    # Tenter reconnexion MT5
                    try:
                        self.mt5.connect()
                    except Exception: pass
                    continue

                df_bars = self.mt5.get_latest_bars(n_bars=config.LOOKBACK_BARS + 250)
                if df_bars is None or len(df_bars) < config.LOOKBACK_BARS:
                    for _ in range(int(config.TICK_INTERVAL_SEC * 10)):
                        if not self._run_event.is_set(): break
                        time.sleep(0.1)
                    continue

                current_bar_time = df_bars.index[-1]
                mid_price = (tick["bid"] + tick["ask"]) / 2

                # ── Check arrêt immédiat ───────────────────────
                if not self._run_event.is_set():
                    break

                # ── B. Sentiment News ──────────────────────────
                sentiment_score, recent_news = self.news_mod.get_current_sentiment()

                # ── C. Compte ──────────────────────────────────
                account_stats = self.mt5.get_account_stats()
                positions     = self.mt5.get_open_positions()
                session_stats = self.risk_mgr.get_session_stats()

                # ── D. Protection Compte (hard-coded, pas configurable) ──
                if not self._run_event.is_set(): break

                if account_stats:
                    equity   = account_stats.get("equity", 0)
                    balance  = account_stats.get("balance", 0)
                    # Arrêt si perte > 5% sur la session (protection absolue)
                    if balance > 0 and equity > 0:
                        session_loss = (equity - balance) / balance
                        if session_loss <= -0.05:
                            logger.critical(f"PROTECTION COMPTE : perte session {session_loss*100:.1f}% > 5%")
                            self._run_event.clear()
                            self.running = False
                            self.mt5.close_all_positions()
                            self.web_dash.broadcast_sync({
                                "log_message": f"STOP — perte session {session_loss*100:.1f}%",
                                "status": "STOPPED"
                            })
                            break

                # ── E. Objectif Journalier ─────────────────────
                if self.risk_mgr.check_daily_profit_target():
                    self.dashboard.add_log("Objectif journalier atteint — pause")
                    self._wait_for_new_day()
                    self.risk_mgr._init_day()
                    continue

                # ── F. Construire l'observation pour l'IA ──────
                if not self._run_event.is_set(): break  # re-check avant inférence

                df_feat      = self.fe.compute_features(df_bars)
                obs          = self._build_live_observation(df_feat, sentiment_score, positions)

                # ── G. Inférence IA ────────────────────────────
                action, log_prob, value = self.agent.predict(obs, deterministic=False)
                action_probs = self.agent.get_action_probabilities(obs)
                action_name  = PPOAgent.ACTION_NAMES.get(action, "HOLD")

                # Partager avec le thread dashboard
                self._last_probs  = action_probs.tolist()
                self._last_action = action_name

                # ── H. Exécution ──────────────────────────────
                if not self._run_event.is_set(): break  # NE PAS trader si arrêt demandé

                is_new_bar = (current_bar_time != last_bar_time)
                if is_new_bar:
                    last_bar_time = current_bar_time

                self._execute_action(
                    action_name,
                    mid_price,
                    df_feat,
                    symbol_info,
                    account_stats,
                    positions,
                    action_probs,
                    sentiment_score
                )

                # Log debug pour voir ce que l'IA décide
                logger.debug(
                    f"IA: {action_name} | "
                    f"H={action_probs[0]:.2f} B={action_probs[1]:.2f} "
                    f"S={action_probs[2]:.2f} C={action_probs[3]:.2f} | "
                    f"Prix={mid_price:.2f}"
                )

                # ── I. Mise à jour du Dashboard ────────────────
                self.risk_mgr.update_daily_high()
                sent_label = self.news_mod.get_summary_string()

                news_for_dash = [
                    {"title": n.title, "sentiment_score": n.sentiment_score}
                    for n in recent_news
                ]

                macro_str    = self.macro_mod.get_dashboard_string()
                macro_data   = self.macro_mod.get_features()
                self.dashboard.update(
                    price       = mid_price,
                    bid         = tick["bid"],
                    ask         = tick["ask"],
                    spread      = tick["spread"],
                    sentiment   = sentiment_score,
                    sent_label  = sent_label,
                    macro_info  = macro_str,
                    balance     = account_stats.get("balance", 0),
                    equity      = account_stats.get("equity", 0),
                    daily_pnl   = session_stats["pnl_abs"],
                    daily_pnl_pct = session_stats["pnl_pct"],
                    drawdown    = session_stats["drawdown_pct"],
                    open_trades = len(positions),
                    positions   = positions,
                    ai_action   = action_name,
                    ai_probs    = action_probs.tolist(),
                    last_news   = news_for_dash,
                    status      = "🟢 EN COURS",
                    total_steps = self.agent.total_steps,
                )
                # ── Broadcast vers le Dashboard Web ──────────
                # Dashboard mis à jour par _dashboard_update_loop (thread 500ms)

                # Sleep interruptible : vérifie running toutes les 100ms
                for _ in range(int(config.TICK_INTERVAL_SEC * 10)):
                    if not self._run_event.is_set(): break
                    time.sleep(0.1)

            except Exception as e:
                logger.exception(f"Erreur boucle : {e}")
                self.dashboard.add_log(f"Erreur: {str(e)[:60]}")
                for _ in range(50):   # 5s max
                    if not self._run_event.is_set(): break
                    time.sleep(0.1)

    # ── Exécution des Ordres ───────────────────────────────────

    def _execute_action(
        self,
        action_name:  str,
        price:        float,
        df_feat,
        symbol_info:  dict,
        account_stats: dict,
        positions:    list,
        probs:        np.ndarray,
        sentiment:    float,
    ):
        """Traduit la décision IA en ordre MT5 réel."""
        has_position = len(positions) > 0
        reason       = ""
        lot_size     = 0.0
        sl_price     = 0.0
        tp_price     = 0.0

        logger.info(
            f"EXECUTE: action={action_name} | "
            f"H={probs[0]:.2f} B={probs[1]:.2f} S={probs[2]:.2f} C={probs[3]:.2f} | "
            f"has_pos={has_position} | prix={price:.2f}"
        )

        pos_type = positions[0]["type"] if has_position else None

        # ── Logique de gestion des positions ──────────────────
        # Si position ouverte + signal opposé → fermer (reverse)
        if has_position and action_name == "SELL" and pos_type == "BUY":
            # Blocage des reversals trop rapides (minimum 5 bars M15 = 75 min)
            min_hold_seconds = self._min_hold_bars * 15 * 60
            if self._position_open_time is not None:
                hold_time = time.time() - self._position_open_time
                if hold_time < min_hold_seconds:
                    bars_held = hold_time / (15 * 60)
                    logger.info(f"SELL reverse ignore : position tenue {bars_held:.1f} bars < {self._min_hold_bars} min")
                    return
            
            logger.info("Signal SELL avec BUY ouvert → fermeture forcee")
            for pos in positions:
                self.mt5.close_position(pos["ticket"])
            self._position_open_time = None  # Reset timer
            return

        if has_position and action_name == "BUY" and pos_type == "SELL":
            # Blocage des reversals trop rapides (minimum 5 bars M15 = 75 min)
            min_hold_seconds = self._min_hold_bars * 15 * 60
            if self._position_open_time is not None:
                hold_time = time.time() - self._position_open_time
                if hold_time < min_hold_seconds:
                    bars_held = hold_time / (15 * 60)
                    logger.info(f"BUY reverse ignore : position tenue {bars_held:.1f} bars < {self._min_hold_bars} min")
                    return
            
            logger.info("Signal BUY avec SELL ouvert → fermeture forcee")
            for pos in positions:
                self.mt5.close_position(pos["ticket"])
            self._position_open_time = None  # Reset timer
            return

        # Si position ouverte + même signal → ignorer
        if has_position and action_name == "BUY" and pos_type == "BUY":
            logger.debug("BUY ignore : position BUY deja ouverte")
            return

        if has_position and action_name == "SELL" and pos_type == "SELL":
            logger.debug("SELL ignore : position SELL deja ouverte")
            return

        # HOLD = aucune action
        if action_name == "HOLD":
            return

        # ── Seuils de confiance ────────────────────────────────
        min_prob_trade = 0.26
        min_prob_close = 0.25

        if action_name == "BUY"   and probs[1] < min_prob_trade:
            logger.info(f"BUY ignore : prob={probs[1]:.3f} < {min_prob_trade}")
            return
        if action_name == "SELL"  and probs[2] < min_prob_trade:
            logger.info(f"SELL ignore : prob={probs[2]:.3f} < {min_prob_trade}")
            return
        if action_name == "CLOSE" and probs[3] < min_prob_close:
            logger.info(f"CLOSE ignore : prob={probs[3]:.3f} < {min_prob_close}")
            return

        # ── Cooldown anti-surtrading ───────────────────────────
        now = time.time()
        if action_name in ("BUY", "SELL"):
            since_last = now - self._last_trade_time
            if since_last < self._trade_cooldown:
                logger.info(f"{action_name} ignore : cooldown {since_last:.0f}s < {self._trade_cooldown:.0f}s")
                return
            # Re-fetch positions directement depuis MT5 (pas le cache)
            live_positions = self.mt5.get_open_positions()
            if len(live_positions) > 0:
                logger.info(f"{action_name} ignore : {len(live_positions)} position(s) ouverte(s)")
                return

            # ── Filtre de tendance EMA ─────────────────────────
            # DÉSACTIVÉ TEMPORAIREMENT : trop restrictif, bloquait tous les trades
            # À réactiver après avec un filtre moins strict
            # try:
            #     close = df_feat["Close"].values
            #     # Filtre tendance multi-timeframe
            #     # H1 : EMA20 sur les 20 dernières heures (80 bougies M15)
            #     # M15 : EMA50 sur les 50 dernières bougies M15
            #     close_vals = df_feat["Close"].values
            #     
            #     # Tendance H1 (court terme robuste)
            #     h1_period = 80  # 80 bougies M15 = 20h
            #     if len(close_vals) >= h1_period:
            #         ema_h1_fast = float(df_feat["Close"].ewm(span=20).mean().iloc[-1])
            #         ema_h1_slow = float(df_feat["Close"].ewm(span=80).mean().iloc[-1])
            #         trend_up   = ema_h1_fast > ema_h1_slow and price > ema_h1_fast
            #         trend_down = ema_h1_fast < ema_h1_slow and price < ema_h1_fast
            #     else:
            #         trend_up = trend_down = True  # Pas assez de données → pas de filtre
            #
            #     if action_name == "BUY" and not trend_up:
            #         logger.info(f"BUY BLOQUE | tendance H1 baissiere EMA20={ema_h1_fast:.1f} < EMA80={ema_h1_slow:.1f}")
            #         return
            #     if action_name == "SELL" and not trend_down:
            #         logger.info(f"SELL BLOQUE | tendance H1 haussiere EMA20={ema_h1_fast:.1f} > EMA80={ema_h1_slow:.1f}")
            #         return
            #
            #     logger.info(f"Filtre H1 OK : {'HAUSSIER' if trend_up else 'BAISSIER'} | EMA20={ema_h1_fast:.1f} EMA80={ema_h1_slow:.1f}")
            # except Exception as e:
            #     logger.warning(f"Filtre tendance erreur : {e} — trade autorise quand meme")

        # ── BUY ───────────────────────────────────────────────
        if action_name == "BUY" and not has_position:
            atr      = self.fe.get_atr(df_feat)
            sl, tp   = self.risk_mgr.calculate_sl_tp("BUY", price, atr, symbol_info["point"])
            sl_pips  = self.risk_mgr.sl_to_pips(price, sl, symbol_info["point"])
            # Lot sizing : 1% du compte réel MT5, pas du capital config
            equity   = account_stats.get("equity", 10000)
            lot_size = self.risk_mgr.calculate_lot_size(sl_pips, symbol_info, equity)

            reason = (
                f"IA BUY | Prob={probs[1]:.3f} | "
                f"Equity={equity:.0f}$ | ATR={atr:.2f}"
            )

            result = self.mt5.place_order("BUY", lot_size, sl, tp, comment="AI_PPO_BUY")
            if result:
                self._last_trade_time = time.time()  # Cooldown démarre ici
                self._position_open_time = time.time()  # Track position open time
                sl_price = result["sl"]
                tp_price = result["tp"]
                self.dashboard.add_log(
                    f"BUY {lot_size:.2f} lots @ {price:.2f} | SL={sl:.2f} TP={tp:.2f}"
                )

        # ── SELL ──────────────────────────────────────────────
        elif action_name == "SELL" and not has_position:

            atr      = self.fe.get_atr(df_feat)
            sl, tp   = self.risk_mgr.calculate_sl_tp("SELL", price, atr, symbol_info["point"])
            sl_pips  = self.risk_mgr.sl_to_pips(price, sl, symbol_info["point"])
            lot_size = self.risk_mgr.calculate_lot_size(
                sl_pips, symbol_info, account_stats.get("equity")
            )

            reason = (
                f"IA décide SELL | Prob={probs[2]:.3f} | "
                f"Sentiment={sentiment:+.3f} | ATR={atr:.2f}"
            )

            result = self.mt5.place_order("SELL", lot_size, sl, tp, comment="AI_PPO_SELL")
            if result:
                self._last_trade_time = time.time()  # Cooldown démarre ici
                self._position_open_time = time.time()  # Track position open time
                sl_price = result["sl"]
                tp_price = result["tp"]
                self.dashboard.add_log(
                    f"SELL {lot_size:.2f} lots @ {price:.2f} | SL={sl:.2f} TP={tp:.2f}"
                )

        # ── CLOSE ─────────────────────────────────────────────
        elif action_name == "CLOSE" and has_position:
            # Blocage des fermetures trop rapides (minimum 5 bars M15 = 75 min)
            min_hold_seconds = self._min_hold_bars * 15 * 60  # 5 bars * 15 min * 60 sec = 4500 sec
            if self._position_open_time is not None:
                hold_time = time.time() - self._position_open_time
                if hold_time < min_hold_seconds:
                    bars_held = hold_time / (15 * 60)
                    logger.info(f"CLOSE ignore : position tenue {bars_held:.1f} bars < {self._min_hold_bars} min")
                    return
            
            for pos in positions:
                self.mt5.close_position(pos["ticket"])
            self._position_open_time = None  # Reset timer
            reason = f"IA décide CLOSE | Prob={probs[3]:.3f} | PnL={sum(p['profit'] for p in positions):.2f}"
            self.dashboard.add_log(
                f"🔵 CLOSE {len(positions)} position(s) @ {price:.2f}"
            )

        else:
            # HOLD ou action non applicable
            return

        # Logger la décision
        self.dec_log.log_decision(
            action_name    = action_name,
            price          = price,
            probs          = probs.tolist(),
            sentiment      = sentiment,
            account_stats  = account_stats,
            reason         = reason,
            lot_size       = lot_size,
            sl             = sl_price,
            tp             = tp_price,
        )

    # ── Observation Live ───────────────────────────────────────

    def _build_live_observation(
        self,
        df_feat,
        sentiment: float,
        positions: list
    ) -> np.ndarray:
        """Construit le vecteur d'observation pour l'inférence live."""
        feat_cols = [c for c in self.fe.get_feature_columns() if c in df_feat.columns]
        window    = df_feat.iloc[-config.LOOKBACK_BARS:][feat_cols].values.astype(np.float32)

        if len(window) < config.LOOKBACK_BARS:
            pad    = np.zeros((config.LOOKBACK_BARS - len(window), len(feat_cols)), dtype=np.float32)
            window = np.vstack([pad, window])

        mean   = window.mean(axis=0)
        std    = window.std(axis=0) + 1e-8
        window = (window - mean) / std

        # Info position
        position_code = 0
        unrealized_pnl = 0.0
        if positions:
            pos = positions[0]
            position_code = 1 if pos["type"] == "BUY" else -1
            unrealized_pnl = pos["profit"] / (self.risk_mgr.start_balance + 1e-9)

        drawdown = self.risk_mgr.get_current_drawdown()

        extra = np.array([[
            float(position_code),
            float(sentiment),
            float(np.clip(unrealized_pnl, -1, 1)),
            float(np.clip(-drawdown, -1, 0)),
        ]] * config.LOOKBACK_BARS, dtype=np.float32)

        obs = np.hstack([window, extra]).flatten()

        # Ajouter les features macro (DXY, taux, sessions)
        macro_vector = self.macro_mod.get_feature_vector()
        obs = np.concatenate([obs, macro_vector])
        return obs

    # ── Utilitaires ────────────────────────────────────────────

    def _wait_for_new_day(self):
        """Attend la prochaine journée de trading."""
        today = date.today()
        logger.info("En attente de la prochaine journée de trading...")
        while date.today() == today and self._run_event.is_set():
            time.sleep(60)

    def stop(self):
        """Arrêt propre du bot depuis le dashboard web."""
        if not self._run_event.is_set():
            logger.warning("Bot deja arrete")
            return
        logger.info("Arret du bot demande...")
        self._run_event.clear()
        self.running = False
        # Attendre max 5s que la boucle se termine
        for _ in range(50):
            if not self.running:
                break
            time.sleep(0.1)
        logger.info("Bot arrete proprement.")

    def _handle_shutdown(self, signum, frame):
        """Arrêt propre sur signal (Ctrl+C)."""
        logger.warning(f"Signal {signum} recu. Arret en cours...")
        self.running = False
        self._run_event.clear()
        print("\n\n⚠️  Arrêt demandé. Fermeture propre en cours...")

        # Fermer les positions ouvertes ?
        positions = self.mt5.get_open_positions()
        if positions:
            user_input = input(
                f"\n{len(positions)} position(s) ouverte(s). Fermer tout ? [o/N] : "
            ).strip().lower()
            if user_input == "o":
                n_closed = self.mt5.close_all_positions()
                print(f"✅ {n_closed} position(s) fermée(s).")

        self.news_mod.stop()
        self.mt5.disconnect()
        print("✅ Arrêt propre terminé.")
        sys.exit(0)

    def shutdown(self):
        """Arrêt programmatique."""
        self.running = False
        self.news_mod.stop()
        self.mt5.disconnect()


# ── Point d'entrée ─────────────────────────────────────────────
def _idle_dashboard_loop(bot):
    """
    Thread léger qui tourne en permanence — même avant de cliquer Démarrer.
    Envoie prix + balance MT5 au dashboard toutes les 2 secondes.
    """
    logger.info("Thread idle dashboard démarre")
    while True:
        try:
            # Seulement si le bot n'est PAS en cours (sinon _dashboard_update_loop gère)
            if not bot._run_event.is_set():
                # Connecter MT5 si besoin
                if not bot.mt5.connected:
                    try:
                        bot.mt5.connect()
                    except Exception:
                        time.sleep(5)
                        continue

                tick          = bot.mt5.get_tick()
                account_stats = bot.mt5.get_account_stats()

                if tick and account_stats:
                    mid_price = (tick["bid"] + tick["ask"]) / 2
                    bot.web_dash.broadcast_sync({
                        "price":       round(float(mid_price), 2),
                        "bid":         round(float(tick["bid"]), 2),
                        "ask":         round(float(tick["ask"]), 2),
                        "spread":      round(float(tick.get("spread", 0)), 1),
                        "balance":     round(float(account_stats.get("balance", 0)), 2),
                        "equity":      round(float(account_stats.get("equity", 0)), 2),
                        "daily_pnl":   0.0,
                        "open_trades": 0,
                        "positions":   [],
                        "status":      "STOPPED",
                        "ai_action":   "—",
                        "ai_probs":    [0.25, 0.25, 0.25, 0.25],
                    })
        except Exception as e:
            logger.debug(f"idle_dashboard_loop: {e}")

        time.sleep(2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Démarrer le bot automatiquement")
    args = parser.parse_args()

    bot = XAUUSDBot()

    # ── Démarrer le dashboard web UNE SEULE FOIS ──────────────
    bot.web_dash.set_bot(bot)
    bot.web_dash.start()

    # ── Démarrer le dashboard terminal UNE SEULE FOIS ─────────
    bot.dashboard.start_background()

    # ── Thread idle : affiche prix + balance avant démarrage ──
    idle_thread = threading.Thread(
        target=_idle_dashboard_loop,
        args=(bot,),
        daemon=True,
        name="IdleDashboardThread"
    )
    idle_thread.start()

    print("=" * 60)
    print("   XAUUSD AI BOT — Dashboard Web")
    print("=" * 60)
    print("Dashboard : http://localhost:8765")
    print("Clique sur [DEMARRER] dans le navigateur pour lancer le bot.")
    print("Ctrl+C pour quitter.")
    print("=" * 60)

    if args.auto:
        bot.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nArret demande...")
        try:
            if bot.running:
                bot.stop()
        except Exception: pass
        print("Bye.")