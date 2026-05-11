#!/usr/bin/env python3
# ============================================================
# evaluate_model.py — Génère un rapport sur le modèle entraîné
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import config
from mt5_connector import MT5Connector
from trading_env import XAUUSDTradingEnv
from ppo_agent import PPOAgent
from macro_features import MacroFeaturesModule
import numpy as np
import torch
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

def evaluate_agent(agent, env, n_episodes=5):
    """Évalue l'agent sur n épisodes."""
    rewards = []
    win_rates = []
    pnls = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_pnl = 0
        ep_trades = 0
        ep_wins = 0
        
        done = False
        step = 0
        while not done and step < 1000:
            with torch.no_grad():
                action, _, _ = agent.predict(obs, deterministic=True)
            result = env.step(action)
            # Gymnasium retourne 5 valeurs, ancienne Gym en retourne 4
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            ep_reward += reward
            ep_pnl += info.get("pnl", 0)
            if info.get("trade_executed"):
                ep_trades += 1
                if info.get("pnl", 0) > 0:
                    ep_wins += 1
            step += 1
        
        wr = (ep_wins / max(ep_trades, 1)) * 100
        rewards.append(ep_reward)
        pnls.append(ep_pnl)
        win_rates.append(wr)
        logger.info(f"Episode {ep+1}/{n_episodes}: Reward={ep_reward:.2f}, PnL={ep_pnl:.2f}$, WR={wr:.1f}%")
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_pnl": np.mean(pnls),
        "mean_winrate": np.mean(win_rates) / 100,
    }

def main():
    print("\n" + "=" * 70)
    print("RAPPORT D'ÉVALUATION DU MODÈLE PPO XAUUSD")
    print("=" * 70 + "\n")
    
    # 1. Connexion MT5 et données
    logger.info("Connexion à MetaTrader5...")
    mt5 = MT5Connector()
    if not mt5.connect():
        logger.error("Impossible de se connecter à MT5")
        return
    
    logger.info("Téléchargement des données historiques...")
    df_bars = mt5.get_historical_data(years=config.TRAINING_YEARS)
    if df_bars is None or len(df_bars) < config.LOOKBACK_BARS:
        logger.error("Données insuffisantes")
        return
    
    # 2. Split train/val
    n_train = int(len(df_bars) * 0.85)
    df_train = df_bars.iloc[:n_train]
    df_val = df_bars.iloc[n_train:]
    
    logger.info(f"📊 Train: {len(df_train)} barres | Val: {len(df_val)} barres")
    
    # 3. Environnements
    logger.info("Création des environnements...")
    macro_mod = MacroFeaturesModule()
    macro_mod.start()
    
    env_train = XAUUSDTradingEnv(df_train, lookback=config.LOOKBACK_BARS, sentiment_score=0.0)
    env_val = XAUUSDTradingEnv(df_val, lookback=config.LOOKBACK_BARS, sentiment_score=0.0)
    
    # 4. Agent
    logger.info("Chargement du modèle...")
    obs_size = env_train.observation_space.shape[0]
    agent = PPOAgent(obs_size=obs_size, n_actions=4)
    
    try:
        agent.load(config.MODEL_PATH)
        logger.info(f"✅ Modèle chargé : {config.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur chargement : {e}")
        return
    
    # 5. Évaluation
    print("\n--- ÉVALUATION IN-SAMPLE (Train) ---")
    eval_train = evaluate_agent(agent, env_train, n_episodes=3)
    
    print("\n--- ÉVALUATION OUT-OF-SAMPLE (Val, données non vues) ---")
    eval_val = evaluate_agent(agent, env_val, n_episodes=5)
    
    # 6. Profit Factor
    pf_ratio = "N/A"
    try:
        wins = eval_val["mean_pnl"] * eval_val["mean_winrate"]
        loss = abs(eval_val["mean_pnl"]) * (1 - eval_val["mean_winrate"])
        pf = wins / (loss + 1e-8)
        pf_ratio = f"{pf:.2f}"
    except:
        pass
    
    overfitting_gap = eval_train["mean_reward"] - eval_val["mean_reward"]
    
    # 7. Rapport
    print("\n" + "=" * 70)
    print("   RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"   Modèle                 : {config.MODEL_PATH}")
    print(f"   Taille observation    : {obs_size:,}")
    print(f"")
    print(f"   -- IN-SAMPLE (train) --")
    print(f"   Reward moyen           : {eval_train['mean_reward']:+.3f} (±{eval_train['std_reward']:.3f})")
    print(f"   Win Rate               : {eval_train['mean_winrate']*100:.1f}%")
    print(f"")
    print(f"   -- OUT-OF-SAMPLE (val, données non vues) --")
    print(f"   Reward moyen           : {eval_val['mean_reward']:+.3f}")
    print(f"   PnL moyen / épisode    : {eval_val['mean_pnl']:+.2f}$")
    print(f"   Win Rate               : {eval_val['mean_winrate']*100:.1f}%")
    print(f"   Profit Factor          : {pf_ratio}")
    print(f"")
    print(f"   Overfitting gap        : {overfitting_gap:.2f} (< 5 = bon)")
    print("=" * 70)
    
    if overfitting_gap > 10:
        print("   ⚠️  ATTENTION : Overfitting détecté (gap train/val > 10)")
    elif eval_val["mean_winrate"] > 0.45:
        print("   ✅ Modèle PRÊT pour le live trading !")
    else:
        print("   ⚠️  Continuer l'entraînement ou ajuster la reward")
    
    print("\n   Prochaine étape: python live_bot.py\n")
    print("=" * 70 + "\n")
    
    macro_mod.stop()
    mt5.disconnect()

if __name__ == "__main__":
    main()
