# ============================================================
# train.py — Script d'Entraînement du Bot XAUUSD
# ============================================================
# Utilisation :
#   python train.py
#   python train.py --resume          (reprend depuis le dernier checkpoint)
#   python train.py --steps 500000    (nombre de steps custom)
# ============================================================

import sys
import os
import argparse
import logging
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import multiprocessing

# ── Optimisation CPU ───────────────────────────────────────
n_cores = multiprocessing.cpu_count()
torch.set_num_threads(max(1, n_cores // 2))
torch.set_num_interop_threads(max(1, n_cores // 4))

import config
from mt5_connector import MT5Connector
from trading_env import XAUUSDTradingEnv
from ppo_agent import PPOAgent
from macro_features import MacroFeaturesModule

# ── Logging ────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

import io, sys as _sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(stream=io.TextIOWrapper(
            _sys.stdout.buffer, encoding="utf-8", errors="replace"
        )),
        logging.FileHandler("logs/training.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("TRAIN")


def evaluate_agent(agent: PPOAgent, env: XAUUSDTradingEnv, n_episodes: int = 5) -> dict:
    """Évalue l'agent sur N épisodes sans exploration."""
    total_rewards = []
    total_pnls    = []
    win_rates     = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_reward = 0.0

        while not done:
            action, _, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)
        total_pnls.append(info.get("episode_pnl", 0))
        win_rates.append(info.get("win_rate", 0))

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward":  np.std(total_rewards),
        "mean_pnl":    np.mean(total_pnls),
        "mean_winrate": np.mean(win_rates),
    }


def train(total_steps: int = config.TOTAL_TRAIN_STEPS, resume: bool = False):
    """Boucle d'entraînement principale."""

    print("=" * 70)
    print("   🤖 XAUUSD AI BOT — ENTRAÎNEMENT PPO")
    print("=" * 70)

    # ── 1. Chargement des données MT5 RÉELLES (priorité absolue) ──
    # Le CSV synthétique est désactivé — trop différent du vrai marché
    df = None
    import pandas as pd

    logger.info("Connexion a MetaTrader5 pour donnees historiques reelles...")
    mt5 = MT5Connector()
    if not mt5.connect():
        logger.error("Impossible de se connecter a MT5. Lance MT5 + Algo Trading vert.")
        sys.exit(1)

    # Télécharger max de barres M15 disponibles (~2 ans = 70 000 barres)
    df = mt5.get_historical_data(
        symbol   = config.SYMBOL,
        timeframe= config.TIMEFRAME,
        years    = config.TRAINING_YEARS
    )
    mt5.disconnect()

    # Fallback : essayer avec moins d'années si le broker limite
    if df is None or len(df) < 5000:
        logger.warning("Peu de donnees — tentative sur 1 an...")
        mt5_2 = MT5Connector()
        mt5_2.connect()
        df = mt5_2.get_historical_data(
            symbol   = config.SYMBOL,
            timeframe= config.TIMEFRAME,
            years    = 1
        )
        mt5_2.disconnect()

    if df is not None:
        logger.info(f"[OK] {len(df)} barres M15 reelles chargees ({df.index[0]} -> {df.index[-1]})")
    else:
        logger.error("Aucune donnee MT5 disponible.")
        sys.exit(1)

    if df is None or len(df) < 1000:
        logger.error("Donnees insuffisantes pour l entrainement.")
        sys.exit(1)

    logger.info(f"[OK] {len(df)} barres de donnees XAUUSD chargees pour entrainement.")

    # ── 2. Split Train / Validation ────────────────────────────
    split_idx = int(len(df) * 0.85)
    df_train  = df.iloc[:split_idx].copy()
    df_val    = df.iloc[split_idx:].copy()
    logger.info(
        f"📊 Train: {len(df_train)} barres | "
        f"Val: {len(df_val)} barres"
    )

    # ── 3. Environnements ──────────────────────────────────────
    macro_mod = MacroFeaturesModule()
    macro_mod.start()
    macro_vector = macro_mod.get_feature_vector()
    logger.info(f"MacroFeatures actives : {len(macro_vector)} features")

    env_train = XAUUSDTradingEnv(df_train, lookback=config.LOOKBACK_BARS)
    env_train.update_macro(macro_vector)
    env_val   = XAUUSDTradingEnv(df_val,   lookback=config.LOOKBACK_BARS)
    env_val.update_macro(macro_vector)

    obs_size = env_train.observation_space.shape[0]
    logger.info(f"Taille observation : {obs_size}")

    # ── 4. Agent ───────────────────────────────────────────────
    agent = PPOAgent(obs_size=obs_size, n_actions=4, training_mode=True)

    if resume:
        if agent.load(config.MODEL_PATH):
            logger.info(f"▶️  Reprise depuis {agent.total_steps:,} steps.")
        else:
            logger.info("Démarrage d'un nouvel entraînement.")

    # ── 5. Boucle d'entraînement ───────────────────────────────
    best_eval_reward  = -float("inf")
    steps_done        = agent.total_steps
    update_count      = 0
    start_time        = time.time()

    # Early Stopping — basé sur la variance de la reward
    eval_rewards_history = []
    early_stop_patience  = 20   # Nombre d'évaluations sans amélioration
    early_stop_counter   = 0
    early_stop_min_delta = 0.5  # Amélioration minimale requise

    pbar = tqdm(
        total=total_steps,
        initial=steps_done,
        desc="Entrainement PPO",
        unit="step",
        ncols=100
    )

    logger.info(f"Demarrage entrainement — Objectif: {total_steps:,} steps | Early stop patience={early_stop_patience}")

    while steps_done < total_steps:
        rollout_info = agent.collect_rollout(env_train)
        metrics      = agent.update(rollout_info["returns"], rollout_info["advantages"])

        steps_done   = agent.total_steps
        update_count += 1

        pbar.update(config.PPO_ROLLOUT_STEPS)
        pbar.set_postfix({
            "rew":     f"{rollout_info['mean_ep_reward']:.2f}",
            "p_loss":  f"{metrics['policy_loss']:.4f}",
            "entropy": f"{metrics['entropy']:.3f}",
        })

        # Logging périodique
        if update_count % 10 == 0:
            elapsed = time.time() - start_time
            fps     = steps_done / elapsed
            logger.info(
                f"Update #{update_count:4d} | Steps: {steps_done:7,} | "
                f"FPS: {fps:.0f} | "
                f"Loss: {metrics['policy_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.3f} | "
                f"ClipFrac: {metrics['clip_frac']:.3f}"
            )

        # Évaluation + Early Stopping
        if update_count % 50 == 0:
            eval_metrics = evaluate_agent(agent, env_val)
            eval_rewards_history.append(eval_metrics["mean_reward"])

            # Variance de la reward (détecte l'overfitting = variance qui monte)
            reward_variance = float(np.std(eval_rewards_history[-10:])) if len(eval_rewards_history) >= 10 else 0.0

            logger.info(
                f"EVAL | Steps={steps_done:,} | "
                f"Reward={eval_metrics['mean_reward']:.2f}±{eval_metrics['std_reward']:.2f} | "
                f"PnL={eval_metrics['mean_pnl']:.2f} | "
                f"WR={eval_metrics['mean_winrate']*100:.1f}% | "
                f"Variance={reward_variance:.3f}"
            )

            # Meilleur modèle
            if eval_metrics["mean_reward"] > best_eval_reward + early_stop_min_delta:
                best_eval_reward = eval_metrics["mean_reward"]
                early_stop_counter = 0
                agent.save(config.MODEL_PATH)
                logger.info(f"Nouveau meilleur modele ! Reward={best_eval_reward:.2f}")
            else:
                early_stop_counter += 1
                logger.info(f"Pas d'amelioration ({early_stop_counter}/{early_stop_patience})")

            # Early stopping si variance trop haute (overfitting) ou stagnation
            if early_stop_counter >= early_stop_patience:
                logger.info(f"EARLY STOPPING a {steps_done:,} steps — stagnation detectee")
                break

            if len(eval_rewards_history) >= 10 and reward_variance > 500:
                logger.info(f"EARLY STOPPING — variance trop haute ({reward_variance:.1f}) = overfitting")
                break

        # Checkpoint régulier
        if steps_done % config.SAVE_EVERY_STEPS < config.PPO_ROLLOUT_STEPS:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"ppo_xauusd_{steps_done:08d}.pt"
            )
            agent.save(ckpt_path)

    pbar.close()

    # ── 6. Sauvegarde Finale ───────────────────────────────────
    agent.save(config.MODEL_PATH)

    # ── 7. Rapport Final + Out-of-Sample ─────────────────────
    elapsed    = time.time() - start_time
    eval_train = evaluate_agent(agent, env_train, n_episodes=5)
    eval_val   = evaluate_agent(agent, env_val,   n_episodes=10)

    # Profit Factor out-of-sample
    pf_ratio = "N/A"
    try:
        wins  = eval_val["mean_pnl"] * eval_val["mean_winrate"]
        loss  = abs(eval_val["mean_pnl"]) * (1 - eval_val["mean_winrate"])
        pf    = wins / (loss + 1e-8)
        pf_ratio = f"{pf:.2f}"
    except Exception:
        pass

    overfitting_gap = eval_train["mean_reward"] - eval_val["mean_reward"]

    print("\n" + "=" * 70)
    print("   RAPPORT D ENTRAINEMENT FINAL")
    print("=" * 70)
    print(f"   Steps total      : {steps_done:,}")
    print(f"   Duree            : {elapsed/3600:.1f}h")
    print(f"   FPS moyen        : {steps_done/elapsed:.0f}")
    print(f"   Early stop count : {early_stop_counter}/{early_stop_patience}")
    print(f"")
    print(f"   -- IN-SAMPLE (train) --")
    print(f"   Reward moyen     : {eval_train['mean_reward']:.3f}")
    print(f"   Win Rate         : {eval_train['mean_winrate']*100:.1f}%")
    print(f"")
    print(f"   -- OUT-OF-SAMPLE (val, donnees non vues) --")
    print(f"   Reward moyen     : {eval_val['mean_reward']:.3f}")
    print(f"   PnL moyen        : {eval_val['mean_pnl']:.2f}")
    print(f"   Win Rate         : {eval_val['mean_winrate']*100:.1f}%")
    print(f"   Profit Factor    : {pf_ratio}")
    print(f"")
    print(f"   Overfitting gap  : {overfitting_gap:.2f} (< 5 = bon)")
    print(f"   Modele final     : {config.MODEL_PATH}")
    print("=" * 70)

    if overfitting_gap > 10:
        print("   ATTENTION : Overfitting detecte (gap train/val > 10)")
    elif eval_val["mean_winrate"] > 0.45:
        print("   Modele pret pour le live trading !")
    else:
        print("   Continuer l entrainement ou ajuster la reward")

    print("\nEntrainement termine ! Lance : python live_bot.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement PPO XAUUSD")
    parser.add_argument("--steps",  type=int,  default=config.TOTAL_TRAIN_STEPS)
    parser.add_argument("--resume", action="store_true", help="Reprendre depuis le dernier checkpoint")
    args = parser.parse_args()
    train(total_steps=args.steps, resume=args.resume)