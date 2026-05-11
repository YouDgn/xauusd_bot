#!/usr/bin/env python
# ============================================================
# diagnose_model.py — Diagnostic du modèle PPO
# ============================================================

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from ppo_agent import PPOAgent, get_device

def diagnose_model():
    """Teste le modèle et affiche les probabilités d'action."""
    
    print("\n" + "="*60)
    print("DIAGNOSTIC MODÈLE PPO XAUUSD")
    print("="*60 + "\n")
    
    # Charger le modèle
    print("[1] Chargement du modèle...")
    try:
        agent = PPOAgent(obs_size=80, n_actions=4, training_mode=False)
        agent.load(config.MODEL_PATH)
        print(f"✅ Modèle chargé : {config.MODEL_PATH}\n")
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        return
    
    # Test 1: Observations aléatoires
    print("[2] Test sur 100 observations aléatoires...")
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
    prob_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    
    for i in range(100):
        obs = np.random.randn(80).astype(np.float32)
        
        with torch.no_grad():
            action, _, _ = agent.predict(obs, deterministic=False)
            probs = agent.get_action_probabilities(obs)
        
        action_counts[action] += 1
        for j in range(4):
            prob_sums[j] += probs[j]
    
    print("\nDistribution des actions (100 samples):")
    for action_id in range(4):
        count = action_counts[action_id]
        avg_prob = prob_sums[action_id] / 100
        print(f"  {action_names[action_id]:<6} : {count:3d}x ({count:3.0f}%) | Prob moyenne: {avg_prob:.3f}")
    
    # Test 2: Observation déterministe
    print("\n[3] Test déterministe (même observation, 10x)...")
    obs = np.zeros(80, dtype=np.float32)
    obs[0] = 1.0  # Feature particulière
    
    print(f"Observation : {obs[:5]}... (80 dims)")
    probs_list = []
    
    for i in range(10):
        with torch.no_grad():
            action, _, _ = agent.predict(obs, deterministic=False)
            probs = agent.get_action_probabilities(obs)
        probs_list.append(probs)
        print(f"  Iter {i+1}: Action={action_names[action]:<6} | Probs: H={probs[0]:.3f} B={probs[1]:.3f} S={probs[2]:.3f} C={probs[3]:.3f}")
    
    # Verdict
    print("\n" + "="*60)
    print("VERDICT :")
    print("="*60)
    
    avg_close_prob = prob_sums[3] / 100
    if avg_close_prob > 0.35:
        print(f"⚠️  PROBLÈME : Modèle output trop de CLOSE ({avg_close_prob:.1%})")
        print("   → Le modèle a probablement convergé vers une stratégie de fermeture rapide")
        print("   → Réentraînement recommandé avec récompense modifiée")
    else:
        print(f"✅ Modèle OK : Distribution équilibrée (CLOSE={avg_close_prob:.1%})")
    
    print()

if __name__ == "__main__":
    diagnose_model()
