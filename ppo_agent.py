# ============================================================
# ppo_agent.py — Agent PPO avec support AMD DirectML
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict
import config

logger = logging.getLogger(__name__)


# ── Device Setup (AMD RX 6700 XT via DirectML) ────────────────
def get_device(force_cpu: bool = False) -> torch.device:
    """
    Retourne le device de calcul optimal.
    force_cpu=True : utilisé pendant l'entraînement (DirectML ne supporte
    pas toutes les opérations backward de PPO).
    force_cpu=False : utilisé en inférence live (DirectML OK).
    """
    if force_cpu:
        logger.info("Device : CPU (mode entrainement - DirectML incompatible avec PPO backward)")
        return torch.device("cpu")

    if config.USE_DIRECTML:
        try:
            import torch_directml
            device = torch_directml.device()
            logger.info(f"AMD DirectML active : {device}")
            return device
        except ImportError:
            logger.warning("torch-directml non trouve. Fallback CPU.")
        except Exception as e:
            logger.warning(f"DirectML indisponible ({e}). Fallback CPU.")

    if torch.cuda.is_available():
        logger.info("CUDA disponible - GPU NVIDIA.")
        return torch.device("cuda")

    logger.info("Device : CPU")
    return torch.device("cpu")


# ── Réseau Actor-Critic ────────────────────────────────────────
class ActorCriticNetwork(nn.Module):
    """
    Réseau de neurones partagé Actor-Critic pour PPO.
    
    Architecture :
        Input → LSTM → Couches Fully-Connected → [Actor Head | Critic Head]
        
    - Actor Head : distribution de probabilité sur les 4 actions
    - Critic Head : estimation de la valeur (V(s))
    """

    def __init__(self, obs_size: int, n_actions: int = 4):
        super().__init__()
        hidden = config.HIDDEN_SIZE

        # ── Couche d'embedding ─────────────────────────────────
        self.embedding = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(config.DROPOUT),
        )

        # ── Couches communes (trunk) ───────────────────────────
        self.trunk = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(config.DROPOUT),

            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.LeakyReLU(0.01),
        )

        # ── Actor Head ─────────────────────────────────────────
        self.actor_head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden // 4, n_actions),
        )

        # ── Critic Head ────────────────────────────────────────
        self.critic_head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden // 4, 1),
        )

        # Initialisation orthogonale (recommandée pour PPO)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne (logits_actions, valeur_état)."""
        emb    = self.embedding(x)
        trunk  = self.trunk(emb)
        logits = self.actor_head(trunk)
        value  = self.critic_head(trunk).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne (action, log_prob, entropy, value).
        Si action=None, échantillonne une nouvelle action.
        """
        logits, value = self.forward(x)
        dist  = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


# ── Rollout Buffer ─────────────────────────────────────────────
class RolloutBuffer:
    """Stocke les expériences collectées avant la mise à jour PPO."""

    def __init__(self, rollout_steps: int, obs_size: int, device):
        self.rollout_steps = rollout_steps
        self.obs_size      = obs_size
        self.device        = device
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.rollout_steps, self.obs_size), dtype=np.float32)
        self.actions      = np.zeros(self.rollout_steps, dtype=np.int64)
        self.log_probs    = np.zeros(self.rollout_steps, dtype=np.float32)
        self.rewards      = np.zeros(self.rollout_steps, dtype=np.float32)
        self.values       = np.zeros(self.rollout_steps, dtype=np.float32)
        self.dones        = np.zeros(self.rollout_steps, dtype=np.float32)
        self.ptr          = 0

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool
    ):
        self.observations[self.ptr] = obs
        self.actions[self.ptr]      = action
        self.log_probs[self.ptr]    = log_prob
        self.rewards[self.ptr]      = reward
        self.values[self.ptr]       = value
        self.dones[self.ptr]        = float(done)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.rollout_steps

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = config.PPO_GAMMA,
        gae_lambda: float = config.PPO_GAE_LAMBDA
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule les returns et advantages via GAE (Generalized Advantage Estimation)."""
        advantages = np.zeros_like(self.rewards)
        last_gae   = 0.0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_val  = last_value
                next_done = 0.0
            else:
                next_val  = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta     = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            last_gae  = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return returns, advantages

    def get_batches(
        self,
        returns:    np.ndarray,
        advantages: np.ndarray,
        batch_size: int = config.PPO_BATCH_SIZE
    ):
        """Génère des mini-batches aléatoires."""
        indices = np.random.permutation(self.rollout_steps)
        for start in range(0, self.rollout_steps, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield (
                torch.FloatTensor(self.observations[batch_idx]).to(self.device),
                torch.LongTensor(self.actions[batch_idx]).to(self.device),
                torch.FloatTensor(self.log_probs[batch_idx]).to(self.device),
                torch.FloatTensor(returns[batch_idx]).to(self.device),
                torch.FloatTensor(advantages[batch_idx]).to(self.device),
            )


# ── Agent PPO ─────────────────────────────────────────────────
class PPOAgent:
    """
    Agent Proximal Policy Optimization (PPO) complet.
    - Entrainement : CPU (DirectML ne supporte pas tous les ops backward)
    - Inference live : AMD DirectML (RX 6700 XT)
    """

    def __init__(self, obs_size: int, n_actions: int = 4, training_mode: bool = False):
        self.device   = get_device(force_cpu=training_mode)
        self.network  = ActorCriticNetwork(obs_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.PPO_LR,
            eps=1e-5
        )

        self.buffer   = RolloutBuffer(config.PPO_ROLLOUT_STEPS, obs_size, self.device)
        self.n_actions = n_actions

        # Métriques d'entraînement
        self.total_steps      = 0
        self.training_history = {
            "policy_loss": [], "value_loss": [],
            "entropy": [],    "kl_div": [],
            "clip_frac": [],
        }

        logger.info(
            f"🧠 Agent PPO initialisé | Device: {self.device} | "
            f"Paramètres: {sum(p.numel() for p in self.network.parameters()):,}"
        )

    # ── Inférence ──────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Prédit une action à partir d'une observation.
        Retourne (action, log_prob, value).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, _, value = self.network.get_action_and_value(obs_t)

        if deterministic:
            logits, value = self.network(obs_t)
            action = logits.argmax(dim=-1)
            dist   = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        _, value = self.network(obs_t)
        return float(value.item())

    # ── Entraînement ───────────────────────────────────────────

    def collect_rollout(self, env) -> Dict:
        """Collecte PPO_ROLLOUT_STEPS steps d'expérience."""
        self.buffer.reset()
        obs, _  = env.reset()
        done     = False
        ep_rewards = []
        ep_reward  = 0.0

        while not self.buffer.is_full():
            action, log_prob, value = self.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            self.buffer.store(obs, action, log_prob, reward, value, terminated or truncated)
            obs = next_obs
            ep_reward += reward
            self.total_steps += 1

            if terminated or truncated:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        # Valeur bootstrap du dernier état
        last_value = self.get_value(obs)
        returns, advantages = self.buffer.compute_returns_and_advantages(last_value)

        return {
            "mean_ep_reward": np.mean(ep_rewards) if ep_rewards else 0.0,
            "returns":        returns,
            "advantages":     advantages,
        }

    def update(self, returns: np.ndarray, advantages: np.ndarray) -> Dict:
        """Met à jour le réseau de neurones avec PPO."""
        # Normaliser les advantages
        adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_clip_frac   = 0.0
        n_batches         = 0

        for epoch in range(config.PPO_UPDATE_EPOCHS):
            for obs_b, act_b, old_log_b, ret_b, adv_b in self.buffer.get_batches(
                returns, adv_normalized
            ):
                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(obs_b, act_b)

                # ── Policy Loss (PPO Clip) ─────────────────────
                ratio       = torch.exp(new_log_prob - old_log_b)
                obj_clip    = torch.clamp(ratio, 1 - config.PPO_CLIP_EPS, 1 + config.PPO_CLIP_EPS) * adv_b
                obj_unclip  = ratio * adv_b
                policy_loss = -torch.min(obj_unclip, obj_clip).mean()

                # ── Value Loss ─────────────────────────────────
                value_loss  = 0.5 * ((new_value - ret_b) ** 2).mean()

                # ── Entropie (exploration) ─────────────────────
                entropy_loss = entropy.mean()

                # ── Loss Totale ────────────────────────────────
                loss = (
                    policy_loss
                    + config.PPO_VALUE_COEF  * value_loss
                    - config.PPO_ENTROPY_COEF * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                # Métriques
                clip_frac = ((ratio - 1.0).abs() > config.PPO_CLIP_EPS).float().mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy_loss.item()
                total_clip_frac   += clip_frac
                n_batches         += 1

        metrics = {
            "policy_loss": total_policy_loss / max(n_batches, 1),
            "value_loss":  total_value_loss  / max(n_batches, 1),
            "entropy":     total_entropy     / max(n_batches, 1),
            "clip_frac":   total_clip_frac   / max(n_batches, 1),
        }

        for k, v in metrics.items():
            self.training_history[k].append(v)

        return metrics

    # ── Sauvegarde / Chargement ────────────────────────────────

    def save(self, path: str = config.MODEL_PATH):
        """Sauvegarde le modèle et l'état de l'optimiseur."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network_state":   self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps":     self.total_steps,
            "history":         self.training_history,
        }, path)
        logger.info(f"💾 Modèle sauvegardé : {path} ({self.total_steps:,} steps)")

    def load(self, path: str = config.MODEL_PATH) -> bool:
        """Charge un modèle pré-entraîné."""
        if not os.path.exists(path):
            logger.warning(f"Aucun modèle trouvé à {path}.")
            return False

        try:
            # Toujours charger sur CPU d'abord pour éviter les conflits de device
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
            self.network.load_state_dict(checkpoint["network_state"])
            self.network.to(self.device)

            # Charger l'optimizer séparément (peut causer des erreurs de device)
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as opt_e:
                logger.warning(f"Optimizer non chargé ({opt_e}) — réinitialisé")
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(), lr=config.PPO_LR
                )

            self.total_steps      = checkpoint.get("total_steps", 0)
            self.training_history = checkpoint.get("history", self.training_history)
            logger.info(f"Modele charge : {path} ({self.total_steps:,} steps)")
            return True
        except Exception as e:
            logger.error(f"Erreur chargement modele : {e}")
            return False

    def get_action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Retourne les probabilités pour chaque action (debug/logging)."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits, _ = self.network(obs_t)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy().flatten()

    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}