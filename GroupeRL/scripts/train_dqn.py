"""
Entraînement d'un agent DQN sur CartPole-v1
"""

import gymnasium as gym
from stable_baselines3 import DQN
import os

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("🚀 Entraînement DQN sur CartPole-v1")
print("=" * 60)

# Créer l'environnement
env = gym.make("CartPole-v1")
print(f"✅ Environnement créé : CartPole-v1")
print(f"   - Espace d'observation : {env.observation_space}")
print(f"   - Espace d'action : {env.action_space}")

# Créer le modèle DQN avec les hyperparamètres
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,         # Taux d'apprentissage
    buffer_size=10000,          # Taille du replay buffer
    learning_starts=1000,       # Commencer à apprendre après 1000 steps
    target_update_interval=500, # Mettre à jour le réseau cible
    verbose=1,                  # Afficher les logs
    device="cpu"                # "cuda" si GPU disponible
)

print(f"\n✅ Modèle DQN créé avec les hyperparamètres")
print(f"   - Learning rate : 1e-3")
print(f"   - Buffer size : 10000")
print(f"   - Learning starts : 1000")
print(f"   - Target update interval : 500")

# Entraîner le modèle
print(f"\n⏳ Entraînement en cours... (50,000 timesteps)")
print(f"   Cela devrait prendre environ 2-3 minutes...")
print("-" * 60)

model.learn(total_timesteps=50000)

# Sauvegarder le modèle
model.save("models/dqn_cartpole")
print("-" * 60)
print(f"\n✅ Entraînement DQN terminé avec succès !")
print(f"   Modèle sauvegardé : models/dqn_cartpole.zip")

env.close()
print("✅ Environnement fermé")
print("=" * 60)
