"""
Entraînement d'un agent PPO sur CartPole-v1
"""

import gymnasium as gym
from stable_baselines3 import PPO
import os

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("🚀 Entraînement PPO sur CartPole-v1")
print("=" * 60)

# Créer l'environnement
env = gym.make("CartPole-v1")
print(f"✅ Environnement créé : CartPole-v1")
print(f"   - Espace d'observation : {env.observation_space}")
print(f"   - Espace d'action : {env.action_space}")

# Créer le modèle PPO avec les hyperparamètres
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,           # Nombre de steps avant mise à jour
    batch_size=64,          # Taille du batch
    n_epochs=10,            # Nombre d'epochs d'optimisation
    learning_rate=3e-4,     # Taux d'apprentissage
    verbose=1,              # Afficher les logs
    device="cpu"            # "cuda" si GPU disponible
)

print(f"\n✅ Modèle PPO créé avec les hyperparamètres")
print(f"   - Learning rate : 3e-4")
print(f"   - N steps : 2048")
print(f"   - Batch size : 64")
print(f"   - N epochs : 10")

# Entraîner le modèle
print(f"\n⏳ Entraînement en cours... (50,000 timesteps)")
print(f"   Cela devrait prendre environ 2-3 minutes...")
print("-" * 60)

model.learn(total_timesteps=50000)

# Sauvegarder le modèle
model.save("models/ppo_cartpole")
print("-" * 60)
print(f"\n✅ Entraînement PPO terminé avec succès !")
print(f"   Modèle sauvegardé : models/ppo_cartpole.zip")

env.close()
print("✅ Environnement fermé")
print("=" * 60)
