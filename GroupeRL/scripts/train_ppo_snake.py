"""
Entraînement d'un agent PPO sur Snake
"""

import sys
import os

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
from stable_baselines3 import PPO

# Importer l'environnement personnalisé
from envs.snake_env import SnakeEnv

os.makedirs("models", exist_ok=True)

print("=" * 60)
print("🚀 Entraînement PPO sur Snake-v0")
print("=" * 60)

# Créer l'environnement Snake
env = SnakeEnv(grid_size=10, render_mode=None)
print(f"✅ Environnement créé : Snake-v0")
print(f"   - Grille : 10x10")
print(f"   - Actions : 4 (Haut, Droite, Bas, Gauche)")
print(f"   - Observation : 6 variables (position, pomme, direction, longueur)")

# Créer le modèle PPO
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1,
    device="cpu"
)

print(f"\n✅ Modèle PPO créé")
print(f"   - Learning rate : 3e-4")
print(f"   - N steps : 2048")
print(f"   - Batch size : 64")

# Entraîner
print(f"\n⏳ Entraînement en cours... (500,000 timesteps)")
print(f"   Cela devrait prendre environ 10-15 minutes...")
print("-" * 60)

model.learn(total_timesteps=1000000)

# Sauvegarder
model.save("models/ppo_snake")
print("-" * 60)
print(f"\n✅ Entraînement PPO terminé !")
print(f"   Modèle sauvegardé : models/ppo_snake.zip")

env.close()
print("=" * 60)
