"""
Entraînement d'un agent SAC sur Pendulum-v1

Note: SAC est conçu pour les environnements continus.
Pendulum-v1 a des actions continues (contrairement à CartPole qui est discret).
"""

import gymnasium as gym
from stable_baselines3 import SAC
import os

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("🚀 Entraînement SAC sur Pendulum-v1")
print("=" * 60)

# Créer l'environnement
env = gym.make("Pendulum-v1")
print(f"✅ Environnement créé : Pendulum-v1")
print(f"   - Espace d'observation : {env.observation_space}")
print(f"   - Espace d'action : {env.action_space}")

# Créer le modèle SAC avec les hyperparamètres
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Taux d'apprentissage
    buffer_size=10000,       # Taille du replay buffer
    learning_starts=100,     # Commencer à apprendre rapidement
    verbose=1,               # Afficher les logs
    device="cpu"             # "cuda" si GPU disponible
)

print(f"\n✅ Modèle SAC créé avec les hyperparamètres")
print(f"   - Learning rate : 3e-4")
print(f"   - Buffer size : 10000")
print(f"   - Learning starts : 100")

# Entraîner le modèle
print(f"\n⏳ Entraînement en cours... (50,000 timesteps)")
print(f"   Cela devrait prendre environ 2-3 minutes...")
print("-" * 60)

model.learn(total_timesteps=50000)

# Sauvegarder le modèle
model.save("models/sac_pendulum")
print("-" * 60)
print(f"\n✅ Entraînement SAC terminé avec succès !")
print(f"   Modèle sauvegardé : models/sac_pendulum.zip")

env.close()
print("✅ Environnement fermé")
print("=" * 60)
