"""
Entraînement d'un agent DQN sur Doom (VizDoom)
Utilise le WAD Ultimate Doom fourni
"""

import os
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import DQN

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("🚀 Entraînement DQN sur Doom (VizDoom)")
print("=" * 60)

# Vérifier si VizDoom est installé
try:
    import vizdoom
    print("✅ VizDoom détecté")
except ImportError:
    print("❌ VizDoom n'est pas installé. Installez-le avec : pip install vizdoom")
    print("   Note: Nécessite Python 3.11 ou antérieur pour pygame.")
    exit(1)

# Copier le WAD si nécessaire (assumer qu'il est dans ../games/DOOM.WAD)
wad_path = "../games/DOOM.WAD"
if not os.path.exists(wad_path):
    print(f"❌ WAD non trouvé à {wad_path}")
    print("   Placez DOOM.WAD dans le dossier games/")
    exit(1)

print(f"✅ WAD trouvé : {wad_path}")

# Enregistrer un environnement personnalisé VizDoom
register(
    id='VizdoomBasicCustom-v0',
    entry_point='vizdoom.gymnasium_wrapper:VizdoomEnv',
    kwargs={'scenario': 'basic', 'wad': wad_path}
)

# Créer l'environnement
env = gymnasium.make('VizdoomBasicCustom-v0')
print(f"✅ Environnement créé : VizdoomBasicCustom-v0")
print(f"   - Espace d'observation : {env.observation_space}")
print(f"   - Espace d'action : {env.action_space}")

# Créer le modèle DQN avec CNN pour les images
model = DQN(
    "CnnPolicy",  # Utilise CNN pour traiter les images
    env,
    learning_rate=1e-4,  # Plus petit pour stabilité
    buffer_size=10000,
    learning_starts=1000,
    target_update_interval=500,
    verbose=1,
    device="cpu"
)

print(f"\n✅ Modèle DQN créé avec les hyperparamètres")
print(f"   - Policy : CnnPolicy (pour images)")
print(f"   - Learning rate : 1e-4")
print(f"   - Buffer size : 10000")
print(f"   - Learning starts : 1000")
print(f"   - Target update interval : 500")

# Entraîner le modèle
print(f"\n⏳ Entraînement en cours... (50,000 timesteps)")
print(f"   Doom est complexe, cela peut prendre du temps...")
print("-" * 60)

model.learn(total_timesteps=50000)

# Sauvegarder le modèle
model.save("models/dqn_doom")
print("-" * 60)
print(f"\n✅ Entraînement DQN sur Doom terminé avec succès !")
print(f"   Modèle sauvegardé : models/dqn_doom.zip")

env.close()
print(f"   Modèle sauvegardé : models/dqn_cartpole.zip")

env.close()
print("✅ Environnement fermé")
print("=" * 60)
