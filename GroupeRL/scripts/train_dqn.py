"""
Entraînement d'un agent DQN sur Doom (VizDoom) - Scénario Custom
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

# Copier le WAD si nécessaire (assumer qu'il est dans games/DOOM.WAD)
script_dir = os.path.dirname(os.path.abspath(__file__))
wad_path = os.path.join(script_dir, "../../games/DOOM.WAD")
wad_path = os.path.abspath(wad_path)
if not os.path.exists(wad_path):
    print(f"❌ WAD non trouvé à {wad_path}")
    print("   Placez DOOM.WAD dans le dossier games/")
    exit(1)

print(f"✅ WAD trouvé : {wad_path}")

# Enregistrer un environnement personnalisé VizDoom
register(
    id='VizdoomBasicCustom-v0',
    entry_point='vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv',
    kwargs={'scenario_file': os.path.join(script_dir, 'basic_custom.cfg')}
)

# Créer l'environnement
# render_mode=None est beaucoup plus rapide pour l'entraînement (pas d'affichage fenêtre)
env = gymnasium.make('VizdoomBasicCustom-v0', render_mode=None)
print(f"✅ Environnement créé : VizdoomBasicCustom-v0")
print(f"   - Espace d'observation : {env.observation_space}")
print(f"   - Espace d'action : {env.action_space}")

# Créer le modèle DQN avec MultiInputPolicy pour les dict observations
model = DQN(
    "MultiInputPolicy",  # Utilise MultiInputPolicy pour traiter les dict observations
    env,
    learning_rate=1e-4,  # Plus petit pour stabilité
    buffer_size=100000,   # Augmenté pour l'exploration
    learning_starts=5000, # Laisser l'agent explorer avant d'apprendre
    batch_size=2048,      # Augmenté pour 8Go VRAM
    target_update_interval=1000,
    verbose=1,
    device="cuda"  # Utilise GPU NVIDIA
)

print(f"\n✅ Modèle DQN créé avec les hyperparamètres")
print(f"   - Policy : MultiInputPolicy (pour dict observations)")
print(f"   - Learning rate : 1e-4")
print(f"   - Buffer size : 100000")
print(f"   - Learning starts : 5000")
print(f"   - Target update interval : 1000")

# Entraîner le modèle
print(f"\n⏳ Entraînement en cours... (10,000 timesteps)")
print(f"   Doom est complexe, cela peut prendre du temps...")
print("-" * 60)

model.learn(total_timesteps=10000)

# Sauvegarder le modèle
model.save("models/dqn_doom_custom")
print("-" * 60)
print(f"\n✅ Entraînement DQN sur Doom terminé avec succès !")
print(f"   Modèle sauvegardé : models/dqn_doom_custom.zip")

env.close()
