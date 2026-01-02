"""
Test des agents entraînés sur Doom
"""

import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
import os

print("=" * 70)
print("🎮 TEST DES AGENTS ENTRAÎNÉS SUR DOOM")
print("=" * 70)

# Vérifier si VizDoom est installé
try:
    import vizdoom
    print("✅ VizDoom détecté")
except ImportError:
    print("❌ VizDoom n'est pas installé.")
    exit(1)

# Vérifier le WAD
wad_path = "../games/DOOM.WAD"
if not os.path.exists(wad_path):
    print(f"❌ WAD non trouvé à {wad_path}")
    exit(1)

print(f"✅ WAD trouvé : {wad_path}")

# Enregistrer l'environnement
register(
    id='VizdoomBasicCustom-v0',
    entry_point='vizdoom.gymnasium_wrapper:VizdoomEnv',
    kwargs={'scenario': 'basic', 'wad': wad_path}
)

# Créer l'environnement
env = gymnasium.make('VizdoomBasicCustom-v0')

# Charger le modèle
model_path = "models/ppo_doom"
if not os.path.exists(f"{model_path}.zip"):
    print(f"❌ Modèle non trouvé : {model_path}.zip")
    print("   Entraînez d'abord avec train_doom.py")
    exit(1)

model = PPO.load(model_path, env=env)

print(f"\n🎬 Test de PPO sur Doom...")
print(f"   Simulation en cours...")

# Test sur quelques épisodes
scores = []
for episode in range(3):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    scores.append(total_reward)
    print(f"   Episode {episode+1}: Score = {total_reward:.0f}, Étapes = {steps}")

avg_score = sum(scores) / len(scores)
print(f"   ✅ Score moyen PPO sur Doom : {avg_score:.1f}")

env.close()

print("\n" + "=" * 70)
print("✅ TESTS TERMINÉS !")
print("   L'IA joue maintenant à Doom ! 🎮")
print("=" * 70)