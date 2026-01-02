"""
Test des 3 agents entraînés sur Doom (VizDoom)
"""

import os
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, DQN, SAC

print("=" * 70)
print("🎮 TEST DES AGENTS ENTRAÎNÉS SUR DOOM")
print("=" * 70)

# Vérifier si VizDoom est installé
try:
    import vizdoom
    print("✅ VizDoom détecté")
except ImportError:
    print("❌ VizDoom n'est pas installé. Installez-le avec : pip install vizdoom")
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
env = gymnasium.make('VizdoomBasicCustom-v0', render_mode="human")
print(f"✅ Environnement créé : VizdoomBasicCustom-v0")

# Charger les modèles
models = {
    "PPO": PPO.load("models/ppo_doom", env=env),
    "DQN": DQN.load("models/dqn_doom", env=env),
    "SAC": SAC.load("models/sac_doom", env=env)
}

for algo_name, model in models.items():
    print(f"\n🎬 Test de {algo_name} sur Doom...")
    print(f"   Vous verrez une fenêtre avec le jeu Doom !")
    
    # 3 épisodes de test
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
    print(f"   ✅ Score moyen {algo_name} : {avg_score:.1f}")
    print()

env.close()

print("\n" + "=" * 70)
print("✅ TESTS TERMINÉS !")
print("=" * 70)
print("\n💡 Résumé :")
print("   - Tous les agents jouent à Doom (Ultimate Doom)")
print("   - Objectif : Survivre et tuer des ennemis")
print("\n   Les fenêtres que vous venez de voir = l'IA en action dans Doom ! 🎮")
print("=" * 70)
