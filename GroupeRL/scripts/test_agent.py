"""
Test des 3 agents entraînés avec visualisation
"""

import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC

print("=" * 70)
print("🎮 TEST DES AGENTS ENTRAÎNÉS")
print("=" * 70)

# Test PPO et DQN sur CartPole
print("\n🎯 Environnement 1 : CartPole-v1 (PPO et DQN)")
print("-" * 70)

env_cartpole = gym.make("CartPole-v1", render_mode="human")

# Charger les modèles pour CartPole
models_cartpole = {
    "PPO": PPO.load("models/ppo_cartpole", env=env_cartpole),
    "DQN": DQN.load("models/dqn_cartpole", env=env_cartpole),
}

for algo_name, model in models_cartpole.items():
    print(f"\n🎬 Test de {algo_name} sur CartPole-v1...")
    print(f"   Vous verrez une fenêtre avec le jeu !")
    
    # 3 épisodes de test
    scores = []
    for episode in range(3):
        obs, info = env_cartpole.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_cartpole.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        scores.append(total_reward)
        print(f"   Episode {episode+1}: Score = {total_reward:.0f}, Étapes = {steps}")
    
    avg_score = sum(scores) / len(scores)
    print(f"   ✅ Score moyen {algo_name} : {avg_score:.1f}")
    print()

env_cartpole.close()

# Test SAC sur Pendulum
print("\n🎯 Environnement 2 : Pendulum-v1 (SAC)")
print("-" * 70)

env_pendulum = gym.make("Pendulum-v1", render_mode="human")

model_sac = SAC.load("models/sac_pendulum", env=env_pendulum)

print(f"\n🎬 Test de SAC sur Pendulum-v1...")
print(f"   Vous verrez une fenêtre avec le pendule !")

# 3 épisodes de test
scores_sac = []
for episode in range(3):
    obs, info = env_pendulum.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action, _ = model_sac.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_pendulum.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    scores_sac.append(total_reward)
    print(f"   Episode {episode+1}: Score = {total_reward:.0f}, Étapes = {steps}")

avg_score_sac = sum(scores_sac) / len(scores_sac)
print(f"   ✅ Score moyen SAC : {avg_score_sac:.1f}")

env_pendulum.close()

print("\n" + "=" * 70)
print("✅ TESTS TERMINÉS !")
print("=" * 70)
print("\n💡 Résumé :")
print("   - PPO et DQN : Équilibrer un bâton sur CartPole")
print("   - SAC : Faire tourner un pendule")
print("\n   Les fenêtres que vous venez de voir = l'IA en action ! 🎮")
print("=" * 70)
