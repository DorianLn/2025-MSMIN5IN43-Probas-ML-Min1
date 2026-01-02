"""
Comparaison des performances des 3 algorithmes
Génère des graphiques de comparaison
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, SAC
import os

os.makedirs("results", exist_ok=True)

print("=" * 70)
print("📊 BENCHMARK ET COMPARAISON DES ALGORITHMES")
print("=" * 70)

def evaluate_agent(model, env, num_episodes=20):
    """Évalue un agent sur plusieurs épisodes"""
    scores = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        scores.append(episode_reward)
    
    return scores

# ============================================
# ÉVALUATION PPO et DQN sur CartPole
# ============================================
print("\n📈 Évaluation CartPole-v1 (PPO vs DQN)...")
print("-" * 70)

env_cartpole = gym.make("CartPole-v1")

models_cartpole = {
    "PPO": PPO.load("models/ppo_cartpole"),
    "DQN": DQN.load("models/dqn_cartpole"),
}

results_cartpole = {}
for algo_name, model in models_cartpole.items():
    print(f"\n🔄 Évaluation de {algo_name} (20 épisodes)...")
    scores = evaluate_agent(model, env_cartpole, num_episodes=20)
    results_cartpole[algo_name] = scores
    
    print(f"   ✅ {algo_name} sur CartPole :")
    print(f"      - Moyenne    : {np.mean(scores):.2f}")
    print(f"      - Écart-type : {np.std(scores):.2f}")
    print(f"      - Min        : {np.min(scores):.0f}")
    print(f"      - Max        : {np.max(scores):.0f}")

env_cartpole.close()

# ============================================
# ÉVALUATION SAC sur Pendulum
# ============================================
print(f"\n📈 Évaluation Pendulum-v1 (SAC)...")
print("-" * 70)

env_pendulum = gym.make("Pendulum-v1")

model_sac = SAC.load("models/sac_pendulum")

print(f"\n🔄 Évaluation de SAC (20 épisodes)...")
scores_sac = evaluate_agent(model_sac, env_pendulum, num_episodes=20)

print(f"   ✅ SAC sur Pendulum :")
print(f"      - Moyenne    : {np.mean(scores_sac):.2f}")
print(f"      - Écart-type : {np.std(scores_sac):.2f}")
print(f"      - Min        : {np.min(scores_sac):.0f}")
print(f"      - Max        : {np.max(scores_sac):.0f}")

env_pendulum.close()

# ============================================
# GRAPHIQUES
# ============================================
print(f"\n📊 Génération des graphiques...")
print("-" * 70)

fig = plt.figure(figsize=(16, 10))

# -------- Graphique 1 : CartPole - Boxplot --------
ax1 = plt.subplot(2, 3, 1)
ax1.boxplot([results_cartpole[algo] for algo in results_cartpole.keys()],
            labels=list(results_cartpole.keys()))
ax1.set_ylabel("Score", fontsize=11, fontweight='bold')
ax1.set_title("CartPole-v1: Distribution des scores\n(Boxplot)", fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 510])

# -------- Graphique 2 : CartPole - Barplot --------
ax2 = plt.subplot(2, 3, 2)
means_cp = [np.mean(results_cartpole[algo]) for algo in results_cartpole.keys()]
stds_cp = [np.std(results_cartpole[algo]) for algo in results_cartpole.keys()]
x = np.arange(len(results_cartpole))
bars = ax2.bar(x, means_cp, yerr=stds_cp, capsize=10, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
ax2.set_xticks(x)
ax2.set_xticklabels(list(results_cartpole.keys()), fontweight='bold')
ax2.set_ylabel("Score moyen", fontsize=11, fontweight='bold')
ax2.set_title("CartPole-v1: Score moyen ± écart-type\n(Plus haut = Mieux)", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 510])

# Ajouter les valeurs sur les barres
for i, (bar, mean) in enumerate(zip(bars, means_cp)):
    ax2.text(bar.get_x() + bar.get_width()/2, mean + 20, f'{mean:.0f}',
             ha='center', va='bottom', fontweight='bold')

# -------- Graphique 3 : CartPole - Violin plot --------
ax3 = plt.subplot(2, 3, 3)
parts = ax3.violinplot([results_cartpole[algo] for algo in results_cartpole.keys()],
                        positions=range(len(results_cartpole)),
                        showmeans=True, showmedians=True)
ax3.set_xticks(range(len(results_cartpole)))
ax3.set_xticklabels(list(results_cartpole.keys()), fontweight='bold')
ax3.set_ylabel("Score", fontsize=11, fontweight='bold')
ax3.set_title("CartPole-v1: Distribution détaillée\n(Violin plot)", fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim([0, 510])

# -------- Graphique 4 : Pendulum - Détails SAC --------
ax4 = plt.subplot(2, 3, 4)
ax4.hist(scores_sac, bins=10, alpha=0.7, color='#2ca02c', edgecolor='black')
ax4.axvline(np.mean(scores_sac), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(scores_sac):.1f}')
ax4.axvline(np.median(scores_sac), color='blue', linestyle='--', linewidth=2, label=f'Médiane: {np.median(scores_sac):.1f}')
ax4.set_xlabel("Score", fontsize=11, fontweight='bold')
ax4.set_ylabel("Fréquence", fontsize=11, fontweight='bold')
ax4.set_title("Pendulum-v1: Distribution des scores SAC\n(Histogramme)", fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# -------- Graphique 5 : Comparaison résumée --------
ax5 = plt.subplot(2, 3, 5)
algo_names_all = list(results_cartpole.keys()) + ["SAC"]
means_all = means_cp + [np.mean(scores_sac)]
env_labels = ["CartPole", "CartPole", "Pendulum"]
colors_map = {'PPO': '#1f77b4', 'DQN': '#ff7f0e', 'SAC': '#2ca02c'}
colors = [colors_map[name] for name in algo_names_all]

bars5 = ax5.bar(algo_names_all, means_all, color=colors, alpha=0.7)
ax5.set_ylabel("Score moyen", fontsize=11, fontweight='bold')
ax5.set_title("Résumé: Tous les algorithmes\n(Scores normalisés par env.)", fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Ajouter les environnements sous les labels
for i, (bar, label) in enumerate(zip(bars5, env_labels)):
    ax5.text(bar.get_x() + bar.get_width()/2, -20, f'({label})',
             ha='center', va='top', fontsize=9, style='italic')

# Ajouter les valeurs
for bar, mean in zip(bars5, means_all):
    ax5.text(bar.get_x() + bar.get_width()/2, mean + 15, f'{mean:.0f}',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# -------- Graphique 6 : Tableau récapitulatif --------
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Créer un tableau de résumé
tableau_data = []
tableau_data.append(['Algorithme', 'Environnement', 'Moyenne', 'Écart-type', 'Min/Max'])
tableau_data.append(['-'*15, '-'*15, '-'*10, '-'*12, '-'*15])

for algo in results_cartpole.keys():
    scores = results_cartpole[algo]
    tableau_data.append([
        algo,
        'CartPole-v1',
        f'{np.mean(scores):.1f}',
        f'{np.std(scores):.1f}',
        f'{np.min(scores):.0f}/{np.max(scores):.0f}'
    ])

tableau_data.append(['-'*15, '-'*15, '-'*10, '-'*12, '-'*15])
tableau_data.append([
    'SAC',
    'Pendulum-v1',
    f'{np.mean(scores_sac):.1f}',
    f'{np.std(scores_sac):.1f}',
    f'{np.min(scores_sac):.0f}/{np.max(scores_sac):.0f}'
])

table = ax6.table(cellText=tableau_data, cellLoc='center', loc='center',
                  colWidths=[0.15, 0.20, 0.15, 0.15, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style des en-têtes
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax6.set_title("Résumé des résultats", fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("results/comparaison_algos.png", dpi=150, bbox_inches='tight')
print(f"✅ Graphique sauvegardé : results/comparaison_algos.png")

plt.show()

# ============================================
# RÉSUMÉ FINAL
# ============================================
print("\n" + "=" * 70)
print("✅ BENCHMARK TERMINÉ !")
print("=" * 70)

print("\n🏆 RÉSULTATS RÉSUMÉS :")
print("-" * 70)

print("\n📊 CartPole-v1 (Actions discrètes):")
for algo in results_cartpole.keys():
    scores = results_cartpole[algo]
    print(f"\n   {algo}:")
    print(f"      • Moyenne     : {np.mean(scores):>6.1f}")
    print(f"      • Écart-type  : {np.std(scores):>6.1f}")
    print(f"      • Meilleur    : {np.max(scores):>6.0f}")
    print(f"      • Pire        : {np.min(scores):>6.0f}")

print(f"\n📊 Pendulum-v1 (Actions continues):")
print(f"\n   SAC:")
print(f"      • Moyenne     : {np.mean(scores_sac):>6.1f}")
print(f"      • Écart-type  : {np.std(scores_sac):>6.1f}")
print(f"      • Meilleur    : {np.max(scores_sac):>6.0f}")
print(f"      • Pire        : {np.min(scores_sac):>6.0f}")

print("\n" + "=" * 70)
print("💡 INTERPRÉTATION :")
print("-" * 70)
print("   • Score HAUT = Agent performant")
print("   • Écart-type BAS = Agent stable et consistant")
print("   • Min/Max proches = Agent prévisible")
print("\n📁 Tous les graphiques sont dans 'results/comparaison_algos.png'")
print("=" * 70)
