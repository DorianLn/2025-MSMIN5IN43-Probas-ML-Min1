"""
Comparaison des performances des 3 algorithmes sur Doom
Génère des graphiques de comparaison
"""

import os
import gymnasium
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, SAC

os.makedirs("results", exist_ok=True)

print("=" * 70)
print("📊 BENCHMARK ET COMPARAISON DES ALGORITHMES SUR DOOM")
print("=" * 70)

# Vérifier si VizDoom est installé
try:
    import vizdoom
    print("✅ VizDoom détecté")
except ImportError:
    print("❌ VizDoom n'est pas installé. Installez-le avec : pip install vizdoom")
    exit(1)

# Vérifier le WAD
script_dir = os.path.dirname(os.path.abspath(__file__))
wad_path = os.path.join(script_dir, "../../games/DOOM.WAD")
wad_path = os.path.abspath(wad_path)
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
# ÉVALUATION SUR DOOM
# ============================================
print("\n📈 Évaluation VizdoomBasicCustom-v0 (PPO vs DQN vs SAC)...")
print("-" * 70)

env = gymnasium.make('VizdoomBasicCustom-v0')

models = {
    "PPO": PPO.load("models/ppo_doom"),
    "DQN": DQN.load("models/dqn_doom"),
    "SAC": SAC.load("models/sac_doom")
}

results = {}
for algo_name, model in models.items():
    print(f"\n🔄 Évaluation de {algo_name} (20 épisodes)...")
    scores = evaluate_agent(model, env, num_episodes=20)
    results[algo_name] = scores
    
    print(f"   ✅ {algo_name} sur Doom :")
    print(f"      - Moyenne    : {np.mean(scores):.2f}")
    print(f"      - Écart-type : {np.std(scores):.2f}")
    print(f"      - Min        : {np.min(scores):.0f}")
    print(f"      - Max        : {np.max(scores):.0f}")

env.close()

# ============================================
# GRAPHIQUES
# ============================================
print(f"\n📊 Génération des graphiques...")
print("-" * 70)

fig = plt.figure(figsize=(16, 10))

# -------- Graphique 1 : Doom - Boxplot --------
ax1 = plt.subplot(2, 3, 1)
ax1.boxplot([results[algo] for algo in results.keys()],
            labels=list(results.keys()))
ax1.set_ylabel("Score", fontsize=11, fontweight='bold')
ax1.set_title("Doom: Distribution des scores\n(Boxplot)", fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# -------- Graphique 2 : Doom - Barplot --------
ax2 = plt.subplot(2, 3, 2)
means = [np.mean(results[algo]) for algo in results.keys()]
stds = [np.std(results[algo]) for algo in results.keys()]
x = np.arange(len(results))
bars = ax2.bar(x, means, yerr=stds, capsize=10, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax2.set_xticks(x)
ax2.set_xticklabels(list(results.keys()), fontweight='bold')
ax2.set_ylabel("Score moyen", fontsize=11, fontweight='bold')
ax2.set_title("Doom: Score moyen ± écart-type\n(Plus haut = Mieux)", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for i, (bar, mean) in enumerate(zip(bars, means)):
    ax2.text(bar.get_x() + bar.get_width()/2, mean + max(stds)*0.1, f'{mean:.0f}',
             ha='center', va='bottom', fontweight='bold')

# -------- Graphique 3 : Doom - Violin plot --------
ax3 = plt.subplot(2, 3, 3)
parts = ax3.violinplot([results[algo] for algo in results.keys()],
                        positions=range(len(results)),
                        showmeans=True, showmedians=True)
ax3.set_xticks(range(len(results)))
ax3.set_xticklabels(list(results.keys()), fontweight='bold')
ax3.set_ylabel("Score", fontsize=11, fontweight='bold')
ax3.set_title("Doom: Distribution détaillée\n(Violin plot)", fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# -------- Graphique 4 : Comparaison détaillée --------
ax4 = plt.subplot(2, 3, 4)
for i, algo in enumerate(results.keys()):
    ax4.hist(results[algo], bins=10, alpha=0.5, label=algo, edgecolor='black')
ax4.set_xlabel("Score", fontsize=11, fontweight='bold')
ax4.set_ylabel("Fréquence", fontsize=11, fontweight='bold')
ax4.set_title("Doom: Histogrammes superposés", fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# -------- Graphique 5 : Résumé comparatif --------
ax5 = plt.subplot(2, 3, 5)
algo_names_all = list(results.keys())
colors_map = {'PPO': '#1f77b4', 'DQN': '#ff7f0e', 'SAC': '#2ca02c'}
colors = [colors_map[name] for name in algo_names_all]

bars5 = ax5.bar(algo_names_all, means, color=colors, alpha=0.7)
ax5.set_ylabel("Score moyen", fontsize=11, fontweight='bold')
ax5.set_title("Résumé: Tous les algorithmes\n(DOOM)", fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs
for bar, mean in zip(bars5, means):
    ax5.text(bar.get_x() + bar.get_width()/2, mean + max(stds)*0.1, f'{mean:.0f}',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# -------- Graphique 6 : Tableau récapitulatif --------
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Créer un tableau de résumé
tableau_data = []
tableau_data.append(['Algorithme', 'Environnement', 'Moyenne', 'Écart-type', 'Min/Max'])
tableau_data.append(['-'*15, '-'*15, '-'*10, '-'*12, '-'*15])

for algo in results.keys():
    scores = results[algo]
    tableau_data.append([
        algo,
        'VizdoomBasic-v0',
        f'{np.mean(scores):.1f}',
        f'{np.std(scores):.1f}',
        f'{np.min(scores):.0f}/{np.max(scores):.0f}'
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

print("\n📊 VizdoomBasic-v0 (Actions discrètes - Doom):")
for algo in results.keys():
    scores = results[algo]
    print(f"\n   {algo}:")
    print(f"      • Moyenne     : {np.mean(scores):>6.1f}")
    print(f"      • Écart-type  : {np.std(scores):>6.1f}")
    print(f"      • Meilleur    : {np.max(scores):>6.0f}")
    print(f"      • Pire        : {np.min(scores):>6.0f}")

print("\n" + "=" * 70)
print("💡 INTERPRÉTATION :")
print("-" * 70)
print("   • Score HAUT = Agent performant (survie + kills)")
print("   • Écart-type BAS = Agent stable et consistant")
print("   • Min/Max proches = Agent prévisible")
print("\n📁 Tous les graphiques sont dans 'results/comparaison_algos.png'")
print("=" * 70)
