# 📊 Marketing Mix Modeling (MMM) Bayésien

> Système d'attribution et d'optimisation budgétaire pour campagnes marketing multi-canaux

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyMC](https://img.shields.io/badge/PyMC-5.10%2B-orange)](https://www.pymc.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Objectif du projet

Ce projet implémente un **système de Marketing Mix Modeling (MMM) bayésien** pour résoudre des problèmes clés en marketing digital :

1. **Attribution des ventes** : Mesurer l'impact réel de chaque canal publicitaire (TV, Facebook, Google Ads, radio, etc.)
2. **Effets de saturation** : Modéliser les rendements décroissants (loi des rendements marginaux)
3. **Effets d'adstock** : Capturer la persistance temporelle de l'impact publicitaire
4. **Optimisation budgétaire** : Recommander l'allocation optimale des investissements marketing

### 💡 Cas d'usage

- **Analyse d'attribution** : "Quel canal marketing génère le plus de ROI ?"
- **Planification budgétaire** : "Comment répartir 1M€ entre TV, digital et radio ?"
- **Prévisions de ventes** : "Quelles seront les ventes si on double le budget Facebook ?"

---

## 🏗️ Architecture du projet

```
mmm-bayesien/
├── data/
│   ├── raw/                    # Données brutes (non versionnées)
│   └── processed/              # Données prétraitées
├── src/
│   ├── data/
│   │   ├── loader.py           # Chargement des données
│   │   └── preprocessing.py    # Nettoyage et feature engineering
│   ├── models/
│   │   ├── base_mmm.py         # Modèle MMM bayésien (PyMC)
│   │   ├── transformations.py  # ✅ Adstock & Saturation
│   │   └── priors.py           # Distributions a priori
│   ├── inference/
│   │   ├── sampler.py          # MCMC sampling
│   │   └── diagnostics.py      # Convergence & posterior checks
│   ├── optimization/
│   │   └── budget_allocator.py # Optimisation budget
│   └── visualization/
│       ├── exploratory.py      # EDA
│       ├── posterior_plots.py  # Visualisations bayésiennes
│       └── contribution.py     # Attribution par canal
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # Analyse exploratoire
├── tests/
│   └── test_transformations.py # ✅ Tests unitaires
├── results/                    # Outputs, graphiques, métriques
├── config/                     # Fichiers de configuration
├── docs/                       # Documentation supplémentaire
├── slides/                     # Présentation finale
├── README.md                   # 📄 Ce fichier
└── requirements.txt            # ✅ Dépendances Python
```

**Légende** :
- ✅ = Implémenté
- 🚧 = En cours
- ⏳ = À venir

---

## 🧠 Concepts clés

### 1. **Adstock géométrique** (Koyck transformation)

Modélise la **persistance temporelle** de l'effet publicitaire :

$$
y_t = x_t + \alpha \cdot x_{t-1} + \alpha^2 \cdot x_{t-2} + ... + \alpha^{l_{max}} \cdot x_{t-l_{max}}
$$

- **α (alpha)** : Taux de rétention ∈ [0, 1)
  - α = 0 : effet immédiat uniquement
  - α = 0.5 : 50% de l'effet persiste à la période suivante
  - α = 0.9 : forte persistance (ex: campagnes de branding)

### 2. **Saturation de Hill** (courbe sigmoïde)

Modélise les **rendements décroissants** :

$$
y = \frac{x^s}{k^s + x^s}
$$

- **k (half_saturation)** : Point où l'effet = 50% du maximum
- **s (slope)** : Pente de la courbe (contrôle la vitesse de saturation)

### 3. **Modèle hiérarchique bayésien**

```
Ventes ~ Distribution_Likelihood(μ, σ)
μ = β₀ + Σᵢ βᵢ · f(dépenses_canal_i)
βᵢ ~ Prior_Distribution
```

Où `f(·)` = transformation adstock + saturation

---

## 🚀 Installation

### Prérequis

- Python 3.9+
- pip ou conda

### Étapes

1. **Cloner le dépôt** (ou extraire l'archive)

```bash
git clone <repo-url>
cd mmm-bayesien
```

2. **Créer un environnement virtuel** (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Vérifier l'installation**

```bash
pytest tests/ -v
```

✅ Si tous les tests passent, l'installation est réussie !

---

## 📚 Utilisation

### Exemple minimal : Transformations

```python
import numpy as np
from src.models.transformations import geometric_adstock, hill_saturation

# Données de dépenses publicitaires (en milliers €)
spend = np.array([100, 80, 60, 40, 20])

# Appliquer l'adstock (alpha=0.5 → persistance modérée)
adstocked = geometric_adstock(spend, alpha=0.5, l_max=4)
print("Dépenses avec adstock:", adstocked)
# Output: [100.0, 130.0, 125.0, 102.5, 71.25]

# Appliquer la saturation (k=80 → demi-saturation à 80k€)
saturated = hill_saturation(adstocked, half_saturation=80, slope=1.0)
print("Effet saturé:", saturated)
# Output: [0.555, 0.619, 0.610, 0.561, 0.471]
```

### Pipeline complet

```python
from src.models.transformations import adstock_and_saturation

# Transformation complète en une ligne
transformed = adstock_and_saturation(
    spend,
    alpha=0.5,
    half_saturation=80,
    l_max=4,
    slope=1.0
)
```

### Visualisation de la saturation

```python
import matplotlib.pyplot as plt
from src.models.transformations import get_effective_reach_curve

spend_range = np.linspace(0, 500, 100)
curve = get_effective_reach_curve(spend_range, half_saturation=100)

plt.plot(spend_range, curve)
plt.xlabel("Dépenses publicitaires (k€)")
plt.ylabel("Effet saturé (0-1)")
plt.title("Courbe de saturation de Hill")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🧪 Tests

Les tests unitaires valident :
- ✅ Propriétés mathématiques (bornes, monotonie)
- ✅ Cas limites (valeurs nulles, grandes valeurs)
- ✅ Multi-canaux
- ✅ Gestion des erreurs

**Exécuter les tests** :

```bash
# Tous les tests
pytest tests/ -v

# Avec couverture de code
pytest tests/ --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_transformations.py::TestGeometricAdstock -v
```

---

## 📊 Dataset recommandé

**Robyn Dataset** (Facebook/Meta) :
- 208 semaines de données simulées
- 5 canaux publicitaires : TV, online_banners, facebook, search, newsletter
- Variables de contrôle : tendances, saisonnalité, événements

**Source** : [Facebook Robyn GitHub](https://github.com/facebookexperimental/Robyn)

---

## 🛠️ Stack technique

| Catégorie | Outils |
|-----------|--------|
| **Inférence bayésienne** | PyMC 5.10+, PyTensor, ArviZ |
| **Data manipulation** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn, plotly |
| **Calcul scientifique** | scipy |
| **Tests** | pytest, pytest-cov |
| **Qualité de code** | black, flake8, mypy |

---

## 📖 Références théoriques

### Papers fondateurs

1. **Jin et al. (2017)** - *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects*
   - Introduction de l'adstock géométrique et de la saturation de Hill en MMM

2. **Chan & Perry (2017)** - *Challenges and Opportunities in Media Mix Modeling*
   - Revue des défis pratiques en attribution marketing

3. **Hill (1910)** - *The possible effects of the aggregation of the molecules of haemoglobin*
   - Origine de l'équation de Hill (biologie → marketing)

### Ressources en ligne

- [PyMC-Marketing Documentation](https://www.pymc.io/projects/marketing/en/stable/)
- [Google LightweightMMM](https://github.com/google/lightweight_mmm)
- [Facebook Robyn](https://github.com/facebookexperimental/Robyn)

---

## 📈 Roadmap

### Phase 1 : Fondations ✅
- [x] Structure du projet
- [x] Transformations (adstock & saturation)
- [x] Tests unitaires
- [x] Documentation initiale

### Phase 2 : Modélisation 🚧
- [ ] Modèle MMM bayésien (PyMC)
- [ ] Définition des priors
- [ ] MCMC sampling & diagnostics

### Phase 3 : Analyse & Visualisation ⏳
- [ ] Contribution par canal
- [ ] Courbes ROI
- [ ] Posterior predictive checks

### Phase 4 : Optimisation ⏳
- [ ] Optimiseur budgétaire
- [ ] Scénarios what-if
- [ ] Recommandations d'allocation

### Phase 5 : Déploiement ⏳
- [ ] Interface interactive (Streamlit/Gradio)
- [ ] Rapport automatisé
- [ ] Présentation finale

---

## 🎓 Contexte académique

**Cours** : IA probabiliste, théorie de jeux et machine learning  
**École** : EPF Engineering School (5ème année)  
**Étudiant** : Ivan - Spécialisation AI & Cloud Computing  
**Type** : Projet individuel (bonus +3 points)  
**Deadline** : 5 janvier 2026 (présentation le 6 janvier)

### Critères d'évaluation

- **Présentation/Communication** (25%) : Clarté, pédagogie, qualité slides
- **Contenu théorique** (25%) : Compréhension MMM, état de l'art
- **Contenu technique** (25%) : Qualité code, résultats, démo
- **Organisation** (25%) : Structure Git, documentation, reproductibilité

---

## 🤝 Contribution

Projet académique individuel. Les suggestions et retours sont bienvenus via issues.

---

## 📝 License

MIT License - voir [LICENSE](LICENSE) pour détails.

---

## 📧 Contact

**Auteur** : Ivan  
**École** : EPF Engineering School  
**Projet** : MSMIN5IN43 - MMM Bayésien

---

## 🙏 Remerciements

- **PyMC Team** : Pour le framework d'inférence bayésienne
- **Meta/Facebook** : Pour le dataset Robyn et la librairie de référence
- **Google Research** : Pour LightweightMMM
- **Communauté PyMC-Marketing** : Pour les exemples et la documentation

---

**⭐ N'oubliez pas de documenter votre progression et de commiter régulièrement !**
