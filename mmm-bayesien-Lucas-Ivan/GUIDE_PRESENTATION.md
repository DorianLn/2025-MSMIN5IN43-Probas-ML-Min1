# 🎯 Guide de Présentation - MMM Bayésien
## 6 janvier 2026 - Ivan

---

## 📋 Checklist avant présentation

### Jour J - 30 minutes avant
- [ ] Ouvrir un terminal dans le dossier du projet
- [ ] Activer l'environnement virtuel: `source venv/bin/activate`
- [ ] Lancer l'application: `streamlit run app.py`
- [ ] Vérifier que l'app s'ouvre sur `http://localhost:8501`
- [ ] Tester la navigation entre les sections
- [ ] Préparer un navigateur en plein écran

---

## 🎬 Déroulement de la présentation (10-15 minutes)

### 1. **ACCUEIL** (1-2 min)
**Page:** 🏠 Accueil

**À dire:**
- "Bonjour, je vais vous présenter mon projet de Marketing Mix Modeling bayésien"
- "L'objectif: mesurer l'impact réel de chaque canal publicitaire et optimiser l'allocation budgétaire"
- Montrer rapidement les concepts clés et la stack technique

**Points à souligner:**
- Approche bayésienne avec PyMC
- 4 objectifs clairs: Attribution, Saturation, Adstock, Optimisation

---

### 2. **DONNÉES** (2 min) - OPTIONNEL
**Page:** 📊 Données & EDA

**À dire:**
- "Voici nos données: X périodes, Y canaux media"
- Montrer rapidement:
  - Séries temporelles (tendance visible)
  - Corrélations (quels canaux sont corrélés aux ventes)
  - Distribution du budget

**Points à souligner:**
- Données réalistes avec tendances et variations
- Corrélations positives entre media et ventes

**💡 Conseil:** Si le temps presse, passer cette section

---

### 3. **TRANSFORMATIONS** (3-4 min) ⭐ CORE
**Page:** 🔬 Transformations

**À dire:**
- "Le cœur du MMM: 2 transformations essentielles"

**Adstock (1.5 min):**
- "L'adstock capture la persistance de l'effet publicitaire dans le temps"
- Jouer avec le slider alpha (montrer 0.3 vs 0.8)
- "Plus alpha est élevé, plus l'effet persiste longtemps"

**Saturation (1.5 min):**
- "La saturation modélise les rendements décroissants"
- Jouer avec k (half-saturation)
- "Au-delà d'un certain budget, l'effet marginal diminue"

**Combiné (1 min):**
- "En pratique, on applique les deux transformations"
- Montrer le graphique combiné

**Points à souligner:**
- Ces transformations sont le CŒUR du MMM
- Elles capturent la réalité économique du marketing

---

### 4. **MODÈLE MMM** (3 min) ⭐ CORE
**Page:** 🧠 Modèle MMM

**À dire:**
- "J'ai entraîné un modèle bayésien avec PyMC et MCMC sampling"

**Diagnostics (1 min):**
- Montrer les métriques de convergence
- "R-hat < 1.01 et ESS > 400 = convergence OK"
- "Le modèle est fiable"

**Performance (1 min):**
- Montrer le scatter plot
- "MAE, RMSE, R² → le modèle prédit bien les ventes"

**Paramètres (1 min):**
- Montrer les paramètres estimés
- "Chaque canal a son propre alpha (adstock) et k (saturation)"

**Points à souligner:**
- Inférence bayésienne rigoureuse
- Diagnostics prouvent la qualité du modèle
- Bonne performance prédictive

---

### 5. **RÉSULTATS & ATTRIBUTION** (2-3 min) ⭐ CORE
**Page:** 📈 Résultats & Attribution

**À dire:**
- "Voici la contribution de chaque canal aux ventes"

**Contributions:**
- Montrer le barplot/pie chart
- "Canal X contribue à Y% des ventes"
- "C'est l'attribution bayésienne"

**Insights:**
- Lire les insights clés affichés
- "Le canal le plus/moins performant"

**Points à souligner:**
- Attribution objective basée sur le modèle
- Résultats actionnables pour le marketing

---

### 6. **OPTIMISATION BUDGÉTAIRE** (3-4 min) ⭐ BUSINESS VALUE
**Page:** 💰 Optimisation Budgétaire

**À dire:**
- "Maintenant, la vraie valeur ajoutée: l'optimisation du budget"

**Allocation optimale (1.5 min):**
- Montrer le tableau actuel vs optimal
- "En réallouant le budget, on gagne X% de ventes SANS budget supplémentaire"
- Montrer le graphique de comparaison

**Scénarios (1 min):**
- Jouer avec le slider de budget
- "Si on augmente/diminue le budget, voici l'impact"

**Recommandations (1.5 min):**
- Lire les recommandations affichées
- "Augmenter canal X, réduire canal Y"
- "Plan d'action concret"

**Points à souligner:**
- Optimisation mathématique rigoureuse
- ROI marginal égalisé entre canaux
- Recommandations concrètes et actionnables

---

## 💡 Conseils de présentation

### À FAIRE ✅
- Parler clairement et pas trop vite
- Montrer l'interactivité (sliders dans Transformations)
- Mettre l'accent sur la VALEUR BUSINESS (optimisation)
- Préparer 2-3 phrases clés par section
- Avoir une bouteille d'eau à portée de main

### À ÉVITER ❌
- Ne pas lire les slides/textes à l'écran
- Ne pas s'attarder sur les détails techniques (sauf si question)
- Ne pas passer trop de temps sur l'EDA
- Ne pas stresser si une question vous bloque

---

## 🎯 Messages clés à retenir

1. **Problème:** Attribution marketing et optimisation budgétaire
2. **Solution:** MMM bayésien avec transformations adstock + saturation
3. **Méthode:** PyMC, inférence MCMC, diagnostics rigoureux
4. **Résultats:** Attribution par canal + optimisation → +X% de ventes
5. **Valeur:** Recommandations actionnables pour améliorer le ROI

---

## 🔥 Points différenciants de votre projet

1. **Application web interactive** (Streamlit) - pas juste des notebooks
2. **Détection automatique** des colonnes media (upload CSV)
3. **Visualisations interactives** (sliders pour comprendre les transformations)
4. **Approche complète**: de l'EDA à l'optimisation
5. **Code propre et structuré** (architecture src/, tests/)

---

## ❓ Questions potentielles & Réponses

### Q: Pourquoi bayésien et pas juste une régression linéaire?
**R:** L'approche bayésienne permet:
- D'incorporer des priors sur les paramètres
- D'obtenir des distributions de probabilité (incertitude)
- De faire de l'inférence robuste même avec données limitées

### Q: Comment choisir les paramètres alpha et k?
**R:** Dans mon implémentation:
- Alpha (adstock): peut être estimé par le modèle ou fixé selon la connaissance métier
- k (saturation): calculé à partir de la moyenne des dépenses
- En production: validation croisée pour optimiser ces hyperparamètres

### Q: Quelle est la complexité du modèle?
**R:**
- Modèle hiérarchique avec N canaux
- Transformations non-linéaires (adstock + saturation)
- Inférence MCMC: 2000 itérations (1000 tune + 1000 draw) × 2 chains
- Temps d'entraînement: 2-3 minutes

### Q: Comment valider les résultats?
**R:**
- Diagnostics MCMC (R-hat, ESS)
- Métriques de performance (MAE, RMSE, R²)
- Posterior predictive checks
- Validation croisée (train/test split)

### Q: Peut-on utiliser ce modèle en production?
**R:** Oui, avec quelques ajustements:
- Automatiser le ré-entraînement périodique
- Monitoring des métriques de performance
- A/B testing des recommandations d'optimisation
- Intégration avec outils BI existants

---

## ⏱️ Timing recommandé

| Section | Temps | Priorité |
|---------|-------|----------|
| Accueil | 1-2 min | ⭐⭐ |
| Données & EDA | 0-2 min | ⭐ (optionnel) |
| Transformations | 3-4 min | ⭐⭐⭐ |
| Modèle MMM | 3 min | ⭐⭐⭐ |
| Résultats | 2-3 min | ⭐⭐⭐ |
| Optimisation | 3-4 min | ⭐⭐⭐ |
| **TOTAL** | **12-18 min** | |

---

## 🎉 Derniers conseils

1. **Respirez** - vous connaissez votre sujet
2. **Souriez** - vous avez fait un super projet
3. **Soyez fier** - l'application est impressionnante
4. **Profitez** - c'est votre moment de briller

**Bonne chance! 🚀**
