# 🚀 Guide de lancement de l'application Streamlit

## Installation rapide

Si vous n'avez pas encore installé Streamlit:

```bash
# Activer l'environnement virtuel
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer Streamlit
pip install streamlit
```

## Lancer l'application

```bash
# Depuis le dossier racine du projet
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse:
```
http://localhost:8501
```

## Utiliser vos propres données

### Format requis

Votre fichier CSV doit contenir au minimum:
- **date**: Dates des périodes (format: YYYY-MM-DD)
- **sales**: Ventes observées
- **Colonnes media**: Dépenses publicitaires par canal (ex: `media_1_spend`, `tv_spend`, `facebook_spend`, etc.)

### Exemple de format

```csv
date,sales,media_1_spend,media_2_spend,media_3_spend
2023-01-01,1000,100,150,50
2023-01-08,1050,110,140,60
2023-01-15,1100,105,160,55
```

### Comment charger vos données

1. Dans la sidebar, sélectionnez "📁 Uploader un fichier CSV"
2. Cliquez sur "📥 Télécharger le template CSV" pour obtenir un exemple
3. Modifiez le template avec vos données
4. Uploadez votre fichier via le bouton "Browse files"

L'application détectera automatiquement les colonnes media et générera les noms de canaux!

## Sections de l'application

### 🏠 Accueil
- Présentation du projet
- Objectifs et concepts clés
- Aperçu des données (avec détection automatique des canaux)

### 📊 Données & EDA
- Séries temporelles des ventes et dépenses
- Corrélations entre canaux
- Statistiques descriptives

### 🔬 Transformations
- **Adstock**: Visualisation interactive de la persistance temporelle
- **Saturation**: Courbes de Hill avec paramètres ajustables
- **Combiné**: Transformation complète adstock + saturation

### 🧠 Modèle MMM
- Diagnostics de convergence MCMC
- Paramètres estimés (distributions a posteriori)
- Performance du modèle (MAE, RMSE, R²)

### 📈 Résultats & Attribution
- Contributions de chaque canal aux ventes
- Visualisations (barplot, pie chart)
- Insights actionnables

### 💰 Optimisation Budgétaire
- Allocation optimale du budget
- Scénarios what-if (budget variable)
- Recommandations stratégiques détaillées

## Conseils pour la présentation (6 janvier)

1. **Démarrer l'application avant la présentation**
   ```bash
   streamlit run app.py
   ```

2. **Navigation fluide**:
   - Utilisez la barre latérale pour naviguer entre les sections
   - Suivez l'ordre logique: Accueil → Données → Transformations → Modèle → Résultats → Optimisation

3. **Points à mettre en avant**:
   - Section **Transformations**: Démontrez l'interactivité avec les sliders
   - Section **Modèle MMM**: Montrez la convergence et la qualité des diagnostics
   - Section **Optimisation**: Présentez les recommandations concrètes

4. **Temps de chargement**:
   - Le modèle est entraîné au premier accès et mis en cache
   - Les pages suivantes seront instantanées
   - Budget 2-3 minutes pour la première fois

## Troubleshooting

### L'application ne démarre pas
```bash
# Vérifier que vous êtes dans le bon dossier
pwd
# Devrait afficher: .../mmm-bayesien

# Vérifier que l'environnement virtuel est activé
which python
# Devrait contenir 'venv'
```

### Erreur de dépendances manquantes
```bash
pip install -r requirements.txt
```

### Port déjà utilisé
```bash
# Utiliser un autre port
streamlit run app.py --server.port 8502
```

## Personnalisation (optionnel)

Pour modifier les couleurs ou le style, éditez la section CSS dans `app.py`:
```python
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;  # Modifier ici
    }
</style>
""", unsafe_allow_html=True)
```

---

**Bon courage pour la présentation! 🎉**
