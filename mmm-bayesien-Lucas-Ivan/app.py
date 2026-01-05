"""
Application Streamlit pour la démonstration du Marketing Mix Modeling Bayésien

Auteur: Ivan
Projet: MMM Bayésien - MSMIN5IN43
Date: Janvier 2026
"""

import sys
sys.path.insert(0, 'src')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from data.loader import load_csv_data, create_sample_data
from data.preprocessing import prepare_mmm_data
from models.base_mmm import BayesianMMM
from models.transformations import geometric_adstock, hill_saturation, adstock_and_saturation
from inference.diagnostics import check_convergence
from visualization.exploratory import (
    plot_time_series, plot_sales_vs_media, plot_correlation_matrix
)
from visualization.contribution import (
    calculate_channel_contributions, plot_contribution_bars, plot_contribution_pie
)
from visualization.posterior_plots import plot_trace, plot_posterior
from optimization.budget_allocator import (
    optimize_budget_allocation, calculate_marginal_roi, compare_scenarios
)
from visualization.optimization_plots import plot_budget_comparison, plot_budget_scenarios

# Configuration de la page
st.set_page_config(
    page_title="MMM Bayésien - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    
    st.markdown("### 📂 Source de données")

    data_source = st.radio(
        "Choisissez la source:",
        ["📊 Données exemple", "📁 Uploader un fichier CSV"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "📁 Uploader un fichier CSV":
        # Bouton de téléchargement du template
        try:
            with open('template_dataset.csv', 'rb') as f:
                st.download_button(
                    label="📥 Télécharger le template CSV",
                    data=f,
                    file_name="template_mmm_dataset.csv",
                    mime="text/csv",
                    help="Téléchargez ce fichier exemple pour voir le format attendu"
                )
        except:
            pass

        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV",
            type=['csv'],
            help="Le fichier doit contenir: 'date', 'sales', et les colonnes media (ex: 'media_1_spend', 'media_2_spend', etc.)"
        )

        if uploaded_file is not None:
            st.success("✅ Fichier chargé!")
        else:
            st.info("💡 En attente d'un fichier...")

    st.markdown("---")
    st.markdown("### 📋 Navigation")

    page = st.radio(
        "Choisissez une section:",
        [
            "🏠 Accueil",
            "📊 Données & EDA",
            "🔬 Transformations",
            "🧠 Modèle & Diagnostics",
            "📈 Attribution & Performance",
            "💰 Optimisation Budgétaire"
        ],
        label_visibility="collapsed"
    )

# Fonction pour charger les données avec cache
@st.cache_data
def load_data_default():
    """Charge ou génère les données par défaut"""
    try:
        df = load_csv_data('data/raw/sample_data.csv')
    except:
        df = create_sample_data(n_periods=104, n_media_channels=3, seed=42)
    return df

@st.cache_data
def load_data_from_upload(uploaded_file):
    """Charge les données depuis un fichier uploadé"""
    df = pd.read_csv(uploaded_file)
    # Convertir la colonne date si elle existe
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def detect_media_columns(df):
    """Détecte automatiquement les colonnes media"""
    # Chercher les colonnes qui contiennent 'media', 'spend', 'tv', 'facebook', etc.
    potential_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['media', 'spend', 'tv', 'facebook', 'google', 'radio', 'digital']):
            if col != 'sales' and col != 'date':
                potential_cols.append(col)

    # Si pas trouvé, chercher toutes les colonnes numériques sauf 'sales' et 'date'
    if not potential_cols:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64'] and col not in ['sales', 'date']:
                potential_cols.append(col)

    return potential_cols

@st.cache_resource
def train_model(X_media, y, alpha, k, s):
    """Entraîne le modèle MMM (cached)"""
    mmm = BayesianMMM(use_adstock=True, use_saturation=True)
    trace = mmm.fit(
        X_media, y,
        alpha=alpha, k=k, s=s,
        draws=1000, tune=1000, chains=2,
        random_seed=42
    )
    return mmm, trace

# Charger les données selon la source choisie
if uploaded_file is not None:
    df = load_data_from_upload(uploaded_file)
    media_cols = detect_media_columns(df)

    # Générer des noms de canaux automatiquement
    if media_cols:
        channel_names = [col.replace('_spend', '').replace('media_', 'Canal ').replace('_', ' ').title()
                        for col in media_cols]
    else:
        st.error("❌ Aucune colonne media détectée dans le fichier. Assurez-vous que le fichier contient des colonnes de dépenses publicitaires.")
        st.stop()

    # Vérifier que 'sales' existe
    if 'sales' not in df.columns:
        st.error("❌ La colonne 'sales' est manquante dans le fichier.")
        st.stop()
else:
    df = load_data_default()
    media_cols = ['media_1_spend', 'media_2_spend', 'media_3_spend']
    channel_names = ['TV', 'Facebook', 'Google Ads']

# PAGE 1: ACCUEIL
if page == "🏠 Accueil":
    st.markdown('<h1 class="main-header">📊 Marketing Mix Modeling Bayésien</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Système d\'attribution et d\'optimisation budgétaire pour campagnes marketing multi-canaux</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Objectifs")
        st.markdown("""
        - **Attribution des ventes**: Mesurer l'impact réel de chaque canal
        - **Effets de saturation**: Modéliser les rendements décroissants
        - **Effets d'adstock**: Capturer la persistance temporelle
        - **Optimisation budgétaire**: Recommander l'allocation optimale
        """)

    with col2:
        st.markdown("### 🧠 Concepts clés")
        st.markdown("""
        - **Adstock géométrique**: Persistance de l'effet pub
        - **Saturation de Hill**: Rendements décroissants
        - **Inférence bayésienne**: Modèle probabiliste avec PyMC
        - **MCMC Sampling**: Estimation des paramètres
        """)

    with col3:
        st.markdown("### 🛠️ Stack technique")
        st.markdown("""
        - **PyMC 5.10+**: Inférence bayésienne
        - **ArviZ**: Diagnostics et visualisations
        - **Pandas/NumPy**: Manipulation de données
        - **Streamlit**: Interface interactive
        """)

    st.markdown("---")

    # Afficher la source des données
    if uploaded_file is not None:
        st.success(f"📁 Données chargées depuis: **{uploaded_file.name}**")
    else:
        st.info("📊 Utilisation des données exemple")

    st.markdown("### 📊 Aperçu du dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Périodes", f"{len(df)}")
    col2.metric("Canaux media", len(media_cols))
    col3.metric("Ventes moyennes", f"{df['sales'].mean():.0f}")
    col4.metric("Budget total moyen", f"{df[media_cols].sum(axis=1).mean():.0f}€")

    # Afficher les noms des canaux détectés
    st.markdown("**Canaux détectés:**")
    for i, (col, name) in enumerate(zip(media_cols, channel_names)):
        st.write(f"{i+1}. **{name}** (`{col}`)")

    st.markdown("### 📋 Prévisualisation des données")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.info("👈 Utilisez la barre latérale pour charger vos propres données ou naviguer entre les sections")

# PAGE 2: DONNÉES & EDA
elif page == "📊 Données & EDA":
    st.markdown('<h1 class="main-header">📊 Analyse Exploratoire des Données</h1>', unsafe_allow_html=True)

    # Séries temporelles (toujours visible)
    st.markdown("### 📈 Évolution temporelle")
    fig = plot_time_series(df, ['sales'] + media_cols, title='')
    st.pyplot(fig)

    st.markdown("---")

    # Statistiques en 2 colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Statistiques descriptives")
        st.dataframe(df[['sales'] + media_cols].describe(), use_container_width=True)

        # Corrélations
        st.markdown("### 🔗 Corrélations avec les ventes")
        corr_with_sales = df[['sales'] + media_cols].corr()['sales'].sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['green' if x > 0 else 'red' for x in corr_with_sales[1:]]
        corr_with_sales[1:].plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        ax.set_xlabel('Corrélation', fontsize=12)
        ax.set_ylabel('')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(0, color='black', linewidth=0.8)
        st.pyplot(fig)

    with col2:
        st.markdown("### 💰 Dépenses par canal")

        # Bar chart des totaux
        totals = df[media_cols].sum()
        totals.index = channel_names
        fig, ax = plt.subplots(figsize=(8, 4))
        totals.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Dépenses totales (€)', fontsize=12)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Pie chart de la distribution
        st.markdown("### 📊 Distribution du budget")
        budget_pct = (df[media_cols].sum() / df[media_cols].sum().sum() * 100)
        budget_pct.index = channel_names
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pie(budget_pct, labels=budget_pct.index, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

# PAGE 3: TRANSFORMATIONS
elif page == "🔬 Transformations":
    st.markdown('<h1 class="main-header">🔬 Transformations Adstock & Saturation</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Adstock", "📈 Saturation", "🔄 Combiné"])

    with tab1:
        st.markdown("### Transformation Adstock Géométrique")
        st.latex(r"y_t = x_t + \alpha \cdot y_{t-1}")
        st.markdown("**α (alpha)** : Taux de rétention ∈ [0, 1)")

        col1, col2 = st.columns([1, 2])

        with col1:
            alpha_demo = st.slider("Taux de rétention (α)", 0.0, 0.95, 0.5, 0.05)
            l_max_demo = st.slider("Longueur maximale", 1, 10, 4)

            st.markdown(f"""
            **Interprétation:**
            - α = {alpha_demo:.2f}
            - Effet persiste ≈ {int(1/(1-alpha_demo)) if alpha_demo < 1 else '∞'} périodes
            """)

        with col2:
            # Exemple de dépenses
            spend_example = np.array([100, 80, 60, 40, 20, 10, 5])
            adstocked = geometric_adstock(spend_example, alpha=alpha_demo, l_max=l_max_demo)

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(spend_example))
            ax.plot(x, spend_example, marker='o', label='Dépenses originales', linewidth=2)
            ax.plot(x, adstocked, marker='s', label='Avec adstock', linewidth=2)
            ax.set_xlabel('Période', fontsize=12)
            ax.set_ylabel('Valeur', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    with tab2:
        st.markdown("### Transformation Saturation de Hill")
        st.latex(r"y = \frac{x^s}{k^s + x^s}")
        st.markdown("**k** : Half-saturation (point où effet = 50% du max), **s** : Slope (pente)")

        col1, col2 = st.columns([1, 2])

        with col1:
            k_demo = st.slider("Half-saturation (k)", 10.0, 200.0, 80.0, 10.0)
            s_demo = st.slider("Slope (s)", 0.5, 3.0, 1.0, 0.1)

        with col2:
            spend_range = np.linspace(0, 300, 100)
            saturated = hill_saturation(spend_range, half_saturation=k_demo, slope=s_demo)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(spend_range, saturated, linewidth=2, color='#ff7f0e')
            ax.axvline(k_demo, color='red', linestyle='--', alpha=0.5, label=f'k={k_demo}')
            ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='50% saturation')
            ax.set_xlabel('Dépenses publicitaires', fontsize=12)
            ax.set_ylabel('Effet saturé (0-1)', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    with tab3:
        st.markdown("### Transformation combinée : Adstock + Saturation")

        col1, col2 = st.columns([1, 2])

        with col1:
            alpha_comb = st.slider("Alpha", 0.0, 0.95, 0.5, 0.05, key='alpha_comb')
            k_comb = st.slider("Half-saturation", 10.0, 200.0, 80.0, 10.0, key='k_comb')
            s_comb = st.slider("Slope", 0.5, 3.0, 1.0, 0.1, key='s_comb')

        with col2:
            spend_example = np.array([100, 80, 60, 40, 20, 10, 5])

            # Étapes
            step1_adstock = geometric_adstock(spend_example, alpha=alpha_comb, l_max=4)
            step2_saturation = hill_saturation(step1_adstock, half_saturation=k_comb, slope=s_comb)

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(spend_example))
            ax.plot(x, spend_example, marker='o', label='1. Original', linewidth=2)
            ax.plot(x, step1_adstock, marker='s', label='2. + Adstock', linewidth=2)
            ax.plot(x, step2_saturation, marker='^', label='3. + Saturation', linewidth=2)
            ax.set_xlabel('Période', fontsize=12)
            ax.set_ylabel('Valeur', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# PAGE 4: MODÈLE & DIAGNOSTICS
elif page == "🧠 Modèle & Diagnostics":
    st.markdown('<h1 class="main-header">🧠 Modèle MMM Bayésien</h1>', unsafe_allow_html=True)

    with st.spinner('Entraînement du modèle en cours... (peut prendre 2-3 minutes)'):
        X_media = df[media_cols].values
        y = np.log1p(df['sales'].values)

        alpha = np.array([0.5, 0.6, 0.4])
        k = X_media.mean(axis=0)
        s = np.array([1.0, 1.0, 1.0])

        mmm, trace = train_model(X_media, y, alpha, k, s)

    st.success("✅ Modèle entraîné avec succès!")

    # Diagnostics de convergence
    st.markdown("### 📊 Diagnostics de convergence MCMC")
    report = check_convergence(trace)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Convergence", "✅ OUI" if report['converged'] else "❌ NON")
    col2.metric("R-hat max", f"{report['r_hat_max']:.4f}")
    col3.metric("ESS bulk min", f"{report['ess_bulk_min']:.0f}")
    col4.metric("Divergences", report['n_divergences'])

    if report['converged']:
        st.markdown('<div class="success-box">✅ Le modèle a bien convergé. Les résultats sont fiables.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ Problème de convergence détecté. Augmentez le nombre d\'itérations.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Paramètres estimés
    st.markdown("### 📈 Paramètres estimés")

    summary = mmm.summary()
    st.dataframe(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']].round(4), use_container_width=True)

    st.markdown("---")

    # Paramètres de transformation en 2 colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚙️ Adstock (α) - Taux de rétention")
        for i, name in enumerate(channel_names):
            persistence = int(1/(1-alpha[i])) if alpha[i] < 1 else np.inf
            st.write(f"• **{name}**: α = {alpha[i]:.2f} → effet persiste ≈{persistence} périodes")

    with col2:
        st.markdown("### ⚙️ Saturation (k) - Half-saturation")
        for i, name in enumerate(channel_names):
            st.write(f"• **{name}**: k = {k[i]:.0f}€")

# PAGE 5: ATTRIBUTION & PERFORMANCE
elif page == "📈 Attribution & Performance":
    st.markdown('<h1 class="main-header">📈 Résultats & Attribution par Canal</h1>', unsafe_allow_html=True)

    with st.spinner('Calcul des contributions...'):
        X_media = df[media_cols].values
        y = np.log1p(df['sales'].values)

        alpha = np.array([0.5, 0.6, 0.4])
        k = X_media.mean(axis=0)
        s = np.array([1.0, 1.0, 1.0])

        mmm, trace = train_model(X_media, y, alpha, k, s)
        X_transformed = mmm.apply_transformations(X_media, alpha, k, s)
        contributions = calculate_channel_contributions(trace, X_transformed, channel_names)

        # Performance
        y_pred = mmm.predict(X_media)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))

    # Performance du modèle en haut
    st.markdown("### 🎯 Performance du modèle")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("R²", f"{r2:.4f}")

    if r2 > 0.8:
        st.success(f"✅ Excellent fit! Le modèle explique {r2*100:.1f}% de la variance des ventes.")
    elif r2 > 0.6:
        st.info(f"ℹ️ Bon fit. Le modèle explique {r2*100:.1f}% de la variance des ventes.")
    else:
        st.warning(f"⚠️ Le modèle pourrait être amélioré (R²={r2:.3f}).")

    st.markdown("---")

    tab1, tab2 = st.tabs(["💰 Contributions par canal", "📊 Insights & Recommandations"])

    with tab1:
        st.markdown("### Contribution de chaque canal aux ventes")

        col1, col2, col3 = st.columns(3)
        for i, row in contributions.iterrows():
            with [col1, col2, col3][i]:
                st.metric(row['channel'], f"{row['pct_total']:.1f}%")

        st.dataframe(contributions[['channel', 'total_contribution', 'pct_total', 'coefficient']].round(3), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Barplot des contributions")
            fig = plot_contribution_bars(contributions)
            st.pyplot(fig)

        with col2:
            st.markdown("### Pie chart des contributions")
            fig = plot_contribution_pie(contributions)
            st.pyplot(fig)

    with tab2:
        st.markdown("### 📊 Insights clés")

        best_channel = contributions.iloc[0]
        worst_channel = contributions.iloc[-1]

        st.markdown(f"""
        <div class="success-box">
        <h4>✅ Canal le plus performant: {best_channel['channel']}</h4>
        <ul>
            <li>Contribution: {best_channel['pct_total']:.1f}% des ventes</li>
            <li>Coefficient: {best_channel['coefficient']:.4f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warning-box">
        <h4>⚠️ Canal le moins performant: {worst_channel['channel']}</h4>
        <ul>
            <li>Contribution: {worst_channel['pct_total']:.1f}% des ventes</li>
            <li>Coefficient: {worst_channel['coefficient']:.4f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 💡 Recommandations initiales")
        st.markdown(f"""
        1. **Prioriser {best_channel['channel']}**: Ce canal génère le plus de ventes
        2. **Analyser {worst_channel['channel']}**: Identifier pourquoi la contribution est faible
        3. **Optimiser l'allocation**: Voir section "Optimisation Budgétaire" pour des recommandations détaillées
        """)

# PAGE 6: OPTIMISATION BUDGÉTAIRE
elif page == "💰 Optimisation Budgétaire":
    st.markdown('<h1 class="main-header">💰 Optimisation Budgétaire</h1>', unsafe_allow_html=True)

    with st.spinner('Calcul de l\'allocation optimale...'):
        X_media = df[media_cols].values
        y = np.log1p(df['sales'].values)

        alpha = np.array([0.5, 0.6, 0.4])
        k = X_media.mean(axis=0)
        s = np.array([1.0, 1.0, 1.0])

        mmm, trace = train_model(X_media, y, alpha, k, s)
        coefficients = trace.posterior['beta_media'].mean(dim=['chain', 'draw']).values

        current_spend = X_media.mean(axis=0)
        total_budget = current_spend.sum()

        result = optimize_budget_allocation(
            total_budget=total_budget,
            coefficients=coefficients,
            alpha=alpha, k=k, s=s,
            current_spend=current_spend
        )

        optimal_spend = result['optimal_spend']

    tab1, tab2, tab3 = st.tabs(["🎯 Allocation optimale", "📊 Scénarios", "💡 Recommandations"])

    with tab1:
        st.markdown("### Allocation budgétaire optimale")

        col1, col2, col3 = st.columns(3)
        col1.metric("Budget total", f"{total_budget:.0f}€")
        col2.metric("Ventes actuelles", f"{result['current_sales']:.2f}")
        col3.metric("Amélioration", f"+{result['improvement']:.2f}%", delta=f"{result['improvement']:.2f}%")

        st.markdown("### Comparaison Actuel vs Optimal")

        comparison_df = pd.DataFrame({
            'Canal': channel_names,
            'Actuel (€)': current_spend,
            'Optimal (€)': optimal_spend,
            'Changement (€)': optimal_spend - current_spend,
            'Changement (%)': ((optimal_spend - current_spend) / current_spend * 100)
        })

        st.dataframe(comparison_df.round(2), use_container_width=True)

        fig = plot_budget_comparison(current_spend, optimal_spend, channel_names)
        st.pyplot(fig)

    with tab2:
        st.markdown("### Analyse de scénarios budgétaires")

        budget_multiplier = st.slider(
            "Multiplicateur de budget",
            0.5, 2.0, 1.0, 0.1,
            help="1.0 = budget actuel, 1.5 = +50%, 0.8 = -20%"
        )

        budgets_to_test = [total_budget * m for m in [0.5, 0.75, 1.0, 1.25, 1.5, budget_multiplier]]
        budgets_to_test = sorted(list(set(budgets_to_test)))

        scenarios = compare_scenarios(budgets_to_test, coefficients, alpha, k, s, channel_names)

        st.dataframe(scenarios.round(2), use_container_width=True)

        fig = plot_budget_scenarios(scenarios)
        st.pyplot(fig)

        # Insight sur le budget sélectionné
        if budget_multiplier != 1.0:
            new_budget = total_budget * budget_multiplier
            change_pct = (budget_multiplier - 1) * 100
            st.info(f"💡 Avec un budget de {new_budget:.0f}€ ({change_pct:+.0f}%), vous pourriez optimiser davantage votre allocation.")

    with tab3:
        st.markdown("### 📋 Recommandations stratégiques")

        increases = []
        decreases = []

        for i, name in enumerate(channel_names):
            change = optimal_spend[i] - current_spend[i]
            pct = (change / current_spend[i]) * 100

            if change > 0:
                increases.append((name, change, pct))
            elif change < 0:
                decreases.append((name, change, pct))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Canaux à augmenter")
            if increases:
                for name, change, pct in sorted(increases, key=lambda x: x[1], reverse=True):
                    st.markdown(f"**{name}**: +{change:.0f}€ ({pct:+.1f}%)")
            else:
                st.markdown("*Aucun canal à augmenter*")

        with col2:
            st.markdown("### ⚠️ Canaux à réduire")
            if decreases:
                for name, change, pct in sorted(decreases, key=lambda x: x[1]):
                    st.markdown(f"**{name}**: {change:.0f}€ ({pct:.1f}%)")
            else:
                st.markdown("*Aucun canal à réduire*")

        st.markdown("---")
        st.markdown("### 🎯 Plan d'action")
        st.markdown(f"""
        1. **Réallocation immédiate**: Implémenter les changements recommandés ci-dessus
        2. **Monitoring**: Suivre les performances sur 4-6 semaines
        3. **Ajustement progressif**: Adapter si nécessaire
        4. **Ré-entraînement**: Mettre à jour le modèle avec les nouvelles données

        **Impact attendu**: +{result['improvement']:.2f}% de ventes sans budget supplémentaire
        """)

        st.success(f"💡 En optimisant votre allocation, vous pourriez gagner {result['improvement']:.2f}% de ventes!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Marketing Mix Modeling Bayésien</strong></p>
    <p>Ivan - EPF Engineering School - MSMIN5IN43</p>
    <p>Projet réalisé avec PyMC, ArviZ, Streamlit</p>
</div>
""", unsafe_allow_html=True)
