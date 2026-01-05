# 🛡️ FactGuard - Frontend

Le frontend de **FactGuard** est une application web moderne conçue pour la détection de fake news assistée par IA. L'interface est optimisée pour la clarté, la performance et l'accessibilité.

## 🚀 Technologies utilisées

* **Framework :** [React](https://reactjs.org/) avec [TypeScript](https://www.typescriptlang.org/)
* **Build Tool :** [Vite](https://vitejs.dev/)
* **Styling :** [Tailwind CSS v3](https://tailwindcss.com/)
* **Composants UI :** [shadcn/ui](https://ui.shadcn.com/) (basé sur Radix UI)
* **Animations :** [Lucide React](https://lucide.dev/) pour les icônes et `tailwindcss-animate`
* **Polices :** [Fontsource](https://fontsource.org/) (Crimson Pro & DM Sans)

## 📦 Installation et démarrage

Assurez-vous d'avoir [Node.js](https://nodejs.org/) installé sur votre machine.

1.  **Cloner le dépôt :**
    ```bash
    git clone <url-du-repo>
    cd frontend
    ```

2.  **Installer les dépendances :**
    ```bash
    npm install
    ```

3.  **Lancer le serveur de développement :**
    ```bash
    npm run dev
    ```
    L'application sera disponible sur `http://localhost:5173`.

## 🎨 Design System

L'application utilise une palette de couleurs spécifique définie via des variables CSS dans `src/index.css` :

* **Accent (Aqua/Turquoise) :** Utilisé pour les éléments clés et le branding (Fake News).
* **Primary (Deep Blue) :** Utilisé pour les boutons d'action principaux.
* **Reliable / Unreliable :** Codes couleurs (Vert/Rouge) pour les indicateurs de score de fiabilité.

## 📂 Structure du projet

```text
src/
├── components/     # Composants réutilisables (Boutons, Cards, Navbar)
├── pages/          # Pages de l'application (Index, Analyzer, etc.)
├── lib/            # Configuration utilitaire (utils.ts pour tailwind-merge)
├── hooks/          # Hooks React personnalisés
├── index.css       # Styles globaux et variables CSS
├── main.tsx        # Point d'entrée de l'application
└── tailwind.config.ts # Configuration avancée de Tailwind