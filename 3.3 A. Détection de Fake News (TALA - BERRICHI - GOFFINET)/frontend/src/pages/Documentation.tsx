import { motion } from "framer-motion";
import { ArrowLeft, Book, Code, Zap, Shield, AlertCircle, CheckCircle, Copy, ExternalLink, Globe2, Brain } from "lucide-react";
import { Link } from "react-router-dom";
import { useState } from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const Documentation = () => {
  const [activeSection, setActiveSection] = useState("getting-started");

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copié dans le presse-papiers !");
  };

  const sections = [
    { id: "getting-started", label: "Démarrage rapide", icon: Zap },
    { id: "models", label: "Nos Modèles IA", icon: Brain },
    { id: "api", label: "API Reference", icon: Code },
    { id: "interpretation", label: "Interpréter les résultats", icon: Book },
    { id: "faq", label: "FAQ", icon: AlertCircle },
  ];

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Header />

      <main className="flex-1">
        {/* Hero */}
        <section className="border-b bg-muted/30 py-12 md:py-16">
          <div className="container">
            <Link
              to="/"
              className="mb-6 inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <ArrowLeft className="h-4 w-4" />
              Retour à l'accueil
            </Link>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="font-display text-4xl font-bold text-foreground md:text-5xl">Documentation de <span className="text-accent text-4xl md:text-5xl">FactGuard</span></h1>
              <p className="mt-4 max-w-2xl text-lg text-muted-foreground">
                Comprendre le fonctionnement technique de FactGuard et intégrer nos capacités d'analyse IA.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Content */}
        <section className="py-12">
          <div className="container">
            <div className="grid gap-12 lg:grid-cols-[250px_1fr]">
              {/* Sidebar */}
              <aside className="hidden lg:block">
                <nav className="sticky top-24 space-y-1">
                  {sections.map((section) => (
                    <button
                      key={section.id}
                      onClick={() => {
                        setActiveSection(section.id);
                        document.getElementById(section.id)?.scrollIntoView({ behavior: 'smooth' });
                      }}
                      className={`flex w-full items-center gap-3 rounded-lg px-4 py-2.5 text-left text-sm transition-colors ${
                        activeSection === section.id
                          ? "bg-accent text-accent-foreground shadow-sm"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <section.icon className="h-4 w-4" />
                      {section.label}
                    </button>
                  ))}
                </nav>
              </aside>

              {/* Main Content */}
              <div className="max-w-3xl">
                {/* Getting Started */}
                <motion.section
                  id="getting-started"
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  className="mb-16 scroll-mt-24"
                >
                  <h2 className="font-display text-2xl font-bold text-foreground">Démarrage rapide</h2>
                  <div className="mt-6 space-y-6 text-muted-foreground">
                    <p>
                      FactGuard analyse la fiabilité d'un texte en utilisant des modèles de traitement du langage naturel (NLP) spécialisés.
                    </p>

                    <div className="space-y-4">
                      <h3 className="font-display text-lg font-semibold text-foreground ">1. Saisie du texte</h3>
                      <p>
                        Collez le corps de l'article ou le contenu suspect dans le champ d'analyse. Pour des résultats optimaux, le texte doit contenir au moins <strong className="text-foreground">200 caractères</strong>.
                      </p>
                    </div>

                    <div className="space-y-4">
                      <h3 className="font-display text-lg font-semibold text-foreground ">2. Sélection du modèle</h3>
                      <p>
                        FactGuard détecte automatiquement la langue. Il utilisera <strong className="text-foreground">CamemBERT</strong> pour le français et <strong className="text-foreground">RoBERTa/BERT</strong> pour l'anglais afin de garantir une analyse sémantique précise.
                      </p>
                    </div>

                    <div className="space-y-4">
                      <h3 className="font-display text-lg font-semibold text-foreground ">3. Rapport d'analyse</h3>
                      <p>
                        Le système génère un score de probabilité et identifie les segments de texte potentiellement trompeurs ou excessivement sensationnalistes.
                      </p>
                    </div>
                  </div>
                </motion.section>

                {/* Models Section - Ajouté pour ton projet */}
                <motion.section id="models" className="mb-16 scroll-mt-24">
                  <h2 className="font-display text-2xl font-bold text-foreground">Nos Modèles IA</h2>
                  <div className="mt-6 grid gap-4 sm:grid-cols-2">
                    <div className="rounded-xl border bg-card p-5">
                      <div className="flex items-center gap-2 text-accent mb-3">
                        <Globe2 className="h-5 w-5" />
                        <span className="font-bold">CamemBERT (FR)</span>
                      </div>
                      <p className="text-sm text-muted-foreground">Optimisé pour la langue française, capable de détecter les nuances et les sarcasmes propres au discours médiatique francophone.</p>
                    </div>
                    <div className="rounded-xl border bg-card p-5">
                      <div className="flex items-center gap-2 text-blue-500 mb-3">
                        <Brain className="h-5 w-5" />
                        <span className="font-bold">RoBERTa / BERT (EN)</span>
                      </div>
                      <p className="text-sm text-muted-foreground">Standards mondiaux du NLP, entraînés sur des milliards de mots pour une compréhension profonde des structures syntaxiques anglaises.</p>
                    </div>
                  </div>
                </motion.section>

                {/* Comparatif des Modèles */}
                <motion.section 
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  className="mb-16"
                >
                  <h3 className="font-display text-lg font-semibold text-foreground mb-6 italic">Comparatif des performances NLP</h3>
                  <div className="overflow-hidden rounded-xl border bg-card">
                    <table className="w-full text-left text-sm">
                      <thead className="bg-muted/50 text-muted-foreground">
                        <tr>
                          <th className="px-4 py-3 font-semibold">Caractéristique</th>
                          <th className="px-4 py-3 font-semibold text-accent">CamemBERT</th>
                          <th className="px-4 py-3 font-semibold">BERT / RoBERTa</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y">
                        <tr>
                          <td className="px-4 py-3 font-medium">Langue cible</td>
                          <td className="px-4 py-3">Français (natif)</td>
                          <td className="px-4 py-3">Anglais / Multilingue</td>
                        </tr>
                        <tr>
                          <td className="px-4 py-3 font-medium">Précision (F1-Score)</td>
                          <td className="px-4 py-3 text-emerald-500 font-bold">~94%</td>
                          <td className="px-4 py-3">~91%</td>
                        </tr>
                        <tr>
                          <td className="px-4 py-3 font-medium">Contexte média</td>
                          <td className="px-4 py-3 italic">Articles FR, Presse locale</td>
                          <td className="px-4 py-3 italic">Reuters, AP, Tweets EN</td>
                        </tr>
                        <tr>
                          <td className="px-4 py-3 font-medium">Spécificité</td>
                          <td className="px-4 py-3">Gère les nuances de la langue de Molière</td>
                          <td className="px-4 py-3">Idéal pour le fact-checking global</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-3 text-xs text-muted-foreground">
                    * Résultats basés sur nos tests internes de classification binaire (Fiable vs Trompeur).
                  </p>
                </motion.section>

                {/* API Reference */}
                <motion.section id="api" className="mb-16 scroll-mt-24">
                  <h2 className="font-display text-2xl font-bold text-foreground">API Reference</h2>
                  <div className="mt-6 space-y-6">
                    <p className="text-muted-foreground">
                      Endpoint pour intégrer l'analyse automatique dans vos flux de modération.
                    </p>

                    <div className="rounded-xl border bg-card overflow-hidden">
                      <div className="flex items-center justify-between bg-muted/50 px-6 py-3 border-b">
                        <span className="font-mono text-xs font-bold uppercase tracking-wider">Requête POST</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyToClipboard("https://api.factguard.app/v1/predict")}
                          className="h-8 px-2"
                        >
                          <Copy className="h-3.5 w-3.5" />
                        </Button>
                      </div>

                      <div className="p-6">
                        <div className="flex items-center gap-2 mb-4">
                           <span className="rounded bg-accent/20 px-2 py-1 text-xs font-bold text-accent">POST</span>
                           <code className="text-sm font-mono italic">/api/v1/predict</code>
                        </div>

                        <div className="rounded-lg bg-slate-950 p-4 text-slate-200">
                          <pre className="overflow-x-auto text-xs">
{`{
  "text": "L'article suspect à analyser ici...",
  "model_type": "camembert", // ou "bert", "roberta"
  "detailed_analysis": true
}`}
                          </pre>
                        </div>

                        <h4 className="mt-6 text-sm font-bold text-foreground">Exemple de réponse</h4>
                        <div className="mt-2 rounded-lg bg-slate-950 p-4 text-emerald-400">
                          <pre className="overflow-x-auto text-xs">
{`{
  "label": "Trompeur",
  "probability": 0.892,
  "language_detected": "fr",
  "processing_time": "142ms"
}`}
                          </pre>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.section>

                {/* Interpretation */}
                <motion.section id="interpretation" className="mb-16 scroll-mt-24">
                  <h2 className="font-display text-2xl font-bold text-foreground">Interpréter les résultats</h2>
                  <div className="mt-6 space-y-6">
                    <div className="space-y-4">
                      <div className="flex items-center gap-3">
                        <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
                        <h3 className="font-display font-semibold text-foreground">Score &gt; 0.80 : Haute Confiance</h3>
                      </div>
                      <p className="text-sm text-muted-foreground ml-5 italic">
                        Le modèle a identifié des patterns très clairs de désinformation (lexique hyperbolique, biais cognitifs marqués).
                      </p>
                    </div>

                    <div className="rounded-xl border border-accent/20 bg-accent/5 p-6">
                       <h3 className="font-display text-lg font-semibold text-foreground flex items-center gap-2">
                         <Shield className="h-5 w-5 text-accent" />
                         Facteurs de Style
                       </h3>
                       <ul className="mt-4 grid gap-4 sm:grid-cols-2">
                         <li className="flex gap-2 text-sm text-muted-foreground">
                           <CheckCircle className="h-4 w-4 text-accent shrink-0" />
                           Biais de confirmation identifié
                         </li>
                         <li className="flex gap-2 text-sm text-muted-foreground">
                           <CheckCircle className="h-4 w-4 text-accent shrink-0" />
                           Analyse des majuscules abusives
                         </li>
                         <li className="flex gap-2 text-sm text-muted-foreground">
                           <CheckCircle className="h-4 w-4 text-accent shrink-0" />
                           Score de sentiment négatif
                         </li>
                         <li className="flex gap-2 text-sm text-muted-foreground">
                           <CheckCircle className="h-4 w-4 text-accent shrink-0" />
                           Patterns de "Clickbait"
                         </li>
                       </ul>
                    </div>
                  </div>
                </motion.section>

                {/* FAQ */}
                <motion.section id="faq" className="mb-16 scroll-mt-24">
                  <h2 className="font-display text-2xl font-bold text-foreground">Questions fréquentes</h2>
                  <div className="mt-6 space-y-4">
                    {[
                      {
                        q: "FactGuard remplace-t-il les journalistes ",
                        a: "Absolument pas. C'est un outil d'aide à la décision. Il signale des anomalies statistiques dans le langage, mais l'esprit critique humain reste indispensable.",
                      },
                      {
                        q: "Pourquoi utiliser CamemBERT plutôt que BERT ",
                        a: "CamemBERT a été entraîné spécifiquement sur des corpus de textes français (OSCAR). Il comprend mieux les expressions idiomatiques et la grammaire française que les modèles globaux.",
                      },
                      {
                        q: "L'outil peut-il analyser des images ou vidéos ",
                        a: "Pour le moment, FactGuard se concentre exclusivement sur l'analyse de texte (articles, tweets, posts).",
                      },
                      {
                        q: "Quelle est la précision réelle ",
                        a: "En laboratoire, nos modèles atteignent 92 à 94% de F1-score. En conditions réelles, cela varie selon la longueur du texte fourni.",
                      }
                    ].map((faq, index) => (
                      <div key={index} className="group rounded-xl border bg-card p-6 transition-all hover:border-accent/30">
                        <h3 className="font-semibold text-foreground flex items-center gap-2 italic">
                           {faq.q}<span className="text-accent">?</span>
                        </h3>
                        <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{faq.a}</p>
                      </div>
                    ))}
                  </div>
                </motion.section>

                {/* Help CTA */}
                <div className="rounded-2xl border-2 border-dashed border-accent/20 bg-accent/5 p-8 text-center">
                  <h3 className="font-display text-xl font-semibold text-foreground italic">Une question technique spécifique ?</h3>
                  <p className="mt-2 text-muted-foreground">
                    Notre équipe de 3 développeurs est à votre disposition pour détailler l'architecture NLP.
                  </p>
                  <div className="mt-6 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
                    <a
                      href="mailto:dev@factguard.app"
                      className="inline-flex h-11 items-center gap-2 rounded-xl bg-primary px-8 text-sm font-bold text-primary-foreground transition-all hover:scale-105 active:scale-95 shadow-lg"
                    >
                      Contacter l'équipe Dev
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Documentation;