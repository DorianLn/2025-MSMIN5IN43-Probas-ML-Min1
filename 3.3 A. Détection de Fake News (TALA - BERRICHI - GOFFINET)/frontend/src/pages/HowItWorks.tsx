import { motion } from "framer-motion";
import { ArrowLeft, Brain, FileText, TrendingUp, Globe, CheckCircle, AlertTriangle, BarChart3, ShieldCheck, Zap, Lock } from "lucide-react";
import { Link } from "react-router-dom";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const HowItWorks = () => {
  const steps = [
    {
      number: "01",
      title: "Soumission du contenu",
      description:
        "Vous collez le texte suspect. FactGuard détecte automatiquement la langue pour attribuer le meilleur processeur IA.",
      icon: FileText,
    },
    {
      number: "02",
      title: "Analyse CamemBERT / BERT",
      description:
        "Le texte est décomposé en vecteurs mathématiques (embeddings). Notre IA analyse la structure sémantique profonde.",
      icon: Brain,
    },
    {
      number: "03",
      title: "Évaluation Stylistique",
      description:
        "Détection des patterns de manipulation : sensationnalisme, clickbait, et biais cognitifs automatisés.",
      icon: BarChart3,
    },
    {
      number: "04",
      title: "Verdict & Confiance",
      description:
        "Génération d'un score de fiabilité basé sur la probabilité statistique que le texte soit une fake news.",
      icon: CheckCircle,
    },
  ];

  const factors = [
    {
      icon: ShieldCheck,
      title: "Analyse Sémantique",
      description: "Étude de la structure du discours",
      details: [
        "Détection du lexique hyperbolique",
        "Analyse de la cohérence argumentative",
        "Identification des biais de confirmation",
        "Repérage des structures de clickbait",
        "Évaluation de la neutralité du ton",
      ],
    },
    {
      icon: TrendingUp,
      title: "Marqueurs Linguistiques",
      description: "Signaux faibles de désinformation",
      details: [
        "Usage excessif de la ponctuation (!?) ",
        "Analyse des majuscules abusives",
        "Détection de l'urgence émotionnelle",
        "Repérage des généralisations hâtives",
        "Calcul du score de sentiment",
      ],
    },
    {
      icon: Globe,
      title: "Spécificité Locale",
      description: "Adaptation selon la langue",
      details: [
        "Modèle CamemBERT dédié au Français",
        "Modèle RoBERTa dédié à l'Anglais",
        "Gestion des expressions idiomatiques",
        "Analyse du contexte culturel médiatique",
        "Précision accrue sur la presse locale",
      ],
    },
  ];

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Header />

      <main className="flex-1">
        {/* Hero */}
        <section className="border-b bg-muted/30 py-16 md:py-24">
          <div className="container">
            <Link
              to="/"
              className="mb-8 inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <ArrowLeft className="h-4 w-4" />
              Retour à l'accueil
            </Link>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="font-display text-4xl font-bold text-foreground md:text-5xl">
                La science derrière <span className="text-accent italic">FactGuard</span>
              </h1>
              <p className="mt-6 text-lg text-muted-foreground leading-relaxed">
                Notre technologie repose sur le <strong>Natural Language Processing (NLP)</strong> de pointe. 
                Nous ne nous contentons pas de lire le texte ; nous analysons son ADN linguistique.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Process Steps */}
        <section className="py-16 md:py-24">
          <div className="container">
            <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
              {steps.map((step, index) => (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="relative group"
                >
                  <div className="rounded-2xl border bg-card p-6 transition-all group-hover:border-accent/50 group-hover:shadow-md">
                    <span className="font-display text-4xl font-bold text-accent/10">{step.number}</span>
                    <div className="mt-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white transition-colors">
                      <step.icon className="h-6 w-6" />
                    </div>
                    <h3 className="mt-4 font-display text-xl font-semibold text-foreground italic">{step.title}</h3>
                    <p className="mt-2 text-sm text-muted-foreground">{step.description}</p>
                  </div>
                  {index < steps.length - 1 && (
                    <div className="absolute right-0 top-1/2 hidden h-px w-8 -translate-y-1/2 translate-x-full bg-border lg:block" />
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Comparatif Technique */}
        <section className="py-16 bg-muted/20 border-y">
          <div className="container max-w-4xl">
            <div className="mb-12 text-center">
              <h2 className="font-display text-3xl font-bold">Optimisation Multi-Modèles</h2>
              <p className="mt-3 text-muted-foreground italic">Pourquoi nous séparons le traitement FR et EN</p>
            </div>
            
            <div className="overflow-hidden rounded-2xl border bg-card shadow-sm">
              <table className="w-full text-left text-sm">
                <thead className="bg-muted/50 text-muted-foreground border-b">
                  <tr>
                    <th className="px-6 py-4 font-semibold uppercase tracking-wider text-xs">Caractéristique</th>
                    <th className="px-6 py-4 font-semibold text-accent">CamemBERT (FR)</th>
                    <th className="px-6 py-4 font-semibold">RoBERTa / BERT (EN)</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  <tr>
                    <td className="px-6 py-4 font-medium">Dataset d'origine</td>
                    <td className="px-6 py-4 italic text-xs">OSCAR (Corpus Français)</td>
                    <td className="px-6 py-4 italic text-xs">CommonCrawl / Wikipedia</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 font-medium">Précision Labo (F1)</td>
                    <td className="px-6 py-4 font-bold text-emerald-500 text-base">94%</td>
                    <td className="px-6 py-4 font-bold text-emerald-400 text-base">91%</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 font-medium">Spécialité</td>
                    <td className="px-6 py-4">Nuances sémantiques FR</td>
                    <td className="px-6 py-4">Structure factuelle globale</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Analysis Factors */}
        <section className="py-16 md:py-24">
          <div className="container">
            <div className="grid gap-8 lg:grid-cols-3">
              {factors.map((factor, index) => (
                <motion.div
                  key={factor.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="rounded-2xl border bg-card p-8 hover:shadow-lg transition-shadow"
                >
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent text-accent-foreground shadow-lg shadow-accent/20">
                    <factor.icon className="h-7 w-7" />
                  </div>
                  <h3 className="mt-6 font-display text-2xl font-semibold text-foreground italic">{factor.title}</h3>
                  <p className="mt-2 text-muted-foreground">{factor.description}</p>

                  <ul className="mt-6 space-y-3">
                    {factor.details.map((detail, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm group">
                        <CheckCircle className="mt-0.5 h-4 w-4 shrink-0 text-accent opacity-60" />
                        <span className="text-muted-foreground">{detail}</span>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Data Privacy Section */}
        <section className="py-16 md:py-24 border-t bg-muted/10">
          <div className="container max-w-4xl">
            <div className="flex flex-col md:flex-row items-center gap-12">
              <div className="flex-1">
                <div className="inline-flex items-center gap-2 rounded-full bg-accent/10 px-4 py-1 text-xs font-bold text-accent mb-6 uppercase tracking-widest">
                  <Lock className="h-4 w-4" /> Confidentialité
                </div>
                <h2 className="font-display text-3xl font-bold text-foreground mb-6">
                  Vos données vous <span className="text-accent italic">appartiennent</span>
                </h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    Chez FactGuard, nous analysons le contenu, pas l'utilisateur. Notre architecture est conçue selon le principe du <strong>Privacy by Design</strong>.
                  </p>
                  <ul className="space-y-3">
                    <li className="flex gap-3 text-sm italic">
                      <Zap className="h-4 w-4 text-accent shrink-0" />
                      <strong>Zéro Stockage:</strong> Les textes soumis sont analysés en mémoire vive et supprimés immédiatement après génération du score.
                    </li>
                    <li className="flex gap-3 text-sm italic">
                      <Zap className="h-4 w-4 text-accent shrink-0" />
                      <strong>Anonymat total :</strong> Aucune donnée personnelle n'est extraite de vos requêtes pour l'entraînement de nos modèles.
                    </li>
                    <li className="flex gap-3 text-sm italic">
                      <Zap className="h-4 w-4 text-accent shrink-0" />
                      <strong>Conformité RGPD :</strong> Nos serveurs d'inférence sont situés en Europe pour garantir une protection juridique maximale.
                    </li>
                  </ul>
                </div>
              </div>
              <div className="w-full md:w-72 aspect-square rounded-3xl bg-gradient-to-br from-accent/20 to-primary/10 border-2 border-dashed border-accent/30 flex items-center justify-center p-8 text-center">
                <div className="space-y-2">
                  <Lock className="h-12 w-12 text-accent mx-auto mb-4" />
                  <p className="font-display font-bold text-foreground">Analyse Éphémère</p>
                  <p className="text-xs text-muted-foreground">Chaque analyse est unique et ne laisse aucune trace numérique sur nos serveurs.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Training Info */}
        <section className="py-16 md:py-24 bg-primary text-primary-foreground">
          <div className="container text-center">
            <h2 className="font-display text-3xl font-bold">Un entraînement sur-mesure</h2>
            <p className="mt-4 mx-auto max-w-2xl opacity-90 leading-relaxed">
              FactGuard n'est pas une simple API de traduction. Nos modèles ont été "fine-tunés" sur plus de 
              <strong> 500 000 articles</strong> étiquetés par des organismes de fact-checking officiels.
            </p>
            <div className="mt-12 grid gap-6 sm:grid-cols-3 max-w-3xl mx-auto">
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <p className="font-display text-4xl font-bold text-accent italic">500K+</p>
                <p className="text-xs uppercase tracking-widest mt-2 opacity-80 font-semibold">Articles d'analyse</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <p className="font-display text-4xl font-bold text-accent italic">94%</p>
                <p className="text-xs uppercase tracking-widest mt-2 opacity-80 font-semibold">F1-Score (FR)</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
                <p className="font-display text-4xl font-bold text-accent italic">&lt;2s</p>
                <p className="text-xs uppercase tracking-widest mt-2 opacity-80 font-semibold">Temps de réponse</p>
              </div>
            </div>
          </div>
        </section>

        {/* Warning */}
        <section className="py-16">
          <div className="container max-w-3xl">
            <div className="flex items-start gap-4 rounded-2xl border border-warning/30 bg-warning-soft/10 p-8 shadow-sm">
              <AlertTriangle className="h-6 w-6 shrink-0 text-warning" />
              <div>
                <h4 className="font-bold text-foreground italic uppercase text-sm tracking-wider">Avertissement Éthique</h4>
                <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
                  L'IA est un outil de signalement, pas un juge de vérité absolue. FactGuard analyse les 
                  <strong> structures linguistiques</strong> associées à la désinformation. 
                  L'esprit critique humain reste votre rempart final le plus puissant.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default HowItWorks;