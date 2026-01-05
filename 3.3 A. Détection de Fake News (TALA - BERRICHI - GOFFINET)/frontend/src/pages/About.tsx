import { motion } from "framer-motion";
import { ArrowLeft, Shield, Users, Target, Award, Lightbulb, Globe2, Cpu } from "lucide-react";
import { Link } from "react-router-dom";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const About = () => {
  const values = [
    {
      icon: Cpu,
      title: "Précision Technique",
      description: "Utilisation de modèles Transformers (CamemBERT, BERT, RoBERTa) pour une analyse fine du langage.",
    },
    {
      icon: Globe2,
      title: "Multilinguisme",
      description: "Une approche dédiée pour traiter les spécificités linguistiques du français et de l'anglais.",
    },
    {
      icon: Shield,
      title: "Objectivité IA",
      description: "Des algorithmes entraînés pour détecter les patterns de désinformation sans biais humain.",
    },
    {
      icon: Lightbulb,
      title: "Innovation Open",
      description: "Exploitation des dernières recherches en NLP pour rester à la pointe de la détection.",
    },
  ];

  const team = [
    {
      name: "Développeur IA",
      role: "Architecte NLP & Backend",
      bio: "Expert en fine-tuning des modèles Transformers. Gère l'architecture du serveur et l'inférence des modèles de détection.",
    },
    {
      name: "Développeur Frontend",
      role: "Architecte UI/UX",
      bio: "Responsable de l'interface utilisateur sous React. Spécialiste de la visualisation de données et de l'expérience utilisateur.",
    },
    {
      name: "Data Scientist",
      role: "Analyste de Données",
      bio: "En charge de la collecte et du nettoyage des datasets d'entraînement ainsi que de l'évaluation de la précision statistique.",
    },
  ];

  const partners = [
    "Hugging Face",
    "PyTorch",
    "Tailwind CSS",
    "React Router",
    "Lucide Icons",
    "Framer Motion",
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
              className="max-w-3xl"
            >
              <h1 className="font-display text-4xl font-bold text-foreground md:text-5xl">
                À propos de <span className="text-accent">FactGuard</span>
              </h1>
              <p className="mt-6 text-lg text-muted-foreground">
                FactGuard est une plateforme d'analyse hybride née de la nécessité de protéger l'intégrité de l'information à l'ère du numérique. 
              </p>
              <p className="mt-4 text-lg text-muted-foreground">
                Notre technologie repose sur une architecture multi-modèles permettant d'analyser les contenus en français et en anglais avec une précision chirurgicale grâce aux architectures Transformers.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Mission Section */}
        <section className="py-16 md:py-24">
          <div className="container">
            <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
              >
                <div className="inline-flex items-center gap-2 rounded-full bg-accent/10 px-4 py-1.5 text-sm text-accent font-medium">
                  <Target className="h-4 w-4" />
                  Notre mission technique
                </div>
                <h2 className="mt-6 font-display text-3xl font-bold text-foreground">
                  Démocratiser la vérification par l'Intelligence Artificielle
                </h2>
                <div className="mt-6 space-y-4 text-muted-foreground leading-relaxed">
                  <p>
                    La désinformation évolue vite. FactGuard utilise le <strong>Deep Learning</strong> pour comprendre le contexte, le ton et les subtilités d'un texte.
                  </p>
                  <p>
                    En proposant des modèles spécifiques comme <strong>CamemBERT</strong> pour le français et <strong>BERT/RoBERTa</strong> pour l'anglais, nous respectons les nuances sémantiques propres à chaque langue.
                  </p>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="grid grid-cols-1 sm:grid-cols-2 gap-4"
              >
                {values.map((value) => (
                  <div
                    key={value.title}
                    className="rounded-xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
                      <value.icon className="h-5 w-5" />
                    </div>
                    <h3 className="mt-4 font-display text-lg font-semibold text-foreground">{value.title}</h3>
                    <p className="mt-2 text-sm text-muted-foreground leading-tight">{value.description}</p>
                  </div>
                ))}
              </motion.div>
            </div>
          </div>
        </section>

        {/* Team - Configuration pour 3 personnes */}
        <section className="border-t bg-muted/30 py-16 md:py-24">
          <div className="container">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="mb-12 text-center"
            >
              <h2 className="font-display text-3xl font-bold text-foreground">Notre équipe</h2>
              <p className="mt-3 text-muted-foreground">
                Trois profils complémentaires unis pour une information plus fiable.
              </p>
            </motion.div>

            <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3 max-w-5xl mx-auto">
              {team.map((member, index) => (
                <motion.div
                  key={member.name}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="rounded-xl border bg-card p-8 text-center hover:border-accent/30 transition-all hover:shadow-md"
                >
                  <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-accent/10 text-accent mb-4">
                    <Users className="h-10 w-10" />
                  </div>
                  <h3 className="font-display text-xl font-bold text-foreground">{member.name}</h3>
                  <p className="text-xs font-bold uppercase tracking-wider text-accent mt-1">{member.role}</p>
                  <p className="mt-4 text-sm text-muted-foreground leading-relaxed italic italic">
                    "{member.bio}"
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Tech Stack */}
        <section className="py-16 md:py-24">
          <div className="container">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center"
            >
              <div className="inline-flex items-center gap-2 rounded-full bg-accent/10 px-4 py-1.5 text-sm text-accent font-medium">
                <Award className="h-4 w-4" />
                Technologies utilisées
              </div>
              <h2 className="mt-6 font-display text-3xl font-bold text-foreground">
                Propulsé par les standards de l'industrie
              </h2>
              
              <div className="mt-12 flex flex-wrap items-center justify-center gap-4">
                {partners.map((partner) => (
                  <div
                    key={partner}
                    className="rounded-lg border bg-card px-8 py-4 text-sm font-bold text-muted-foreground hover:text-accent hover:border-accent/50 transition-all cursor-default"
                  >
                    {partner}
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </section>

        {/* Contact CTA */}
        <section className="border-t bg-primary py-16 text-primary-foreground">
          <div className="container">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="mx-auto max-w-2xl text-center"
            >
              <h2 className="font-display text-3xl font-bold">Plus d'informations sur les modèles ?</h2>
              <p className="mt-4 text-primary-foreground/80">
                Vous êtes chercheur ou développeur ? Contactez-nous pour en savoir plus sur l'architecture de nos modèles.
              </p>
              <a
                href="mailto:contact@factguard.app"
                className="mt-8 inline-flex h-12 items-center justify-center rounded-lg bg-primary-foreground px-8 font-bold text-primary transition-all hover:scale-105 active:scale-95 shadow-lg"
              >
                Nous contacter
              </a>
            </motion.div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default About;