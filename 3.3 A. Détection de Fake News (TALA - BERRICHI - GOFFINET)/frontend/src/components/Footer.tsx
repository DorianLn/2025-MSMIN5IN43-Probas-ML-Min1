import { Shield, Github, Twitter } from "lucide-react";
import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="border-t bg-muted/30">
      <div className="container py-12">
        <div className="grid gap-8 md:grid-cols-4">
          <div className="md:col-span-2">
            <Link to="/" className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-accent-foreground">
                <Shield className="h-5 w-5" />
              </div>
              <span className="font-display text-xl font-bold text-foreground">FactGuard</span>
            </Link>
            <p className="mt-4 max-w-md text-sm text-muted-foreground">
            Propulsé par les architectures <strong className="text-foreground font-medium">CamemBERT</strong> et <strong className="text-foreground font-medium">RoBERTa</strong>. 
            Notre moteur NLP analyse les structures sémantiques pour identifier les patterns de désinformation en temps réel.            </p>
          </div>

          <div>
            <h4 className="font-semibold text-foreground">Ressources</h4>
            <ul className="mt-4 space-y-2">
              <li>
                <Link to="/documentation" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                  Documentation
                </Link>
              </li>
              <li>
                <Link to="/documentation#api" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                  API
                </Link>
              </li>
              <li>
                <Link to="/comment-ca-marche" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                  Méthodologie
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-foreground">Légal</h4>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="/comment-ca-marche" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                  Confidentialité
                </a>
              </li>
              <li>
                <a href="/documentation" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
                  Conditions d'utilisation
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Section du bas centrée et adaptée au back */}
        <div className="mt-12 border-t pt-8 flex flex-col items-center gap-2">
          <p className="text-sm text-muted-foreground font-medium">
            © 2026 FactGuard <span className="mx-2 text-accent">|</span> <span className="italic">Intelligence Artificielle au service de la vérité.</span>
          </p>
          <p className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground/50">
            Modèles d'inférence haute performance (FR/EN)
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
