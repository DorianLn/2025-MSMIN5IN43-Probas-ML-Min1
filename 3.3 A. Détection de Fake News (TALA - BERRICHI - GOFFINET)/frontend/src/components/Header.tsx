import { Shield } from "lucide-react";
import { Link } from "react-router-dom";

const Header = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/80 backdrop-blur-lg">
      <div className="container flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-accent-foreground">
            <Shield className="h-5 w-5" />
          </div>
          <span className="font-display text-xl font-bold text-foreground">FactGuard</span>
        </Link>

        <nav className="hidden items-center gap-6 md:flex">
          <Link to="/comment-ca-marche" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
            Comment ça marche
          </Link>
          <Link to="/a-propos" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
            À propos
          </Link>
          <Link to="/documentation" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
            Documentation
          </Link>
        </nav>

        <div className="flex items-center gap-3">
          <Link
            to="/#analyzer"
            className="inline-flex h-9 items-center justify-center rounded-lg bg-accent px-4 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent/90"
          >
            Analyser un article
          </Link>
        </div>
      </div>
    </header>
  );
};

export default Header;
