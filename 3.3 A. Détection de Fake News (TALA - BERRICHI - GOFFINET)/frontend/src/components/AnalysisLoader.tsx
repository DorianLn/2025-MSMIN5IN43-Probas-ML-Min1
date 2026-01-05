import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

const AnalysisLoader = () => {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        className="relative"
      >
        <div className="h-16 w-16 rounded-full border-4 border-muted" />
        <div className="absolute inset-0 h-16 w-16 rounded-full border-4 border-transparent border-t-accent" />
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mt-6 text-center"
      >
        <h3 className="font-display text-xl font-semibold text-foreground">Analyse en cours...</h3>
        <p className="mt-2 text-muted-foreground">Notre modèle IA examine votre article</p>
      </motion.div>

      <div className="mt-6 flex items-center gap-2">
        {["Style", "Vocabulaire", "Source"].map((step, index) => (
          <motion.div
            key={step}
            initial={{ opacity: 0.3 }}
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              delay: index * 0.3,
            }}
            className="flex items-center gap-1.5 rounded-full bg-accent/10 px-3 py-1.5 text-sm text-accent"
          >
            <Loader2 className="h-3 w-3 animate-spin" />
            {step}
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default AnalysisLoader;
