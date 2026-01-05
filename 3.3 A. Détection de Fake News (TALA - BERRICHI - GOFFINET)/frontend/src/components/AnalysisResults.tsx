import { motion } from "framer-motion";
import { Shield, AlertTriangle, CheckCircle, XCircle, TrendingUp, FileText, Globe } from "lucide-react";

interface AnalysisResultProps {
  result: {
    isReliable: boolean;
    confidence: number;
    factors: {
      style: { score: number; label: string };
      vocabulary: { score: number; label: string };
      source: { score: number; label: string };
    };
    summary: string;
  };
}

const AnalysisResult = ({ result }: AnalysisResultProps) => {
  const { isReliable, confidence, factors, summary } = result;

  const getConfidenceColor = (score: number) => {
    if (score >= 70) return "text-reliable";
    if (score >= 40) return "text-warning";
    return "text-unreliable";
  };

  const getFactorIcon = (key: string) => {
    switch (key) {
      case "style":
        return FileText;
      case "vocabulary":
        return TrendingUp;
      case "source":
        return Globe;
      default:
        return FileText;
    }
  };

  const factorLabels: Record<string, string> = {
    style: "Style d'écriture",
    vocabulary: "Vocabulaire",
    source: "Fiabilité de la source",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="space-y-6"
    >
      {/* Main Result Card */}
      <motion.div
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.3, delay: 0.1 }}
        className={`relative overflow-hidden rounded-2xl p-8 ${
          isReliable ? "bg-reliable-soft border border-reliable/20" : "bg-unreliable-soft border border-unreliable/20"
        }`}
      >
        <div className="flex items-start gap-6">
          <div
            className={`flex h-16 w-16 items-center justify-center rounded-2xl ${
              isReliable ? "bg-reliable text-reliable-foreground" : "bg-unreliable text-unreliable-foreground"
            }`}
          >
            {isReliable ? <CheckCircle className="h-8 w-8" /> : <XCircle className="h-8 w-8" />}
          </div>

          <div className="flex-1">
            <h3 className="font-display text-2xl font-bold text-foreground">
              {isReliable ? "Article probablement fiable" : "Article potentiellement trompeur"}
            </h3>
            <p className="mt-2 text-muted-foreground">{summary}</p>

            {/* Confidence Meter */}
            <div className="mt-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Niveau de confiance</span>
                <span className={`font-semibold ${getConfidenceColor(confidence)}`}>{confidence}%</span>
              </div>
              <div className="mt-2 h-3 overflow-hidden rounded-full bg-muted">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidence}%` }}
                  transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
                  className={`h-full rounded-full ${isReliable ? "bg-reliable" : "bg-unreliable"}`}
                />
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Analysis Factors */}
      <div className="grid gap-4 md:grid-cols-3">
        {Object.entries(factors).map(([key, factor], index) => {
          const Icon = getFactorIcon(key);
          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.2 + index * 0.1 }}
              className="rounded-xl border bg-card p-5"
            >
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent">
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">{factorLabels[key]}</p>
                  <p className="font-semibold text-foreground">{factor.label}</p>
                </div>
              </div>

              <div className="mt-4">
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${factor.score}%` }}
                    transition={{ duration: 0.6, delay: 0.4 + index * 0.1, ease: "easeOut" }}
                    className={`h-full rounded-full ${
                      factor.score >= 70
                        ? "bg-reliable"
                        : factor.score >= 40
                        ? "bg-warning"
                        : "bg-unreliable"
                    }`}
                  />
                </div>
                <p className={`mt-1 text-right text-sm font-medium ${getConfidenceColor(factor.score)}`}>
                  {factor.score}%
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Warning Notice */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.6 }}
        className="flex items-start gap-3 rounded-lg border border-warning/20 bg-warning-soft p-4"
      >
        <AlertTriangle className="h-5 w-5 shrink-0 text-warning" />
        <p className="text-sm text-muted-foreground">
          <span className="font-medium text-foreground">Note :</span> Cette analyse est basée sur un modèle IA et ne
          remplace pas la vérification humaine. Nous vous encourageons à consulter plusieurs sources avant de tirer des
          conclusions.
        </p>
      </motion.div>
    </motion.div>
  );
};

export default AnalysisResult;
