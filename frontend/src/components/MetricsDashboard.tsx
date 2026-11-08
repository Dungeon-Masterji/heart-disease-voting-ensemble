import type { ModelInfo } from "../types";
import CurvesChart from "./CurvesChart";

const pct = (x?: number | null) => (x == null ? "n/a" : x.toFixed(3));

function Card({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded border border-slate-300 bg-white p-4">
      <p className="text-sm text-slate-700">{label}</p>
      <p className="text-3xl font-bold text-slate-900">{value}</p>
      {note && <p className="text-xs text-slate-700">{note}</p>}
    </div>
  );
}

export default function MetricsDashboard({ info }: { info: ModelInfo }) {
  const m = info.metrics;
  const cm = info.confusion_matrix;
  const cv = info.cv_summary;
  const importance = info.feature_importance
    ? Object.entries(info.feature_importance).sort((a, b) => b[1] - a[1])
    : [];
  const maxImp = importance.length ? Math.max(...importance.map(([, v]) => Math.abs(v)), 1e-9) : 1;

  return (
    <div className="space-y-8">
      <p className="text-sm text-slate-800">
        Model <strong>{info.model_version}</strong> ({info.voting_strategy} voting), trained {info.training_date} on{" "}
        {info.n_training_rows ?? "n/a"} patients. Operating threshold <strong>{info.threshold}</strong>.
        <span className="block">{info.threshold_rationale}</span>
      </p>

      <section aria-label="Headline metrics" className="grid gap-4 sm:grid-cols-4">
        <Card label="Sensitivity" value={pct(m.sensitivity)} note="out-of-fold, at threshold" />
        <Card label="Specificity" value={pct(m.specificity)} note="out-of-fold, at threshold" />
        <Card label="F1-score" value={pct(m.f1)} note="out-of-fold, at threshold" />
        <Card label="ROC AUC" value={pct(m.roc_auc)} note="threshold-independent" />
      </section>

      {info.honest_check && (
        <p className="text-sm text-slate-800">
          Honest check (threshold chosen without seeing the scored fold): sensitivity{" "}
          {pct(info.honest_check.sensitivity)}, specificity {pct(info.honest_check.specificity)}.
        </p>
      )}

      {cm && (
        <section aria-label="Confusion matrix">
          <h3 className="mb-2 font-semibold text-slate-900">Confusion matrix (out-of-fold)</h3>
          <table className="border-collapse text-center">
            <caption className="sr-only">Counts of true/false positives and negatives</caption>
            <thead>
              <tr><th className="p-2" /><th className="p-2">Predicted Low</th><th className="p-2">Predicted High</th></tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row" className="p-2 text-left">Actually no disease</th>
                <td className="border border-slate-400 bg-green-50 p-4 text-lg text-green-950">TN {cm.tn}</td>
                <td className="border border-slate-400 bg-amber-50 p-4 text-lg text-amber-950">FP {cm.fp}</td>
              </tr>
              <tr>
                <th scope="row" className="p-2 text-left">Actually disease</th>
                <td className="border border-slate-400 bg-red-50 p-4 text-lg text-red-950">FN {cm.fn}</td>
                <td className="border border-slate-400 bg-green-50 p-4 text-lg text-green-950">TP {cm.tp}</td>
              </tr>
            </tbody>
          </table>
        </section>
      )}

      {info.curves?.roc?.length && info.curves?.precision_recall?.length && (
        <section aria-label="Evaluation curves" className="grid gap-4 lg:grid-cols-2">
          <CurvesChart
            title="ROC curve"
            points={info.curves.roc.map(({ fpr, tpr }) => ({ x: fpr, y: tpr }))}
            threshold={info.threshold}
            xLabel="False positive rate"
            yLabel="True positive rate"
            summary={`Out-of-fold ROC curve. Area under the curve is ${pct(m.roc_auc)}.`}
          />
          <CurvesChart
            title="Precision-Recall curve"
            points={info.curves.precision_recall.map(({ recall, precision }) => ({ x: recall, y: precision }))}
            threshold={info.threshold}
            xLabel="Recall"
            yLabel="Precision"
            summary={`Out-of-fold precision-recall curve. Operating threshold is ${info.threshold.toFixed(3)}.`}
          />
        </section>
      )}

      {cv && (
        <section aria-label="Cross-validation summary" className="overflow-x-auto">
          <h3 className="mb-2 font-semibold text-slate-900">Nested cross-validation (5 folds, threshold 0.5)</h3>
          <table className="min-w-[28rem] border-collapse text-sm">
            <thead>
              <tr className="bg-slate-100 text-left">
                <th className="p-2">Fold</th><th className="p-2">Sensitivity</th>
                <th className="p-2">Specificity</th><th className="p-2">F1</th><th className="p-2">ROC AUC</th>
              </tr>
            </thead>
            <tbody>
              {cv.folds.map((f, i) => (
                <tr key={i} className="border-t border-slate-300">
                  <td className="p-2">{String(f.fold ?? i + 1)}</td>
                  <td className="p-2">{pct(f.sensitivity)}</td>
                  <td className="p-2">{pct(f.specificity)}</td>
                  <td className="p-2">{pct(f.f1)}</td>
                  <td className="p-2">{pct(f.roc_auc)}</td>
                </tr>
              ))}
              <tr className="border-t-2 border-slate-500 font-semibold">
                <td className="p-2">mean ± std</td>
                {(["sensitivity", "specificity", "f1", "roc_auc"] as const).map((k) => (
                  <td key={k} className="p-2">{pct(cv.mean[k])} ± {pct(cv.std[k])}</td>
                ))}
              </tr>
            </tbody>
          </table>
          {info.baseline_logreg_cv && (
            <p className="mt-2 text-sm text-slate-800">
              Baseline (plain logistic regression, same CV): F1 {pct(info.baseline_logreg_cv.f1)}, sensitivity{" "}
              {pct(info.baseline_logreg_cv.sensitivity)}, ROC AUC {pct(info.baseline_logreg_cv.roc_auc)}.
            </p>
          )}
        </section>
      )}

      {importance.length > 0 && (
        <section aria-label="Feature importance">
          <h3 className="mb-2 font-semibold text-slate-900">Permutation importance (drop in AUC when shuffled)</h3>
          <ul className="space-y-1">
            {importance.map(([name, v]) => (
              <li key={name} className="flex items-center gap-2 text-sm">
                <span className="w-20 shrink-0 font-mono">{name}</span>
                <div className="h-3 flex-1 rounded bg-slate-100">
                  <div className="h-3 rounded bg-blue-700" style={{ width: `${Math.max(0, (v / maxImp) * 100)}%` }} />
                </div>
                <span className="w-14 text-right tabular-nums">{v.toFixed(3)}</span>
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs text-slate-700">Predictive association only - not a causal effect.</p>
        </section>
      )}
    </div>
  );
}
