import type { ScoreResponse } from "../types";

const LONG_DISCLAIMER =
  "This is a clinical decision-support signal, not a medical diagnosis. " +
  "Please consult clinical guidelines and qualified professionals.";

export default function RiskResult({ result }: { result: ScoreResponse }) {
  const high = result.risk_flag === "High Risk";
  const pct = Math.round(result.risk_probability * 1000) / 10;
  const thrPct = result.threshold * 100;

  // Icon + text + colour: never colour alone (accessibility requirement).
  const palette = high
    ? "border-red-700 bg-red-50 text-red-900"
    : "border-green-700 bg-green-50 text-green-900";

  return (
    <section aria-label="Risk result" role="status" aria-live="polite" className="space-y-4">
      <div className={`rounded border-2 p-5 ${palette}`}>
        <p className="flex items-center gap-3 text-2xl font-bold">
          <span aria-hidden="true">{high ? "\u26A0" : "\u2713"}</span>
          <span>{result.risk_flag}</span>
        </p>
        <p className="mt-1 text-sm">
          Estimated probability: <strong>{pct}%</strong> (flag threshold {thrPct.toFixed(0)}%)
        </p>
        <div
          role="progressbar"
          aria-label="Predicted risk probability"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          className="relative mt-3 h-4 w-full rounded bg-white ring-1 ring-slate-400"
        >
          <div
            className={`h-4 rounded ${high ? "bg-red-700" : "bg-green-700"}`}
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          />
          <div
            title="Flag threshold"
            className="absolute top-[-4px] h-6 w-0.5 bg-slate-900"
            style={{ left: `${Math.min(100, Math.max(0, thrPct))}%` }}
          />
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-2 text-sm text-slate-900 sm:grid-cols-4">
        <div><dt className="font-semibold">Model version</dt><dd>{result.model_version}</dd></div>
        <div><dt className="font-semibold">Voting</dt><dd>{result.voting_strategy}</dd></div>
        <div><dt className="font-semibold">Threshold</dt><dd>{result.threshold}</dd></div>
        <div><dt className="font-semibold">Timestamp</dt><dd>{new Date(result.timestamp).toLocaleString()}</dd></div>
      </dl>

      <p className="rounded border border-amber-700 bg-amber-50 p-3 text-sm text-amber-950">
        {LONG_DISCLAIMER} <span className="block text-xs">{result.disclaimer}</span>
      </p>
    </section>
  );
}
