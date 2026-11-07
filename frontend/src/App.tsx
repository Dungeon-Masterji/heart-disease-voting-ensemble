import { useCallback, useEffect, useState } from "react";
import DashboardPage from "./pages/DashboardPage";
import ModelCardPage from "./pages/ModelCardPage";
import ScorePage from "./pages/ScorePage";
import { getHealth } from "./services/api";

type Tab = "score" | "dashboard" | "card";
const TABS: { id: Tab; label: string }[] = [
  { id: "score", label: "Patient scoring" },
  { id: "dashboard", label: "Model performance" },
  { id: "card", label: "Model card" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("score");
  const [available, setAvailable] = useState<boolean | null>(null);

  const checkHealth = useCallback(() => {
    getHealth().then((h) => setAvailable(h.model_loaded)).catch(() => setAvailable(false));
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return (
    <div className="mx-auto max-w-4xl p-4 text-slate-900">
      <header className="mb-4">
        <h1 className="text-2xl font-bold">HeartGuard AI - Diagnosis Support</h1>
        <p className="rounded border border-amber-700 bg-amber-50 p-2 text-sm text-amber-950">
          Educational prototype - not a medical device. Outputs are advisory risk signals for
          qualified clinicians, not diagnoses.
        </p>
      </header>

      {available === false && (
        <div role="alert" className="mb-4 rounded border-2 border-red-700 bg-red-50 p-3 text-red-950">
          The scoring service is unavailable (under maintenance or not running).
          <button type="button" onClick={checkHealth} className="ml-3 rounded bg-red-800 px-2 py-1 text-white">
            Check again
          </button>
        </div>
      )}

      <nav aria-label="Main" className="mb-4 flex gap-2" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            type="button"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-600 ${
              tab === t.id ? "bg-blue-700 text-white" : "border border-slate-400 bg-white text-slate-900"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main role="tabpanel">
        {tab === "score" && <ScorePage />}
        {tab === "dashboard" && <DashboardPage />}
        {tab === "card" && <ModelCardPage />}
      </main>
    </div>
  );
}
