import { useCallback, useEffect, useState } from "react";
import MetricsDashboard from "../components/MetricsDashboard";
import { getModel, toApiFailure } from "../services/api";
import type { ApiFailure, ModelInfo } from "../types";

export default function DashboardPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);

  const load = useCallback(() => {
    setError(null);
    getModel().then(setInfo).catch((e) => setError(toApiFailure(e)));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (error)
    return (
      <div role="alert" className="rounded border-2 border-red-700 bg-red-50 p-4 text-red-950">
        <p>Could not load model performance: {error.message} ({error.code})</p>
        <button type="button" onClick={load} className="mt-2 rounded bg-red-800 px-3 py-1 text-white">Retry</button>
      </div>
    );
  if (!info) return <p role="status">Loading model performance...</p>;
  return <MetricsDashboard info={info} />;
}
