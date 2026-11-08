import { useRef } from "react";
import RiskForm from "../components/RiskForm";
import RiskResult from "../components/RiskResult";
import { usePrediction } from "../hooks/usePrediction";
import type { PatientFeatures } from "../types";

export default function ScorePage() {
  const { state, submit, reset } = usePrediction();
  const last = useRef<PatientFeatures | null>(null);

  const handleSubmit = (f: PatientFeatures) => {
    last.current = f;
    void submit(f).catch(() => undefined);
  };

  return (
    <div className="space-y-6">
      <RiskForm
        onSubmit={handleSubmit}
        onReset={reset}
        loading={state.status === "loading"}
        serverErrors={state.status === "error" ? state.error.fieldErrors : undefined}
      />

      {state.status === "loading" && (
        <div role="status" className="flex items-center gap-3 text-slate-900">
          <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-slate-400 border-t-blue-700" />
          Analyzing risk profile...
        </div>
      )}

      {state.status === "success" && <RiskResult result={state.result} />}

      {state.status === "error" && (
        <div role="alert" className="rounded border-2 border-red-700 bg-red-50 p-4 text-red-950">
          <p className="font-semibold">
            {state.error.code === "MODEL_UNAVAILABLE"
              ? "The scoring service is under maintenance."
              : "The request could not be completed."}
          </p>
          <p className="text-sm">{state.error.message}</p>
          <p className="text-xs">Error code: {state.error.code}{state.error.status ? ` (HTTP ${state.error.status})` : ""}</p>
          {state.error.code !== "VALIDATION_ERROR" && (
            <button
              type="button"
              onClick={() => last.current && void submit(last.current).catch(() => undefined)}
              className="mt-3 rounded bg-red-800 px-3 py-1 text-white hover:bg-red-900 focus:outline-none focus:ring-2 focus:ring-red-600"
            >
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}
