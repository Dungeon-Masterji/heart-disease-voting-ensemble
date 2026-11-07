import { useCallback, useEffect, useState } from "react";
import { getModelCard, toApiFailure } from "../services/api";
import type { ApiFailure } from "../types";

export default function ModelCardPage() {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);

  const load = useCallback(() => {
    setError(null);
    getModelCard().then((c) => setText(c.content)).catch((e) => setError(toApiFailure(e)));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (error)
    return (
      <div role="alert" className="rounded border-2 border-red-700 bg-red-50 p-4 text-red-950">
        <p>Could not load the model card: {error.message} ({error.code})</p>
        <button type="button" onClick={load} className="mt-2 rounded bg-red-800 px-3 py-1 text-white">Retry</button>
      </div>
    );
  if (text === null) return <p role="status">Loading model card...</p>;
  return (
    <article aria-label="Model card">
      <pre className="whitespace-pre-wrap rounded border border-slate-300 bg-white p-4 text-sm text-slate-900">{text}</pre>
    </article>
  );
}
