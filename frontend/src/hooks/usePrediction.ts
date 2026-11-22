import { useCallback, useRef, useState } from "react";
import { scorePatient, toApiFailure } from "../services/api";
import type { ApiFailure, PatientFeatures, ScoreResponse } from "../types";

export type PredictionState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; result: ScoreResponse }
  | { status: "error"; error: ApiFailure };

export function usePrediction() {
  const [state, setState] = useState<PredictionState>({ status: "idle" });

  // Identifies the newest request: answers to older or abandoned ones are ignored.
  const latest = useRef(0);

  const submit = useCallback(async (features: PatientFeatures) => {
    const request = ++latest.current;
    setState({ status: "loading" });
    try {
      const result = await scorePatient(features);
      if (request !== latest.current) return;
      if ("risk_probability" in result === false && "code" in result && "message" in result) {
        setState({ status: "error", error: result as unknown as ApiFailure });
        return;
      }
      setState({ status: "success", result });
    } catch (e) {
      if (request !== latest.current) return;
      let failure: ApiFailure;
      try {
        failure = toApiFailure(e);
      } catch {
        failure = { status: null, code: "UNKNOWN", message: "Unexpected error." };
      }
      setState({ status: "error", error: failure });
    }
  }, []);

  const reset = useCallback(() => {
    latest.current += 1;
    setState({ status: "idle" });
  }, []);
  return { state, submit, reset };
}
