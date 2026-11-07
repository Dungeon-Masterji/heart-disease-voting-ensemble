import axios from "axios";
import type {
  ApiFailure, HealthResponse, ModelCard, ModelInfo, PatientFeatures, ScoreResponse,
} from "../types";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1",
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

/** Turn any thrown error into one predictable shape the UI can render. */
export function toApiFailure(err: unknown): ApiFailure {
  if (axios.isAxiosError(err)) {
    if (err.response) {
      const data = err.response.data as
        | { error_code?: string; detail?: unknown; errors?: ApiFailure["fieldErrors"] }
        | undefined;
      return {
        status: err.response.status,
        code: data?.error_code ?? `HTTP_${err.response.status}`,
        message: typeof data?.detail === "string" ? data.detail : "The request failed.",
        fieldErrors: data?.errors,
      };
    }
    return { status: null, code: "NETWORK_ERROR", message: "Cannot reach the scoring service." };
  }
  return { status: null, code: "UNKNOWN", message: "Unexpected error." };
}

export async function scorePatient(features: PatientFeatures): Promise<ScoreResponse> {
  const { data } = await client.post<ScoreResponse>("/score", features);
  return data;
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}

export async function getModel(): Promise<ModelInfo> {
  const { data } = await client.get<ModelInfo>("/model");
  return data;
}

export async function getModelCard(): Promise<ModelCard> {
  const { data } = await client.get<ModelCard>("/model-card");
  return data;
}
