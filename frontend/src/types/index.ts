export interface PatientFeatures {
  age: number; sex: number; cp: number; trestbps: number; chol: number; fbs: number;
  restecg: number; thalach: number; exang: number; oldpeak: number; slope: number;
  ca: number; thal: number;
}

export type RiskFlag = "High Risk" | "Low Risk";

export interface ScoreResponse {
  risk_probability: number;
  risk_flag: RiskFlag;
  model_version: string;
  voting_strategy: string;
  threshold: number;
  timestamp: string;
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  timestamp: string;
}

export interface MetricBlock {
  sensitivity?: number; specificity?: number; precision?: number; f1?: number; roc_auc?: number;
}

export interface CvSummary {
  folds: Array<Record<string, number | null>>;
  mean: Record<string, number>;
  std: Record<string, number>;
}

export interface ModelInfo {
  model_version: string;
  voting_strategy: string;
  threshold: number;
  threshold_rationale: string;
  training_date: string;
  n_training_rows: number | null;
  class_balance: Record<string, number> | null;
  metrics: MetricBlock;
  honest_check: { sensitivity: number; specificity: number } | null;
  cv_summary: CvSummary | null;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number } | null;
  feature_importance: Record<string, number> | null;
  baseline_logreg_cv: MetricBlock | null;
  curves: {
    roc: Array<{ fpr: number; tpr: number }> | null;
    precision_recall: Array<{ precision: number; recall: number }> | null;
    threshold_marker: { threshold: number } | null;
  } | null;
}

export interface ModelCard { format: string; content: string; }

export interface FieldError { field: string; message: string; }

export interface ApiFailure {
  status: number | null;
  code: string;
  message: string;
  fieldErrors?: FieldError[];
}
