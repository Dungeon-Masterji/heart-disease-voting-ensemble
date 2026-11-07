import type { PatientFeatures } from "../types";

export interface Option { value: number; label: string; }

export interface FieldSpec {
  name: keyof PatientFeatures;
  label: string;
  group: "Demographics" | "Vitals" | "Lab values" | "Symptoms & tests";
  kind: "number" | "select";
  min: number;
  max: number;
  step?: number;
  integer: boolean;
  unit?: string;
  options?: Option[];
}

const yesNo: Option[] = [{ value: 0, label: "No" }, { value: 1, label: "Yes" }];
const codes = (from: number, to: number, prefix: string): Option[] =>
  Array.from({ length: to - from + 1 }, (_, i) => ({ value: from + i, label: `${prefix} ${from + i}` }));

// Ranges mirror backend/app/models/schemas.py and training/src/data/schema.py.
export const FIELDS: FieldSpec[] = [
  { name: "age", label: "Age", group: "Demographics", kind: "number", min: 0, max: 120, integer: true, unit: "years" },
  { name: "sex", label: "Sex", group: "Demographics", kind: "select", min: 0, max: 1, integer: true,
    options: [{ value: 1, label: "Male" }, { value: 0, label: "Female" }] },
  { name: "trestbps", label: "Resting blood pressure", group: "Vitals", kind: "number", min: 60, max: 260, integer: true, unit: "mm Hg" },
  { name: "thalach", label: "Maximum heart rate achieved", group: "Vitals", kind: "number", min: 60, max: 250, integer: true, unit: "bpm" },
  { name: "chol", label: "Serum cholesterol", group: "Lab values", kind: "number", min: 50, max: 700, integer: true, unit: "mg/dl" },
  { name: "fbs", label: "Fasting blood sugar > 120 mg/dl", group: "Lab values", kind: "select", min: 0, max: 1, integer: true, options: yesNo },
  { name: "cp", label: "Chest pain type (dataset code)", group: "Symptoms & tests", kind: "select", min: 0, max: 3, integer: true, options: codes(0, 3, "Type") },
  { name: "exang", label: "Exercise-induced angina", group: "Symptoms & tests", kind: "select", min: 0, max: 1, integer: true, options: yesNo },
  { name: "restecg", label: "Resting ECG result (code)", group: "Symptoms & tests", kind: "select", min: 0, max: 2, integer: true, options: codes(0, 2, "Result") },
  { name: "oldpeak", label: "ST depression (oldpeak)", group: "Symptoms & tests", kind: "number", min: 0, max: 7, step: 0.1, integer: false },
  { name: "slope", label: "Peak exercise ST slope (code)", group: "Symptoms & tests", kind: "select", min: 0, max: 2, integer: true, options: codes(0, 2, "Slope") },
  { name: "ca", label: "Major vessels coloured (0-3)", group: "Symptoms & tests", kind: "select", min: 0, max: 3, integer: true, options: codes(0, 3, "Vessels:") },
  { name: "thal", label: "Thalassemia (code 1-3)", group: "Symptoms & tests", kind: "select", min: 1, max: 3, integer: true, options: codes(1, 3, "Code") },
];

export const GROUPS = ["Demographics", "Vitals", "Lab values", "Symptoms & tests"] as const;

export const SAMPLE_PATIENT: Record<string, string> = {
  age: "63", sex: "1", cp: "3", trestbps: "145", chol: "233", fbs: "1", restecg: "0",
  thalach: "150", exang: "0", oldpeak: "2.3", slope: "0", ca: "0", thal: "1",
};

export function validateField(spec: FieldSpec, raw: string): string | null {
  if (raw === undefined || raw.trim() === "") return "Required";
  const n = Number(raw);
  if (Number.isNaN(n)) return "Enter a number";
  if (spec.integer && !Number.isInteger(n)) return "Enter a whole number";
  if (n < spec.min || n > spec.max) return `Must be between ${spec.min} and ${spec.max}`;
  return null;
}

export function validateAll(values: Record<string, string>): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const spec of FIELDS) {
    const err = validateField(spec, values[spec.name] ?? "");
    if (err) errors[spec.name] = err;
  }
  return errors;
}

export function toFeatures(values: Record<string, string>): PatientFeatures {
  const out: Record<string, number> = {};
  for (const spec of FIELDS) out[spec.name] = Number(values[spec.name]);
  return out as unknown as PatientFeatures;
}
