import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import type { FieldError, PatientFeatures } from "../types";
import { FIELDS, GROUPS, SAMPLE_PATIENT, toFeatures, validateField } from "../utils/validation";

interface Props {
  onSubmit: (features: PatientFeatures) => void;
  onReset: () => void;
  loading: boolean;
  serverErrors?: FieldError[];
}

const emptyValues = (): Record<string, string> =>
  Object.fromEntries(FIELDS.map((f) => [f.name, ""]));

export default function RiskForm({ onSubmit, onReset, loading, serverErrors }: Props) {
  const [values, setValues] = useState<Record<string, string>>(emptyValues);
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitted, setSubmitted] = useState(false);

  const errors = useMemo(() => {
    const out: Record<string, string> = {};
    for (const spec of FIELDS) {
      if (submitted || touched[spec.name]) {
        const err = validateField(spec, values[spec.name]);
        if (err) out[spec.name] = err;
      }
    }
    for (const se of serverErrors ?? []) out[se.field] = out[se.field] ?? se.message;
    return out;
  }, [values, touched, submitted, serverErrors]);

  const allEmpty = FIELDS.every((f) => values[f.name] === "");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    const hasError = FIELDS.some((spec) => validateField(spec, values[spec.name]) !== null);
    if (!hasError) onSubmit(toFeatures(values));
  };

  const handleReset = () => {
    setValues(emptyValues());
    setTouched({});
    setSubmitted(false);
    onReset();
  };

  return (
    <form onSubmit={handleSubmit} noValidate aria-label="Patient diagnostic scoring form" className="space-y-6">
      {allEmpty && (
        <p className="rounded border border-slate-300 bg-slate-50 p-3 text-slate-800">
          Enter the patient&apos;s values below (or load a sample patient) to get a risk flag.
        </p>
      )}

      {GROUPS.map((group) => (
        <fieldset key={group} className="rounded border border-slate-300 p-4">
          <legend className="px-2 font-semibold text-slate-900">{group}</legend>
          <div className="grid gap-4 sm:grid-cols-2">
            {FIELDS.filter((f) => f.group === group).map((spec) => {
              const id = `field-${spec.name}`;
              const err = errors[spec.name];
              return (
                <div key={spec.name} className="flex flex-col gap-1">
                  <label htmlFor={id} className="text-sm font-medium text-slate-900">
                    {spec.label}
                    {spec.unit ? ` (${spec.unit})` : ""}
                  </label>
                  {spec.kind === "select" ? (
                    <select
                      id={id}
                      value={values[spec.name]}
                      onChange={(e) => setValues({ ...values, [spec.name]: e.target.value })}
                      onBlur={() => setTouched({ ...touched, [spec.name]: true })}
                      aria-invalid={err ? true : undefined}
                      aria-describedby={err ? `${id}-error` : undefined}
                      className="rounded border border-slate-400 bg-white p-2 focus:outline-none focus:ring-2 focus:ring-blue-600"
                    >
                      <option value="">Select...</option>
                      {spec.options?.map((o) => (
                        <option key={o.value} value={String(o.value)}>{o.label}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={id}
                      type="number"
                      inputMode="decimal"
                      min={spec.min}
                      max={spec.max}
                      step={spec.step ?? 1}
                      value={values[spec.name]}
                      onChange={(e) => setValues({ ...values, [spec.name]: e.target.value })}
                      onBlur={() => setTouched({ ...touched, [spec.name]: true })}
                      aria-invalid={err ? true : undefined}
                      aria-describedby={err ? `${id}-error` : `${id}-hint`}
                      className="rounded border border-slate-400 p-2 focus:outline-none focus:ring-2 focus:ring-blue-600"
                    />
                  )}
                  {!err && spec.kind === "number" && (
                    <span id={`${id}-hint`} className="text-xs text-slate-700">
                      Valid range: {spec.min}-{spec.max}
                    </span>
                  )}
                  {err && (
                    <span id={`${id}-error`} className="text-sm font-medium text-red-800">
                      {err}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </fieldset>
      ))}

      <div className="flex flex-wrap gap-3">
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-blue-700 px-4 py-2 font-semibold text-white hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 disabled:opacity-60"
        >
          Get risk flag
        </button>
        <button
          type="button"
          onClick={handleReset}
          className="rounded border border-slate-500 px-4 py-2 text-slate-900 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
        >
          Reset form
        </button>
        <button
          type="button"
          onClick={() => setValues({ ...SAMPLE_PATIENT })}
          className="rounded border border-slate-500 px-4 py-2 text-slate-900 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
        >
          Load sample patient
        </button>
      </div>
    </form>
  );
}
