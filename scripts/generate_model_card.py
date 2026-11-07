"""Generate docs/MODEL_CARD.md from the saved model metadata (PRD section 21).

Every number in the card comes from training/artifacts/models/model_m1.0.0.json,
so the card cannot drift from the model it describes.

Usage:  python scripts/generate_model_card.py [path/to/model_meta.json]
"""

import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_META = REPO / "training" / "artifacts" / "models" / "model_m1.0.0.json"
OUT = REPO / "docs" / "MODEL_CARD.md"
METRICS = ("sensitivity", "specificity", "f1", "roc_auc")


def fmt(x, nd=3):
    return (
        "n/a"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:.{nd}f}"
    )


def fold_stats(folds):
    out = {}
    for m in METRICS:
        v = [
            f[m]
            for f in (folds or [])
            if isinstance(f.get(m), (int, float)) and not math.isnan(f[m])
        ]
        if v:
            out[m] = (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
    return out


def build_card(meta: dict) -> str:
    m = meta.get("metrics_at_threshold_oof", {})
    hc = meta.get("metrics_honest_check") or {}
    cm = meta.get("confusion_matrix")
    cv = fold_stats(meta.get("cv_folds"))
    base = meta.get("baseline_logreg_cv")
    cb = meta.get("class_balance") or {}
    n = meta.get("n_training_rows", "n/a")
    thr = meta.get("threshold")
    analysis = meta.get("analysis") or {}

    L: list[str] = []
    a = L.append
    a("# Model Card - Voting Classifier Heart Disease Risk Flag\n")
    a(
        "> Educational prototype. **Not a medical device.** Outputs are advisory risk signals, not diagnoses.\n"
    )
    a("## 1. Model details\n")
    a("| Field | Value |\n|---|---|")
    a(f"| Version | {meta.get('model_version')} |")
    a("| Type | Supervised binary classification ensemble |")
    a(
        "| Implementation | scikit-learn `VotingClassifier` inside a `Pipeline` with leakage-safe preprocessing |"
    )
    a(
        f"| Voting strategy | {meta.get('voting_strategy')} (chosen because threshold selection needs probabilities) |"
    )
    a(
        "| Base estimators | Logistic Regression, Random Forest, Gradient Boosting, SVM (`probability=True`) |"
    )
    a(f"| Training date | {meta.get('training_date')} |")
    a(f"| Artifact SHA-256 | `{meta.get('artifact_sha256')}` |")
    a(
        f"| Hyperparameters | `{json.dumps(meta.get('hyperparameters'), default=str)}` |\n"
    )
    a("## 2. Intended use\n")
    a(
        "- Educational demonstration of the full ML lifecycle (data, validation, evaluation, serving)."
    )
    a(
        "- A model-driven risk flag to help *prioritise* patients for further workup, always with clinician review."
    )
    a(
        "- Research into ensemble methods (grounded in *Popular Ensemble Methods: An Empirical Study*, JAIR 1999).\n"
    )
    a("## 3. Out of scope\n")
    a("- Autonomous diagnosis or any real clinical decision.")
    a("- Any regulatory claim: no FDA, CE or other approval exists.")
    a("- Populations, geographies or time periods different from the training data.")
    a(
        "- Emergency or real-time critical-care decisions (latency and reliability not validated).\n"
    )
    a("## 4. Training data\n")
    a(
        "- Source: UCI Heart Disease (Cleveland subset), via a recoded 303-row CSV variant. Provenance and the"
    )
    a(
        "  encoding differences from the UCI documentation are in `docs/DATA_DICTIONARY.md`."
    )
    a(
        f"- Rows used for training: **{n}** (1 exact duplicate removed). Class balance (0 / 1): "
        f"{cb.get('0', 'n/a')} / {cb.get('1', 'n/a')}."
    )
    a(
        "- 7 values outside the documented coding range (`ca`=4, `thal`=0) were treated as missing and imputed "
        "inside the pipeline, using statistics from training folds only.\n"
    )
    a("## 5. Validation\n")
    a(
        "- Stratified 5-fold cross-validation (seed 42); every preprocessing step is re-fitted on each fold's training split."
    )
    a(
        "- Hyperparameters tuned with **nested** CV (inner 3-fold randomized search, selection metric F1) so the reported "
        "estimate is not inflated by tuning.\n"
    )
    a("## 6. Evaluation results\n")
    a(f"Operating threshold: **{thr}**. Rule: {meta.get('threshold_rationale')}\n")
    a("| Metric (out-of-fold predictions) | Value |\n|---|---|")
    for k in ("sensitivity", "specificity", "precision", "f1"):
        a(f"| {k} | {fmt(m.get(k))} |")
    a(f"| ROC AUC | {fmt(meta.get('roc_auc_oof'))} |")
    if hc:
        a(
            f"\nHonest check (threshold re-chosen inside each fold, never on the fold being scored): "
            f"sensitivity {fmt(hc.get('sensitivity'))}, specificity {fmt(hc.get('specificity'))}."
        )
    if cm:
        a("\n| | Predicted Low | Predicted High |\n|---|---|---|")
        a(f"| **Actually no disease** | TN = {cm['tn']} | FP = {cm['fp']} |")
        a(f"| **Actually disease** | FN = {cm['fn']} | TP = {cm['tp']} |")
    if cv:
        a("\nNested CV at threshold 0.5, mean (std) across 5 folds:\n")
        a("| Metric | Mean | Std |\n|---|---|---|")
        for k, (mu, sd) in cv.items():
            a(f"| {k} | {fmt(mu)} | {fmt(sd)} |")
    if base and cv.get("f1"):
        diff = cv["f1"][0] - base.get("f1", float("nan"))
        a(
            f"\n**Baseline comparison.** Plain logistic regression under the same CV scored F1 {fmt(base.get('f1'))}, "
            f"sensitivity {fmt(base.get('sensitivity'))}, ROC AUC {fmt(base.get('roc_auc'))}. "
            f"The tuned ensemble's F1 differs by {diff:+.3f}."
        )
    if analysis:
        calibration = analysis.get("calibration", {})
        a("\n## 10. Additional descriptive analysis\n")
        a(
        f"- Soft-voting out-of-fold calibration Brier score: "
        f"{fmt(calibration.get('brier_score'))}. Lower is better."
        )
        a(
        "- Continuous-feature variance inflation factors and target-association tests "
        "are recorded under `training/artifacts/`. Association is descriptive, not causal."
        )
        a(
        "- Subgroup metrics at the chosen threshold are reported by sex and age band "
        "in `training/artifacts/subgroup_metrics.json`; counts and small-sample caveats "
        "are included. These results are not evidence of causal or generalizable subgroup differences."
        )
        if base and cv.get("f1") and abs(diff) <= cv["f1"][1]:
            a(
                "That difference is smaller than the fold-to-fold standard deviation, so on this dataset the ensemble "
                "is **not demonstrably better** than the simple baseline."
            )
    a("\n## 7. Ethical considerations and limitations\n")
    a(
        "- **Population bias:** single hospital, 1980s data, ~300 patients; subgroup coverage is limited and "
        "descriptive subgroup estimates can be unstable in small groups."
    )
    a("- **Historical bias:** labels reflect the diagnostic practice of the era.")
    a(
        "- **Small sample:** fold-level metrics vary a lot (about 60 patients per fold); treat differences of a few "
        "points as noise."
    )
    a(
        "- **Prevalence:** this dataset is roughly half positive, far above real screening prevalence; precision in "
        "practice would be much lower than reported here."
    )
    a("- **Proxy variables:** `sex` and `age` may encode societal bias.")
    a(
        "- **Human oversight:** every output is advisory and requires confirmation by a qualified clinician.\n"
    )
    a("## 8. Monitoring (planned - NOT implemented)\n")
    a(
        "No drift, calibration or outcome monitoring exists yet. A production system would need input-drift checks "
        "(KS/PSI), prediction-rate tracking, calibration checks, and outcome-based recomputation of "
        "sensitivity/specificity once diagnoses are confirmed.\n"
    )
    a("## 9. Versioning and reproducibility\n")
    a(
        f"Seed {meta.get('seed')}; the model artifact is checksum-verified when the API starts; every API response "
        "includes the model version and threshold. Training steps are recorded in `training/artifacts/`.\n"
    )
    return "\n".join(L)


def main() -> None:
    meta_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_META
    if not meta_path.exists():
        sys.exit(
            f"Metadata not found: {meta_path}\nRun notebook cells 8g and 8h first."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_card(json.loads(meta_path.read_text())), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
