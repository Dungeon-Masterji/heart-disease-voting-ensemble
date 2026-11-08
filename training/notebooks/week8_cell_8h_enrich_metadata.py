# PASTE THIS INTO A NEW NOTEBOOK CELL (after cells 8a-8g). It is not meant to be run as a script.
# Adds everything the dashboard and model card need to the model metadata JSON.
# The model file itself is untouched, so its SHA-256 stays valid.

meta_path = MODEL_DIR / "model_m1.0.0.json"
meta = json.loads(meta_path.read_text())

# confusion matrix at the chosen threshold (cm comes from cell 8d)
tn_, fp_, fn_, tp_ = [int(v) for v in cm.ravel()]
meta["confusion_matrix"] = {"tn": tn_, "fp": fp_, "fn": fn_, "tp": tp_}

# honest check, recomputed from pred_cv (cell 8c) so it cannot be overwritten by later cells
tn_h, fp_h, fn_h, tp_h = confusion_matrix(y_true, pred_cv).ravel()
meta["metrics_honest_check"] = {
    "sensitivity": float(tp_h / (tp_h + fn_h)),
    "specificity": float(tn_h / (tn_h + fp_h)),
}

meta["cv_folds"] = soft_nested.drop(columns="model").round(4).to_dict("records")
meta["feature_importance"] = {k: round(float(v), 4) for k, v in importance.items()}
meta["class_balance"] = {
    str(k): int(v) for k, v in y.value_counts().sort_index().items()
}

cols = ["sensitivity", "specificity", "f1", "roc_auc"]
base_rows = w6[w6.model == "logreg"][cols]
meta["baseline_logreg_cv"] = {
    k: round(float(v), 4) for k, v in base_rows.mean().items()
}
meta["baseline_logreg_cv_std"] = {
    k: round(float(v), 4) for k, v in base_rows.std().items()
}

meta_path.write_text(json.dumps(meta, indent=2, default=str))
print("metadata enriched:", sorted(meta.keys()))
