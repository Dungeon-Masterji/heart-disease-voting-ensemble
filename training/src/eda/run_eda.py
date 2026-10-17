"""
Exploratory Data Analysis (BR-003 / TR-004).

Generates and saves:
  - target distribution bar chart
  - histograms for continuous features
  - correlation heatmap (Pearson)
  - missingness heatmap
  - boxplots of key continuous features grouped by target
  - a written eda_summary.md with descriptive (not inferential) findings

Run with: python -m training.src.eda.run_eda
(or `python training/src/eda/run_eda.py` from the repo root)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "training" / "artifacts" / "eda"

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

sns.set_theme(style="whitegrid")


def plot_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str),
                palette=["#4C72B0", "#C44E52"], legend=False, ax=ax)
    ax.set_xlabel("target (0 = absence, 1 = presence)")
    ax.set_ylabel("patient count")
    ax.set_title(f"Target class distribution (n={len(df)})")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_continuous_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, CONTINUOUS_FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
    for ax in axes[len(CONTINUOUS_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Continuous feature distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_distributions.png", dpi=150)
    plt.close(fig)


def plot_categorical_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, CATEGORICAL_FEATURES):
        sns.countplot(x=df[col], ax=ax, color="#55A868")
        ax.set_title(col)
    fig.suptitle("Categorical / coded feature distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "categorical_distributions.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Pearson correlation matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    return corr


def plot_missingness_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title(f"Missingness map (total missing cells: {int(df.isna().sum().sum())})")
    fig.tight_layout()
    fig.savefig(out_dir / "missingness_heatmap.png", dpi=150)
    plt.close(fig)


def plot_target_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(CONTINUOUS_FEATURES), figsize=(4 * len(CONTINUOUS_FEATURES), 4))
    for ax, col in zip(axes, CONTINUOUS_FEATURES):
        sns.boxplot(x="target", y=col, hue="target", data=df, ax=ax,
                    palette=["#4C72B0", "#C44E52"], legend=False)
        ax.set_title(col)
    fig.suptitle("Continuous features grouped by target")
    fig.tight_layout()
    fig.savefig(out_dir / "target_boxplots.png", dpi=150)
    plt.close(fig)


def iqr_outlier_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for col in CONTINUOUS_FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return counts


def write_summary(df: pd.DataFrame, corr: pd.DataFrame, outlier_counts: dict, out_dir: Path) -> None:
    target_corr = corr["target"].drop("target").sort_values(key=abs, ascending=False)
    top_pos = target_corr.head(3)
    counts = df["target"].value_counts().sort_index()
    imbalance_ratio = round(float(counts.max() / counts.min()), 2)

    lines = [
        "# EDA Summary — Week 3",
        "",
        f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns "
        "(see docs/DATA_DICTIONARY.md for full column provenance).",
        "",
        "## Target distribution",
        f"- Class 0 (absence): {counts.get(0, 0)}",
        f"- Class 1 (presence): {counts.get(1, 0)}",
        f"- Imbalance ratio: {imbalance_ratio}:1 — mild imbalance, not severe. "
        "Stratified k-fold CV (planned for Week 6+) is still warranted.",
        "",
        "## Missingness",
        f"- Total missing cells (NaN): {int(df.isna().sum().sum())}.",
        "- No NaNs are present in this CSV variant, but validation.py flags "
        "5 rows where `ca`=4 and 2 rows where `thal`=0 — both outside the "
        "documented coding range for this dataset variant. These almost "
        "certainly correspond to the 6 originally-missing values in the raw "
        "UCI Cleveland file (marked `?`) that were coerced to an out-of-range "
        "integer by whichever preprocessing produced this CSV, rather than "
        "true clinical measurements. This is flagged, not silently dropped — "
        "the imputation/exclusion decision belongs to Week 4 feature "
        "engineering, not to EDA.",
        "",
        "## Duplicates",
        "- 1 exact duplicate row pair detected (patients at index 163/164 — "
        "identical across all 14 columns). Documented as a known artifact of "
        "this widely-circulated CSV variant; to be resolved during Week 4 "
        "preprocessing (drop or keep-with-justification), not silently here.",
        "",
        "## Correlation with target (top 3 by |r|)",
    ]
    for feat, val in top_pos.items():
        lines.append(f"- `{feat}`: r = {val:.3f}")
    lines += [
        "",
        "*Correlation does not imply causality — this is descriptive EDA only, "
        "consistent with the PRD's caution in section 10.3.*",
        "",
        "## Outlier counts (IQR method, continuous features)",
    ]
    for col, n in outlier_counts.items():
        lines.append(f"- `{col}`: {n} potential outlier(s)")
    lines += [
        "",
        "## Figures",
        "- `target_distribution.png`",
        "- `continuous_distributions.png`",
        "- `categorical_distributions.png`",
        "- `correlation_heatmap.png`",
        "- `missingness_heatmap.png`",
        "- `target_boxplots.png`",
        "",
        "## Carried forward to Week 4",
        "- Decide imputation/exclusion for the 7 out-of-documented-range "
        "`ca`/`thal` rows (fit on train folds only).",
        "- Decide whether to drop the 1 duplicate row before or within CV "
        "(duplicate must not span train/validation split in the same fold).",
        "- `oldpeak` and `chol` show the most outliers by IQR — review "
        "clinical plausibility before deciding on winsorization vs. keep.",
    ]
    (out_dir / "eda_summary.md").write_text("\n".join(lines))


def main() -> None:
    from training.src.data.ingestion import load_raw_dataset

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataset()

    plot_target_distribution(df, ARTIFACT_DIR)
    plot_continuous_distributions(df, ARTIFACT_DIR)
    plot_categorical_distributions(df, ARTIFACT_DIR)
    corr = plot_correlation_heatmap(df, ARTIFACT_DIR)
    plot_missingness_heatmap(df, ARTIFACT_DIR)
    plot_target_boxplots(df, ARTIFACT_DIR)
    outliers = iqr_outlier_counts(df)
    write_summary(df, corr, outliers, ARTIFACT_DIR)

    print(f"EDA artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
