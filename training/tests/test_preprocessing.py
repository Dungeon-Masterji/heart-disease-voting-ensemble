"""Week 4-8 data-science tests (PRD TEST-007, TEST-008). No dataset file needed:
they run on synthetic data with the same schema, so they also work in CI."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from training.src.features.preprocessing import (
    build_preprocessor,
    clean_and_engineer,
    load_xy,
)


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        dict(
            age=rng.integers(30, 77, n),
            sex=rng.integers(0, 2, n),
            cp=rng.integers(0, 4, n),
            trestbps=rng.integers(94, 200, n),
            chol=rng.integers(126, 564, n),
            fbs=rng.integers(0, 2, n),
            restecg=rng.integers(0, 3, n),
            thalach=rng.integers(71, 202, n),
            exang=rng.integers(0, 2, n),
            oldpeak=rng.random(n) * 4,
            slope=rng.integers(0, 3, n),
            ca=rng.integers(0, 4, n),
            thal=rng.integers(1, 4, n),
        )
    )
    df["target"] = (rng.random(n) < 0.5).astype(int)
    return df


def test_load_xy_drops_exact_duplicates():
    df = make_df()
    df.loc[len(df)] = df.loc[3]  # add an exact duplicate row
    X, y = load_xy(df)
    assert len(X) == len(y) == 200 and "target" not in X.columns


def test_out_of_range_codes_become_nan_and_feature_is_added():
    X, _ = load_xy(make_df())
    X.loc[0, "ca"] = 4
    X.loc[1, "thal"] = 0
    out = clean_and_engineer(X)
    assert np.isnan(out.loc[0, "ca"]) and np.isnan(out.loc[1, "thal"])
    expected = X.loc[2, "thalach"] / (220 - X.loc[2, "age"])
    assert np.isclose(out.loc[2, "hr_pct_of_max"], expected)
    assert X.loc[0, "ca"] == 4  # input was not mutated


# TEST-008: preprocessing statistics come from the training split only
def test_scaler_statistics_use_training_data_only():
    X, _ = load_xy(make_df())
    X_tr, X_va = X.iloc[:150], X.iloc[150:]
    prep = build_preprocessor().fit(X_tr)
    scaler = prep.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    train_age_mean = X_tr["age"].mean()
    assert np.isclose(scaler.mean_[0], train_age_mean)
    assert not np.isclose(
        scaler.mean_[0], X["age"].mean()
    )  # would match if data had leaked


def test_transform_does_not_refit():
    X, _ = load_xy(make_df())
    prep = build_preprocessor().fit(X.iloc[:150])
    scaler = prep.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    before = scaler.mean_.copy()
    prep.transform(X.iloc[150:])
    assert np.array_equal(before, scaler.mean_)


def test_train_and_validation_have_same_columns():
    X, _ = load_xy(make_df())
    prep = build_preprocessor().fit(X.iloc[:150])
    assert (
        prep.transform(X.iloc[:150]).shape[1] == prep.transform(X.iloc[150:]).shape[1]
    )


# TEST-007: reproducible, correct stratified CV
def test_cv_folds_reproducible_and_each_row_validated_once():
    X, y = load_xy(make_df())

    def folds(seed):
        return [
            va.tolist()
            for _, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
        ]

    assert folds(42) == folds(42)
    assert folds(42) != folds(7)
    seen = sorted(i for f in folds(42) for i in f)
    assert seen == list(range(len(X)))
    for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        assert abs(y.iloc[va].mean() - y.mean()) < 0.1  # class ratio preserved
