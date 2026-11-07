from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from training.src.models.voting_ensemble import make_tuning_pipeline

PARAM_DIST = {
    "clf__voting": ["soft"],
    "clf__weights": [
        None,
        [1, 1, 1, 1],
        [2, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2],
        [2, 2, 1, 1],
    ],
    "clf__lr__C": [0.01, 0.1, 1, 10],
    "clf__rf__max_depth": [3, 5, 8, None],
    "clf__rf__min_samples_leaf": [1, 3, 5],
    "clf__gb__learning_rate": [0.03, 0.1, 0.3],
    "clf__gb__n_estimators": [50, 100, 200],
    "clf__gb__max_depth": [2, 3],
    "clf__svm__C": [0.1, 1, 10],
    "clf__svm__gamma": ["scale", 0.01, 0.1],
}


def nested_soft_cv(X, y, seed=42, n_iter=20, quick=False):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), 1):
        search = RandomizedSearchCV(
            make_tuning_pipeline(seed),
            PARAM_DIST,
            n_iter=2 if quick else n_iter,
            scoring="f1",
            cv=inner,
            random_state=seed,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        row = {"fold": fold, "mean_test_f1": search.best_score_, **search.best_params_}
        row["model"] = search.best_estimator_
        rows.append(row)
    return rows


def final_soft_search(X, y, seed=42, n_iter=40, quick=False):
    search = RandomizedSearchCV(
        make_tuning_pipeline(seed),
        PARAM_DIST,
        n_iter=2 if quick else n_iter,
        scoring="f1",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X, y)
    return search
