from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from training.src.features.preprocessing import build_preprocessor


def base_estimators(seed: int = 42):
    return [
        ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=seed)),
        ("gb", GradientBoostingClassifier(random_state=seed)),
        ("svm", SVC(probability=True, random_state=seed)),
    ]


def build_voting(strategy: str = "soft", seed: int = 42) -> VotingClassifier:
    if strategy not in {"hard", "soft"}:
        raise ValueError("strategy must be 'hard' or 'soft'")
    return VotingClassifier(base_estimators(seed), voting=strategy)


def make_tuning_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", build_voting("soft", seed)),
    ])
