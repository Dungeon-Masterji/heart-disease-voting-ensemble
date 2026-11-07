from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from training.src.features.preprocessing import build_preprocessor


def make_baseline(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
