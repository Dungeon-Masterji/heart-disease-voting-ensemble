"""Week 4: leakage-safe preprocessing + clinically justified feature engineering."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak", "hr_pct_of_max"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_xy(df: pd.DataFrame):
    """Drop exact duplicate rows (1 pair found in validation), split X / y."""
    df = df.drop_duplicates().reset_index(drop=True)
    return df.drop(columns="target"), df["target"]


def clean_and_engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise -> cannot leak information between rows/folds."""
    X = X.copy()
    X.loc[X["ca"] > 3, "ca"] = np.nan      # 5 rows coded 4 (documented max is 3)
    X.loc[X["thal"] < 1, "thal"] = np.nan  # 2 rows coded 0 (documented min is 1)
    # Clinical rationale: fraction of age-predicted max heart rate (220 - age)
    # actually achieved during the exercise test.
    X["hr_pct_of_max"] = X["thalach"] / (220 - X["age"])
    return X


def build_preprocessor() -> Pipeline:
    """Unfitted pipeline. Imputer/scaler/encoder statistics are learned only
    when .fit() is called on a training split."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    columns = ColumnTransformer([("num", numeric, CONTINUOUS),
                                 ("cat", categorical, CATEGORICAL)])
    return Pipeline([("clean", FunctionTransformer(clean_and_engineer, validate=False)),
                     ("columns", columns)])