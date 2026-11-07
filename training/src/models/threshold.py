import pandas as pd


def select_threshold(sweep: pd.DataFrame, target_sensitivity: float = 0.90):
    eligible = sweep.loc[sweep["sensitivity"] >= target_sensitivity]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["specificity", "threshold"], ascending=[False, False]
    ).iloc[0].to_dict()
