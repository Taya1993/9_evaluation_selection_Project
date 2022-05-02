from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline_LogisticRegression(
    use_scaler: bool, max_iter: int, logreg_C: float, use_feature_selection: int, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if use_feature_selection == 1:
        pipeline_steps.append(("feature_selection", SelectFromModel(RandomForestClassifier(random_state=2022))))
    if use_feature_selection == 2:
        pipeline_steps.append(("feature_selection", VarianceThreshold(threshold = 0.20)))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)

def create_pipeline_RandomForest(
    use_scaler: bool, n_estimators: int, use_feature_selection: int, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if use_feature_selection == 1:
        pipeline_steps.append(("feature_selection", SelectFromModel(RandomForestClassifier(random_state=2022))))
    if use_feature_selection == 2:
        pipeline_steps.append(("feature_selection", VarianceThreshold(threshold = 0.20)))
    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
        )
    )
    return Pipeline(steps=pipeline_steps)