from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_selection import SelectFromModel, VarianceThreshold  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def create_pipeline_LogisticRegression(
    use_scaler: bool,
    max_iter: int,
    logreg_C: float,
    use_feature_selection: int,
    grid_search: bool,
    random_state: int,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if use_feature_selection == 1:
        pipeline_steps.append(
            (
                "feature_selection",
                SelectFromModel(RandomForestClassifier(random_state=2022)),
            )
        )
    if use_feature_selection == 2:
        pipeline_steps.append(("feature_selection", VarianceThreshold(threshold=0.20)))
    if grid_search is False:
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_C
                ),
            )
        )
    if grid_search is True:
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(random_state=random_state),
            )
        )
    return Pipeline(steps=pipeline_steps)


def create_pipeline_RandomForest(
    use_scaler: bool,
    n_estimators: int,
    use_feature_selection: int,
    grid_search: bool,
    random_state: int,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if use_feature_selection == 1:
        pipeline_steps.append(
            (
                "feature_selection",
                SelectFromModel(RandomForestClassifier(random_state=2022)),
            )
        )
    if use_feature_selection == 2:
        pipeline_steps.append(("feature_selection", VarianceThreshold(threshold=0.20)))
    if grid_search is False:
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators, random_state=random_state
                ),
            )
        )
    if grid_search is True:
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(random_state=random_state),
            )
        )
    return Pipeline(steps=pipeline_steps)
