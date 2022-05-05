from pathlib import Path
from joblib import dump  # type: ignore[unused-ignore]

import click
import mlflow  # type: ignore[unused-ignore]
import mlflow.sklearn  # type: ignore[unused-ignore]
from sklearn.model_selection import KFold  # type: ignore[unused-ignore]
from sklearn.model_selection import GridSearchCV  # type: ignore[unused-ignore]
from sklearn.model_selection import cross_val_score  # type: ignore[unused-ignore]

from .data import get_dataset
from .pipeline import create_pipeline_LogisticRegression
from .pipeline import create_pipeline_RandomForest


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option("--random-state", default=42, type=int)
@click.option("--test-split-ratio", default=10, type=int)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
)
@click.option("--max-iter", default=100, type=int)
@click.option("--logreg-c", default=1.0, type=float)
@click.option("--use-feature-selection", default=0, type=int)
@click.option("--ml-model", default=1, type=int)
@click.option("--n-estimators", default=100, type=int)
@click.option("--grid-search", default=False, type=bool)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    use_feature_selection: int,
    ml_model: int,
    n_estimators: int,
    grid_search: bool,
) -> None:
    features, target = get_dataset(dataset_path)

    with mlflow.start_run():
        if ml_model == 1:
            pipeline = create_pipeline_LogisticRegression(
                use_scaler,
                max_iter,
                logreg_c,
                use_feature_selection,
                grid_search,
                random_state,
            )
            mlflow.log_param("ml_model", "LogisticRegression")
            if grid_search is False:
                mlflow.log_param("max_iter", max_iter)
                mlflow.log_param("logreg_c", logreg_c)
        if ml_model == 2:
            pipeline = create_pipeline_RandomForest(
                use_scaler,
                n_estimators,
                use_feature_selection,
                grid_search,
                random_state,
            )
            mlflow.log_param("ml_model", "RandomForestClassifier")
            if grid_search is False:
                mlflow.log_param("n_estimators", n_estimators)

        if grid_search is True and ml_model == 1:
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
            param_grid = {
                "penalty": ["l1", "l2"],
                "C": [1, 10, 100],
                "solver": ["newton-cg", "lbfgs", "liblinear"],
                "max_iter": [100, 1000, 1500, 2000],
            }
            scoring = [
                "accuracy",
                "f1_weighted",
                "precision_weighted",
                "recall_weighted",
            ]
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                n_jobs=1,
                cv=cv_inner,
                scoring=scoring,
                refit="f1_weighted",
            )

        if grid_search is True and ml_model == 2:
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
            param_grid = {
                "classifier__n_estimators": [10, 100, 150, 200],
                "classifier__max_features": ["auto", "sqrt", "log2"],
                "classifier__max_depth": [2, 4, 5, 6, None],
                "classifier__criterion": ["gini", "entropy"],
            }
            scoring = [
                "accuracy",
                "f1_weighted",
                "precision_weighted",
                "recall_weighted",
            ]
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                n_jobs=1,
                cv=cv_inner,
                scoring=scoring,
                refit="f1_weighted",
            )

        cv = KFold(n_splits=test_split_ratio, shuffle=True, random_state=random_state)

        if grid_search == 1:
            f1 = cross_val_score(
                search, features, target, scoring="f1_weighted", cv=cv, n_jobs=1
            ).mean()
            if ml_model == 1:
                mlflow.log_param("max_iter", search.best_params_["max_iter"])
                mlflow.log_param("logreg_c", search.best_params_["C"])
            if ml_model == 2:
                mlflow.log_param("n_estimators", search.best_params_["n_estimators"])
            search.fit(features, target)
        else:
            f1 = cross_val_score(
                pipeline, features, target, scoring="f1_weighted", cv=cv, n_jobs=1
            ).mean()
            pipeline.fit(features, target)

        if use_feature_selection == 0:
            mlflow.log_param("use_feature_selection", "None")
        if use_feature_selection == 1:
            mlflow.log_param("use_feature_selection", "SelectFromModel")
        if use_feature_selection == 2:
            mlflow.log_param("use_feature_selection", "VarianceThreshold")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("splits", test_split_ratio)

        mlflow.log_metric("f1-score", f1)

        click.echo(f"f1-score: {f1}.")

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
