from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


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
) -> None:
    features, target = get_dataset(dataset_path)

    with mlflow.start_run():
        if (ml_model == 1):
            pipeline = create_pipeline_LogisticRegression(use_scaler, max_iter, logreg_c, use_feature_selection, random_state)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
            mlflow.log_param("ml_model", "LogisticRegression")
        if (ml_model == 2):
            pipeline = create_pipeline_RandomForest(use_scaler, n_estimators, use_feature_selection, random_state)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("ml_model", "RandomForestClassifier")

        pipeline.fit(features, target)

        cv = KFold(n_splits=test_split_ratio, shuffle=True, random_state=random_state)

        f1 = cross_val_score(pipeline, features, target, scoring='f1_weighted', cv=cv, n_jobs = 1).mean()
        
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