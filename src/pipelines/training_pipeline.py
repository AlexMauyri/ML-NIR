"""
training_pipeline.py – Orchestrates model tuning, evaluation, MLflow logging,
and optional model registry registration.

Interface
---------
    tp = TrainingPipeline(model=SVRModel(), n_trials=50)
    tp.run(data_pipeline)           # tunes + evaluates + logs everything to MLflow
    print(tp.run_id)                # MLflow run id for later retrieval
    tp.register_model("SVR_prod")   # promote to MLflow Model Registry (optional)
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from config.mlflow import get_or_create_experiment
from config.settings import N_TRIALS
from models.base_model import BaseModel
from visualization.plots import plot_predictions
from .data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Single-responsibility: take a DataPipeline, tune a model, evaluate it,
    log everything to MLflow, and optionally register the model.

    What gets logged to MLflow per run
    -----------------------------------
    Tags    : model_type, n_trials, n_features, train_rows, test_rows
    Params  : all best hyperparameters from Optuna
    Metrics : test_mape, train_mape
    Artifact: fitted model (via the model's native MLflow flavor)

    Parameters
    ----------
    model    : an untuned BaseModel subclass instance
    n_trials : number of Optuna optimisation trials
    """

    def __init__(self, model: BaseModel, n_trials: int = N_TRIALS) -> None:
        self.model = model
        self.n_trials = n_trials

        self.y_pred:    np.ndarray | None = None
        self.test_mape: float | None = None
        self.train_mape: float | None = None
        self.run_id:    str | None = None

    def run(self, data: DataPipeline) -> "TrainingPipeline":
        """
        Tune the model, evaluate on the test split, and log everything to
        MLflow.  Nested child runs for each Optuna trial are opened
        automatically inside ``time_series_cv``.
        """
        X_train, X_test, y_train, y_test = data.get_splits()
        model_name = type(self.model).__name__
        experiment_id = get_or_create_experiment(model_name)
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{model_name}_optuna_{self.n_trials}trials",
        ) as run:
            self.run_id = run.info.run_id
            logger.info(
                "MLflow run started — experiment: Sberbank_%s | run_id: %s",
                model_name, self.run_id,
            )

            mlflow.set_tags({
                "model_type": model_name,
                "n_trials":   str(self.n_trials),
            })

            train_dataset, test_dataset = data.mlflow_datasets()
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset,  context="testing")
            logger.info("Dataset inputs logged to MLflow (train + test splits)")

            logger.info("TrainingPipeline: tuning %s", model_name)
            self.model.tune(X_train, y_train, X_test, y_test, n_trials=self.n_trials)

            self.y_pred = self.model.predict(X_test)
            self.test_mape = mean_absolute_percentage_error(y_test, self.y_pred)

            mlflow.log_params({
                k: str(v) for k, v in self.model.best_params.items()
            })

            mlflow.log_metric("test_mape", self.test_mape)
            self.model.log_to_mlflow(name="model")
            logger.info("Predictions plot logged to MLflow (plots/predictions.png)")

            logger.info(
                "%s — best test MAPE: %.4f%%  |  run_id: %s",
                model_name, self.test_mape * 100, self.run_id,
            )

        return self

    def register_model(self, registered_name: str | None = None) -> str:
        """
        Register the logged model in the MLflow Model Registry.

        This promotes the artifact from the run into a named, versioned entry
        in the registry so it can be loaded by name (e.g. for production).

        Parameters
        ----------
        registered_name : registry name, e.g. ``"SVRModel_production"``.
                          Defaults to the model class name.

        Returns
        -------
        str – the registered model version
        """
        if self.run_id is None:
            raise RuntimeError("Call TrainingPipeline.run() before register_model().")

        if registered_name is None:
            registered_name = type(self.model).__name__

        model_uri = f"runs:/{self.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=registered_name)

        logger.info(
            "Model registered as '%s' version %s (run_id=%s)",
            registered_name, result.version, self.run_id,
        )
        return result.version

    def plot(self, data: DataPipeline) -> None:
        """Show the actual-vs-predicted chart interactively."""
        _, _, _, y_test = data.get_splits()
        if self.y_pred is None:
            raise RuntimeError("Call TrainingPipeline.run() before plot().")
        fig = plot_predictions(
            self.y_pred,
            y_test,
            title=f"{type(self.model).__name__} – Predicted vs Actual",
        )
        plt.show()
        plt.close(fig)

    @property
    def summary(self) -> dict:
        return {
            "model":       type(self.model).__name__,
            "run_id":      self.run_id,
            "best_params": self.model.best_params,
            "test_mape":   self.test_mape,
        }
