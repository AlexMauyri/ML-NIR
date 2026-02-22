"""
training_pipeline.py – Orchestrates model tuning, evaluation and persistence.

Interface
---------
    tp = TrainingPipeline(model=SVRModel(), n_trials=50)
    tp.run(data_pipeline)
    tp.save_model()
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from config.settings import N_TRIALS, SVR_MODEL_FILE, XGB_MODEL_FILE, RF_MODEL_FILE
from models import BaseModel, SVRModel, XGBModel, RFModel
from visualization import plot_predictions
from .data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

_DEFAULT_PATHS: dict[type, Path] = {
    SVRModel: SVR_MODEL_FILE,
    XGBModel: XGB_MODEL_FILE,
    RFModel: RF_MODEL_FILE,
}


class TrainingPipeline:
    """
    Single-responsibility: take a DataPipeline, tune a model, evaluate it,
    and optionally save it.

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

    def run(self, data: DataPipeline) -> "TrainingPipeline":
        """
        Tune the model on *data*'s train split, then evaluate on the test split.
        """
        X_train, X_test, y_train, y_test = data.get_splits()
        model_name = type(self.model).__name__

        logger.info("TrainingPipeline: tuning %s", model_name)
        self.model.tune(X_train, y_train, X_test, y_test, n_trials=self.n_trials)

        self.y_pred = self.model.predict(X_test)
        self.test_mape = mean_absolute_percentage_error(y_test, self.y_pred)

        logger.info("%s final test MAPE: %.4f%%", model_name, self.test_mape * 100)
        return self

    def save_model(self, path: Path | None = None) -> None:
        """Persist the fitted model. Uses a sensible default path if none given."""
        if path is None:
            path = _DEFAULT_PATHS.get(type(self.model))
            if path is None:
                raise ValueError(
                    f"No default save path for {type(self.model).__name__}. "
                    "Pass an explicit path."
                )
        self.model.save(path)

    def plot(self, data: DataPipeline) -> None:
        """Render actual-vs-predicted chart for the test set."""
        _, _, _, y_test = data.get_splits()
        if self.y_pred is None:
            raise RuntimeError("Call TrainingPipeline.run() before plot().")
        plot_predictions(
            self.y_pred,
            y_test,
            title=f"{type(self.model).__name__} – Predicted vs Actual",
        )

    @property
    def summary(self) -> dict:
        return {
            "model":       type(self.model).__name__,
            "best_params": self.model.best_params,
            "test_mape":   self.test_mape,
        }
