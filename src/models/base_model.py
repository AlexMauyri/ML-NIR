"""
base_model.py – Abstract contract that every concrete model must satisfy.

Enforcing this interface means pipelines can swap SVR for XGBoost (or any
future model) without changing any pipeline code.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Minimal interface for all predictive models in this project.

    Concrete subclasses must implement ``tune`` and ``predict``.
    ``save`` / ``load`` work via joblib (local fallback).
    ``log_to_mlflow`` / ``from_mlflow_run`` handle MLflow persistence.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._best_params: dict = {}

    @abstractmethod
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_trials: int,
    ) -> "BaseModel":
        """
        Run hyperparameter search, fit the best model, store it in
        ``self._model``.  Must return *self* to allow method chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for *X*. Requires a prior call to ``tune``."""

    @classmethod
    def get_mlflow_flavor(cls):
        """
        Return the mlflow flavor module used to log/load this model type.

        Default: ``mlflow.sklearn``  (SVRModel and RFModel inherit this).
        XGBModel overrides this to return ``mlflow.xgboost``.
        """
        import mlflow.sklearn
        return mlflow.sklearn

    def log_to_mlflow(self, artifact_path: str = "model") -> None:
        """
        Log the fitted inner model as an MLflow artifact inside the
        *currently active* run.  Must be called inside a ``mlflow.start_run``
        context.

        Parameters
        ----------
        artifact_path : subdirectory inside the run's artifact store
        """
        if self._model is None:
            raise RuntimeError("Cannot log to MLflow: model has not been fitted yet.")
        flavor = self.__class__.get_mlflow_flavor()
        flavor.log_model(self._model, artifact_path=artifact_path)
        logger.info(
            "Logged %s to MLflow artifact path '%s'",
            type(self).__name__, artifact_path,
        )

    @classmethod
    def from_mlflow_run(cls, run_id: str, artifact_path: str = "model") -> "BaseModel":
        """
        Restore a fitted model from an MLflow run.

        The inner sklearn/xgb model is loaded via the correct flavor and
        wrapped back into a ``cls`` instance so that the rest of the pipeline
        can treat it identically to a freshly tuned model.

        Parameters
        ----------
        run_id        : MLflow run id (shown in the UI or returned by TrainingPipeline)
        artifact_path : artifact subdirectory used when the model was logged

        Returns
        -------
        An instance of *cls* with ``_model`` and ``_best_params`` populated.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        flavor = cls.get_mlflow_flavor()

        logger.info("Loading %s from MLflow — run_id=%s", cls.__name__, run_id)
        inner_model = flavor.load_model(model_uri)

        run = mlflow.get_run(run_id)
        params = dict(run.data.params)

        instance = cls.__new__(cls)
        BaseModel.__init__(instance)
        instance._model = inner_model
        instance._best_params = params
        logger.info("Successfully restored %s (params: %s)", cls.__name__, params)
        return instance

    def save(self, path: Path) -> None:
        """Persist the fitted model to *path* using joblib."""
        if self._model is None:
            raise RuntimeError("Cannot save: model has not been fitted yet.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved locally to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Restore a previously saved model from a local joblib file."""
        logger.info("Loading model from %s", path)
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    @property
    def best_params(self) -> dict:
        return self._best_params

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self._model is not None})"
