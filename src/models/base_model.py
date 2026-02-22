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
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Minimal interface for all predictive models in this project.

    Concrete subclasses must implement ``tune`` and ``predict``.
    ``save`` / ``load`` work out-of-the-box via joblib.
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

    def save(self, path: Path) -> None:
        """Persist the fitted model to *path* using joblib."""
        if self._model is None:
            raise RuntimeError("Cannot save: model has not been fitted yet.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Restore a previously saved model from *path*."""
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
