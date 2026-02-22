"""
svr_model.py – Support Vector Regressor with Optuna hyperparameter tuning.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from config.settings import (
    N_TRIALS,
    SVR_C_LOW, SVR_C_HIGH,
    SVR_EPS_LOW, SVR_EPS_HIGH,
)
from .base_model import BaseModel
from .validation import BestTracker, time_series_cv

logger = logging.getLogger(__name__)


class SVRModel(BaseModel):
    """
    MinMaxScaler → SVR pipeline tuned with Optuna (TPE + Hyperband pruner).

    Scaling is embedded inside the sklearn Pipeline so it is applied
    consistently to any new data passed to ``predict``.
    """

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
        n_trials: int = N_TRIALS,
    ) -> "SVRModel":
        logger.info("Starting SVR hyperparameter search (%d trials)", n_trials)
        tracker = BestTracker()

        def objective(trial: optuna.Trial) -> float:
            params = {
                "C":       trial.suggest_float("C", SVR_C_LOW, SVR_C_HIGH),
                "epsilon": trial.suggest_float("epsilon", SVR_EPS_LOW, SVR_EPS_HIGH),
                "kernel":  "rbf",
            }
            model = Pipeline([
                ("scaler", MinMaxScaler()),
                ("svr",    SVR(**params)),
            ])
            return time_series_cv(model,
                                  params,
                                  X_train,
                                  y_train,
                                  X_test,
                                  y_test,
                                  tracker)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self._best_params = tracker.best_params
        logger.info(
            "SVR best params: %s  (test MAPE=%.6f)",
            self._best_params, tracker.best.test_mape,
        )

        self._model = Pipeline([
            ("scaler", MinMaxScaler()),
            ("svr",    SVR(**self._best_params)),
        ])
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SVRModel must be tuned before predicting.")
        return self._model.predict(X)
