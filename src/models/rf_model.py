"""
rf_model.py – Random Forest Regressor with Optuna hyperparameter tuning.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config.settings import (
    N_TRIALS,
    RF_N_EST_LOW,   RF_N_EST_HIGH,
    RF_MAX_DEPTH_LOW, RF_MAX_DEPTH_HIGH,
    RF_MIN_SAMPLES_SPLIT_LOW, RF_MIN_SAMPLES_SPLIT_HIGH,
    RF_MIN_SAMPLES_LEAF_LOW,  RF_MIN_SAMPLES_LEAF_HIGH,
    RF_MAX_FEATURES,
)
from models.base_model import BaseModel
from models.validation import BestTracker, time_series_cv

logger = logging.getLogger(__name__)


class RFModel(BaseModel):
    """
    Random Forest Regressor tuned with Optuna (TPE + Hyperband pruner).

    No scaling is needed – tree-based models are scale-invariant.
    """

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
        n_trials: int = N_TRIALS,
    ) -> "RFModel":
        logger.info("Starting Random Forest hyperparameter search (%d trials)", n_trials)
        tracker = BestTracker()

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators",
                                                  RF_N_EST_LOW,
                                                  RF_N_EST_HIGH),
                "max_depth": trial.suggest_int("max_depth",
                                               RF_MAX_DEPTH_LOW,
                                               RF_MAX_DEPTH_HIGH),
                "min_samples_split": trial.suggest_int("min_samples_split",
                                                       RF_MIN_SAMPLES_SPLIT_LOW,
                                                       RF_MIN_SAMPLES_SPLIT_HIGH),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf",
                                                      RF_MIN_SAMPLES_LEAF_LOW,
                                                      RF_MIN_SAMPLES_LEAF_HIGH),
                "max_features": trial.suggest_categorical("max_features",
                                                          RF_MAX_FEATURES),
                "n_jobs": -1,
            }
            model = RandomForestRegressor(**params)
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
            "Random Forest best params: %s  (test MAPE=%.6f)",
            self._best_params, tracker.best.test_mape,
        )

        self._model = RandomForestRegressor(**self._best_params)
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RFModel must be tuned before predicting.")
        return self._model.predict(X)

    @property
    def feature_importances(self) -> dict[str, float]:
        """
        Return a dict mapping feature name → importance score.
        Useful for feeding into ``visualization.plots.plot_feature_importance``.
        """
        if self._model is None:
            raise RuntimeError(
                "RFModel must be tuned before accessing feature importances.",
                )
        return dict(zip(self._model.feature_names_in_, self._model.feature_importances_))
