"""
xgb_model.py – XGBoost Regressor with Optuna hyperparameter tuning.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from config.settings import (
    N_TRIALS,
    XGB_N_EST_LOW,   XGB_N_EST_HIGH,
    XGB_MAX_DEPTH_LOW, XGB_MAX_DEPTH_HIGH,
    XGB_MCW_LOW,     XGB_MCW_HIGH,
    XGB_SUBSAMPLE_LOW, XGB_SUBSAMPLE_HIGH,
    XGB_COLSAMPLE_LOW, XGB_COLSAMPLE_HIGH,
    XGB_LAMBDA_LOW,  XGB_LAMBDA_HIGH,
    XGB_EARLY_STOPPING,
    XGB_LEARNING_RATE,
)
from .base_model import BaseModel
from .validation import BestTracker, time_series_cv

logger = logging.getLogger(__name__)


class XGBModel(BaseModel):
    """
    XGBoost Regressor tuned with Optuna (TPE + Hyperband pruner).

    Overrides ``get_mlflow_flavor()`` to use ``mlflow.xgboost`` so that the
    model is logged and loaded with the native XGBoost MLflow flavor instead
    of the generic sklearn flavor.
    """

    @classmethod
    def get_mlflow_flavor(cls):
        """Use the native XGBoost MLflow flavor for richer artifact metadata."""
        import mlflow.xgboost
        return mlflow.xgboost

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
        n_trials: int = N_TRIALS,
    ) -> "XGBModel":
        logger.info("Starting XGBoost hyperparameter search (%d trials)", n_trials)
        tracker = BestTracker()

        def objective(trial: optuna.Trial) -> float:
            params = {
                "tree_method": "exact",
                "n_estimators": trial.suggest_int("n_estimators",
                                                  XGB_N_EST_LOW,
                                                  XGB_N_EST_HIGH),
                "max_depth": trial.suggest_int("max_depth",
                                               XGB_MAX_DEPTH_LOW,
                                               XGB_MAX_DEPTH_HIGH),
                "min_child_weight": trial.suggest_float("min_child_weight",
                                                        XGB_MCW_LOW,
                                                        XGB_MCW_HIGH,
                                                        log=True),
                "learning_rate": XGB_LEARNING_RATE,
                "subsample": trial.suggest_float("subsample",
                                                 XGB_SUBSAMPLE_LOW,
                                                 XGB_SUBSAMPLE_HIGH),
                "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                        XGB_COLSAMPLE_LOW,
                                                        XGB_COLSAMPLE_HIGH),
                "reg_lambda": trial.suggest_float("reg_lambda",
                                                  XGB_LAMBDA_LOW,
                                                  XGB_LAMBDA_HIGH,
                                                  log=True),
                "early_stopping_rounds": XGB_EARLY_STOPPING,
                "n_jobs": -1,
            }
            model = xgb.XGBRegressor(**params)
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
            "XGBoost best params: %s  (test MAPE=%.6f)",
            self._best_params, tracker.best.test_mape,
        )

        self._model = xgb.XGBRegressor(**self._best_params)
        self._model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBModel must be tuned before predicting.")
        return self._model.predict(X)
