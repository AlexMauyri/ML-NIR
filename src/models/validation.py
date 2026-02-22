"""
validation.py – Walk-forward cross-validation used by all model tuners.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from config.settings import N_CV_SPLITS

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    train_mape: float
    test_mape:  float
    params:     dict[str, Any]


@dataclass
class BestTracker:
    """Thread-safe-enough tracker of the best trial seen across Optuna trials."""
    best: TrialResult | None = field(default=None)

    def update(self, result: TrialResult) -> None:
        if self.best is None or result.test_mape < self.best.test_mape:
            self.best = result
            logger.debug(
                "New best → test MAPE %.6f with params %s",
                result.test_mape, result.params,
            )

    @property
    def best_params(self) -> dict:
        if self.best is None:
            raise RuntimeError("No trial has been recorded yet.")
        return self.best.params


def _fit(model, X_tr, y_tr, X_val=None, y_val=None) -> None:
    """Fit *model*, supplying an eval set when it is an XGBRegressor."""
    if isinstance(model, xgb.XGBRegressor) and X_val is not None:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_tr, y_tr)


def time_series_cv(
    model,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    tracker: BestTracker,
    n_splits: int = N_CV_SPLITS,
) -> float:
    """
    Walk-forward cross-validation on the training set, followed by a
    held-out test evaluation.

    Parameters
    ----------
    model    : sklearn-compatible estimator (unfitted)
    params   : hyperparameter dict used to build *model* (for logging only)
    tracker  : updated in-place when a new best is found
    n_splits : number of TimeSeriesSplit folds

    Returns
    -------
    float – mean train MAPE across CV folds (Optuna minimises this)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_mapes: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        _fit(model, X_tr, y_tr, X_val, y_val)
        fold_mape = mean_absolute_percentage_error(y_val, model.predict(X_val))
        fold_mapes.append(fold_mape)
        logger.debug("  Fold %d MAPE: %.6f", fold, fold_mape)

    _fit(model, X_train, y_train, X_test, y_test)
    test_mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
    train_mape = float(np.mean(fold_mapes))

    result = TrialResult(train_mape=train_mape, test_mape=test_mape, params=params)
    tracker.update(result)

    logger.info("Trial → train MAPE: %.6f | test MAPE: %.6f", train_mape, test_mape)
    return train_mape
