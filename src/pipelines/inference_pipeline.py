"""
inference_pipeline.py – Load a saved model and produce predictions on new data.

Interface
---------
    ip = InferencePipeline.from_saved(SVRModel, path=SVR_MODEL_FILE)
    predictions = ip.run(new_data_df)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import SVR_MODEL_FILE, XGB_MODEL_FILE, TARGET_COLUMN
from features import create_ta_features
from models import BaseModel, SVRModel, XGBModel
from visualization import plot_predictions

logger = logging.getLogger(__name__)

_DEFAULT_PATHS: dict[type, Path] = {
    SVRModel: SVR_MODEL_FILE,
    XGBModel: XGB_MODEL_FILE,
}


class InferencePipeline:
    """
    Load a persisted model and run predictions on raw or processed data.

    Typical use-case: you already have a trained model on disk and want to
    score new market data without re-training.
    """

    def __init__(self, model: BaseModel) -> None:
        self.model = model
        self._predictions: np.ndarray | None = None
        self._index = None

    @classmethod
    def from_saved(
        cls,
        model_cls: type[BaseModel],
        path: Path | None = None,
    ) -> "InferencePipeline":
        """
        Restore a model from disk.

        Parameters
        ----------
        model_cls : SVRModel or XGBModel (used to locate the default path)
        path      : explicit file path; falls back to the default for *model_cls*
        """
        if path is None:
            path = _DEFAULT_PATHS.get(model_cls)
            if path is None:
                raise ValueError(
                    f"No default path known for {model_cls.__name__}. "
                    "Pass an explicit path."
                )
        model = model_cls.load(path)
        logger.info("InferencePipeline: loaded %s from %s", model_cls.__name__, path)
        return cls(model)

    def run(
        self,
        raw_df: pd.DataFrame,
        apply_feature_engineering: bool = True,
        target: str = TARGET_COLUMN,
    ) -> np.ndarray:
        """
        Score *raw_df* and return an array of predictions.

        Parameters
        ----------
        raw_df                    : raw price DataFrame (same format as data/raw/)
        apply_feature_engineering : set False if *raw_df* is already a feature matrix
        target                    : name of the target column to drop before scoring
        """
        if apply_feature_engineering:
            logger.info("InferencePipeline: applying feature engineering")
            df = create_ta_features(raw_df, target=target)
        else:
            df = raw_df.copy()

        X = df.drop(columns=[target], errors="ignore")
        self._index = X.index
        self._predictions = self.model.predict(X)

        logger.info(
            "InferencePipeline: produced %d predictions (range %.2f – %.2f)",
            len(self._predictions),
            float(self._predictions.min()),
            float(self._predictions.max()),
        )
        return self._predictions

    def as_series(self) -> pd.Series:
        """Return predictions as a Series with the original DatetimeIndex."""
        if self._predictions is None:
            raise RuntimeError("Call InferencePipeline.run() first.")
        return pd.Series(self._predictions, index=self._index, name="prediction")

    def plot(self, y_actual: pd.Series | None = None) -> None:
        """
        Plot predictions.  If *y_actual* is provided, overlay it for comparison.
        """
        if self._predictions is None:
            raise RuntimeError("Call InferencePipeline.run() first.")
        if y_actual is not None:
            plot_predictions(
                self._predictions, y_actual,
                title=f"{type(self.model).__name__} – Inference vs Actual",
            )
        else:
            import matplotlib.pyplot as plt
            import seaborn as sns
            preds = self.as_series()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=preds, ax=ax, color="tomato", linewidth=2)
            ax.set_title(f"{type(self.model).__name__} – Predictions",
                         fontsize=14, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Predicted Close", fontsize=12)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()
