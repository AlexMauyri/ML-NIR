"""
inference_pipeline.py – Load a model from MLflow and produce predictions.

Interface
---------
    # Load by run id
    ip = InferencePipeline.from_mlflow(SVRModel, run_id="abc123")

    # Load by registered model name + version or stage
    ip = InferencePipeline.from_registry(SVRModel, name="SVRModel", version=3)
    ip = InferencePipeline.from_registry(SVRModel, name="SVRModel", stage="Production")

    predictions = ip.run(raw_df)
    ip.plot(y_actual=y_test)        # overlay actuals if available
    series = ip.as_series()         # predictions as a DatetimeIndex Series
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import TARGET_COLUMN
from features.engineering import create_ta_features
from models.base_model import BaseModel
from visualization.plots import plot_predictions

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Load a trained model from MLflow and score new data.

    The model source is always MLflow – either a specific run or a registered
    model version/stage.  There is no dependency on local joblib files.
    """

    def __init__(self, model: BaseModel) -> None:
        self.model = model
        self._predictions: np.ndarray | None = None
        self._index = None

    @classmethod
    def from_mlflow(
        cls,
        model_cls: type[BaseModel],
        run_id: str,
        artifact_path: str = "model",
    ) -> "InferencePipeline":
        """
        Load a model that was logged inside a specific MLflow run.

        Parameters
        ----------
        model_cls      : SVRModel, XGBModel, or RFModel
        run_id         : the MLflow run id (printed by train.py, or found in the UI)
        artifact_path  : artifact subdirectory used during training (default "model")
        """
        logger.info(
            "InferencePipeline: loading %s from MLflow run %s",
            model_cls.__name__, run_id,
        )
        model = model_cls.from_mlflow_run(run_id, artifact_path=artifact_path)
        return cls(model)

    @classmethod
    def from_registry(
        cls,
        model_cls: type[BaseModel],
        name: str,
        version: int | None = None,
        stage: str | None = None,
    ) -> "InferencePipeline":
        """
        Load a model from the MLflow Model Registry by name and version or stage.

        Exactly one of *version* or *stage* must be provided.

        Parameters
        ----------
        model_cls : SVRModel, XGBModel, or RFModel  (determines the MLflow flavor)
        name      : registered model name (e.g. ``"SVRModel"``)
        version   : integer version number
        stage     : lifecycle stage – ``"Staging"``, ``"Production"``, or ``"Archived"``
        """
        if (version is None) == (stage is None):
            raise ValueError(
                "Provide exactly one of version or stage, not both (or neither).",
                )

        if version is not None:
            model_uri = f"models:/{name}/{version}"
        else:
            model_uri = f"models:/{name}/{stage}"

        logger.info(
            "InferencePipeline: loading %s from registry — uri: %s",
            model_cls.__name__, model_uri,
        )

        flavor = model_cls.get_mlflow_flavor()
        inner_model = flavor.load_model(model_uri)

        instance = model_cls.__new__(model_cls)
        BaseModel.__init__(instance)
        instance._model = inner_model

        logger.info("Loaded %s from registry: %s", model_cls.__name__, model_uri)
        return cls(instance)

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
        Plot predictions, overlaying actuals if provided.
        """
        if self._predictions is None:
            raise RuntimeError("Call InferencePipeline.run() first.")
        model_name = type(self.model).__name__
        if y_actual is not None:
            plot_predictions(
                self._predictions, y_actual,
                title=f"{model_name} – Inference vs Actual",
            )
        else:
            preds = self.as_series()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=preds, ax=ax, color="tomato", linewidth=2)
            ax.set_title(f"{model_name} – Predictions", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Predicted Close", fontsize=12)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
