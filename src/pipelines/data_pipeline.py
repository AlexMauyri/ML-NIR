"""
data_pipeline.py – Responsible for one thing: producing a clean, feature-rich
dataset ready for model training or inference.

Interface
---------
    dp = DataPipeline()
    dp.run()                        # load → engineer → split
    X_train, X_test, y_train, y_test = dp.get_splits()
    dp.save_processed()             # persist features to data/processed/
"""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import (
    TARGET_COLUMN,
    TEST_SIZE,
    PROCESSED_FEATURES_FILE,
)
from features import load_sberbank, create_ta_features

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Encapsulates raw data loading → feature engineering → train/test split.

    Attributes set after ``run()``
    --------------------------------
    features   : full feature DataFrame (X + y together)
    X_train, X_test, y_train, y_test : ready-to-use splits
    """

    def __init__(
        self,
        target: str = TARGET_COLUMN,
        test_size: float = TEST_SIZE,
    ) -> None:
        self.target = target
        self.test_size = test_size

        self.features: pd.DataFrame | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def run(self) -> "DataPipeline":
        """Execute the full data preparation sequence."""
        logger.info("DataPipeline started")
        raw = load_sberbank()
        self.features = create_ta_features(raw, target=self.target)
        self._split()
        logger.info("DataPipeline finished — train=%d, test=%d",
                    len(self.X_train), len(self.X_test))
        return self

    def get_splits(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return (X_train, X_test, y_train, y_test). Raises if not run yet."""
        self._require_run()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def mlflow_datasets(self) -> tuple:
        """
        Return a (train_dataset, test_dataset) pair of ``mlflow.data.PandasDataset``
        objects ready to be passed to ``mlflow.log_input()``.

        Each dataset is constructed with ``mlflow.data.from_pandas()``:
          - ``source`` points to the Sberbank raw CSV so MLflow records data lineage
          - ``name``   identifies the split ("sberbank_train" / "sberbank_test")
          - ``targets`` is set to the target column name

        Usage inside a run::

            train_ds, test_ds = data.mlflow_datasets()
            mlflow.log_input(train_ds, context="training")
            mlflow.log_input(test_ds,  context="testing")
        """
        self._require_run()

        from config.settings import SBERBANK_CSV

        train_df = self.X_train.copy()
        train_df[self.target] = self.y_train

        test_df = self.X_test.copy()
        test_df[self.target] = self.y_test

        source = str(SBERBANK_CSV)

        train_dataset = mlflow.data.from_pandas(
            train_df,
            source=source,
            name="sberbank_train",
            targets=self.target,
        )
        test_dataset = mlflow.data.from_pandas(
            test_df,
            source=source,
            name="sberbank_test",
            targets=self.target,
        )

        logger.debug(
            "MLflow datasets prepared — train: %d rows | test: %d rows",
            len(train_df), len(test_df),
        )
        return train_dataset, test_dataset

    def save_processed(self, path: Path = PROCESSED_FEATURES_FILE) -> None:
        """Persist the full feature DataFrame to *path* as Parquet."""
        self._require_run()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.features.to_parquet(path)
        logger.info("Processed features saved to %s", path)

    @classmethod
    def from_processed(
        cls,
        path: Path = PROCESSED_FEATURES_FILE,
        target: str = TARGET_COLUMN,
        test_size: float = TEST_SIZE,
    ) -> "DataPipeline":
        """
        Fast-path: skip feature engineering and load a previously saved
        Parquet file from data/processed/.
        """
        logger.info("Loading processed features from %s", path)
        dp = cls(target=target, test_size=test_size)
        dp.features = pd.read_parquet(path)
        dp._split()
        logger.info(
            "DataPipeline loaded from cache — train=%d, test=%d",
            len(dp.X_train), len(dp.X_test),
        )
        return dp

    def _split(self) -> None:
        X = self.features.drop(columns=[self.target])
        y = self.features[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        logger.debug(
            "Split → X_train %s | X_test %s | features: %s",
            self.X_train.shape, self.X_test.shape, list(X.columns),
        )

    def _require_run(self) -> None:
        if self.features is None:
            raise RuntimeError("Call DataPipeline.run() (or from_processed()) first.")
