"""
train.py – Top-level training script.

Every run is automatically logged to MLflow (params, metrics, model artifact).
The run_id is printed at the end – copy it for use with inference.py.

Usage
-----
    # Train all three models (logged to MLflow automatically)
    python entrypoint/train.py

    # Train only SVR with 100 Optuna trials
    python entrypoint/train.py --model svr --trials 100

    # Train XGBoost and register the best model in the MLflow Model Registry
    python entrypoint/train.py --model xgb --register-model

    # Register under a custom name
    python entrypoint/train.py --model rf --register-model --register-name RF_production

    # Use cached feature data (skip re-engineering)
    python entrypoint/train.py --model all --use-cache

Run from the project root:
    cd project/
    python entrypoint/train.py --model svr
"""
import argparse
import logging
import sys
from pathlib import Path

from config.logger import setup_logging
from config.settings import N_TRIALS
from config.mlflow import setup_mlflow
from models import RFModel, SVRModel, XGBModel
from pipelines import DataPipeline, TrainingPipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Sberbank price prediction models and log to MLflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["svr", "xgb", "rf", "all"],
        default="all",
        help="Model(s) to train",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help="Number of Optuna hyperparameter search trials",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the trained model(s) in the MLflow Model Registry",
    )
    parser.add_argument(
        "--register-name",
        type=str,
        default=None,
        help="Registry name override (only used with --register-model\
and a single --model)",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Cache the processed feature matrix to data/processed/",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load features from data/processed/ instead of recomputing them",
    )
    return parser.parse_args()


def run_model(
    model_instance,
    data: DataPipeline,
    n_trials: int,
    register: bool,
    register_name: str | None,
) -> None:
    name = type(model_instance).__name__
    logger.info("=" * 60)
    logger.info("Training %s", name)
    logger.info("=" * 60)

    tp = TrainingPipeline(model=model_instance, n_trials=n_trials)
    tp.run(data)

    logger.info("Summary: %s", tp.summary)
    logger.info("MLflow run_id: %s", tp.run_id)
    tp.plot(data)

    if register:
        version = tp.register_model(registered_name=register_name)
        logger.info(
            "%s registered in MLflow Model Registry (version %s)", name, version
        )


def main() -> None:
    args = parse_args()

    setup_mlflow()
    logger.info(
        "train.py started | model=%s | trials=%d", args.model, args.trials
    )

    if args.use_cache:
        data = DataPipeline.from_processed()
    else:
        data = DataPipeline().run()
        if args.save_data:
            data.save_processed()

    logger.info(
        "Dataset ready — train: %d rows | test: %d rows | features: %s",
        len(data.X_train), len(data.X_test), list(data.X_train.columns),
    )

    if args.model in ("svr", "all"):
        run_model(SVRModel(), data, args.trials, args.register_model, args.register_name)

    if args.model in ("xgb", "all"):
        run_model(XGBModel(), data, args.trials, args.register_model, args.register_name)

    if args.model in ("rf", "all"):
        run_model(RFModel(), data, args.trials, args.register_model, args.register_name)

    logger.info("train.py finished")


if __name__ == "__main__":
    main()
