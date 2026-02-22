"""
train.py – Top-level script for training and evaluating models.

Usage
-----
    # Train both models with default settings
    python entrypoint/train.py

    # Train only SVR with 100 Optuna trials, cache processed data
    python entrypoint/train.py --model svr --trials 100 --save-data

    # Train XGBoost and save the fitted model to disk
    python entrypoint/train.py --model xgb --save-model

Run from the project root so relative imports resolve correctly:
    cd project/
    python entrypoint/train.py --model all
"""
import argparse
import logging
import sys
from pathlib import Path

from config.logger import setup_logging
from config.settings import N_TRIALS

from models import SVRModel, XGBModel
from pipelines import DataPipeline
from pipelines import TrainingPipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

setup_logging()

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Sberbank price prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["svr", "xgb", "all"],
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
        "--save-model",
        action="store_true",
        help="Save the fitted model(s) to models/saved/",
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
    save: bool,
) -> None:
    name = type(model_instance).__name__
    logger.info("=" * 60)
    logger.info("Training %s", name)
    logger.info("=" * 60)

    tp = TrainingPipeline(model=model_instance, n_trials=n_trials)
    tp.run(data)

    logger.info("Summary: %s", tp.summary)
    tp.plot(data)

    if save:
        tp.save_model()
        logger.info("%s saved successfully", name)


def main() -> None:
    args = parse_args()
    logger.info("train.py started | model=%s | trials=%d", args.model, args.trials)

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
        run_model(SVRModel(), data, args.trials, args.save_model)

    if args.model in ("xgb", "all"):
        run_model(XGBModel(), data, args.trials, args.save_model)

    logger.info("train.py finished")


if __name__ == "__main__":
    main()
