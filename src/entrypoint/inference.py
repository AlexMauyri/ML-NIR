"""
inference.py – Score new data using a model loaded from MLflow.

Two ways to specify the model source:
  1. By run id          →  --run-id <run_id>
  2. By registry entry  →  --model-name <name> --model-version <n>
                       or  --model-name <name> --model-stage Production

The run_id is printed by train.py at the end of every training run.
You can also find it in the MLflow UI at the tracking URI configured in
config/settings.py (default: project/mlruns/).

Usage
-----
    # Score with SVR, loading from a specific training run
    python entrypoint/inference.py --model svr --run-id abc123def456

    # Score with XGBoost from the Model Registry (specific version)
    python entrypoint/inference.py --model xgb --model-name XGBModel --model-version 2

    # Score with RF from registry Production stage and save predictions
    python entrypoint/inference.py --model rf --model-name RFModel
    --model-stage Production --output preds.csv

    # Plot predictions against actual values
    python entrypoint/inference.py --model svr --run-id abc123 --plot

Run from the project root:
    cd project/
    python entrypoint/inference.py --model svr --run-id <run_id>
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config.logger import setup_logging
from config.mlflow import setup_mlflow
from config.settings import SBERBANK_CSV, TARGET_COLUMN
from models.svr_model import RFModel, SVRModel, XGBModel
from pipelines import InferencePipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

setup_logging()
logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "svr": SVRModel,
    "xgb": XGBModel,
    "rf":  RFModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score new data using a model loaded from MLflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["svr", "xgb", "rf"],
        required=True,
        help="Model type to load",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-id",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="MLflow run id from which to load the model artifact",
    )
    source.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Registered model name (requires --model-version or --model-stage)",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        metavar="VERSION",
        help="Registry model version (used with --model-name)",
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        default=None,
        choices=["Staging", "Production", "Archived"],
        help="Registry lifecycle stage (used with --model-name)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SBERBANK_CSV,
        help="Path to raw input CSV (same format as data/raw/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save predictions as CSV",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot predictions (overlays actuals if input has a Close column)",
    )
    return parser.parse_args()


def _validate_registry_args(args: argparse.Namespace) -> None:
    """Raise if --model-name is given without a version or stage."""
    if args.model_name is not None:
        if args.model_version is None and args.model_stage is None:
            raise SystemExit(
                "error: --model-name requires --model-version or --model-stage"
            )


def _load_pipeline(args: argparse.Namespace) -> InferencePipeline:
    """Construct an InferencePipeline from the CLI source arguments."""
    model_cls = _MODEL_MAP[args.model]

    if args.run_id is not None:
        logger.info("Loading %s from run_id=%s", model_cls.__name__, args.run_id)
        return InferencePipeline.from_mlflow(model_cls, run_id=args.run_id)

    logger.info(
        "Loading %s from registry — name=%s version=%s stage=%s",
        model_cls.__name__, args.model_name, args.model_version, args.model_stage,
    )
    return InferencePipeline.from_registry(
        model_cls,
        name=args.model_name,
        version=args.model_version,
        stage=args.model_stage,
    )


def main() -> None:
    args = parse_args()
    _validate_registry_args(args)

    setup_mlflow()
    logger.info("inference.py started | model=%s", args.model)

    pipeline = _load_pipeline(args)

    raw = pd.read_csv(args.input, delimiter=";", dtype={"<DATE>": str})
    raw = raw[["<DATE>", "<CLOSE>"]].rename(
        columns={col: col[1:-1].capitalize() for col in raw.columns}
    )
    raw["Date"] = pd.to_datetime(raw["Date"], format="%y%m%d")
    raw = raw.set_index("Date")

    pipeline.run(raw, apply_feature_engineering=True, target=TARGET_COLUMN)
    series = pipeline.as_series()
    logger.info("Inference complete — %d predictions", len(series))

    if args.output:
        series.to_csv(args.output, header=True)
        logger.info("Predictions saved to %s", args.output)

    if args.plot:
        actuals = None
        if TARGET_COLUMN in raw.columns:
            actuals = raw.loc[series.index, TARGET_COLUMN].dropna()
        pipeline.plot(y_actual=actuals)

    logger.info("inference.py finished")


if __name__ == "__main__":
    main()
