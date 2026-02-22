"""
inference.py – Top-level script for running predictions with a saved model.

Usage
-----
    # Score new data with the saved SVR model
    python entrypoint/inference.py --model svr --input data/raw/sberbank_stock_prices.csv

    # Score and plot against known actuals
    python entrypoint/inference.py --model xgb --input data/raw/new_data.csv --plot

    # Save predictions to a CSV file
    python entrypoint/inference.py --model svr
    --input data/raw/new_data.csv --output preds.csv

Run from the project root:
    cd project/
    python entrypoint/inference.py --model svr
"""
import argparse
import sys
from pathlib import Path

from config.logger import setup_logging

import logging
import pandas as pd

from config.settings import SBERBANK_CSV, TARGET_COLUMN
from models import SVRModel, XGBModel
from pipelines import InferencePipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

setup_logging()


logger = logging.getLogger(__name__)

_MODEL_MAP = {"svr": SVRModel, "xgb": XGBModel}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sberbank price predictions using a saved model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["svr", "xgb"],
        required=True,
        help="Which saved model to load",
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
        help="Plot predictions (if the input contains a Close column, overlay it)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(
        "inference.py started | model=%s | input=%s", args.model, args.input
    )

    model_cls = _MODEL_MAP[args.model]
    pipeline = InferencePipeline.from_saved(model_cls)

    raw = pd.read_csv(args.input, delimiter=";", dtype={"<DATE>": str})
    raw = raw[["<DATE>", "<CLOSE>"]].rename(
        columns={col: col[1:-1].capitalize() for col in raw.columns}
    )
    raw["Date"] = pd.to_datetime(raw["Date"], format="%y%m%d")
    raw = raw.set_index("Date")

    predictions = pipeline.run(raw, apply_feature_engineering=True, target=TARGET_COLUMN)
    series = pipeline.as_series()

    logger.info("Inference complete – %d predictions", len(predictions))

    if args.output:
        series.to_csv(args.output, header=True)
        logger.info("Predictions saved to %s", args.output)

    if args.plot:
        actuals = None
        if TARGET_COLUMN in raw.columns and raw.index.isin(series.index).any():
            actuals = raw.loc[series.index, TARGET_COLUMN].dropna()
        pipeline.plot(y_actual=actuals)

    logger.info("inference.py finished")


if __name__ == "__main__":
    main()
