"""
loader.py – Functions that read raw CSV files from data/raw/ and return
clean DataFrames.  No feature engineering happens here.
"""
import logging
from pathlib import Path

import pandas as pd

from config.settings import SBERBANK_CSV, CB_RATE_CSV, MOSCOW_IDX_CSV

logger = logging.getLogger(__name__)


def load_sberbank(path: Path = SBERBANK_CSV) -> pd.DataFrame:
    """
    Read Sberbank OHLCV data.

    Returns a DataFrame with a DatetimeIndex named 'Date' and a single
    numeric column 'Close'.
    """
    logger.info("Loading Sberbank prices from %s", path)
    df = pd.read_csv(path, delimiter=";", dtype={"<DATE>": str})
    df = df[["<DATE>", "<CLOSE>"]]
    df = df.rename(columns={col: col[1:-1].capitalize() for col in df.columns})
    df["Date"] = pd.to_datetime(df["Date"], format="%y%m%d")
    df = df.set_index("Date")
    logger.debug("Loaded %d rows of Sberbank data", len(df))
    return df


def load_cb_rate(path: Path = CB_RATE_CSV) -> pd.DataFrame:
    """
    Read the Central Bank key-rate table.

    Returns a DataFrame indexed by date with a single column 'Ставка'.
    """
    logger.info("Loading CB key rate from %s", path)
    df = pd.read_csv(path, parse_dates=["Дата"]).set_index("Дата")
    logger.debug("Loaded %d CB rate records", len(df))
    return df


def load_moscow_index(path: Path = MOSCOW_IDX_CSV) -> pd.DataFrame:
    """
    Read the MOEX (Moscow Exchange) index closing prices.

    Returns a DataFrame indexed by trade date with a column 'index_close'.
    """
    logger.info("Loading MOEX index from %s", path)
    df = (
        pd.read_csv(path, parse_dates=["TRADEDATE"], delimiter=";", decimal=",")
        .set_index("TRADEDATE")
        .rename(columns={"CLOSE": "index_close"})
    )
    logger.debug("Loaded %d MOEX index records", len(df))
    return df
