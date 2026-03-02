"""
engineering.py – All feature-construction logic.

Every function here is a pure transformation: it receives a DataFrame and
returns a new one. No I/O, no side effects.
"""
import logging

import pandas as pd
import talib as ta

from config.settings import (
    NUMBER_OF_LAGS,
    ROLLING_STD,
    ROLLING_MEAN,
    EMA_TIME_PERIOD,
    RSI_TIME_PERIOD,
    TARGET_COLUMN,
)
from .loader import load_cb_rate, load_moscow_index

logger = logging.getLogger(__name__)


def join_key_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the CB key rate onto *df* as column 'Rate'.

    The rate is shifted forward by one period and expressed as a daily
    percentage change so no future information leaks into the model.
    """
    df_rate = load_cb_rate()
    df_rate = df_rate.shift(1).pct_change()
    df.index = pd.to_datetime(df.index)
    df = df.join(df_rate["Ставка"].rename("Rate"), how="left")
    logger.debug("Joined CB key rate (%d non-null values)", df["Rate"].notna().sum())
    return df


def join_moscow_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the MOEX index close (shifted 1 day) onto *df* as 'index_close'.
    """
    df_index = load_moscow_index().shift(1)
    df = df.join(df_index["index_close"], how="left")
    logger.debug(
        "Joined MOEX index (%d non-null values)", df["index_close"].notna().sum()
        )
    return df


def create_lags(df: pd.DataFrame, column: str, n_lags: int) -> pd.DataFrame:
    """
    Append *n_lags* shifted copies of *column* named '<column>_lag_<n>'.
    """
    out = df.copy()
    for lag in range(1, n_lags + 1):
        out[f"{column}_lag_{lag}"] = out[column].shift(lag)
    logger.debug("Created %d lag features for column '%s'", n_lags, column)
    return out


def create_ta_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    n_lags: int = NUMBER_OF_LAGS,
    ema_period: int = EMA_TIME_PERIOD,
    rsi_period: int = RSI_TIME_PERIOD,
    rolling_std: int = ROLLING_STD,
    rolling_mean: int = ROLLING_MEAN,
) -> pd.DataFrame:
    """
    Build the full feature matrix from a raw price DataFrame.

    Transformation order
    --------------------
    1. Join CB key rate
    2. Add price lags 1..n_lags
    3. EMA, ROC, RSI  (TA-Lib)
    4. Rolling std / mean
    5. Daily percentage price change
    6. Drop rows with any NaN

    All indicators that are *not* naturally lag-safe are computed on
    ``close_prev = Close.shift(1)`` to prevent look-ahead bias.
    """
    logger.info("Building feature matrix (target='%s', n_lags=%d)", target, n_lags)

    out = df.copy()
    out = join_key_rate(out)
    out = create_lags(out, target, n_lags)

    close_prev = out[target].shift(1)

    out[f"EMA_{ema_period}"] = ta.EMA(out[target], timeperiod=ema_period)
    out["ROC_5"] = ta.ROC(close_prev, timeperiod=5)
    out[f"RSI_{rsi_period}"] = ta.RSI(out[target], timeperiod=rsi_period)
    out[f"Std_{rolling_std}"] = close_prev.rolling(rolling_std).std()
    out[f"Mean_{rolling_mean}"] = close_prev.rolling(rolling_mean).mean()
    out["Price_change"] = close_prev.pct_change()

    rows_before = len(out)
    out.dropna(inplace=True)
    logger.info(
        "Feature matrix ready: %d rows, %d features (dropped %d NaN rows)",
        len(out), out.shape[1], rows_before - len(out),
    )
    return out
