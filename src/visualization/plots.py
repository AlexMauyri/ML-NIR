"""
plots.py – All plotting functions for the project.

Every function is self-contained: it receives data, renders a chart, and
returns nothing. No state is stored here.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error

logger = logging.getLogger(__name__)


def plot_predictions(
    y_pred: np.ndarray,
    y_test: pd.Series,
    title: str = "Predicted vs Actual Closing Price",
    save_path: str | None = None,
) -> None:
    """
    Overlay actual and predicted closing prices on a single line chart.

    Parameters
    ----------
    y_pred    : array of predicted values
    y_test    : Series with a DatetimeIndex (actual values)
    title     : chart title (MAPE is appended automatically)
    save_path : if given, save the figure to this path instead of showing it
    """
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info("Plotting predictions – MAPE=%.4f%%", mape * 100)

    df_plot = pd.DataFrame({
        "Date":      y_test.index,
        "Actual":    y_test.values,
        "Predicted": y_pred,
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_plot, x="Date", y="Actual",
                 color="steelblue", label="Actual", linewidth=2, ax=ax)
    sns.lineplot(data=df_plot, x="Date", y="Predicted",
                 color="tomato", label="Predicted", linewidth=2, ax=ax)

    ax.set_title(f"{title}  (MAPE = {mape:.4%})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Plot saved to %s", save_path)

    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "Feature Importances",
    top_n: int = 15,
    save_path: str | None = None,
) -> None:
    """
    Horizontal bar chart of the top-*n* features by importance score.

    Parameters
    ----------
    feature_names : list of feature name strings
    importances   : array of importance scores (same length as feature_names)
    top_n         : how many features to display
    save_path     : optional file path to save the figure
    """
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(4, top_n // 2)))
    sns.barplot(data=df, y="feature", x="importance", palette="Blues_r", ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Feature importance plot saved to %s", save_path)

    return fig
