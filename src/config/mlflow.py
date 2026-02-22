"""
mlflow_config.py – MLflow initialisation utilities.

Call ``setup_mlflow()`` once at the start of each entry-point script, after
``setup_logging()``.  Every other module that needs the experiment id imports
``get_or_create_experiment()``.
"""
import logging

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from config.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_URI,
    MLFLOW_EXPERIMENT_PREFIX,
)

logger = logging.getLogger(__name__)


def setup_mlflow() -> None:
    """
    Point MLflow at the configured tracking and registry stores.

    Must be called before any mlflow.* calls are made.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    logger.info(
        "MLflow initialised — tracking URI: %s | registry URI: %s",
        MLFLOW_TRACKING_URI,
        MLFLOW_REGISTRY_URI,
    )


def get_or_create_experiment(model_class_name: str) -> str:
    """
    Return the MLflow experiment id for *model_class_name*, creating it if
    it does not yet exist.

    Experiment names follow the pattern: ``Sberbank_<ModelClassName>``
    (e.g. ``Sberbank_SVRModel``, ``Sberbank_XGBModel``, ``Sberbank_RFModel``).
    This keeps runs for each model family grouped together in the UI.

    Parameters
    ----------
    model_class_name : the ``__name__`` of the model class being trained

    Returns
    -------
    str – the experiment id
    """
    experiment_name = f"{MLFLOW_EXPERIMENT_PREFIX}_{model_class_name}"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info("Created MLflow experiment '%s' (id=%s)",
                    experiment_name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Using MLflow experiment '%s' (id=%s)",
                    experiment_name, experiment_id)

    return experiment_id
