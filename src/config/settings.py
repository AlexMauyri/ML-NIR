"""
settings.py – Single source of truth for every constant in the project.

Rule: if a value appears more than once, or might ever need changing,
it lives here. Nothing else is hard-coded anywhere.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models" / "saved"
SBERBANK_CSV = DATA_RAW_DIR / "sberbank_stock_prices.csv"
CB_RATE_CSV = DATA_RAW_DIR / "cb_rate.csv"
MOSCOW_IDX_CSV = DATA_RAW_DIR / "security.csv"
PROCESSED_FEATURES_FILE = DATA_PROC_DIR / "features.parquet"

SVR_MODEL_FILE = MODELS_DIR / "svr_model.joblib"
XGB_MODEL_FILE = MODELS_DIR / "xgb_model.joblib"

TARGET_COLUMN = "Close"
TEST_SIZE = 0.2

NUMBER_OF_LAGS = 4
ROLLING_STD = 11
ROLLING_MEAN = 11
EMA_TIME_PERIOD = 9
RSI_TIME_PERIOD = 9

N_CV_SPLITS = 3

N_TRIALS = 50

SVR_C_LOW = 1e3
SVR_C_HIGH = 1e4
SVR_EPS_LOW = 0.01
SVR_EPS_HIGH = 5.0

XGB_N_EST_LOW = 1500
XGB_N_EST_HIGH = 2000
XGB_MAX_DEPTH_LOW = 4
XGB_MAX_DEPTH_HIGH = 8
XGB_MCW_LOW = 5.0
XGB_MCW_HIGH = 15.0
XGB_SUBSAMPLE_LOW = 0.7
XGB_SUBSAMPLE_HIGH = 0.9
XGB_COLSAMPLE_LOW = 0.7
XGB_COLSAMPLE_HIGH = 0.9
XGB_LAMBDA_LOW = 300
XGB_LAMBDA_HIGH = 500
XGB_EARLY_STOPPING = 100
XGB_LEARNING_RATE = 0.1
