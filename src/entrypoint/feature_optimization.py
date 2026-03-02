"""
feature_optimization.py
═══════════════════════════════════════════════════════════════════════════════
Подбор оптимальных констант для feature engineering через временну́ю
кросс-валидацию (TimeSeriesSplit).

Структура:
  1. ACFPACFAnalyzer   — визуальное и статистическое обоснование числа лагов
  2. FeatureParamOptimizer — grid search периодов для каждого индикатора
  3. LagCountOptimizer    — подбор числа лаговых признаков
  4. FullPipelineOptimizer — совместная оптимизация всех констант через Optuna
  5. OptimizationReport   — сводный отчёт с таблицами и графиками для НИР
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import talib as ta
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf, pacf

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass для хранения результатов
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimalParams:
    """Результат оптимизации — все подобранные константы."""
    n_lags:         int   = 4
    ema_period:     int   = 9
    rsi_period:     int   = 9
    rolling_window: int   = 11
    roc_period:     int   = 5
    mom_period:     int   = 10
    atr_period:     int   = 14
    bb_period:      int   = 20
    adx_period:     int   = 14
    cv_mape:        float = 0.0
    cv_results:     Dict  = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 1. ACF / PACF анализ
# ──────────────────────────────────────────────────────────────────────────────

class ACFPACFAnalyzer:
    """
    Визуальное и статистическое обоснование числа лагов.

    Использование:
        analyzer = ACFPACFAnalyzer(df["Close"])
        analyzer.run_full_analysis(max_lags=20, n_splits=5)
    """

    def __init__(self, price_series: pd.Series):
        self.price  = price_series.dropna()
        self.returns = price_series.pct_change().dropna()

    # ── Графики ACF / PACF ────────────────────────────────────────────────
    def plot_acf_pacf(self, max_lags: int = 20, save_path: Optional[str] = None):
        """
        Строит ACF и PACF для уровней цен И для доходностей.
        Для НИР важнее смотреть на PACF доходностей — именно там
        виден «обрыв», обосновывающий число лагов.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("ACF и PACF: уровни цен и доходности", fontsize=13)

        plot_acf(self.price,   lags=max_lags, ax=axes[0, 0],
                 title="ACF — цена закрытия")
        plot_pacf(self.price,  lags=max_lags, ax=axes[0, 1],
                 title="PACF — цена закрытия")
        plot_acf(self.returns, lags=max_lags, ax=axes[1, 0],
                 title="ACF — дневная доходность")
        plot_pacf(self.returns, lags=max_lags, ax=axes[1, 1],
                 title="PACF — дневная доходность")

        for ax in axes.flat:
            ax.set_xlabel("Лаг (дней)")
            ax.set_ylabel("Автокорреляция")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── Тест Льюнга–Бокса ────────────────────────────────────────────────
    def ljung_box_test(self, max_lags: int = 10) -> pd.DataFrame:
        """
        Формальная проверка гипотезы об отсутствии автокорреляции.
        p < 0.05 → отвергаем H₀ → автокорреляция статистически значима.
        """
        result = acorr_ljungbox(self.returns, lags=list(range(1, max_lags + 1)),
                                return_df=True)
        result.index.name = "Лаг"
        result.columns = ["Q-статистика", "p-value"]
        result["Вывод"] = result["p-value"].apply(
            lambda p: "Отвергаем H₀ (автокорреляция значима)"
                      if p < 0.05 else "Не отвергаем H₀"
        )
        print("\n─── Тест Льюнга–Бокса (доходности) ───")
        print(result.to_string())
        return result

    # ── ADF тест на стационарность ────────────────────────────────────────
    def adf_test(self) -> Dict:
        """Проверка стационарности уровней и доходностей."""
        results = {}
        for name, series in [("Цена (уровень)", self.price),
                              ("Доходность pct_change", self.returns)]:
            stat, pval, _, _, crit, _ = adfuller(series.dropna())
            results[name] = {
                "ADF-статистика": round(stat, 4),
                "p-value": round(pval, 6),
                "Стационарен": "Да ✓" if pval < 0.05 else "Нет ✗",
                "Крит. 5%": round(crit["5%"], 4),
            }
        df = pd.DataFrame(results).T
        print("\n─── Тест Дики–Фуллера (ADF) ───")
        print(df.to_string())
        return results

    # ── Значимые лаги по PACF ────────────────────────────────────────────
    def get_significant_lags(self, max_lags: int = 20,
                              significance: float = 0.05) -> List[int]:
        """
        Возвращает список лагов, значимых на PACF доходностей.
        Граница значимости: ±z_{α/2} / √n.
        """
        n = len(self.returns)
        z = 1.96  # 95% доверительный интервал
        ci_bound = z / np.sqrt(n)

        pacf_vals = pacf(self.returns.dropna(), nlags=max_lags)[1:]  # skip lag 0
        sig_lags = [i + 1 for i, v in enumerate(pacf_vals) if abs(v) > ci_bound]

        print(f"\n─── Значимые лаги PACF (доходности, CI ±{ci_bound:.4f}) ───")
        print(f"Значимые лаги: {sig_lags}")
        print(f"Рекомендуемое число лагов: {max(sig_lags) if sig_lags else 1}")
        return sig_lags

    def run_full_analysis(self, max_lags: int = 20, save_path: Optional[str] = None):
        """Запускает все анализы последовательно."""
        print("=" * 60)
        print("АНАЛИЗ АВТОКОРРЕЛЯЦИОННОЙ СТРУКТУРЫ")
        print("=" * 60)
        self.plot_acf_pacf(max_lags=max_lags, save_path=save_path)
        self.adf_test()
        self.ljung_box_test(max_lags=min(max_lags, 15))
        sig = self.get_significant_lags(max_lags=max_lags)
        return sig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Подбор числа лагов через CV
# ──────────────────────────────────────────────────────────────────────────────

class LagCountOptimizer:
    """
    Эмпирический подбор оптимального числа лагов через временну́ю CV.
    Базовая модель: Ridge (нейтральная относительно признаков).
    """

    def __init__(self, df: pd.DataFrame, target: str = "Close",
                 n_splits: int = 5, max_lags: int = 10):
        self.df       = df
        self.target   = target
        self.n_splits = n_splits
        self.max_lags = max_lags
        self.results_  = {}

    def _build_lag_matrix(self, n_lags: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Строит матрицу из n_lags лаговых признаков."""
        X = pd.DataFrame(index=self.df.index)
        for k in range(1, n_lags + 1):
            X[f"lag_{k}"] = self.df[self.target].shift(k)
        y = self.df[self.target]
        mask = X.notna().all(axis=1)
        return X[mask], y[mask]

    def _cv_mape(self, X: pd.DataFrame, y: pd.Series) -> float:
        tscv   = TimeSeriesSplit(n_splits=self.n_splits)
        scaler = StandardScaler()
        model  = Ridge(alpha=1.0)
        mapes  = []
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
            ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
            Xtr_s  = scaler.fit_transform(Xtr)
            Xval_s = scaler.transform(Xval)
            model.fit(Xtr_s, ytr)
            pred = model.predict(Xval_s)
            mapes.append(mean_absolute_percentage_error(yval, pred))
        return float(np.mean(mapes))

    def optimize(self) -> int:
        print("\n─── Подбор числа лагов (Ridge + TimeSeriesSplit) ───")
        for n_lags in range(1, self.max_lags + 1):
            X, y = self._build_lag_matrix(n_lags)
            mape = self._cv_mape(X, y)
            self.results_[n_lags] = mape
            print(f"  n_lags={n_lags:2d}  →  CV MAPE = {mape:.6f}")

        optimal = min(self.results_, key=self.results_.get)
        print(f"\n✓ Оптимальное число лагов: {optimal} "
              f"(MAPE = {self.results_[optimal]:.6f})")
        return optimal

    def plot(self, save_path: Optional[str] = None):
        if not self.results_:
            raise ValueError("Сначала запустите .optimize()")
        lags  = list(self.results_.keys())
        mapes = list(self.results_.values())
        opt   = min(self.results_, key=self.results_.get)

        plt.figure(figsize=(9, 4))
        plt.plot(lags, mapes, marker="o", linewidth=1.8, color="#2E75B6")
        plt.axvline(opt, color="red", linestyle="--", alpha=0.7,
                    label=f"Оптимум: {opt} лага")
        plt.scatter([opt], [self.results_[opt]], color="red", zorder=5, s=80)
        plt.xlabel("Число лагов")
        plt.ylabel("MAPE (кросс-валидация)")
        plt.title("Зависимость CV MAPE от числа лаговых признаков")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Подбор периодов технических индикаторов
# ──────────────────────────────────────────────────────────────────────────────

class FeatureParamOptimizer:
    """
    Grid search периодов для каждого индикатора независимо.

    Логика: фиксируем базовый набор признаков (лаги),
    добавляем один индикатор с разными периодами,
    измеряем CV MAPE → выбираем минимум.

    Параметры поиска:
        ema_periods    : [5, 7, 9, 11, 14, 21]
        rsi_periods    : [7, 9, 11, 14, 21]
        rolling_windows: [7, 9, 11, 14, 21]
        roc_periods    : [3, 5, 7, 10]
        mom_periods    : [5, 7, 10, 14]
        bb_periods     : [14, 20, 26]
        atr_periods    : [7, 10, 14, 21]
        adx_periods    : [7, 10, 14, 21]
    """

    SEARCH_SPACE = {
        "ema":     [5, 7, 9, 11, 14, 21],
        "rsi":     [7, 9, 11, 14, 21],
        "rolling": [7, 9, 11, 14, 21],
        "roc":     [3, 5, 7, 10],
        "mom":     [5, 7, 10, 14],
        "bb":      [14, 20, 26],
        "atr":     [7, 10, 14, 21],
        "adx":     [7, 10, 14, 21],
    }

    def __init__(self, df: pd.DataFrame, target: str = "Close",
                 n_lags: int = 4, n_splits: int = 5):
        self.df       = df.copy()
        self.target   = target
        self.n_lags   = n_lags
        self.n_splits = n_splits
        self.results_: Dict[str, Dict] = {}

    def _base_features(self) -> pd.DataFrame:
        """Базовый набор: только лаги."""
        X = pd.DataFrame(index=self.df.index)
        for k in range(1, self.n_lags + 1):
            X[f"lag_{k}"] = self.df[self.target].shift(k)
        return X

    def _cv_mape(self, X: pd.DataFrame) -> float:
        y    = self.df[self.target]
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        tscv   = TimeSeriesSplit(n_splits=self.n_splits)
        scaler = StandardScaler()
        model  = Ridge(alpha=1.0)
        mapes  = []
        for tr, val in tscv.split(X):
            Xtr, Xval = X.iloc[tr], X.iloc[val]
            ytr, yval = y.iloc[tr], y.iloc[val]
            model.fit(scaler.fit_transform(Xtr), ytr)
            pred = model.predict(scaler.transform(Xval))
            mapes.append(mean_absolute_percentage_error(yval, pred))
        return float(np.mean(mapes))

    # Каждый метод — один индикатор с grid search

    def optimize_ema(self) -> int:
        return self._search_single("ema", "EMA", lambda p: {
            "EMA": ta.EMA(self.df[self.target].shift(1), timeperiod=p)
        })

    def optimize_rsi(self) -> int:
        return self._search_single("rsi", "RSI", lambda p: {
            "RSI": ta.RSI(self.df[self.target].shift(1), timeperiod=p)
        })

    def optimize_rolling(self) -> int:
        return self._search_single("rolling", "Rolling", lambda p: {
            "Std":  self.df[self.target].shift(1).rolling(p).std(),
            "Mean": self.df[self.target].shift(1).rolling(p).mean(),
        })

    def optimize_roc(self) -> int:
        return self._search_single("roc", "ROC", lambda p: {
            "ROC": ta.ROC(self.df[self.target].shift(1), timeperiod=p)
        })

    def optimize_mom(self) -> int:
        return self._search_single("mom", "MOM", lambda p: {
            "MOM": ta.MOM(self.df[self.target].shift(1), timeperiod=p)
        })

    def optimize_bb(self) -> int:
        def build(p):
            u, m, l = ta.BBANDS(self.df[self.target].shift(1), timeperiod=p)
            denom = (u - l).replace(0, np.nan)
            return {
                "BB_width": (u - l) / m,
                "BB_pos":   (self.df[self.target].shift(1) - l) / denom,
            }
        return self._search_single("bb", "Bollinger Bands", build)

    def optimize_atr(self) -> int:
        c = self.df["Close_lag_1"] if "Close_lag_1" in self.df else self.df[self.target].shift(1)
        h = self.df["High_lag_1"]  if "High_lag_1"  in self.df else c
        l = self.df["Low_lag_1"]   if "Low_lag_1"   in self.df else c
        return self._search_single("atr", "ATR/NATR", lambda p: {
            "NATR": ta.NATR(h, l, c, timeperiod=p)
        })

    def optimize_adx(self) -> int:
        c = self.df["Close_lag_1"] if "Close_lag_1" in self.df else self.df[self.target].shift(1)
        h = self.df["High_lag_1"]  if "High_lag_1"  in self.df else c
        l = self.df["Low_lag_1"]   if "Low_lag_1"   in self.df else c
        return self._search_single("adx", "ADX", lambda p: {
            "ADX": ta.ADX(h, l, c, timeperiod=p)
        })

    def _search_single(self, key: str, label: str,
                       feature_fn) -> int:
        """Универсальный grid search для одного индикатора."""
        print(f"\n─── Подбор периода: {label} ───")
        periods = self.SEARCH_SPACE[key]
        scores  = {}
        base    = self._base_features()

        for p in periods:
            X = base.copy()
            feats = feature_fn(p)
            for name, series in feats.items():
                X[name] = series
            mape = self._cv_mape(X)
            scores[p] = mape
            print(f"  period={p:3d}  →  CV MAPE = {mape:.6f}")

        optimal = min(scores, key=scores.get)
        self.results_[key] = {"scores": scores, "optimal": optimal,
                              "label": label}
        print(f"✓ Оптимальный период {label}: {optimal} "
              f"(MAPE = {scores[optimal]:.6f})")
        return optimal

    def optimize_all(self) -> Dict[str, int]:
        """Запускает подбор для всех индикаторов последовательно."""
        print("\n" + "=" * 60)
        print("ПОДБОР ПЕРИОДОВ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ")
        print("=" * 60)
        return {
            "ema":     self.optimize_ema(),
            "rsi":     self.optimize_rsi(),
            "rolling": self.optimize_rolling(),
            "roc":     self.optimize_roc(),
            "mom":     self.optimize_mom(),
            "bb":      self.optimize_bb(),
            "atr":     self.optimize_atr(),
            "adx":     self.optimize_adx(),
        }

    def plot_all(self, save_path: Optional[str] = None):
        """Сводный график результатов подбора всех периодов."""
        if not self.results_:
            raise ValueError("Сначала запустите .optimize_all()")

        n = len(self.results_)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flat

        for ax, (key, res) in zip(axes, self.results_.items()):
            periods = list(res["scores"].keys())
            mapes   = list(res["scores"].values())
            opt     = res["optimal"]
            ax.plot(periods, mapes, marker="o", linewidth=1.8, color="#2E75B6")
            ax.axvline(opt, color="red", linestyle="--", alpha=0.7)
            ax.scatter([opt], [res["scores"][opt]], color="red", zorder=5, s=80)
            ax.set_title(f'{res["label"]} (оптим. = {opt})')
            ax.set_xlabel("Период N")
            ax.set_ylabel("CV MAPE")
            ax.grid(alpha=0.3)

        # Скрываем пустые subplot'ы
        for ax in list(axes)[n:]:
            ax.set_visible(False)

        plt.suptitle("Grid Search периодов технических индикаторов",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def summary_table(self) -> pd.DataFrame:
        """Таблица результатов для НИР."""
        rows = []
        for key, res in self.results_.items():
            scores = res["scores"]
            best_p = res["optimal"]
            rows.append({
                "Индикатор":      res["label"],
                "Диапазон N":     f"{min(scores.keys())}–{max(scores.keys())}",
                "Лучший N":       best_p,
                "MAPE (лучший)":  f"{scores[best_p]:.6f}",
                "MAPE (худший)":  f"{max(scores.values()):.6f}",
                "Δ MAPE":         f"{max(scores.values()) - scores[best_p]:.6f}",
            })
        df = pd.DataFrame(rows)
        print("\n─── Сводная таблица результатов подбора периодов ───")
        print(df.to_string(index=False))
        return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. Совместная оптимизация через Optuna
# ──────────────────────────────────────────────────────────────────────────────

class FullPipelineOptimizer:
    """
    Совместный подбор ВСЕХ констант через Optuna (TPE).

    Используется после независимого подбора для проверки:
    не теряем ли мы качество из-за взаимодействий между признаками?

    Параметры пространства поиска можно расширить.
    """

    def __init__(self, df: pd.DataFrame, target: str = "Close",
                 n_splits: int = 5, n_trials: int = 80):
        self.df       = df.copy()
        self.target   = target
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.study_   = None

    def _build_features(self, trial) -> pd.DataFrame:
        """Строит матрицу признаков по параметрам из trial."""
        n_lags  = trial.suggest_int("n_lags",  1, 10)
        ema_p   = trial.suggest_categorical("ema_period",     [5,7,9,11,14,21])
        rsi_p   = trial.suggest_categorical("rsi_period",     [7,9,11,14,21])
        roll_p  = trial.suggest_categorical("rolling_window", [7,9,11,14,21])
        roc_p   = trial.suggest_categorical("roc_period",     [3,5,7,10])
        mom_p   = trial.suggest_categorical("mom_period",     [5,7,10,14])
        bb_p    = trial.suggest_categorical("bb_period",      [14,20,26])
        atr_p   = trial.suggest_categorical("atr_period",     [7,10,14,21])
        adx_p   = trial.suggest_categorical("adx_period",     [7,10,14,21])

        X = pd.DataFrame(index=self.df.index)
        c1 = self.df[self.target].shift(1)

        # Лаги
        for k in range(1, n_lags + 1):
            X[f"lag_{k}"] = self.df[self.target].shift(k)

        # OHLC лаги (если есть)
        if all(col in self.df.columns for col in ["High","Low"]):
            h1 = self.df["High"].shift(1)
            l1 = self.df["Low"].shift(1)
        else:
            h1 = l1 = c1

        # Технические индикаторы
        X["EMA"]  = ta.EMA(c1, timeperiod=ema_p)
        X["RSI"]  = ta.RSI(c1, timeperiod=rsi_p)
        X["ROC"]  = ta.ROC(c1, timeperiod=roc_p)
        X["MOM"]  = ta.MOM(c1, timeperiod=mom_p)
        X["NATR"] = ta.NATR(h1, l1, c1, timeperiod=atr_p)
        X["ADX"]  = ta.ADX(h1, l1, c1, timeperiod=adx_p)

        # MACD (периоды стандартные — не оптимизируем, чтобы не раздувать пространство)
        X["MACD"], X["MACD_sig"], X["MACD_hist"] = ta.MACD(c1)
        X["WILLR"] = ta.WILLR(h1, l1, c1, timeperiod=14)

        # Боллинджер
        bb_u, bb_m, bb_l = ta.BBANDS(c1, timeperiod=bb_p)
        denom = (bb_u - bb_l).replace(0, np.nan)
        X["BB_width"] = (bb_u - bb_l) / bb_m
        X["BB_pos"]   = (c1 - bb_l) / denom

        # Статистические
        X["Std"]  = c1.rolling(roll_p).std()
        X["Mean"] = c1.rolling(roll_p).mean()
        X["Ret"]  = c1.pct_change()

        # Свечные (если OHLC есть)
        if "Open" in self.df.columns:
            o1 = self.df["Open"].shift(1)
            X["candle_body"]  = (c1 - o1) / o1
            X["HL_range"]     = (h1 - l1) / l1
            X["gap_open"]     = (o1 - self.df[self.target].shift(2)) / self.df[self.target].shift(2)

        return X

    def _objective(self, trial) -> float:
        X = self._build_features(trial)
        y = self.df[self.target]
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        tscv   = TimeSeriesSplit(n_splits=self.n_splits)
        scaler = StandardScaler()
        model  = Ridge(alpha=1.0)
        mapes  = []
        for tr, val in tscv.split(X):
            Xtr, Xval = X.iloc[tr], X.iloc[val]
            ytr, yval = y.iloc[tr], y.iloc[val]
            model.fit(scaler.fit_transform(Xtr), ytr)
            pred = model.predict(scaler.transform(Xval))
            mapes.append(mean_absolute_percentage_error(yval, pred))
        return float(np.mean(mapes))

    def optimize(self) -> OptimalParams:
        print("\n" + "=" * 60)
        print(f"СОВМЕСТНАЯ ОПТИМИЗАЦИЯ (Optuna TPE, {self.n_trials} trials)")
        print("=" * 60)

        self.study_ = optuna.create_study(direction="minimize",
                                          sampler=optuna.samplers.TPESampler(seed=42))
        self.study_.optimize(self._objective, n_trials=self.n_trials,
                             show_progress_bar=True)

        bp = self.study_.best_params
        result = OptimalParams(
            n_lags         = bp["n_lags"],
            ema_period     = bp["ema_period"],
            rsi_period     = bp["rsi_period"],
            rolling_window = bp["rolling_window"],
            roc_period     = bp["roc_period"],
            mom_period     = bp["mom_period"],
            bb_period      = bp["bb_period"],
            atr_period     = bp["atr_period"],
            adx_period     = bp["adx_period"],
            cv_mape        = self.study_.best_value,
        )

        print(f"\n✓ Лучшее CV MAPE: {result.cv_mape:.6f}")
        print(f"  Параметры: {bp}")
        return result

    def plot_optimization_history(self, save_path: Optional[str] = None):
        if self.study_ is None:
            raise ValueError("Сначала запустите .optimize()")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # История сходимости
        values = [t.value for t in self.study_.trials]
        best_so_far = np.minimum.accumulate(values)
        axes[0].plot(values, alpha=0.4, color="#2E75B6", label="Trial MAPE")
        axes[0].plot(best_so_far, color="red", linewidth=2, label="Best so far")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("CV MAPE")
        axes[0].set_title("История оптимизации Optuna")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Важность параметров
        try:
            imp = optuna.importance.get_param_importances(self.study_)
            pd.Series(imp).sort_values().plot(kind="barh", ax=axes[1],
                                              color="#2E75B6")
            axes[1].set_title("Важность гиперпараметров")
            axes[1].set_xlabel("Relative importance")
            axes[1].grid(alpha=0.3)
        except Exception:
            axes[1].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Сводный отчёт
# ──────────────────────────────────────────────────────────────────────────────

class OptimizationReport:
    """
    Собирает результаты всех этапов оптимизации
    и выводит итоговую таблицу для НИР.
    """

    def __init__(self, lag_results: Dict[int, float],
                 param_results: Dict[str, Dict],
                 optuna_params: Optional[OptimalParams] = None):
        self.lag_results    = lag_results
        self.param_results  = param_results
        self.optuna_params  = optuna_params

    def print_summary(self):
        print("\n" + "=" * 60)
        print("ИТОГОВЫЕ ПАРАМЕТРЫ FEATURE ENGINEERING")
        print("=" * 60)

        opt_lags = min(self.lag_results, key=self.lag_results.get)
        print(f"\n{'Параметр':<25} {'Grid search':^15} {'Optuna':^15}")
        print("-" * 55)
        print(f"{'n_lags':<25} {opt_lags:^15}", end="")
        if self.optuna_params:
            print(f" {self.optuna_params.n_lags:^15}")
        else:
            print()

        for key, res in self.param_results.items():
            gs_val = res["optimal"]
            opt_val = getattr(self.optuna_params, f"{key}_period", "—") \
                      if self.optuna_params else "—"
            print(f"  {res['label']:<23} {gs_val:^15} {opt_val!s:^15}")

        if self.optuna_params:
            print(f"\n{'CV MAPE (Optuna)':<25} {self.optuna_params.cv_mape:.6f}")

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{"Параметр": "n_lags",
                 "Grid search": min(self.lag_results, key=self.lag_results.get),
                 "Optuna": self.optuna_params.n_lags if self.optuna_params else "—",
                 "Финальное значение": min(self.lag_results, key=self.lag_results.get)}]
        for key, res in self.param_results.items():
            opt_val = getattr(self.optuna_params, f"{key}_period", res["optimal"]) \
                      if self.optuna_params else res["optimal"]
            rows.append({
                "Параметр":          f"{res['label']}_period",
                "Grid search":       res["optimal"],
                "Optuna":            opt_val,
                "Финальное значение": opt_val,
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа — пример использования
# ──────────────────────────────────────────────────────────────────────────────

def run_full_optimization(df: pd.DataFrame,
                          target: str = "Close",
                          n_splits: int = 5,
                          n_optuna_trials: int = 80,
                          save_plots: bool = False) -> OptimalParams:
    """
    Полный pipeline оптимизации:
      1. ACF/PACF анализ → рекомендуемое число лагов
      2. Grid search числа лагов через CV
      3. Grid search периодов каждого индикатора
      4. Совместная оптимизация через Optuna
      5. Итоговый отчёт

    Параметры
    ----------
    df             : DataFrame с колонками Close (обязательно), High, Low, Open (опционально)
    target         : имя целевой колонки
    n_splits       : число блоков TimeSeriesSplit
    n_optuna_trials: число итераций Optuna
    save_plots     : сохранять ли графики в файлы

    Возвращает
    ----------
    OptimalParams с итоговыми константами
    """

    # ── Шаг 1: ACF/PACF ────────────────────────────────────────────────────
    acf_analyzer = ACFPACFAnalyzer(df[target])
    sig_lags = acf_analyzer.run_full_analysis(
        max_lags=20,
        save_path="acf_pacf.png" if save_plots else None,
    )

    # ── Шаг 2: Grid search числа лагов ────────────────────────────────────
    lag_opt = LagCountOptimizer(df, target=target, n_splits=n_splits, max_lags=10)
    optimal_lags = lag_opt.optimize()
    lag_opt.plot(save_path="lag_cv.png" if save_plots else None)

    # ── Шаг 3: Grid search периодов ───────────────────────────────────────
    param_opt = FeatureParamOptimizer(df, target=target,
                                      n_lags=optimal_lags, n_splits=n_splits)
    best_periods = param_opt.optimize_all()
    param_opt.plot_all(save_path="periods_cv.png" if save_plots else None)
    param_opt.summary_table()

    # ── Шаг 4: Optuna совместная оптимизация ──────────────────────────────
    full_opt = FullPipelineOptimizer(df, target=target,
                                     n_splits=n_splits, n_trials=n_optuna_trials)
    final_params = full_opt.optimize()
    full_opt.plot_optimization_history(
        save_path="optuna_history.png" if save_plots else None
    )

    # ── Шаг 5: Отчёт ──────────────────────────────────────────────────────
    report = OptimizationReport(
        lag_results=lag_opt.results_,
        param_results=param_opt.results_,
        optuna_params=final_params,
    )
    report.print_summary()
    summary_df = report.to_dataframe()
    print("\n─── DataFrame для вставки в НИР ───")
    print(summary_df.to_string(index=False))

    # Предупреждение при расхождении grid search и Optuna
    gs_lags = optimal_lags
    optuna_lags = final_params.n_lags
    if gs_lags != optuna_lags:
        print(f"\n⚠ Расхождение n_lags: grid search={gs_lags}, Optuna={optuna_lags}.")
        print("  Это может указывать на взаимодействие лагов с другими признаками.")
        print("  Рекомендуется использовать значение Optuna.")

    return final_params


# ──────────────────────────────────────────────────────────────────────────────
# Пример запуска
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Загрузи свои данные:
    # df = pd.read_csv("sber.csv", index_col="Date", parse_dates=True)
    # params = run_full_optimization(df, target="Close", n_splits=5,
    #                                n_optuna_trials=80, save_plots=True)

    # Синтетический пример для демонстрации:
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = 250 + np.cumsum(np.random.randn(n) * 2)
    df_demo = pd.DataFrame({
        "Close": price,
        "High":  price + np.abs(np.random.randn(n)),
        "Low":   price - np.abs(np.random.randn(n)),
        "Open":  price + np.random.randn(n) * 0.5,
    }, index=dates)

    params = run_full_optimization(
        df_demo, target="Close", n_splits=5, n_optuna_trials=30, save_plots=False
    )
    print(f"\n─── Итоговые параметры ───")
    print(f"n_lags         = {params.n_lags}")
    print(f"ema_period     = {params.ema_period}")
    print(f"rsi_period     = {params.rsi_period}")
    print(f"rolling_window = {params.rolling_window}")
    print(f"roc_period     = {params.roc_period}")
    print(f"mom_period     = {params.mom_period}")
    print(f"bb_period      = {params.bb_period}")
    print(f"atr_period     = {params.atr_period}")
    print(f"adx_period     = {params.adx_period}")
    print(f"CV MAPE        = {params.cv_mape:.6f}")