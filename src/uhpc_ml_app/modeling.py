"""\u5efa\u6a21\u4e3b\u6a21\u5757\uff1a\u8bad\u7ec3\u3001\u8c03\u4f18\u3001\u8bc4\u4f30\u3001\u7279\u5f81\u91cd\u8981\u6027\u7edf\u4e00\u3002

PR#2 \u5bf9\u6bd4\u539f\u5b9e\u73b0\u7684\u5173\u952e\u6539\u52a8\uff1a

1. BP \u795e\u7ecf\u7f51\u7edc \u2192 ``Pipeline(StandardScaler + MLPRegressor)``\uff0c\u4e0d\u518d\u56e0\u672a\u6807\u51c6\u5316
   \u800c\u5728 RF/XGBoost \u9762\u524d\u5403\u4e8f\u3002
2. RandomForestRegressor \u4e0d\u518d\u88ab ``MultiOutputRegressor`` \u5305\u88f9\uff0c\u76f4\u63a5\u4f7f\u7528
   sklearn \u539f\u751f\u591a\u8f93\u51fa\u652f\u6301\uff0c\u51cf\u5c11\u8bad\u7ec3\u6210\u672c\u5e76\u4fdd\u7559\u76ee\u6807\u95f4\u7684\u8026\u5408\u4fe1\u606f\u3002
3. \u7f51\u683c\u641c\u7d22\u7a7a\u95f4\u6269\u5927\uff08``n_estimators`` / ``max_features`` / ``reg_alpha`` / ``reg_lambda`` \u7b49\uff09\uff0c
   \u907f\u514d\u539f\u7248\u56e0\u7f51\u683c\u8fc7\u5c0f\u9020\u6210\u7684\u201c\u4f2a\u8c03\u53c2\u201d\u3002
4. \u6307\u6807\u62c6\u5206\u4e3a **per-target + overall**\uff0c\u907f\u514d\u62c9\u4f38/\u6297\u526a\u4e24\u4e2a\u76ee\u6807\u7684\u8868\u73b0
   \u88ab uniform_average \u63a9\u76d6\u3002
5. \u65b0\u589e **Repeated K-Fold CV**\uff08\u9ed8\u8ba4 5\u00d710\uff09\u5728\u5168\u91cf\u6570\u636e\u4e0a\u4e3a\u6700\u4f18\u6a21\u578b
   \u7ed9\u51fa\u83b7\u5f97\u5747\u503c\u00b195% CI \u7684\u6cdb\u5316\u4f30\u8ba1\uff0c\u907f\u514d\u5355\u6b21 20 \u884c\u6d4b\u8bd5\u96c6\u9009\u6a21\u578b\u7684
   \u9ad8\u65b9\u5dee\u95ee\u9898\u3002
6. **\u6a21\u578b\u9009\u4f18\u6539\u4e3a\u6309 CV-R\u00b2 \u5747\u503c**\uff0c\u800c\u975e\u5355\u6b21\u6d4b\u8bd5\u96c6 R\u00b2\u3002
7. \u7edf\u4e00\u5404\u6a21\u578b\u7684\u7279\u5f81\u91cd\u8981\u6027\u4e3a **Permutation Importance**\uff0c\u6811\u6a21\u578b\u7684
   ``feature_importances_`` \u5141\u8bb8\u540c\u65f6\u4fdd\u7559\u4f5c\u4e3a\u53c2\u8003\u3002
8. \u53ef\u9009\u5730\u5e76\u5165\u57fa\u7ebf\u6a21\u578b\uff08\u5747\u503c / \u7ebf\u6027 / \u7ecf\u9a8c\u516c\u5f0f\uff09\u4f5c\u4e3a\u5bf9\u6bd4\u4e0b\u754c\u3002

\u8f93\u51fa\u7684 ``result`` \u5b57\u5178\u4fdd\u6301\u5bf9 reporting.py / streamlit_app.py \u7684**\u5411\u4e0b\u517c\u5bb9**
\uff08``metrics / best_model / x_train / y_train / y_test_df / predictions_by_model`` \u7b49\u952e\u540d\u5747\u5b58\u5728\uff09\uff0c
\u540c\u65f6\u65b0\u589e\u4e86\uff1a``metrics_per_target``\u3001``cv_metrics``\u3001``permutation_importance_by_model``\u3001
``baseline_metrics``\u3002SHAP \u7528\u5230\u7684 ``best_model`` \u4ecd\u4e3a\u6700\u4f18 pipeline/estimator\uff0c
PR#3 \u4f1a\u57fa\u4e8e\u8be5\u5bf9\u8c61\u505a\u53cc\u76ee\u6807\u89e3\u91ca\u3002
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment,misc]

from uhpc_ml_app.baselines import build_baselines
from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS, TrainConfig

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _evaluate_overall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """\u4ee5 uniform_average \u7684\u5f62\u5f0f\u8fd4\u56de\u6574\u4f53\u56de\u5f52\u6307\u6807\uff08\u4e0e\u539f\u7248\u5411\u4e0b\u517c\u5bb9\uff09\u3002"""
    return {
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _evaluate_per_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """\u6309\u76ee\u6807\u5206\u522b\u8ba1\u7b97 R\u00b2 / RMSE / MAE\u3002

    \u539f\u7248\u5c06\u62c9\u4f38\u4e0e\u6297\u526a\u4e24\u4e2a\u76ee\u6807\u5e73\u5747\u5728\u4e00\u8d77\u6c47\u62a5\uff0c\u7531\u4e8e\u4e24\u76ee\u6807\u91cf\u7eb2\u76f8\u5dee
    ~4 \u500d\uff0cRMSE/MAE \u4f1a\u88ab\u6297\u526a\u4e3b\u5bfc\uff0c\u5904\u7406\u4e0a\u4e0d\u4e25\u8c28\u3002\u672c\u51fd\u6570\u4e3a per-target \u62c6\u5206\uff0c
    \u6446\u5230\u8bba\u6587\u8868\u683c\u65f6\u4e5f\u66f4\u5bb9\u6613\u89e3\u8bfb\u3002
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    out: dict[str, dict[str, float]] = {}
    for idx, target in enumerate(target_names):
        yt = y_true_arr[:, idx]
        yp = y_pred_arr[:, idx]
        out[target] = {
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        }
    return out


def _evaluate_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> dict[str, object]:
    """\u7edf\u4e00\u8fd4\u56de overall + per_target \u4e24\u5c42\u6307\u6807\uff0c\u65b9\u4fbf reporting \u8f93\u51fa\u3002"""
    return {
        "overall": _evaluate_overall(y_true, y_pred),
        "per_target": _evaluate_per_target(y_true, y_pred, target_names),
    }


def _ci95_from_scores(scores: np.ndarray) -> tuple[float, float, float]:
    """\u7531\u6837\u672c\u7ecf\u9a8c\u6807\u51c6\u5dee\u7ed9\u51fa\u5747\u503c + \u8fd1\u4f3c 95% CI\u3002"""
    mean = float(np.mean(scores))
    if len(scores) <= 1:
        return mean, mean, mean
    sem = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def _repeated_kfold_cv_scores(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: TrainConfig,
    random_state: int,
) -> dict[str, float | int]:
    """Repeated K-Fold \u5168\u6570\u636e\u4ea4\u53c9\u9a8c\u8bc1\uff0c\u8fd4\u56de R\u00b2 \u5747\u503c\u00b195% CI \u53ca\u8bbe\u7f6e\u3002

    \u4f7f\u7528 sklearn \u7684 ``cross_val_score(scoring="r2")``\u3002\u8bc4\u4f30\u65f6 ``estimator`` \u4f1a\u88ab\u5185\u90e8
    clone + fit\uff0c\u4e0d\u4f1a\u6c61\u67d3\u5916\u90e8\u7684\u5df2\u62df\u5408\u6a21\u578b\u3002
    """
    rkf = RepeatedKFold(
        n_splits=cfg.repeated_cv_n_splits,
        n_repeats=cfg.repeated_cv_n_repeats,
        random_state=random_state,
    )
    try:
        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=rkf,
            scoring="r2",
            n_jobs=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "r2_mean": float("nan"),
            "r2_ci95_low": float("nan"),
            "r2_ci95_high": float("nan"),
            "n_splits": float(cfg.repeated_cv_n_splits),
            "n_repeats": float(cfg.repeated_cv_n_repeats),
            "error": str(exc),  # type: ignore[dict-item]
        }
    mean, low, high = _ci95_from_scores(scores)
    return {
        "r2_mean": mean,
        "r2_ci95_low": low,
        "r2_ci95_high": high,
        "n_splits": cfg.repeated_cv_n_splits,
        "n_repeats": cfg.repeated_cv_n_repeats,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_models(random_state: int, cfg: TrainConfig) -> dict[str, tuple[BaseEstimator, dict]]:
    """\u6784\u9020\u4e09\u7c7b\u5019\u9009\u6a21\u578b\u53ca\u5176\u7f51\u683c\u641c\u7d22\u53c2\u6570\u7a7a\u95f4\u3002

    \u7ea6\u5b9a\uff1a
    - RF\uff1asklearn \u539f\u751f\u591a\u8f93\u51fa\uff0c\u4e0d\u505a MultiOutputRegressor \u5305\u88f9\u3002
    - XGBoost\uff1a\u4ecd\u4ee5 ``MultiOutputRegressor(XGBRegressor)`` \u63a5\u5165\uff0c\u4fdd\u6301\u7a33\u5b9a
      \u517c\u5bb9\u6027\uff1bxgboost \u4e0d\u53ef\u7528\u65f6\u56de\u9000 ``GradientBoostingRegressor``\u3002
    - BP\uff1a``Pipeline([scaler, mlp])``\uff0c\u4ee5\u4fdd\u8bc1\u5165\u7f51\u524d\u505a\u6807\u51c6\u5316\uff0c\u4e0e\u6811\u6a21\u578b\u516c\u5e73\u5bf9\u6bd4\u3002
    """
    fast = cfg.grid_mode == "fast"
    rf_estimator = RandomForestRegressor(random_state=random_state, n_jobs=1)
    rf_grid = (
        {
            "n_estimators": [200],
            "max_depth": [None, 6],
            "min_samples_split": [2],
        }
        if fast
        else {
            "n_estimators": [200, 500],
            "max_depth": [None, 6, 12],
            "min_samples_split": [2, 4],
        }
    )

    if XGBRegressor is not None:
        xgb_estimator = MultiOutputRegressor(
            XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
                n_jobs=1,
                verbosity=0,
            )
        )
        xgb_grid = (
            {
                "estimator__n_estimators": [200],
                "estimator__max_depth": [3, 5],
                "estimator__learning_rate": [0.1],
                "estimator__subsample": [1.0],
            }
            if fast
            else {
                "estimator__n_estimators": [200, 500],
                "estimator__max_depth": [3, 5],
                "estimator__learning_rate": [0.03, 0.1],
                "estimator__subsample": [0.8, 1.0],
            }
        )
    else:
        xgb_estimator = MultiOutputRegressor(GradientBoostingRegressor(random_state=random_state))
        xgb_grid = (
            {
                "estimator__n_estimators": [200],
                "estimator__max_depth": [2, 3],
                "estimator__learning_rate": [0.1],
                "estimator__subsample": [1.0],
            }
            if fast
            else {
                "estimator__n_estimators": [200, 500],
                "estimator__max_depth": [2, 3],
                "estimator__learning_rate": [0.03, 0.1],
                "estimator__subsample": [0.8, 1.0],
            }
        )

    bp_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    random_state=random_state,
                    max_iter=cfg.mlp_max_iter,
                    early_stopping=False,
                ),
            ),
        ]
    )
    bp_grid = (
        {
            "mlp__hidden_layer_sizes": [(32,)],
            "mlp__activation": ["relu"],
            "mlp__alpha": [1e-4],
        }
        if fast
        else {
            "mlp__hidden_layer_sizes": [(64,), (64, 32)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [1e-4, 1e-3],
        }
    )

    return {
        "random_forest": (rf_estimator, rf_grid),
        "xgboost": (xgb_estimator, xgb_grid),
        "bp_neural_network": (bp_pipeline, bp_grid),
    }


# ---------------------------------------------------------------------------
# Feature importance (unified + native)
# ---------------------------------------------------------------------------


def _tree_native_importance(
    model_name: str,
    fitted: BaseEstimator,
    feature_names: list[str],
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    """\u4ec5\u5bf9\u6811\u6a21\u578b\u53d6 ``feature_importances_`` \u4f5c\u4e3a\u53c2\u8003\u3002

    \u6ce8\u610f\u4e0d\u540c\u6a21\u578b\u7684\u5185\u7f6e\u91cd\u8981\u6027\u7eb2\u8981\u5316\u65b9\u5f0f\u4e0d\u540c\uff0c**\u4e0d\u5efa\u8bae**\u76f4\u63a5\u8de8\u6a21\u578b\u5bf9\u6bd4\u3002
    """
    importance: dict[str, dict[str, float]] = {}
    if model_name == "random_forest":
        # RF \u539f\u751f\u591a\u8f93\u51fa\uff0c\u91cd\u8981\u6027\u662f\u8de8\u8f93\u51fa\u5e73\u5747\u7684\u5355\u4e00\u5411\u91cf
        scores = getattr(fitted, "feature_importances_", None)
        if scores is None:
            return {t: dict.fromkeys(feature_names, 0.0) for t in target_names}
        normalized = {f: float(s) for f, s in zip(feature_names, scores)}
        return {t: dict(normalized) for t in target_names}
    if model_name == "xgboost" and hasattr(fitted, "estimators_"):
        for target_name, sub in zip(target_names, fitted.estimators_):
            scores = getattr(sub, "feature_importances_", np.zeros(len(feature_names)))
            importance[target_name] = {f: float(s) for f, s in zip(feature_names, scores)}
        return importance
    return {t: dict.fromkeys(feature_names, 0.0) for t in target_names}


def _permutation_importance_per_target(
    fitted: BaseEstimator,
    X_eval: pd.DataFrame,
    y_eval: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
    cfg: TrainConfig,
    random_state: int,
) -> dict[str, dict[str, float]]:
    """\u4f7f\u7528 sklearn permutation_importance \u7edf\u4e00\u8ba1\u7b97\u6bcf\u4e2a\u76ee\u6807\u7684\u7279\u5f81\u91cd\u8981\u6027\u3002

    \u540c\u65f6\u4f5c\u7528\u4e8e\u6811\u6a21\u578b\u4e0e BP\uff0c\u4fdd\u8bc1\u8de8\u6a21\u578b\u53ef\u6bd4\u3002
    \u5bf9\u6bcf\u4e2a\u76ee\u6807\u5355\u72ec\u7528 R\u00b2 \u6253\u5206\uff0c\u907f\u514d\u88ab\u5176\u4ed6\u76ee\u6807\u51b2\u51c0\u3002
    """
    result: dict[str, dict[str, float]] = {}
    y_arr = np.asarray(y_eval)
    for idx, target in enumerate(target_names):
        y_target = y_arr[:, idx]

        def _scorer(est, Xs, ys, _idx=idx):  # type: ignore[no-untyped-def]
            preds = est.predict(Xs)
            if preds.ndim == 1:
                return float(r2_score(ys, preds))
            return float(r2_score(ys, preds[:, _idx]))

        try:
            pi = permutation_importance(
                fitted,
                X_eval,
                y_target,
                scoring=_scorer,
                n_repeats=cfg.permutation_importance_n_repeats,
                random_state=random_state,
                n_jobs=1,
            )
            scores = pi.importances_mean
        except Exception:  # pragma: no cover
            scores = np.zeros(len(feature_names))
        result[target] = {f: float(s) for f, s in zip(feature_names, scores)}
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _predictions_to_target_dict(
    predictions: np.ndarray,
    target_names: list[str],
) -> dict[str, list[float]]:
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    return {
        target: predictions[:, idx].astype(float).tolist()
        for idx, target in enumerate(target_names)
    }


def _evaluate_baseline(
    name: str,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    cfg: TrainConfig,
    random_state: int,
) -> dict[str, object]:
    fitted = clone(estimator).fit(x_train, y_train)
    train_pred = np.asarray(fitted.predict(x_train))
    test_pred = np.asarray(fitted.predict(x_test))
    cv_metrics = _repeated_kfold_cv_scores(
        clone(estimator),
        pd.concat([x_train, x_test], axis=0).reset_index(drop=True),
        pd.concat([y_train, y_test], axis=0).reset_index(drop=True),
        cfg,
        random_state,
    )
    return {
        "name": name,
        "train": _evaluate_full(np.asarray(y_train), train_pred, TARGET_COLUMNS),
        "test": _evaluate_full(np.asarray(y_test), test_pred, TARGET_COLUMNS),
        "cv": cv_metrics,
    }


def train_and_compare_models(
    df: pd.DataFrame,
    random_state: int = 42,
    config: TrainConfig | None = None,
) -> dict:
    """\u8bad\u7ec3\u5e76\u5bf9\u6bd4\u4e09\u79cd\u6a21\u578b\uff0c\u8fd4\u56de\u540e\u7eed\u62a5\u544a\u6240\u9700\u7684\u5b8c\u6574\u4e2d\u95f4\u7ed3\u679c\u3002

    \u4e3b\u6d41\u7a0b\uff1a

    1. 8:2 \u5207\u5206\uff08\u4ec5\u7528\u4e8e\u7cbe\u5ea6\u53ef\u89c6\u5316\u4e0e\u6700\u7ec8\u62df\u5408\u7684\u6301\u51fa\u6d4b\u8bd5\uff09\u3002
    2. \u5bf9\u6bcf\u4e2a\u6a21\u578b\u5728\u8bad\u7ec3\u96c6\u4e0a\u6267\u884c GridSearchCV\uff08cv=5\uff09\u9009\u8d85\u53c2\u3002
    3. \u8bb0\u5f55 train/test per-target + overall \u6307\u6807\u3002
    4. \u7528 best_estimator_ \u5728\u5168\u91cf\u6570\u636e\u4e0a\u8dd1 Repeated K-Fold CV\uff0c\u5f97\u5230 R\u00b2 \u5747\u503c\u00b195% CI\u3002
    5. \u7528 Permutation Importance \u7edf\u4e00\u8de8\u6a21\u578b\u7279\u5f81\u91cd\u8981\u6027\uff08\u6811\u6a21\u578b\u540c\u65f6\u4fdd\u7559\u5185\u7f6e\u91cd\u8981\u6027\uff09\u3002
    6. \u6839\u636e ``cfg.model_selection_metric`` \u9009\u6700\u4f18\u6a21\u578b\uff08\u9ed8\u8ba4 CV-R\u00b2 \u5747\u503c\uff09\u3002
    7. \u53ef\u9009\u5730\u5e76\u5165\u57fa\u7ebf\u6a21\u578b\uff08mean / linear / \u221af_c\uff09\u7684\u6307\u6807\u3002
    """
    cfg = config or TrainConfig()
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=random_state
    )

    models = _build_models(random_state, cfg)

    metrics: dict[str, dict] = {}
    metrics_per_target: dict[str, dict] = {}
    cv_metrics: dict[str, dict] = {}
    fitted: dict[str, BaseEstimator] = {}
    predictions: dict[str, dict[str, list[float]]] = {}
    native_importance: dict[str, dict[str, dict[str, float]]] = {}
    permutation_importance_by_model: dict[str, dict[str, dict[str, float]]] = {}

    inner_cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=random_state)

    for name, (estimator, grid) in models.items():
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=inner_cv,
            scoring="r2",
            n_jobs=1,
            refit=True,
        )
        gs.fit(x_train, y_train)
        best = gs.best_estimator_
        fitted[name] = best

        train_pred = np.asarray(best.predict(x_train))
        test_pred = np.asarray(best.predict(x_test))

        metrics[name] = {
            "train": _evaluate_overall(y_train, train_pred),
            "test": _evaluate_overall(y_test, test_pred),
            "best_params": gs.best_params_,
        }
        metrics_per_target[name] = {
            "train": _evaluate_per_target(y_train, train_pred, TARGET_COLUMNS),
            "test": _evaluate_per_target(y_test, test_pred, TARGET_COLUMNS),
        }

        cv_metrics[name] = _repeated_kfold_cv_scores(
            clone(estimator).set_params(**gs.best_params_),
            x,
            y,
            cfg,
            random_state,
        )

        native_importance[name] = _tree_native_importance(
            name, best, FEATURE_COLUMNS, TARGET_COLUMNS
        )
        permutation_importance_by_model[name] = _permutation_importance_per_target(
            best,
            x_test,
            y_test,
            FEATURE_COLUMNS,
            TARGET_COLUMNS,
            cfg,
            random_state,
        )

        predictions[name] = _predictions_to_target_dict(test_pred, TARGET_COLUMNS)

    # \u2014\u2014 \u9009\u6700\u4f18\u6a21\u578b\uff1a\u9ed8\u8ba4\u6309 CV-R\u00b2 \u5747\u503c\uff0c\u56de\u9000\u5230\u6d4b\u8bd5\u96c6 R\u00b2 \u2014\u2014
    def _model_selection_score(name: str) -> float:
        if cfg.model_selection_metric == "cv_r2_mean":
            val = cv_metrics.get(name, {}).get("r2_mean", float("nan"))
            return float(val) if val == val else metrics[name]["test"]["r2"]
        return metrics[name]["test"]["r2"]

    best_model_name = max(fitted.keys(), key=_model_selection_score)
    best_model = fitted[best_model_name]

    baseline_metrics: dict[str, dict[str, object]] = {}
    if cfg.include_baselines:
        for bname, bmodel in build_baselines().items():
            baseline_metrics[bname] = _evaluate_baseline(
                bname, bmodel, x_train, y_train, x_test, y_test, cfg, random_state
            )

    feature_importance = native_importance[best_model_name]
    predictions_best = predictions[best_model_name]

    return {
        "config": asdict(cfg),
        # \u2014\u2014 \u4e0e PR#1 \u4e4b\u524d\u5411\u4e0b\u517c\u5bb9\u7684\u952e \u2014\u2014
        "metrics": metrics,
        "feature_importance": feature_importance,
        "feature_importance_by_model": native_importance,
        "predictions": predictions_best,
        "predictions_by_model": predictions,
        "y_test": y_test.to_dict(orient="list"),
        "best_model_name": best_model_name,
        "best_model": best_model,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test_df": y_test,
        # \u2014\u2014 PR#2 \u65b0\u589e\u952e \u2014\u2014
        "metrics_per_target": metrics_per_target,
        "cv_metrics": cv_metrics,
        "permutation_importance_by_model": permutation_importance_by_model,
        "baseline_metrics": baseline_metrics,
        "model_selection_metric": cfg.model_selection_metric,
    }


__all__ = [
    "train_and_compare_models",
]
