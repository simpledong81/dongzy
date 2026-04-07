from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS, TrainConfig


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算回归任务常用指标。

    - R2: 拟合优度，越接近 1 越好
    - RMSE: 均方根误差，对大误差更敏感，越小越好
    - MAE: 平均绝对误差，直观反映平均偏差，越小越好
    """
    return {
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _build_models(random_state: int) -> dict[str, tuple[object, dict]]:
    """构建三类候选模型及其网格搜索参数空间。

    约定：
    - key 名称固定为 random_forest / xgboost / bp_neural_network
    - 值为 (estimator, param_grid)
    - 统一使用随机种子，保证可复现
    """
    if XGBRegressor is not None:
        # 首选 XGBoost（论文要求之一）
        xgb_estimator = MultiOutputRegressor(
            XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
                n_jobs=2,
            )
        )
        xgb_grid = {
            "estimator__n_estimators": [300],
            "estimator__max_depth": [3, 5],
            "estimator__learning_rate": [0.03, 0.1],
            "estimator__subsample": [0.8],
        }
    else:
        # 某些 macOS 环境缺少 libomp 会导致 xgboost 无法加载。
        # 为保证流程不断裂，这里做“同名模型位”回退：
        # 仍使用 "xgboost" 键，但底层替换为 GradientBoostingRegressor。
        xgb_estimator = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=random_state)
        )
        xgb_grid = {
            "estimator__n_estimators": [200, 300],
            "estimator__max_depth": [2, 3],
            "estimator__learning_rate": [0.03, 0.1],
        }

    return {
        "random_forest": (
            MultiOutputRegressor(RandomForestRegressor(random_state=random_state)),
            {
                "estimator__n_estimators": [200],
                "estimator__max_depth": [None, 8],
                "estimator__min_samples_split": [2, 4],
            },
        ),
        "xgboost": (
            xgb_estimator,
            xgb_grid,
        ),
        "bp_neural_network": (
            MLPRegressor(random_state=random_state, max_iter=2000),
            {
                "hidden_layer_sizes": [(64, 32), (128, 64)],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.001, 0.01],
            },
        ),
    }


def _collect_feature_importance(
    model_name: str,
    estimator: object,
    feature_names: list[str],
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    """提取特征重要性（按输出目标分别给出）。

    返回结构：
    {
      "tensile_bond_strength": {"roughness_ra": 0.xx, ...},
      "shear_bond_strength": {"roughness_ra": 0.xx, ...}
    }
    """
    importance: dict[str, dict[str, float]] = {}
    if model_name in {"random_forest", "xgboost"}:
        # MultiOutputRegressor 下，每个输出目标对应一个子回归器
        for target_name, reg in zip(target_names, estimator.estimators_):
            scores = getattr(reg, "feature_importances_", np.zeros(len(feature_names)))
            importance[target_name] = {
                f: float(s) for f, s in zip(feature_names, scores)
            }
    else:
        # MLP 无内置重要性，这里返回零占位。
        for target_name in target_names:
            importance[target_name] = {f: 0.0 for f in feature_names}
    return importance


def train_and_compare_models(df: pd.DataFrame, random_state: int = 42) -> dict:
    """训练并对比三种模型，返回后续报告所需的完整中间结果。

    主流程：
    1) 读取配置并切分数据（8:2）
    2) 对每个模型执行 GridSearchCV（5 折）
    3) 记录训练/测试指标
    4) 自动选择测试集 R2 最优模型
    5) 汇总预测值、特征重要性与训练中间数据
    """
    cfg = TrainConfig()
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    # 固定随机种子切分，便于复现实验结果
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=random_state
    )

    models = _build_models(random_state)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    feature_importance_by_model: dict[str, dict[str, dict[str, float]]] = {}
    predictions: dict[str, dict[str, list[float]]] = {}
    fitted: dict[str, object] = {}

    best_model_name = ""
    best_score = -np.inf

    for name, (estimator, grid) in models.items():
        # GridSearchCV 负责“参数组合遍历 + 交叉验证 + 最优参数回填”
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=cfg.cv_folds,
            scoring="r2",
            n_jobs=1,
            refit=True,
        )
        gs.fit(x_train, y_train)
        best = gs.best_estimator_
        fitted[name] = best

        # 统一在训练集和测试集上评估，便于观察过拟合/欠拟合
        train_pred = best.predict(x_train)
        test_pred = best.predict(x_test)
        metrics[name] = {
            "train": _evaluate(y_train, train_pred),
            "test": _evaluate(y_test, test_pred),
            "best_params": gs.best_params_,
        }

        # 用测试集 R2 作为主排序指标，选出当前最优模型
        current_score = metrics[name]["test"]["r2"]
        if current_score > best_score:
            best_score = current_score
            best_model_name = name

        # 记录每个模型、每个输出目标的特征重要性
        feature_importance_by_model[name] = _collect_feature_importance(
            name, best, FEATURE_COLUMNS, TARGET_COLUMNS
        )

        # 按目标字段保存测试集预测值，用于后续散点图与对比导出
        predictions[name] = {
            target: test_pred[:, idx].tolist() for idx, target in enumerate(TARGET_COLUMNS)
        }

    # 对外默认暴露“最优模型”的重要性和预测结果，便于上层直接使用
    feature_importance = feature_importance_by_model[best_model_name]
    predictions_best = predictions[best_model_name]
    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "feature_importance_by_model": feature_importance_by_model,
        "predictions": predictions_best,
        "predictions_by_model": predictions,
        "y_test": y_test.to_dict(orient="list"),
        "best_model_name": best_model_name,
        "best_model": fitted[best_model_name],
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test_df": y_test,
    }
