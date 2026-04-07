from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS


def save_metrics(result: dict, output_dir: Path) -> None:
    """将三模型指标整理成表格并导出 CSV。"""
    rows = []
    for model, metric in result["metrics"].items():
        rows.append(
            {
                "model": model,
                "train_r2": metric["train"]["r2"],
                "train_rmse": metric["train"]["rmse"],
                "train_mae": metric["train"]["mae"],
                "test_r2": metric["test"]["r2"],
                "test_rmse": metric["test"]["rmse"],
                "test_mae": metric["test"]["mae"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)


def save_prediction_scatter(result: dict, output_dir: Path) -> None:
    """绘制“实测值-预测值”散点图，用于直观评估模型泛化能力。

    图中红色虚线 y=x 代表“理想预测”，点越靠近虚线通常误差越小。
    """
    # y_true 为真实测试标签，作为横轴基准
    y_true = result["y_test_df"].reset_index(drop=True)
    # 兼容两种输入：全部模型预测（predictions_by_model）或仅最优模型预测（predictions）
    pred_source = result.get("predictions_by_model", {result["best_model_name"]: result["predictions"]})
    for model, pred_dict in pred_source.items():
        pred_df = pd.DataFrame(pred_dict)
        for target in TARGET_COLUMNS:
            # 每个模型-目标单独出图，便于论文中横向比较
            plt.figure(figsize=(5, 5))
            plt.scatter(y_true[target], pred_df[target], alpha=0.8)
            min_v = min(y_true[target].min(), pred_df[target].min())
            max_v = max(y_true[target].max(), pred_df[target].max())
            plt.plot([min_v, max_v], [min_v, max_v], "r--")
            plt.xlabel("Measured")
            plt.ylabel("Predicted")
            plt.title(f"{model} - {target}")
            plt.tight_layout()
            plt.savefig(output_dir / f"scatter_{model}_{target}.png", dpi=180)
            plt.close()


def _estimate_threshold(df: pd.DataFrame, value_col: str, target_col: str) -> dict:
    """估算粗糙度推荐区间（教学版规则）。

    思路：
    1) 先取 value_col 的 10%-90% 区间，避免两端极端值影响；
    2) 在该区间内将 value_col 分成 5 个桶；
    3) 计算每个桶内 target_col 的均值；
    4) 选择均值最高的桶作为推荐范围。
    """
    q_low, q_high = df[value_col].quantile([0.1, 0.9]).tolist()
    subset = df[(df[value_col] >= q_low) & (df[value_col] <= q_high)]
    if subset.empty:
        return {"recommended_low": None, "recommended_high": None}
    grouped = subset.groupby(pd.cut(subset[value_col], bins=5, duplicates="drop"))[target_col].mean()
    best_bin = grouped.idxmax()
    return {
        "recommended_low": float(best_bin.left),
        "recommended_high": float(best_bin.right),
    }


def run_shap_analysis(result: dict, output_dir: Path) -> dict:
    """执行 SHAP 解释并输出最优模型信息 JSON。

    若 SHAP 运行失败（例如环境未安装或底层依赖问题），
    会回退到“与拉伸粘结强度的相关系数绝对值”作为近似贡献度，
    保证流程可持续输出。
    """
    best_model = result["best_model"]
    x_train = result["x_train"]

    if isinstance(best_model, MultiOutputRegressor):
        # 多输出回归器按目标拆成多个子模型，这里用第一个目标模型做 SHAP 展示
        base_estimator = best_model.estimators_[0]
    else:
        base_estimator = best_model

    shap_importance: dict[str, float]
    try:
        import shap

        # 用训练集作为背景数据建立解释器
        explainer = shap.Explainer(base_estimator, x_train)
        shap_values = explainer(x_train)
        # summary_plot 会绘制全局特征影响分布图
        shap.summary_plot(shap_values, x_train, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close()

        # 取 |SHAP| 的均值作为全局贡献排序依据
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_importance = {
            feature: float(score) for feature, score in zip(FEATURE_COLUMNS, mean_abs)
        }
    except Exception:
        # 若 shap 未安装或运行失败，回退为相关性贡献近似，保证流程可运行。
        corr = pd.concat([x_train, result["y_train"]], axis=1).corr(numeric_only=True)
        target_corr = corr["tensile_bond_strength"].reindex(FEATURE_COLUMNS).fillna(0.0).abs()
        shap_importance = {k: float(v) for k, v in target_corr.to_dict().items()}

    merged = pd.concat([result["x_train"], result["y_train"]], axis=1)
    # 针对核心变量 roughness_ra 给出工程建议区间
    rec = _estimate_threshold(merged, "roughness_ra", "tensile_bond_strength")

    output = {
        "best_model": result["best_model_name"],
        "shap_importance": shap_importance,
        "roughness_ra_recommendation": rec,
    }

    with (output_dir / "best_model_info.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output
