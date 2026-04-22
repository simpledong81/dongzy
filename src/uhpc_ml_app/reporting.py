"""结果输出模块：指标表、预测散点图、特征重要性、SHAP 解释与最优模型摘要。

PR#2 起本模块的对外 API 保持**向下兼容**：

- ``save_metrics`` / ``save_prediction_scatter`` / ``run_shap_analysis`` 继续
  存在、继续输出 ``metrics_summary.csv`` 与 ``best_model_info.json``；
- 新增 ``save_per_target_metrics`` / ``save_cv_metrics`` /
  ``save_permutation_importance`` / ``save_baseline_metrics``，按目标与基线
  拆分指标，便于论文中写"按目标的 R²/RMSE/MAE 与 Repeated K-Fold 95% CI"表格；
- PR#3 将接管 ``run_shap_analysis`` 做**双目标 + 依赖图驱动阈值**的重写。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS


# ---------------------------------------------------------------------------
# Metrics tables
# ---------------------------------------------------------------------------


def save_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``metrics_summary.csv``：每模型的 overall train/test 指标。

    为了兼容现有 Streamlit UI 与论文表格直接拷贝使用，列顺序与原版保持一致，
    同时补充 best_params 的 JSON 字符串便于审查。
    """
    rows = []
    cv_metrics = result.get("cv_metrics", {})
    for model, metric in result["metrics"].items():
        row = {
            "model": model,
            "train_r2": metric["train"]["r2"],
            "train_rmse": metric["train"]["rmse"],
            "train_mae": metric["train"]["mae"],
            "test_r2": metric["test"]["r2"],
            "test_rmse": metric["test"]["rmse"],
            "test_mae": metric["test"]["mae"],
        }
        cv = cv_metrics.get(model, {})
        row["cv_r2_mean"] = cv.get("r2_mean", float("nan"))
        row["cv_r2_ci95_low"] = cv.get("r2_ci95_low", float("nan"))
        row["cv_r2_ci95_high"] = cv.get("r2_ci95_high", float("nan"))
        row["best_params"] = json.dumps(metric.get("best_params", {}), ensure_ascii=False)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)


def save_per_target_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``metrics_per_target.csv``（长表）：每模型 × 每目标 × 训练/测试 的指标。

    论文正文/附表中"拉伸粘结强度 R²、抗剪粘结强度 R²"需要分列呈现，
    长表比宽表更稳健，便于直接在 Excel/LaTeX 中透视。
    """
    metrics_per_target = result.get("metrics_per_target", {})
    rows: list[dict[str, object]] = []
    for model, splits in metrics_per_target.items():
        for split_name in ("train", "test"):
            per_target = splits.get(split_name, {})
            for target, scores in per_target.items():
                rows.append(
                    {
                        "model": model,
                        "split": split_name,
                        "target": target,
                        "r2": scores.get("r2", float("nan")),
                        "rmse": scores.get("rmse", float("nan")),
                        "mae": scores.get("mae", float("nan")),
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "metrics_per_target.csv", index=False)


def save_cv_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``cv_metrics.csv``：Repeated K-Fold 95% CI。"""
    rows: list[dict[str, object]] = []
    for model, cv in result.get("cv_metrics", {}).items():
        rows.append(
            {
                "model": model,
                "r2_mean": cv.get("r2_mean", float("nan")),
                "r2_ci95_low": cv.get("r2_ci95_low", float("nan")),
                "r2_ci95_high": cv.get("r2_ci95_high", float("nan")),
                "n_splits": cv.get("n_splits"),
                "n_repeats": cv.get("n_repeats"),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "cv_metrics.csv", index=False)


def save_permutation_importance(result: dict, output_dir: Path) -> None:
    """导出 ``permutation_importance.csv`` + 每模型一张条形图。

    Permutation Importance 与 `feature_importances_` 相比的好处是**跨模型统一口径**
    （RF/XGBoost/BP 都能打分），因此适合作为论文中"多模型特征重要性对比"的主要依据；
    树模型内置重要性仍由 ``feature_importance_by_model`` 另行保留。
    """
    pim = result.get("permutation_importance_by_model", {})
    rows: list[dict[str, object]] = []
    for model, per_target in pim.items():
        for target, feature_scores in per_target.items():
            for feature, score in feature_scores.items():
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "feature": feature,
                        "permutation_importance_mean": float(score),
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "permutation_importance.csv", index=False)

    # 每模型画一张横向条形图（按均值大小排序，tensile/shear 并列）
    for model, per_target in pim.items():
        df = pd.DataFrame(per_target).fillna(0.0)
        if df.empty:
            continue
        df = df.reindex(FEATURE_COLUMNS)
        df.index.name = "feature"
        df = df.sort_values(by=df.columns[0], ascending=True)
        ax = df.plot(kind="barh", figsize=(6, 4))
        ax.set_xlabel("Permutation importance (R² drop)")
        ax.set_title(f"Permutation importance — {model}")
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / f"permutation_importance_{model}.png", dpi=180)
        plt.close(ax.figure)


def save_baseline_metrics(result: dict, output_dir: Path) -> None:
    """导出 ``baseline_metrics.csv``：均值/线性/经验公式基线的测试集指标 + CV 均值。

    让读者一眼看到 ML 模型是否真的优于"零学习"基线与经典经验公式。
    """
    baseline_metrics = result.get("baseline_metrics", {})
    rows: list[dict[str, object]] = []
    for name, entry in baseline_metrics.items():
        test_overall = entry.get("test", {}).get("overall", {})
        cv = entry.get("cv", {})
        rows.append(
            {
                "model": name,
                "test_r2": test_overall.get("r2", float("nan")),
                "test_rmse": test_overall.get("rmse", float("nan")),
                "test_mae": test_overall.get("mae", float("nan")),
                "cv_r2_mean": cv.get("r2_mean", float("nan")),
                "cv_r2_ci95_low": cv.get("r2_ci95_low", float("nan")),
                "cv_r2_ci95_high": cv.get("r2_ci95_high", float("nan")),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "baseline_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_prediction_scatter(result: dict, output_dir: Path) -> None:
    """绘制"实测值-预测值"散点图，用于直观评估模型泛化能力。"""
    y_true = result["y_test_df"].reset_index(drop=True)
    pred_source = result.get(
        "predictions_by_model", {result["best_model_name"]: result["predictions"]}
    )
    for model, pred_dict in pred_source.items():
        pred_df = pd.DataFrame(pred_dict)
        for target in TARGET_COLUMNS:
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


# ---------------------------------------------------------------------------
# SHAP  (PR#3 will supersede this with dual-target + dependence-driven logic)
# ---------------------------------------------------------------------------


def _estimate_threshold(df: pd.DataFrame, value_col: str, target_col: str) -> dict:
    """估算粗糙度推荐区间（**教学版规则，非 SHAP 驱动**）。

    PR#3 将用 SHAP 依赖图的符号翻转点/正效应平台段替换此函数。
    当前实现保留是为了让 ``best_model_info.json`` 的字段结构不发生破坏性变更。
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


def _unwrap_for_shap(best_model):
    """取出 SHAP 能直接解释的子估计器（如 RF/XGBRegressor）。"""
    inner = best_model
    if isinstance(inner, Pipeline):
        inner = inner.steps[-1][1]
    if isinstance(inner, MultiOutputRegressor):
        inner = inner.estimators_[0]
    return inner


def run_shap_analysis(result: dict, output_dir: Path) -> dict:
    """执行 SHAP 解释（PR#2 仍保留单目标版本，PR#3 将替换为双目标实现）。"""
    best_model = result["best_model"]
    x_train = result["x_train"]

    base_estimator = _unwrap_for_shap(best_model)

    shap_importance: dict[str, float]
    explanation_method = "shap_tree_or_kernel"
    try:
        import shap

        explainer = shap.Explainer(base_estimator, x_train)
        shap_values = explainer(x_train)
        shap.summary_plot(shap_values, x_train, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close()

        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_importance = {
            feature: float(score) for feature, score in zip(FEATURE_COLUMNS, mean_abs)
        }
    except Exception:
        explanation_method = "correlation_fallback"
        corr = pd.concat([x_train, result["y_train"]], axis=1).corr(numeric_only=True)
        target_corr = (
            corr["tensile_bond_strength"].reindex(FEATURE_COLUMNS).fillna(0.0).abs()
        )
        shap_importance = {k: float(v) for k, v in target_corr.to_dict().items()}

    merged = pd.concat([result["x_train"], result["y_train"]], axis=1)
    rec = _estimate_threshold(merged, "roughness_ra", "tensile_bond_strength")

    output = {
        "best_model": result["best_model_name"],
        "model_selection_metric": result.get("model_selection_metric", "test_r2"),
        "explanation_method": explanation_method,
        "shap_importance": shap_importance,
        "roughness_ra_recommendation": rec,
        # 将 PR#2 的 CV 指标也写入，便于 Streamlit/论文直接引用
        "cv_metrics": result.get("cv_metrics", {}).get(result["best_model_name"], {}),
        "metrics_per_target": result.get("metrics_per_target", {}).get(
            result["best_model_name"], {}
        ),
    }

    with (output_dir / "best_model_info.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "save_metrics",
    "save_per_target_metrics",
    "save_cv_metrics",
    "save_permutation_importance",
    "save_baseline_metrics",
    "save_prediction_scatter",
    "run_shap_analysis",
]
