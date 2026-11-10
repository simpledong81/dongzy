"""数据读取与清洗流水线：规范化 → 中位数填补 → IQR → interface_prep one-hot。"""

from pathlib import Path

import pandas as pd

from uhpc_ml_app.config import (
    CATEGORICAL_FEATURE_COLUMNS,
    INTERFACE_PREP_VALUES,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from uhpc_ml_app.schemas import CLEAN_SCHEMA, INPUT_SCHEMA

# 视为"未提供界面处理方式"的字符串（大小写不敏感，输入先统一为大写再匹配）
_INTERFACE_PREP_MISSING: tuple[str, ...] = (
    "",
    "NAN",
    "NONE",
    "NULL",
    "UNKNOWN",
    "N/A",
    "NA",
    "-",
)


def _normalize_interface_prep(series: pd.Series) -> pd.Series:
    """把界面处理方式统一为大写编号（如 sb → SB）、去空白；缺失/占位符归一为 None。"""
    s = series.astype(str).str.strip().str.upper()
    return s.replace(list(_INTERFACE_PREP_MISSING), None)


def _clip_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """使用 IQR 规则逐列剔除异常值（仅作用于连续数值特征）。"""
    clipped = df.copy()
    for col in columns:
        q1 = clipped[col].quantile(0.25)
        q3 = clipped[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        clipped = clipped[(clipped[col] >= low) & (clipped[col] <= high)]
    return clipped


def _onehot_interface_prep(df: pd.DataFrame) -> pd.DataFrame:
    """把 ``interface_prep`` 展开为 ``interface_prep__<value>`` 0/1 列。

    - 命中 ``INTERFACE_PREP_VALUES`` 的值 → 对应列置 1；
    - 其余值 / 缺失 → 全 0（作为 reference 基准，表示"未提供/其他处理方式"）。
    """
    prep = df["interface_prep"] if "interface_prep" in df.columns else pd.Series(dtype=str)
    out = df.drop(columns=["interface_prep"], errors="ignore").copy()
    for value in INTERFACE_PREP_VALUES:
        col = f"interface_prep__{value}"
        out[col] = (prep == value).astype(int)
    return out


def load_and_clean_data(csv_path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """读取并清洗文献整合 CSV 数据。

    与旧版差异：
    - 界面粗糙度不再用 ``roughness_ra`` / ``roughness_rz``，改用 ``interface_prep``（分类）
      或 ``roughness_sa``（三维数值）表达，二者可二选一、也可同时提供；
    - 连续数值特征（含 ``roughness_sa``）缺失时用中位数填补；``interface_prep`` 缺失归入基准类别；
    - 输出目标（拉伸/抗剪）允许缺失，但至少填一项；不做目标中位数填补，保留 NaN 以便按目标独立建模。
    """
    df = pd.read_csv(csv_path)
    required = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + TARGET_COLUMNS

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    work = df[required].copy()
    for col in NUMERIC_FEATURE_COLUMNS + TARGET_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["interface_prep"] = _normalize_interface_prep(work["interface_prep"])

    if validate:
        work = INPUT_SCHEMA.validate(work, lazy=True)

    work = work.dropna(axis=0, how="all")
    # 至少有一个输出标签
    work = work[work[TARGET_COLUMNS].notna().any(axis=1)]
    # 连续输入特征用中位数填补后仍缺则删行
    work[NUMERIC_FEATURE_COLUMNS] = work[NUMERIC_FEATURE_COLUMNS].fillna(
        work[NUMERIC_FEATURE_COLUMNS].median(numeric_only=True)
    )
    work = work.dropna(subset=NUMERIC_FEATURE_COLUMNS)
    work = _clip_outliers_iqr(work, NUMERIC_FEATURE_COLUMNS)
    work = work.reset_index(drop=True)

    if work.empty:
        raise ValueError("数据清洗后为空，请检查输入数据质量。")

    # interface_prep → one-hot（缺失/其他 → 全 0 基准）
    work = _onehot_interface_prep(work)

    if validate:
        work = CLEAN_SCHEMA.validate(work, lazy=True)
    return work


__all__ = ["load_and_clean_data"]
