from pathlib import Path

import pandas as pd

from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS


def _clip_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """使用 IQR 规则逐列剔除异常值。

    说明：
    - Q1/Q3: 第 25/75 百分位
    - IQR = Q3 - Q1
    - 保留区间: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    - 这里采用“逐列过滤”的策略，某行只要在任意关键列越界就会被剔除。
    """
    clipped = df.copy()
    for col in columns:
        # 计算当前列的箱线图上下界
        q1 = clipped[col].quantile(0.25)
        q3 = clipped[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        # 仅保留该列处于正常区间的样本，再进入下一列筛选
        clipped = clipped[(clipped[col] >= low) & (clipped[col] <= high)]
    return clipped


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    """读取并清洗文献整合 CSV 数据。

    处理流程（与论文中“数据预处理”部分对应）：
    1) 字段完整性校验（必须包含输入特征+输出标签）
    2) 数值化转换（不可转换项变为 NaN）
    3) 删除整行全空记录
    4) 用中位数填补缺失值（比均值更抗异常值）
    5) IQR 异常值剔除
    6) 重置索引，得到连续样本
    """
    # 读取原始文献整合数据
    df = pd.read_csv(csv_path)
    required = FEATURE_COLUMNS + TARGET_COLUMNS

    # 校验字段是否齐全，便于早失败（Fail Fast）
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    # 只保留建模所需字段，避免无关列干扰
    work = df[required].copy()
    # 强制转为数值，无法转成数值的文本会标记为 NaN
    work = work.apply(pd.to_numeric, errors="coerce")
    # 删除“整行都是空”的无效样本
    work = work.dropna(axis=0, how="all")
    # 用中位数填补缺失值，适合材料试验中偏态分布数据
    work = work.fillna(work.median(numeric_only=True))
    # 使用 IQR 去除明显离群样本
    work = _clip_outliers_iqr(work, required)
    # 重置索引，方便后续训练与结果对齐
    work = work.reset_index(drop=True)

    if work.empty:
        raise ValueError("数据清洗后为空，请检查输入数据质量。")
    return work
