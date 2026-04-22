"""pandera 数据模式：对输入 / 清洗后 DataFrame 做强约束校验。

- 输入：UHPC-NC 文献整合原始/半整合 CSV；
- 清洗后：已通过 :func:`uhpc_ml_app.data_pipeline.load_and_clean_data` 的建模就绪表。

两套 schema 的边界尽量贴近论文"数据预处理"节的陈述，并与
:mod:`uhpc_ml_app.config` 的 ``FEATURE_COLUMNS`` / ``TARGET_COLUMNS`` 保持单一事实源。
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

from uhpc_ml_app.config import FEATURE_COLUMNS, TARGET_COLUMNS

# 宽松的"领域物理合理性"边界。真实文献数据通常落在更窄区间，
# 但此处刻意留余量，以便 IQR 去极值 + 中位数填补发生前校验不会误杀。
_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "roughness_ra": (0.0, 30.0),  # μm；文献整合上限一般 <15μm，放到 30 给清洗期
    "roughness_rz": (0.0, 300.0),  # μm；Rz = n × Ra 量级
    "water_binder_ratio": (0.08, 0.60),  # UHPC 远低于普通混凝土；保留 NC 基体可用范围
    "fiber_volume_ratio": (0.0, 6.0),  # vol%
    "nc_strength_grade": (10.0, 120.0),  # MPa
    "curing_age_days": (0.5, 400.0),  # 0.5d = 12h；上限留到 1 年
    "tensile_bond_strength": (0.0, 20.0),  # MPa
    "shear_bond_strength": (0.0, 40.0),  # MPa
}


def _input_column(name: str) -> Column:
    """输入侧：只要求可转换成非负有限 float，允许 NaN；真实的物理上界由 IQR 去掉或在 CLEAN 侧挡。"""
    return Column(
        float,
        checks=[Check.greater_than_or_equal_to(0.0)],
        nullable=True,
        coerce=True,
        description=f"{name} ≥ 0 (nullable; physical bounds enforced by CLEAN_SCHEMA)",
    )


def _clean_column(name: str) -> Column:
    """清洗后：物理合理性硬上下界 + 非空。"""
    low, high = _PHYSICAL_BOUNDS[name]
    return Column(
        float,
        checks=[Check.in_range(low, high, include_min=True, include_max=True)],
        nullable=False,
        coerce=True,
        description=f"{name} ∈ [{low}, {high}]",
    )


# 原始 / 半整合输入：允许缺失值 + 允许超出物理上界（IQR 会清理），交给数据清洗管道处理
INPUT_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={name: _input_column(name) for name in FEATURE_COLUMNS + TARGET_COLUMNS},
    strict="filter",  # 允许额外列进来，但不会被建模使用
    coerce=True,
    ordered=False,
)

# 清洗后：不允许缺失值，且额外要求至少 8 行（样本数下限的硬约束）
CLEAN_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={name: _clean_column(name) for name in FEATURE_COLUMNS + TARGET_COLUMNS},
    strict="filter",
    coerce=True,
    ordered=False,
    checks=[
        Check(
            lambda df: len(df) >= 8,
            name="min_sample_count",
            error="cleaned dataset must have at least 8 rows for an 8:2 split",
        )
    ],
)


__all__ = ["INPUT_SCHEMA", "CLEAN_SCHEMA"]
