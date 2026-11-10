"""pandera 数据模式：对输入 / 清洗后 DataFrame 做强约束校验。

自 v0.3 起，界面粗糙度用 ``interface_prep``（分类）或 ``roughness_sa``（三维数值）表达，
二者可选、可同时提供。因此：

- ``INPUT_SCHEMA`` 在**原始列**上校验：数值特征/目标允许 NaN，``interface_prep`` 为可选字符串；
- ``CLEAN_SCHEMA`` 在**建模列**上校验：数值特征非空且在物理上下界内，
  ``interface_prep`` 已展开为 one-hot 0/1 列，目标允许缺失（拉伸/抗剪可只填其一）。
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

from uhpc_ml_app.config import (
    FEATURE_COLUMNS,
    INTERFACE_PREP_VALUES,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "roughness_sa": (0.0, 50.0),
    "water_binder_ratio": (0.08, 0.60),
    "fiber_volume_ratio": (0.0, 6.0),
    "nc_strength_grade": (10.0, 120.0),
    "curing_age_days": (0.5, 400.0),
    "tensile_bond_strength": (0.0, 20.0),
    "shear_bond_strength": (0.0, 40.0),
}


def _input_numeric_column(name: str) -> Column:
    return Column(
        float,
        checks=[Check.greater_than_or_equal_to(0.0)],
        nullable=True,
        coerce=True,
        description=f"{name} ≥ 0 (nullable)",
    )


def _clean_numeric_column(name: str, *, nullable: bool) -> Column:
    low, high = _PHYSICAL_BOUNDS[name]
    return Column(
        float,
        checks=[Check.in_range(low, high, include_min=True, include_max=True)],
        nullable=nullable,
        coerce=True,
        description=f"{name} ∈ [{low}, {high}]" + (" (nullable)" if nullable else ""),
    )


def _clean_onehot_column(name: str) -> Column:
    return Column(
        int,
        checks=[Check.isin([0, 1])],
        nullable=False,
        coerce=True,
        description=f"{name} ∈ {{0, 1}} (one-hot)",
    )


# 原始列入口校验（宽松）：数值特征/目标 nullable，interface_prep 可选字符串。
INPUT_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={
        **{name: _input_numeric_column(name) for name in NUMERIC_FEATURE_COLUMNS + TARGET_COLUMNS},
        "interface_prep": Column(str, nullable=True, coerce=False),
    },
    strict="filter",
    coerce=True,
    ordered=False,
)

# 清洗后建模列校验（严格）：数值特征非空 + 物理上下界；one-hot 0/1；目标 nullable 但至少一个。
CLEAN_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={
        **{name: _clean_numeric_column(name, nullable=False) for name in NUMERIC_FEATURE_COLUMNS},
        **{name: _clean_numeric_column(name, nullable=True) for name in TARGET_COLUMNS},
        **{
            name: _clean_onehot_column(name)
            for name in FEATURE_COLUMNS
            if name.startswith("interface_prep__")
        },
    },
    strict="filter",
    coerce=True,
    ordered=False,
    checks=[
        Check(
            lambda df: len(df) >= 8,
            name="min_sample_count",
            error="cleaned dataset must have at least 8 rows",
        ),
        Check(
            lambda df: bool(df[TARGET_COLUMNS].notna().any(axis=1).all()),
            name="each_row_has_at_least_one_target",
            error="each row must have at least one of tensile_bond_strength or shear_bond_strength",
        ),
    ],
)


__all__ = ["INPUT_SCHEMA", "CLEAN_SCHEMA", "INTERFACE_PREP_VALUES"]
