"""PR#4 pandera schema 的单测。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors as pe
import pytest

from uhpc_ml_app.config import FEATURE_COLUMNS
from uhpc_ml_app.data_pipeline import load_and_clean_data
from uhpc_ml_app.schemas import CLEAN_SCHEMA, INPUT_SCHEMA


def _good_row(sa: float = 2.0) -> dict:
    return {
        "roughness_sa": sa,
        "water_binder_ratio": 0.22,
        "fiber_volume_ratio": 1.5,
        "nc_strength_grade": 45.0,
        "curing_age_days": 28.0,
        "interface_prep": "SB",
        "tensile_bond_strength": 3.0,
        "shear_bond_strength": 10.0,
    }


def test_input_schema_accepts_nan_and_extreme_outliers() -> None:
    # 原始输入阶段容忍 NaN 与远超物理上界的边界值，IQR 会处理
    df = pd.DataFrame(
        [_good_row(), {**_good_row(), "roughness_sa": 9999.0, "water_binder_ratio": float("nan")}]
    )
    validated = INPUT_SCHEMA.validate(df, lazy=True)
    assert len(validated) == 2


def test_input_schema_rejects_negative_values() -> None:
    df = pd.DataFrame([{**_good_row(), "curing_age_days": -1.0}])
    with pytest.raises(pe.SchemaErrors):
        INPUT_SCHEMA.validate(df, lazy=True)


def test_clean_schema_enforces_physical_upper_bound() -> None:
    # 清洗后不允许物理不合理值。
    df = pd.DataFrame([_good_row(sa=9999.0)] * 8)
    with pytest.raises(pe.SchemaErrors):
        CLEAN_SCHEMA.validate(df, lazy=True)


def test_clean_schema_requires_minimum_sample_count() -> None:
    df = pd.DataFrame([_good_row()] * 3)
    with pytest.raises(pe.SchemaErrors):
        CLEAN_SCHEMA.validate(df, lazy=True)


def test_load_and_clean_data_validated_path(tmp_path: Path) -> None:
    csv = tmp_path / "ok.csv"
    pd.DataFrame(
        [_good_row(sa=sa) for sa in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]]
    ).to_csv(csv, index=False)
    cleaned = load_and_clean_data(csv)
    assert cleaned[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert len(cleaned) >= 8


def test_load_and_clean_data_raises_on_negative(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    bad = [_good_row(sa=sa) for sa in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]]
    bad[0]["water_binder_ratio"] = -0.1
    pd.DataFrame(bad).to_csv(csv, index=False)
    with pytest.raises(pe.SchemaErrors):
        load_and_clean_data(csv)
