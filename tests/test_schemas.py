"""PR#4 pandera schema \u7684\u5355\u6d4b\u3002"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors as pe
import pytest

from uhpc_ml_app.data_pipeline import load_and_clean_data
from uhpc_ml_app.schemas import CLEAN_SCHEMA, INPUT_SCHEMA


def _good_row(ra: float = 2.0, rz: float = 20.0) -> dict:
    return {
        "roughness_ra": ra,
        "roughness_rz": rz,
        "water_binder_ratio": 0.22,
        "fiber_volume_ratio": 1.5,
        "nc_strength_grade": 45.0,
        "curing_age_days": 28.0,
        "tensile_bond_strength": 3.0,
        "shear_bond_strength": 10.0,
    }


def test_input_schema_accepts_nan_and_extreme_outliers() -> None:
    # \u539f\u59cb\u8f93\u5165\u9636\u6bb5\u5bb9\u5fcd NaN \u4e0e\u8fdc\u8d85\u7269\u7406\u4e0a\u754c\u7684\u8fb9\u754c\u503c\uff0cIQR \u4f1a\u5904\u7406
    df = pd.DataFrame(
        [_good_row(), {**_good_row(), "roughness_rz": 9999.0, "water_binder_ratio": float("nan")}]
    )
    validated = INPUT_SCHEMA.validate(df, lazy=True)
    assert len(validated) == 2


def test_input_schema_rejects_negative_values() -> None:
    df = pd.DataFrame([{**_good_row(), "curing_age_days": -1.0}])
    with pytest.raises(pe.SchemaErrors):
        INPUT_SCHEMA.validate(df, lazy=True)


def test_clean_schema_enforces_physical_upper_bound() -> None:
    # \u6e05\u6d17\u540e\u4e0d\u5141\u8bb8\u7269\u7406\u4e0d\u5408\u7406\u503c\u3002
    df = pd.DataFrame([_good_row(rz=9999.0)] * 8)
    with pytest.raises(pe.SchemaErrors):
        CLEAN_SCHEMA.validate(df, lazy=True)


def test_clean_schema_requires_minimum_sample_count() -> None:
    df = pd.DataFrame([_good_row()] * 3)
    with pytest.raises(pe.SchemaErrors):
        CLEAN_SCHEMA.validate(df, lazy=True)


def test_load_and_clean_data_validated_path(tmp_path: Path) -> None:
    csv = tmp_path / "ok.csv"
    pd.DataFrame(
        [_good_row(ra=ra) for ra in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]]
    ).to_csv(csv, index=False)
    cleaned = load_and_clean_data(csv)
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) >= 8


def test_load_and_clean_data_raises_on_negative(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    bad = [_good_row(ra=ra) for ra in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]]
    bad[0]["water_binder_ratio"] = -0.1
    pd.DataFrame(bad).to_csv(csv, index=False)
    with pytest.raises(pe.SchemaErrors):
        load_and_clean_data(csv)
