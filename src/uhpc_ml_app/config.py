from dataclasses import dataclass


FEATURE_COLUMNS = [
    "roughness_ra",
    "roughness_rz",
    "water_binder_ratio",
    "fiber_volume_ratio",
    "nc_strength_grade",
    "curing_age_days",
]

TARGET_COLUMNS = [
    "tensile_bond_strength",
    "shear_bond_strength",
]


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    cv_folds: int = 5
