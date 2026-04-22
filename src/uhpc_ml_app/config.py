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
    """训练与评估配置。

    - test_size / cv_folds：对应论文"8:2 划分 + 5 折 CV"。
    - repeated_cv_n_repeats：Repeated K-Fold 的重复次数，用于得到均值±95% CI 的
      无偏泛化估计，默认 10；小样本科研数据强烈建议保留。
    - permutation_importance_n_repeats：Permutation Importance 的重复次数，
      用于统一三模型（RF/XGBoost/BP）的特征重要性比较口径。
    - include_baselines：是否在指标表中并入 mean/linear/empirical 基线模型。
    """

    test_size: float = 0.2
    cv_folds: int = 5
    repeated_cv_n_splits: int = 5
    repeated_cv_n_repeats: int = 10
    permutation_importance_n_repeats: int = 20
    include_baselines: bool = True
    # 选最优模型时，使用 Repeated K-Fold CV 的 R² 均值，而非单次 8:2 测试集 R²
    model_selection_metric: str = "cv_r2_mean"
    # 网格模式："full" 用于科研运行；"fast" 用于测试/冒烟跑（每参数保留 1-2 个候选值）
    grid_mode: str = "full"
    mlp_max_iter: int = 1500

    @classmethod
    def for_testing(cls) -> "TrainConfig":
        """测试用压缩配置，避免单测运行数分钟。

        在 CI / 本地 pytest 下使用；论文级结果必须用默认 ``TrainConfig()``。
        """
        return cls(
            repeated_cv_n_splits=3,
            repeated_cv_n_repeats=2,
            permutation_importance_n_repeats=3,
            grid_mode="fast",
            mlp_max_iter=400,
        )
