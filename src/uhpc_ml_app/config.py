"""全局配置：输入/输出特征、训练超参与数据增强开关。

自 v0.3 起，界面粗糙度不再使用二维线粗糙度 ``roughness_ra`` / ``roughness_rz``，
改用两种**可选的、可二选一或同时提供**的表达方式：

- ``interface_prep``（分类变量，界面处理方式，如喷砂/凿毛/原浇筑面），建模前 one-hot；
- ``roughness_sa``（连续变量，三维面粗糙度 Sa，μm），作为数值特征直接进入模型。

因此 ``NUMERIC_FEATURE_COLUMNS`` 中包含 ``roughness_sa``，而 ``FEATURE_COLUMNS`` 是
**建模层**使用的完整特征列（数值特征 + interface_prep 的 one-hot 列），供 modeling /
reporting / shap 等下游复用，保持既有接口不变。
"""

from dataclasses import dataclass

# 连续数值特征（清洗阶段做中位数填补 + IQR 剔除）。其中 roughness_sa 允许缺失，
# 与 interface_prep 二选一表达粗糙度；缺失时用中位数填补。
NUMERIC_FEATURE_COLUMNS = [
    "roughness_sa",  # 三维面粗糙度 Sa（μm），可选
    "water_binder_ratio",
    "fiber_volume_ratio",
    "nc_strength_grade",
    "curing_age_days",
]

# 分类特征（建模前 one-hot 编码）
CATEGORICAL_FEATURE_COLUMNS = ["interface_prep"]

# 界面处理方式的合法枚举值（编号，与 data_protocol.md 一致；其余值/缺失归入 reference=全 0）
INTERFACE_PREP_VALUES = (
    "C",  # 未处理（光滑/成膜面）untreated
    "SB",  # 喷砂 sandblasting
    "HW",  # 高压水射流 hydrodemolition
    "CH",  # 凿毛/机械凿毛 chipping
    "M",  # 机械铣刨 milling
)

# 编号 → 中文/英文标签，用于 UI 与文档展示（不改动 one-hot 列名逻辑）
INTERFACE_PREP_LABELS: dict[str, str] = {
    "C": "未处理（untreated）",
    "SB": "喷砂（sandblasting）",
    "HW": "高压水射流（hydrodemolition）",
    "CH": "凿毛/机械凿毛（chipping）",
    "M": "机械铣刨（milling）",
}

# 建模层使用的完整特征列：数值特征 + interface_prep 的 one-hot 列。
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + [f"interface_prep__{v}" for v in INTERFACE_PREP_VALUES]

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
    - augment：是否启用 KNN 插值 + 高斯噪声的数据增强（默认关闭，仅作用于训练集，
      测试集保持真实数据，保证评估诚实可对比）。
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
    # 数据增强（默认关闭）
    augment: bool = False
    augment_n_synthetic: int = 100
    augment_noise_scale: float = 0.03

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
