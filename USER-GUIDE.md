# USER GUIDE

本文档详细解释 `data_pipeline.py`、`modeling.py`、`reporting.py` 三个核心模块的实现细节，对应到论文研究流程。

---

> ⚠️ 本仓库自带的 `data/raw/literature_sample.csv` 为**合成示例数据**（Ra 近似线性递增、龄期按固定顺序循环），仅用于走通程序流程与单元测试。切勿将其产出直接用于论文正文、答辩或工程决策。真实论文结果必须基于按 `docs/data_protocol.md` 规定流程整理的文献数据库。

## 1. 整体流程图（从数据到结论）

1. **数据清洗**（`load_and_clean_data`）
2. **模型训练与对比**（`train_and_compare_models`）
3. **结果输出与解释**（`save_metrics` / `save_prediction_scatter` / `run_shap_analysis`）

对应论文中的四个关键动作：

- 文献数据整合与预处理（缺失值处理 + 异常值剔除；归一化/标准化当前仅在 BP 网络内部做，整体归一化为后续扩展项，见第 6 节）
- RF / XGBoost / BP 三模型构建与调参
- 精度指标对比 + 可视化验证
- SHAP 解释 + 粗糙度工程推荐区间

---

## 2. `data_pipeline.py` 详解

### 2.1 入口函数：`load_and_clean_data(csv_path)`

此函数完成建模前的全部数据准备工作。

#### 步骤 A：字段校验

- 程序要求输入 CSV 必须包含：
  - 特征列：`roughness_ra`、`roughness_rz`、`water_binder_ratio`、`fiber_volume_ratio`、`nc_strength_grade`、`curing_age_days`
  - 目标列：`tensile_bond_strength`、`shear_bond_strength`
- 如果缺列，立刻抛错，避免后续训练得到错误结论。

#### 步骤 B：数据类型标准化

- 使用 `pd.to_numeric(errors="coerce")` 把可转数值的内容统一转成数值；
- 不能转换的字符串会被设为 `NaN`，之后统一处理。

#### 步骤 C：缺失值处理

- 删除“整行全空”的记录；
- 其余缺失值用每列中位数填补。
- 为什么用中位数：材料试验数据常有偏态分布，中位数比均值更稳健。

#### 步骤 D：异常值剔除（IQR）

- `_clip_outliers_iqr` 按列计算箱线图上下界：
  - `low = Q1 - 1.5*IQR`
  - `high = Q3 + 1.5*IQR`
- 每列依次过滤，越界样本被删除。
- 该策略简单、可解释，适合小样本场景。

---

## 3. `modeling.py` 详解

### 3.1 指标函数：`_evaluate`

统一返回 3 个回归指标：

- `R2`：拟合优度（越大越好）
- `RMSE`：均方根误差（越小越好）
- `MAE`：平均绝对误差（越小越好）

### 3.2 模型工厂：`_build_models`

构建三类模型及其参数网格：

1. `random_forest`：随机森林回归（多输出封装）
2. `xgboost`：
   - 若环境可用：`XGBRegressor`
   - 若不可用（例如 macOS 缺 `libomp`）：自动回退 `GradientBoostingRegressor`
3. `bp_neural_network`：`MLPRegressor`

### 3.3 特征重要性：`_collect_feature_importance`

- 对树模型（RF / XGBoost）读取内置 `feature_importances_`；
- 对 BP 网络（MLP）没有直接内置重要性，当前实现返回 0 占位，主要依赖后续 SHAP 分析解释特征贡献。

### 3.4 核心训练函数：`train_and_compare_models`

#### 步骤 A：训练测试划分

- 使用 `train_test_split(test_size=0.2)`，即 8:2 划分。

#### 步骤 B：网格搜索 + 5 折交叉验证

- `GridSearchCV(cv=5, scoring="r2")`
- 每个模型都做超参数组合搜索；
- 自动回填最优参数得到 `best_estimator_`。

#### 步骤 C：训练集/测试集双指标评估

- 防止只看测试集而忽略过拟合迹象。

#### 步骤 D：自动选择最优模型

- 以测试集 `R2` 最大为准。

#### 步骤 E：输出完整结果字典

返回内容包括：

- `metrics`：三模型训练/测试指标
- `feature_importance_by_model`：每模型每目标的重要性
- `predictions_by_model`：每模型预测结果（用于散点图）
- `best_model`、`best_model_name`
- `x_train`、`y_train`、`y_test_df`（用于 SHAP 与报告）

---

## 4. `reporting.py` 详解

### 4.1 `save_metrics`

- 把三模型指标整理成表格导出 `metrics_summary.csv`；
- 便于直接用于论文表格。

### 4.2 `save_prediction_scatter`

- 生成“实测-预测”散点图；
- 红色虚线 `y=x` 为理想预测线；
- 点越集中在虚线附近，说明模型预测越可靠。

输出文件命名示例：

- `scatter_random_forest_tensile_bond_strength.png`
- `scatter_xgboost_shear_bond_strength.png`

### 4.3 `run_shap_analysis`

#### 主流程

1. 读取最优模型与训练特征；
2. 构建 SHAP 解释器并计算 `shap_values`；
3. 保存 `shap_summary.png`；
4. 计算每个特征的 `mean(|SHAP|)` 作为全局贡献度。

#### 异常回退

- 若 SHAP 不可用或报错，回退到“与 `tensile_bond_strength` 的相关系数绝对值”近似贡献。
- 这样可以保证流程稳定，不会因为环境问题中断。

### 4.4 粗糙度推荐区间 `_estimate_threshold`

实现逻辑：

1. 取 `roughness_ra` 的 10%-90% 区间；
2. 分 5 桶；
3. 计算每桶平均拉伸粘结强度；
4. 选择均值最高桶作为推荐区间。

最终与 SHAP 结果一起写入 `best_model_info.json`。

---

## 5. 你在写论文时可如何引用这些结果

- **模型精度对比**：`metrics_summary.csv`
- **可靠性与泛化能力展示**：`scatter_*.png`
- **粗糙度核心影响作用**：`shap_summary.png` + `best_model_info.json`
- **工程推荐区间**：`best_model_info.json` 中 `roughness_ra_recommendation`

---

## 6. 进一步扩展的方向（与 PR 路线图对齐）

1. 全局数据标准化/归一化策略开关（`StandardScaler` / `RobustScaler`），默认关闭，由下游 Pipeline 处理 → **PR#2**
2. 将多输出拆分为两个单输出管道分别优化，支持 per-target 指标 → **PR#2**
3. 最优模型持久化（`joblib.dump`）+ schema 导出 → **PR#2/#4**
4. 不确定性评估（Repeated K-Fold 95% CI / Conformal Prediction） → **PR#2**
5. 用真实文献数据做外部验证集评估 → **PR#5**

---

## 7. 真实文献数据库采集说明（新增）

项目新增 `uhpc_ml_app.literature_crawler`，用于合规获取真实文献元数据并自动提取结构化字段。

### 7.1 为什么不直接“全站爬取”WoS/Scopus/CNKI？

- 这些平台通常有登录、授权、反爬与版权条款；
- 建议优先使用官方 API 或机构授权导出；
- 本项目提供开放 API（OpenAlex/Crossref）自动采集 + 人工补录合并方案，确保可复现且合规。

### 7.2 程序做了什么

1. 用关键词检索 OpenAlex / Crossref；
2. 标准化字段（标题、DOI、年份、来源、摘要等）；
3. 去重（优先 DOI）；
4. 基于标题+摘要关键词筛选相关文献；
5. 从摘要中正则抽取：
   - `roughness_ra` / `roughness_rz`
   - `water_binder_ratio`
   - `fiber_volume_ratio`
   - `curing_age_days`
   - `tensile_bond_strength` / `shear_bond_strength`
6. 输出 `literature_real.csv` 与 `manifest.json`。

### 7.3 命令

```bash
PYTHONPATH=src python -m uhpc_ml_app.literature_crawler \
  --query "UHPC NC repair interface bond roughness Ra Rz water binder ratio fiber volume curing age tensile shear" \
  --output data/raw/literature_real.csv \
  --pages 3
```

### 7.4 后续建议

- 将学校账号可导出的 WoS/Scopus/CNKI 文献结果补录到同一 CSV；
- 对自动抽取出的数值进行人工复核（摘要文本常不完整，全文更可靠）；
- 将复核后的数据作为最终训练集输入主建模流程。

---

## 8. “模型解释性分析”章节模板（占位，禁止直接复制数值）

> ⚠️ **重要提示（请勿忽略）**
>
> 原版本在本节给出了具体数值（养护龄期 0.0927、Ra 0.0706、推荐区间 3.35–3.83 等），这些数值来源于合成示例数据 `literature_sample.csv`，**不具备任何工程或学术意义**。已按评审建议改为占位模板，以避免被误写入论文正文或答辩材料。
>
> **正式使用前必须：**
>
> 1. 用按 `docs/data_protocol.md`（PRISMA）整理的真实文献数据库重新训练；
> 2. 从当次训练产物 `best_model_info.json` 与 `shap_summary_*.png` 中取数替换所有 `<占位>`；
> 3. 基于 SHAP 依赖图（`shap_dependence_*.png`）而非朴素分桶均值来陈述“阈值区间”与“正负作用方向”。

以下为供论文“模型解释性分析”小节参考的**填写模板**：

基于最优模型（`<模型名：RF / XGBoost / BP>`）的 SHAP 解释结果可知，不同输入特征对 UHPC-NC 修复界面 `<目标：拉伸 / 抗剪>` 粘结强度预测的贡献存在明显差异。按平均绝对 SHAP 值排序，特征重要性由高到低依次为：`<特征1>`（`<SHAP均值1>`）、`<特征2>`（`<SHAP均值2>`）、…、`<特征n>`（`<SHAP均值n>`）。该结果表明，在本研究数据集条件下，`<简要说明主导因素>`，界面粗糙度 `<Ra / Rz>` 排序为第 `<位次>` 位，贡献占比约 `<百分比>`，`<说明其是否为核心影响因素及耦合关系>`。

进一步结合 SHAP 依赖图与部分依赖图（PDP）可知，`<Ra / Rz>` 在 `<下界>`–`<上界>` 范围内对应 SHAP 正效应平台段（或符号翻转点之前的正贡献区间），可作为界面处理工艺的推荐控制区间。在 `水胶比 <区间>` 与 `养护龄期 <区间>` 等耦合条件下，该推荐区间可能进一步收窄/平移（详见条件化依赖图）。总体而言，SHAP 分析不仅提升了模型可解释性，也为工程实践中关键参数的定量选取提供了依据。

> 使用步骤：训练完成后打开 `outputs/best_model_info.json` → 按 `shap_importance.<target>` 字段填入各特征均值；按 `roughness_recommendation.<feature>.<target>` 字段填入阈值区间；结合 `outputs/shap_summary_<target>.png` 的红蓝点分布陈述正负作用方向。PR#3 起所有 SHAP 相关产物将按目标分拆输出。

### 8.1 常见错误（切勿出现）

- 直接复制本模板中的 `<占位>` 字样作为数值；
- 使用合成数据 `literature_sample.csv` 跑出的数字写入论文；
- 使用 SHAP 失败回退路径（`explanation_method = correlation_fallback`，见 PR#3）得到的“重要性”作为 SHAP 结论。

### 8.2 图注版（对应 `shap_summary.png`）

图 X 展示了最优模型（XGBoost）的 SHAP 总结图。各特征按平均绝对 SHAP 值由高到低排序，依次为养护龄期、水胶比、纤维体积率、`Ra`、`Rz` 与 NC 强度等级。横坐标为 SHAP 值，表示该特征对预测结果的正负贡献；点的颜色表示特征取值高低。结果表明，样本中养护龄期与配比参数的整体贡献更高，粗糙度参数（`Ra`、`Rz`）仍表现出显著影响。基于模型输出，推荐界面粗糙度 `Ra` 控制在 3.35-3.83 区间。
