# UHPC-NC 修复界面粘结性能预测应用

本项目面向论文主题“多因素耦合下界面粗糙度对 UHPC-NC 修复界面粘结性能演化规律及机理研究”，提供一个可复现实验流程的机器学习应用：

- 文献整合数据清洗（缺失值处理、异常值剔除）
- 三模型对比（RF / XGBoost / BP 神经网络）
- 8:2 划分 + 网格搜索 + 5 折交叉验证
- 指标评估（R2 / RMSE / MAE）
- 预测值-实测值散点图
- SHAP 可解释性分析与粗糙度推荐区间

## 1. 使用 Python 3.12 虚拟环境

```bash
cd ./dongzy
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2. 运行测试（TDD）

```bash
cd ./dongzy
source .venv/bin/activate
pytest -q
```

## 3. 运行建模应用

```bash
cd ./dongzy
source .venv/bin/activate
PYTHONPATH=src python -m uhpc_ml_app.app \
  --input data/raw/literature_sample.csv \
  --output outputs
```

## 4. 运行 Streamlit Web 界面

```bash
cd ./dongzy
source .venv/bin/activate
PYTHONPATH=src streamlit run streamlit_app.py
```

页面能力：

- 上传 CSV 并预览前 10 行
- 一键执行建模与可解释性分析
- 在线查看指标表、最优模型信息、散点图与 SHAP 图
- 下载单个结果文件或一键下载 ZIP

## 5. 输入数据字段规范

必须包含以下列（CSV）：

- `roughness_ra`
- `roughness_rz`
- `water_binder_ratio`
- `fiber_volume_ratio`
- `nc_strength_grade`
- `curing_age_days`
- `tensile_bond_strength`
- `shear_bond_strength`

## 6. 输出结果

运行后将生成：

- `outputs/metrics_summary.csv`：三模型训练/测试指标
- `outputs/scatter_*.png`：预测-实测散点图
- `outputs/shap_summary.png`：SHAP 总结图
- `outputs/best_model_info.json`：最优模型、SHAP 贡献及粗糙度推荐区间

## 7. 真实文献采集（合规 API）

通过 OpenAlex + Crossref 获取真实文献元数据，并自动抽取摘要中的结构化指标（Ra/Rz、水胶比、纤维率、龄期、拉伸/抗剪强度）。

```bash
cd ./dongzy
source .venv/bin/activate
PYTHONPATH=src python -m uhpc_ml_app.literature_crawler \
  --query "UHPC NC repair interface bond roughness Ra Rz water binder ratio fiber volume curing age tensile shear" \
  --output data/raw/literature_real.csv \
  --pages 3 \
  --min-filled 4
```

说明：

- 程序会生成：
  - `data/raw/literature_real.csv`
  - `data/raw/literature_real.manifest.json`
- `--min-filled` 表示 8 个建模字段中最少非空数量。建议设为 `4`，可自动过滤低质量记录。
- 对 WoS/Scopus/Engineering Village/CNKI，建议使用机构授权接口或人工补录后并入同一 CSV（这些平台通常有登录和版权限制，不建议绕过访问控制抓取）。
