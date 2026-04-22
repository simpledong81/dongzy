from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from uhpc_ml_app.app import run_pipeline
from uhpc_ml_app.web_export import build_results_zip

st.set_page_config(page_title="UHPC-NC 界面粘结预测", layout="wide")
st.title("UHPC-NC 修复界面粘结性能预测")
st.caption("上传文献整合 CSV，自动完成清洗、建模、评估、可解释性分析与结果导出。")

with st.expander("输入数据字段要求", expanded=False):
    st.code(
        "\n".join(
            [
                "roughness_ra",
                "roughness_rz",
                "water_binder_ratio",
                "fiber_volume_ratio",
                "nc_strength_grade",
                "curing_age_days",
                "tensile_bond_strength",
                "shear_bond_strength",
            ]
        )
    )

uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
seed = st.number_input("随机种子", min_value=1, value=42, step=1)
run_clicked = st.button("开始训练与分析", type="primary", disabled=uploaded is None)

if uploaded is not None:
    preview = pd.read_csv(uploaded).head(10)
    st.subheader("数据预览（前10行）")
    st.dataframe(preview, use_container_width=True)
    uploaded.seek(0)

if run_clicked and uploaded is not None:
    with tempfile.TemporaryDirectory(prefix="uhpc_streamlit_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.csv"
        output_dir = tmp_path / "outputs"

        input_path.write_bytes(uploaded.getvalue())
        with st.spinner("正在建模与生成图表，请稍候..."):
            run_pipeline(input_path, output_dir, random_state=int(seed))

        st.success("分析完成")

        metrics_path = output_dir / "metrics_summary.csv"
        model_info_path = output_dir / "best_model_info.json"

        st.subheader("模型评估指标")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
        st.download_button(
            "下载 metrics_summary.csv",
            data=metrics_path.read_bytes(),
            file_name="metrics_summary.csv",
            mime="text/csv",
        )

        st.subheader("最优模型与粗糙度推荐区间")
        model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
        st.json(model_info)
        st.download_button(
            "下载 best_model_info.json",
            data=model_info_path.read_bytes(),
            file_name="best_model_info.json",
            mime="application/json",
        )

        st.subheader("图表预览与下载")
        image_files = sorted(output_dir.glob("*.png"))
        cols = st.columns(2)
        for idx, img_path in enumerate(image_files):
            with cols[idx % 2]:
                st.image(str(img_path), caption=img_path.name)
                st.download_button(
                    f"下载 {img_path.name}",
                    data=img_path.read_bytes(),
                    file_name=img_path.name,
                    mime="image/png",
                    key=f"download_{img_path.name}",
                )

        zip_bytes = build_results_zip(output_dir)
        st.download_button(
            "一键下载全部结果（ZIP）",
            data=zip_bytes,
            file_name="uhpc_results.zip",
            mime="application/zip",
        )
