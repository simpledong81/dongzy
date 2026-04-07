from __future__ import annotations

import argparse
from pathlib import Path

from uhpc_ml_app.data_pipeline import load_and_clean_data
from uhpc_ml_app.modeling import train_and_compare_models
from uhpc_ml_app.reporting import run_shap_analysis, save_metrics, save_prediction_scatter


def run_pipeline(input_csv: str | Path, output_dir: str | Path, random_state: int = 42) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cleaned = load_and_clean_data(input_csv)
    result = train_and_compare_models(cleaned, random_state=random_state)

    save_metrics(result, output)
    save_prediction_scatter(result, output)
    run_shap_analysis(result, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="UHPC-NC 修复界面粘结性能预测应用")
    parser.add_argument("--input", required=True, help="输入文献整合数据 CSV 路径")
    parser.add_argument("--output", default="outputs", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.seed)
    print(f"完成建模与解释性分析，结果已输出到: {args.output}")


if __name__ == "__main__":
    main()
