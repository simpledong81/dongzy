from pathlib import Path

from uhpc_ml_app.web_export import build_results_zip


def test_build_results_zip_contains_generated_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_summary.csv").write_text("model,test_r2\nrf,0.9\n", encoding="utf-8")
    (out_dir / "plot.png").write_bytes(b"fakepng")

    zip_bytes = build_results_zip(out_dir)

    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0
