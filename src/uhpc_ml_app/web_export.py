from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_results_zip(output_dir: str | Path) -> bytes:
    output_path = Path(output_dir)
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zf:
        for file_path in sorted(output_path.glob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    return buffer.getvalue()
