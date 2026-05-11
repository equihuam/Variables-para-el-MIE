from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

pytest.importorskip("pyarrow")


def test_assemble_training_table_smoke(tmp_path: Path):
    features_dir = tmp_path / "features"
    (features_dir / "tasa_erosion").mkdir(parents=True)
    (features_dir / "corales_global").mkdir(parents=True)

    base = pd.DataFrame(
        {
            "regionid": ["region_1", "region_1"],
            "pixid": [1, 2],
            "x": [0.5, 1.5],
            "y": [9.5, 8.5],
        }
    )
    tasa = base.assign(erosion=[0.1, 0.2])
    corales = base.assign(corals=[100.0, 50.0])

    tasa.to_parquet(features_dir / "tasa_erosion" / "region_1.parquet", index=False)
    corales.to_parquet(features_dir / "corales_global" / "region_1.parquet", index=False)

    out = tmp_path / "train_dat_2c.parquet"
    script = Path("scripts/workflow/assemble_training_table.py")

    subprocess.run(
        [sys.executable, str(script), "--features-dir", str(features_dir), "--output", str(out)],
        check=True,
    )

    assert out.exists()
    df = pd.read_parquet(out)
    assert "erosion" in df.columns
    assert "corals" in df.columns
    assert len(df) == 2
