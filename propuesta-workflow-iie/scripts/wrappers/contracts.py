from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_SPATIAL_COLUMNS = ["regionid", "pixid", "x", "y"]


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def assert_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def assert_spatial_contract(df: pd.DataFrame) -> None:
    assert_required_columns(df, REQUIRED_SPATIAL_COLUMNS)
    if df[["regionid", "pixid"]].duplicated().any():
        raise ValueError("Duplicated (regionid, pixid) keys found")
