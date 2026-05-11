from pathlib import Path
import pandas as pd
import pytest

from scripts.workflow.contracts import assert_required_columns, assert_spatial_contract, ensure_parent


def test_ensure_parent(tmp_path: Path):
    p = ensure_parent(tmp_path / "a" / "b" / "file.parquet")
    assert p.parent.exists()


def test_assert_required_columns_ok():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert_required_columns(df, ["a", "b"])


def test_assert_required_columns_fail():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        assert_required_columns(df, ["a", "b"])


def test_assert_spatial_contract_ok():
    df = pd.DataFrame(
        {
            "regionid": ["region_1", "region_1"],
            "pixid": [1, 2],
            "x": [0.1, 0.2],
            "y": [1.1, 1.2],
        }
    )
    assert_spatial_contract(df)


def test_assert_spatial_contract_duplicate_key():
    df = pd.DataFrame(
        {
            "regionid": ["region_1", "region_1"],
            "pixid": [1, 1],
            "x": [0.1, 0.2],
            "y": [1.1, 1.2],
        }
    )
    with pytest.raises(ValueError):
        assert_spatial_contract(df)
