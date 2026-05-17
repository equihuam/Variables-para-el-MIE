#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from scripts.workflow.contracts import ensure_parent, assert_spatial_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble feature tables into one training table.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    out = ensure_parent(args.output)

    tasa_files = sorted((features_dir / "tasa_erosion").glob("*.parquet"))
    coral_files = sorted((features_dir / "corales_global").glob("*.parquet"))

    tasa = pd.concat([pd.read_parquet(p) for p in tasa_files], ignore_index=True)
    coral = pd.concat([pd.read_parquet(p) for p in coral_files], ignore_index=True)

    assert_spatial_contract(tasa)
    assert_spatial_contract(coral)

    dat = tasa.merge(coral, on=["regionid", "pixid", "x", "y"], how="inner")
    dat.to_parquet(out, index=False)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
