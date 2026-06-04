#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(description="Stub rule: create final EI raster per region.")
    parser.add_argument("--region", required=True)
    parser.add_argument("--ref-grid", required=True)
    parser.add_argument("--training-table", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.ref_grid) as src:
        meta = src.meta.copy()
        arr = src.read(1).astype("float32") * 0.5
        meta.update(dtype="float32", count=1, compress="lzw")

    with rasterio.open(out, "w", **meta) as dst:
        dst.write(arr, 1)

    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
