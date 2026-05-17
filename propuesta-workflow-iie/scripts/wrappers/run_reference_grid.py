#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


def main() -> None:
    parser = argparse.ArgumentParser(description="Stub rule: create reference grid by region.")
    parser.add_argument("--region", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stub minimal raster 10x10 for workflow wiring.
    arr = np.ones((10, 10), dtype="float32")
    transform = from_origin(0, 10, 1, 1)
    meta = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(arr, 1)

    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
