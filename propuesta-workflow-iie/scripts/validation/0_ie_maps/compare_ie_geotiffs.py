#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import rasterio


NODATA = -9999.0


def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        meta = {
            "path": str(path),
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }
    return arr, meta


def valid_mask(arr, nodata):
    return np.isfinite(arr) & (arr != nodata)


def compare_pair(r_path: Path, py_path: Path, region: str, nodata: float):
    r_arr, r_meta = read_raster(r_path)
    py_arr, py_meta = read_raster(py_path)

    same_shape = r_arr.shape == py_arr.shape
    same_crs = r_meta["crs"] == py_meta["crs"]
    same_transform = np.allclose(r_meta["transform"], py_meta["transform"], rtol=0, atol=1e-12)

    row = {
        "region": region,
        "same_shape": same_shape,
        "same_crs": same_crs,
        "same_transform": same_transform,
        "r_height": r_meta["height"],
        "r_width": r_meta["width"],
        "py_height": py_meta["height"],
        "py_width": py_meta["width"],
    }

    if not same_shape:
        row.update({
            "n_common_valid": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs_diff": np.nan,
            "cor": np.nan,
            "status": "shape_mismatch",
        })
        return row

    r_valid = valid_mask(r_arr, nodata)
    py_valid = valid_mask(py_arr, nodata)
    both = r_valid & py_valid

    only_r = r_valid & ~py_valid
    only_py = py_valid & ~r_valid

    row.update({
        "n_r_valid": int(r_valid.sum()),
        "n_py_valid": int(py_valid.sum()),
        "n_common_valid": int(both.sum()),
        "n_only_r": int(only_r.sum()),
        "n_only_py": int(only_py.sum()),
    })

    if both.sum() == 0:
        row.update({
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs_diff": np.nan,
            "cor": np.nan,
            "status": "no_common_valid_pixels",
        })
        return row

    diff = py_arr[both] - r_arr[both]
    abs_diff = np.abs(diff)

    r_vals = r_arr[both]
    py_vals = py_arr[both]

    row.update({
        "r_min": float(np.min(r_vals)),
        "r_max": float(np.max(r_vals)),
        "py_min": float(np.min(py_vals)),
        "py_max": float(np.max(py_vals)),
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_diff": float(np.max(abs_diff)),
        "cor": float(np.corrcoef(r_vals, py_vals)[0, 1]) if both.sum() > 1 else np.nan,
        "status": "ok",
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-dir", required=True, help="Carpeta con GeoTIFFs producidos por R.")
    parser.add_argument("--py-dir", required=True, help="Carpeta con GeoTIFFs producidos por Python.")
    parser.add_argument("--output", required=True, help="CSV de resumen.")
    parser.add_argument("--regions", default=",".join([f"region_{i}" for i in range(1, 15)]))
    parser.add_argument("--r-pattern", default="eicoastal_{region}.tif")
    parser.add_argument("--py-pattern", default="eicoastal_{region}.tif")
    parser.add_argument("--nodata", type=float, default=NODATA)
    args = parser.parse_args()

    r_dir = Path(args.r_dir)
    py_dir = Path(args.py_dir)
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    rows = []
    for region in regions:
        r_path = r_dir / args.r_pattern.format(region=region)
        py_path = py_dir / args.py_pattern.format(region=region)

        if not r_path.exists() or not py_path.exists():
            rows.append({
                "region": region,
                "status": "missing_file",
                "r_exists": r_path.exists(),
                "py_exists": py_path.exists(),
                "r_path": str(r_path),
                "py_path": str(py_path),
            })
            continue

        rows.append(compare_pair(r_path, py_path, region, args.nodata))

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out)
    print(f"OK -> {args.output}")


if __name__ == "__main__":
    main()