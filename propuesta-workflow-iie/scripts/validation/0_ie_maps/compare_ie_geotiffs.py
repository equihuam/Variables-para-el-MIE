#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import rasterio

DEFAULT_NODATA = -9999.0


def read_raster(path: Path, band: int = 1):
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float64")
        meta = {
            "path": str(path),
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "nodata": src.nodata,
            "dtype": src.dtypes[band - 1],
        }
    return arr, meta


def nodata_for(meta: dict, fallback: float) -> float:
    nd = meta.get("nodata")
    if nd is None:
        return fallback
    try:
        if np.isnan(float(nd)):
            return fallback
    except Exception:
        return fallback
    return float(nd)


def valid_mask(arr: np.ndarray, nodata: float) -> np.ndarray:
    return np.isfinite(arr) & (arr != nodata)


def normalize_with_mask(arr: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi == lo:
        raise ValueError("El rango de normalización no puede tener extremos iguales.")
    out = np.full(arr.shape, np.nan, dtype="float64")
    out[valid] = (arr[valid] - lo) / (hi - lo)
    return out


def parse_range(spec: str | None) -> tuple[float, float] | None:
    if spec is None or str(spec).strip() == "":
        return None
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Rango inválido: {spec}. Use, por ejemplo, 1,5")
    return float(parts[0]), float(parts[1])


def same_grid(meta_a: dict, meta_b: dict) -> dict:
    return {
        "same_shape": (meta_a["height"] == meta_b["height"] and meta_a["width"] == meta_b["width"]),
        "same_crs": meta_a["crs"] == meta_b["crs"],
        "same_transform": np.allclose(meta_a["transform"], meta_b["transform"], rtol=0, atol=1e-12),
    }


def pair_stats(a: np.ndarray, b: np.ndarray, a_valid: np.ndarray, b_valid: np.ndarray, prefix: str) -> dict:
    both = a_valid & b_valid
    only_a = a_valid & ~b_valid
    only_b = b_valid & ~a_valid

    row = {
        f"{prefix}_n_common_valid": int(both.sum()),
        f"{prefix}_n_only_a": int(only_a.sum()),
        f"{prefix}_n_only_b": int(only_b.sum()),
    }

    if both.sum() == 0:
        row.update({
            f"{prefix}_a_min": np.nan,
            f"{prefix}_a_max": np.nan,
            f"{prefix}_b_min": np.nan,
            f"{prefix}_b_max": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_max_abs_diff": np.nan,
            f"{prefix}_cor": np.nan,
        })
        return row

    av = a[both]
    bv = b[both]
    diff = bv - av
    abs_diff = np.abs(diff)

    row.update({
        f"{prefix}_a_min": float(np.min(av)),
        f"{prefix}_a_max": float(np.max(av)),
        f"{prefix}_b_min": float(np.min(bv)),
        f"{prefix}_b_max": float(np.max(bv)),
        f"{prefix}_mae": float(np.mean(abs_diff)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}_max_abs_diff": float(np.max(abs_diff)),
        f"{prefix}_cor": float(np.corrcoef(av, bv)[0, 1]) if both.sum() > 1 else np.nan,
    })
    return row


def compare_region(
    region: str,
    r_path: Path,
    py_path: Path,
    nodata: float,
    reference_path: Path | None = None,
    reference_band: int = 1,
    reference_normalize_from: tuple[float, float] | None = None,
    reference_nodata: float | None = None,
) -> dict:
    r_arr, r_meta = read_raster(r_path)
    py_arr, py_meta = read_raster(py_path)

    grid = same_grid(r_meta, py_meta)
    row = {
        "region": region,
        **grid,
        "r_height": r_meta["height"],
        "r_width": r_meta["width"],
        "py_height": py_meta["height"],
        "py_width": py_meta["width"],
        "r_path": str(r_path),
        "py_path": str(py_path),
    }

    if not grid["same_shape"]:
        row.update({"status": "shape_mismatch"})
        return row

    r_nd = nodata_for(r_meta, nodata)
    py_nd = nodata_for(py_meta, nodata)
    r_valid = valid_mask(r_arr, r_nd)
    py_valid = valid_mask(py_arr, py_nd)

    row.update({
        "n_r_valid": int(r_valid.sum()),
        "n_py_valid": int(py_valid.sum()),
    })
    row.update(pair_stats(r_arr, py_arr, r_valid, py_valid, "r_py"))

    # Backwards-compatible aliases.
    row.update({
        "n_common_valid": row["r_py_n_common_valid"],
        "n_only_r": row["r_py_n_only_a"],
        "n_only_py": row["r_py_n_only_b"],
        "r_min": row["r_py_a_min"],
        "r_max": row["r_py_a_max"],
        "py_min": row["r_py_b_min"],
        "py_max": row["r_py_b_max"],
        "mae": row["r_py_mae"],
        "rmse": row["r_py_rmse"],
        "max_abs_diff": row["r_py_max_abs_diff"],
        "cor": row["r_py_cor"],
    })

    if reference_path is not None:
        if not reference_path.exists():
            row.update({
                "reference_path": str(reference_path),
                "reference_status": "missing_file",
            })
        else:
            ref_arr_raw, ref_meta = read_raster(reference_path, band=reference_band)
            ref_grid_r = same_grid(ref_meta, r_meta)
            ref_grid_py = same_grid(ref_meta, py_meta)

            ref_nd = reference_nodata if reference_nodata is not None else nodata_for(ref_meta, nodata)
            ref_valid_raw = valid_mask(ref_arr_raw, ref_nd)

            ref_arr = ref_arr_raw.copy()
            ref_valid = ref_valid_raw.copy()

            if reference_normalize_from is not None:
                lo, hi = reference_normalize_from
                ref_arr = normalize_with_mask(ref_arr_raw, ref_valid_raw, lo, hi)
                ref_valid = np.isfinite(ref_arr)

            row.update({
                "reference_path": str(reference_path),
                "ref_height": ref_meta["height"],
                "ref_width": ref_meta["width"],
                "ref_count": ref_meta["count"],
                "ref_nodata_used": ref_nd,
                "ref_same_shape_r": ref_grid_r["same_shape"],
                "ref_same_crs_r": ref_grid_r["same_crs"],
                "ref_same_transform_r": ref_grid_r["same_transform"],
                "ref_same_shape_py": ref_grid_py["same_shape"],
                "ref_same_crs_py": ref_grid_py["same_crs"],
                "ref_same_transform_py": ref_grid_py["same_transform"],
                "n_ref_valid": int(ref_valid.sum()),
            })

            if not ref_grid_r["same_shape"] or not ref_grid_py["same_shape"]:
                row.update({"reference_status": "shape_mismatch"})
            else:
                row.update(pair_stats(ref_arr, r_arr, ref_valid, r_valid, "ref_r"))
                row.update(pair_stats(ref_arr, py_arr, ref_valid, py_valid, "ref_py"))
                row.update({
                    "cor_ref_r": row["ref_r_cor"],
                    "cor_ref_py": row["ref_py_cor"],
                    "mae_ref_r": row["ref_r_mae"],
                    "mae_ref_py": row["ref_py_mae"],
                    "rmse_ref_r": row["ref_r_rmse"],
                    "rmse_ref_py": row["ref_py_rmse"],
                    "reference_status": "ok",
                })

    row.update({"status": "ok"})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-dir", required=True, help="Carpeta con GeoTIFFs producidos por R.")
    parser.add_argument("--py-dir", required=True, help="Carpeta con GeoTIFFs producidos por Python.")
    parser.add_argument("--output", required=True, help="CSV de resumen.")
    parser.add_argument("--regions", default=",".join([f"region_{i}" for i in range(1, 15)]))
    parser.add_argument("--r-pattern", default="eicoastal_{region}.tif")
    parser.add_argument("--py-pattern", default="eicoastal_{region}.tif")
    parser.add_argument("--nodata", type=float, default=DEFAULT_NODATA)
    parser.add_argument("--reference-dir", default=None, help="Carpeta con GeoTIFFs de referencia, por ejemplo ei_qnint.")
    parser.add_argument("--reference-pattern", default="{region}.tif")
    parser.add_argument("--reference-band", type=int, default=1)
    parser.add_argument("--reference-normalize-from", default=None, help="Rango para normalizar referencia, por ejemplo 1,5.")
    parser.add_argument("--reference-nodata", type=float, default=None, help="NoData explícito para la referencia. Si se omite, usa el nodata del raster o --nodata.")
    args = parser.parse_args()

    r_dir = Path(args.r_dir)
    py_dir = Path(args.py_dir)
    ref_dir = Path(args.reference_dir) if args.reference_dir else None
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    ref_norm = parse_range(args.reference_normalize_from)

    rows = []
    for region in regions:
        r_path = r_dir / args.r_pattern.format(region=region)
        py_path = py_dir / args.py_pattern.format(region=region)
        ref_path = ref_dir / args.reference_pattern.format(region=region) if ref_dir else None

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

        rows.append(compare_region(
            region=region,
            r_path=r_path,
            py_path=py_path,
            nodata=args.nodata,
            reference_path=ref_path,
            reference_band=args.reference_band,
            reference_normalize_from=ref_norm,
            reference_nodata=args.reference_nodata,
        ))

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out)
    print(f"OK -> {args.output}")


if __name__ == "__main__":
    main()
