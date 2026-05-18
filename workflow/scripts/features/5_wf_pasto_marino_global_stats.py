#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute global seagrass-distance sentinel replacement stats across reference grids."""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree

PASTO_SENTINEL = 999.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calcula maximo global crudo de distancia a pasto.")
    p.add_argument("--pasto-marino-shp", required=True)
    p.add_argument("--ref-grids", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--distance-mode", choices=["kknn_scaled", "raw"], default="kknn_scaled")
    p.add_argument("--validity-mode", choices=["finite", "notnan"], default="finite")
    p.add_argument("--all-touched", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def load_pasto_marino(path: Path) -> gpd.GeoDataFrame:
    pasto = gpd.read_file(path)
    if pasto.empty or pasto.crs is None:
        raise ValueError(f"Shapefile de pasto marino vacio o sin CRS: {path}")
    return pasto[pasto.geometry.notna() & ~pasto.geometry.is_empty].copy()


def reproject_raster_to_crs(src, dst_crs):
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    dst = np.empty((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return dst, transform


def valid_points(arr, transform, validity_mode):
    mask = np.isfinite(arr) if validity_mode == "finite" else ~np.isnan(arr)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)
    xs, ys = xy(transform, rows, cols, offset="center")
    return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])


def rasterize_pasto(shape, transform, pasto, all_touched):
    return rasterize(
        shapes=((geom, 1.0) for geom in pasto.geometry if geom is not None and not geom.is_empty),
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
        all_touched=all_touched,
    )


def pasto_points(pasto_rast, transform):
    rows, cols = np.where(np.isfinite(pasto_rast))
    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)
    xs, ys = xy(transform, rows, cols, offset="center")
    return np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])


def scale_train_test(train_xy, pred_xy):
    sd = train_xy.std(axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd != 0), sd, 1.0)
    return train_xy / sd, pred_xy / sd


def distances_for_grid(ref_grid: Path, pasto, args) -> tuple[float, int, int, bool]:
    with rasterio.open(ref_grid) as src:
        arr, transform = reproject_raster_to_crs(src, pasto.crs)
    pred_xy = valid_points(arr, transform, args.validity_mode)
    cr = rasterize_pasto(arr.shape, transform, pasto, args.all_touched)
    cp_xy = pasto_points(cr, transform)
    if len(cp_xy) == 0 or len(pred_xy) == 0:
        return PASTO_SENTINEL, len(pred_xy), len(cp_xy), True
    if len(cp_xy) == 1:
        cp_xy = np.vstack([cp_xy, cp_xy])
    train_xy, query_xy = cp_xy, pred_xy
    if args.distance_mode == "kknn_scaled":
        train_xy, query_xy = scale_train_test(train_xy, query_xy)
    tree = cKDTree(train_xy)
    dist, _ = tree.query(query_xy, k=1)
    return float(np.nanmax(dist)), len(pred_xy), len(cp_xy), False


def main() -> None:
    args = parse_args()
    pasto_path = Path(args.pasto_marino_shp)
    ref_grids = [Path(p) for p in args.ref_grids]
    missing = [str(p) for p in [pasto_path, *ref_grids] if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))
    pasto = load_pasto_marino(pasto_path)

    rows = []
    maxima = []
    for rg in ref_grids:
        raw_max, n_valid, n_pasto, used_sentinel = distances_for_grid(rg, pasto, args)
        maxima.append(raw_max)
        region_id = rg.parent.name
        rows.append(
            {
                "regionid": region_id,
                "ref_grid": str(rg),
                "region_raw_max": raw_max,
                "valid_points": n_valid,
                "pasto_points": n_pasto,
                "used_sentinel": used_sentinel,
            }
        )
        log(f"{region_id}: max={raw_max}, valid={n_valid}, pasto_points={n_pasto}, sentinel={used_sentinel}", args.verbose)

    global_raw_max = float(np.nanmax(maxima)) if maxima else np.nan
    global_fill_value = 1.5 * global_raw_max
    out = pd.DataFrame(
        [{
            "global_raw_max": global_raw_max,
            "global_fill_value": global_fill_value,
            "distance_mode": args.distance_mode,
            "validity_mode": args.validity_mode,
            "all_touched": bool(args.all_touched),
            "n_regions": len(ref_grids),
            "n_regions_with_no_pastos": int(sum(r["used_sentinel"] for r in rows)),
        }]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    detail = output.with_name(output.stem + "_by_region.csv")
    pd.DataFrame(rows).to_csv(detail, index=False)
    print(f"OK -> {output}")


if __name__ == "__main__":
    main()
