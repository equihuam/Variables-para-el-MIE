#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug/canonical-compatible script for coral distance feature.

Replicates the R workflow in 4_corales_global.R as closely as practical:
- reproject regional ref_grid to coral CRS with nearest-neighbor resampling
- rasterize coral geometries on the regional grid
- convert rasterized coral cells to point centers
- compute k=1 nearest-neighbor distances using kknn-like scaled coordinates
- optionally replace sentinel 999 using a global fill value

Output columns: regionid, pixid, x, y, d_corales
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree

CORALS_SENTINEL = 999.0
OUTPUT_FIELD = "d_corales"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calcula distancia a corales por pixel para una region.")
    p.add_argument("--corals-shp", required=True, help="Ruta al shapefile global de corales.")
    p.add_argument("--ref-grid", required=True, help="Ruta al ref_grid.tif de la region.")
    p.add_argument("--region-id", required=True, help="Identificador de region, por ejemplo region_1.")
    p.add_argument("--output", required=True, help="Ruta de salida .parquet.")
    p.add_argument(
        "--corals-global-stats",
        default=None,
        help="CSV con global_raw_max/global_fill_value para reemplazo global del sentinel 999.",
    )
    p.add_argument(
        "--sentinel-mode",
        choices=["none", "local", "global"],
        default="global",
        help="Modo de reemplazo del sentinel 999. Para equivalencia R completa usar global.",
    )
    p.add_argument(
        "--distance-mode",
        choices=["kknn_scaled", "raw"],
        default="kknn_scaled",
        help="kknn_scaled emula kknn(scale=TRUE); raw usa distancia euclidiana cruda.",
    )
    p.add_argument(
        "--validity-mode",
        choices=["finite", "notnan"],
        default="finite",
        help="Criterio para pixeles validos del ref_grid reproyectado.",
    )
    p.add_argument(
        "--all-touched",
        action="store_true",
        help="Usar all_touched=True en rasterize; por defecto False.",
    )
    p.add_argument("--debug-grid-output", default=None, help="CSV debug de grilla.")
    p.add_argument("--debug-metadata-output", default=None, help="CSV debug de metadatos.")
    p.add_argument("--debug-coral-points-output", default=None, help="CSV debug de puntos coral rasterizados.")
    p.add_argument("--verbose", action="store_true", help="Imprimir diagnosticos detallados.")
    return p.parse_args()


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if p is not None and not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def load_corals(path: Path) -> gpd.GeoDataFrame:
    corales = gpd.read_file(path)
    if corales.empty:
        raise ValueError(f"El shapefile de corales esta vacio: {path}")
    if corales.crs is None:
        raise ValueError(f"El shapefile de corales no tiene CRS: {path}")
    corales = corales[corales.geometry.notna() & ~corales.geometry.is_empty].copy()
    if corales.empty:
        raise ValueError("No quedaron geometrías válidas de corales.")
    return corales


def reproject_raster_to_crs(src: rasterio.io.DatasetReader, dst_crs) -> tuple[np.ndarray, rasterio.Affine, dict]:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
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
    meta = {
        "src_crs": str(src.crs),
        "dst_crs": str(dst_crs),
        "src_width": src.width,
        "src_height": src.height,
        "dst_width": width,
        "dst_height": height,
        "src_nodata": src.nodata,
        "src_total_points": int(src.width * src.height),
        "dst_total_points": int(width * height),
    }
    data = src.read(1, masked=True)
    meta["src_valid_masked"] = int(np.ma.count(data))
    return dst, transform, meta


def valid_raster_points_dataframe(arr: np.ndarray, transform, validity_mode: str = "finite") -> pd.DataFrame:
    if validity_mode == "finite":
        mask = np.isfinite(arr)
    else:
        mask = ~np.isnan(arr)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", "ref_value"])
    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "row": rows.astype(np.int64),
            "col": cols.astype(np.int64),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "ref_value": arr[rows, cols],
        }
    )


def rasterize_corals_on_region(shape, transform, corales: gpd.GeoDataFrame, all_touched: bool = False) -> np.ndarray:
    shapes = ((geom, 1.0) for geom in corales.geometry if geom is not None and not geom.is_empty)
    return rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
        all_touched=all_touched,
    )


def coral_points_from_raster(corales_rast: np.ndarray, transform) -> pd.DataFrame:
    rows, cols = np.where(np.isfinite(corales_rast))
    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", "part"])
    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "row": rows.astype(np.int64),
            "col": cols.astype(np.int64),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "part": 1,
        }
    )


def scale_train_test(train_xy: np.ndarray, pred_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    sd = train_xy.std(axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd != 0), sd, 1.0)
    return train_xy / sd, pred_xy / sd, (float(sd[0]), float(sd[1]))


def nearest_distance(points_xy: np.ndarray, coral_points: pd.DataFrame, distance_mode: str) -> tuple[np.ndarray, tuple[float, float]]:
    if coral_points.empty:
        return np.full(points_xy.shape[0], CORALS_SENTINEL, dtype=float), (np.nan, np.nan)

    cp = coral_points.copy()
    if len(cp) == 1:
        cp = pd.concat([cp, cp], ignore_index=True)

    train_xy = cp[["x", "y"]].to_numpy(dtype=float)
    pred_xy = points_xy.astype(float)

    if distance_mode == "kknn_scaled":
        train_xy, pred_xy, sd = scale_train_test(train_xy, pred_xy)
    else:
        sd = (1.0, 1.0)

    tree = cKDTree(train_xy)
    distances, _ = tree.query(pred_xy, k=1)
    return distances.astype(float), sd


def read_global_fill_value(path: Path) -> float:
    stats = pd.read_csv(path)
    if "global_fill_value" in stats.columns:
        return float(stats["global_fill_value"].iloc[0])
    if "global_raw_max" in stats.columns:
        return 1.5 * float(stats["global_raw_max"].iloc[0])
    raise ValueError("El CSV de estadisticas globales requiere global_fill_value o global_raw_max.")


def finalize_corals(distances: np.ndarray, mode: str, global_stats: Path | None) -> np.ndarray:
    out = distances.copy()
    sentinel_mask = out == CORALS_SENTINEL
    if not np.any(sentinel_mask) or mode == "none":
        return out
    if mode == "local":
        fill = 1.5 * float(np.max(out))
    elif mode == "global":
        if global_stats is None:
            raise ValueError("--sentinel-mode global requiere --corals-global-stats")
        fill = read_global_fill_value(global_stats)
    else:
        raise ValueError(f"Modo sentinel no reconocido: {mode}")
    out[sentinel_mask] = fill
    return out


def write_debug_outputs(
    region_id: str,
    region_points: pd.DataFrame,
    coral_points: pd.DataFrame,
    metadata: dict,
    distances: np.ndarray,
    args: argparse.Namespace,
    sd: tuple[float, float],
) -> None:
    if args.debug_grid_output:
        p = Path(args.debug_grid_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        dbg = region_points.copy()
        dbg.insert(0, "regionid", region_id)
        dbg.insert(1, "pixid", np.arange(1, len(dbg) + 1))
        dbg.to_csv(p, index=False)
        log(f"debug grid -> {p}", args.verbose)

    if args.debug_coral_points_output:
        p = Path(args.debug_coral_points_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        dbg = coral_points.copy()
        dbg.insert(0, "regionid", region_id)
        dbg.to_csv(p, index=False)
        log(f"debug coral points -> {p}", args.verbose)

    if args.debug_metadata_output:
        p = Path(args.debug_metadata_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(metadata)
        meta.update(
            {
                "regionid": region_id,
                "validity_mode": args.validity_mode,
                "distance_mode": args.distance_mode,
                "sentinel_mode": args.sentinel_mode,
                "all_touched": bool(args.all_touched),
                "valid_points": int(len(region_points)),
                "coral_points": int(len(coral_points)),
                "distance_raw_min": float(np.nanmin(distances)) if len(distances) else np.nan,
                "distance_raw_max": float(np.nanmax(distances)) if len(distances) else np.nan,
                "sd_x": sd[0],
                "sd_y": sd[1],
            }
        )
        pd.DataFrame([meta]).to_csv(p, index=False)
        log(f"debug metadata -> {p}", args.verbose)


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {output_path.suffix}")
    df.to_parquet(output_path, index=False, engine="pyarrow")


def main() -> None:
    args = parse_args()
    corals_path = Path(args.corals_shp)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    global_stats_path = Path(args.corals_global_stats) if args.corals_global_stats else None
    validate_inputs(corals_path, ref_grid_path, global_stats_path)

    region_id = str(args.region_id).strip()
    corales = load_corals(corals_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
        region_arr, region_transform, metadata = reproject_raster_to_crs(src, corales.crs)

    region_points = valid_raster_points_dataframe(region_arr, region_transform, args.validity_mode)
    log(f"total puntos GeoTIFF original: {metadata['src_total_points']}", args.verbose)
    log(f"puntos válidos GeoTIFF original masked: {metadata['src_valid_masked']}", args.verbose)
    log(f"total puntos reproyectados: {metadata['dst_total_points']}", args.verbose)
    log(f"puntos válidos usados en malla: {len(region_points)}", args.verbose)

    corales_rast = rasterize_corals_on_region(
        shape=region_arr.shape,
        transform=region_transform,
        corales=corales,
        all_touched=args.all_touched,
    )
    coral_points = coral_points_from_raster(corales_rast, region_transform)

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)
    raw_distances, sd = nearest_distance(pred_xy, coral_points, args.distance_mode)
    final_distances = finalize_corals(raw_distances, args.sentinel_mode, global_stats_path)

    log(f"modo distancia: {args.distance_mode}", args.verbose)
    log(f"modo sentinel: {args.sentinel_mode}", args.verbose)
    log(f"coral points rasterizados: {len(coral_points)}", args.verbose)
    log(f"sd coral points: ({sd[0]}, {sd[1]})", args.verbose)

    write_debug_outputs(region_id, region_points, coral_points, metadata, raw_distances, args, sd)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            OUTPUT_FIELD: final_distances,
        }
    )
    save_output(out, output_path)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
