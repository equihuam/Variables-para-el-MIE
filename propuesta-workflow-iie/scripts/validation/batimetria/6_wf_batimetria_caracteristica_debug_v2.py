#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug v2 - batimetría característica.

Replica el flujo R de 6_batimetria_caracteristica.R para una región:
  - proyecta ref_grid al CRS de batimetría con nearest
  - recorta batimetría al bbox de la región reproyectada
  - convierte batimetría recortada a puntos de entrenamiento
  - predice con emulación de kknn(bat ~ x + y, k=7, distance=2,
    kernel='optimal', scale=TRUE)

Salida principal:
  regionid, pixid, x, y, bati_char

Salidas debug opcionales:
  grid, metadata, bat_points, knn_weights
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from sklearn.neighbors import NearestNeighbors

OUTPUT_FIELD = "bati_char"
K_NEIGHBORS = 7
DISTANCE_POWER = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estima batimetría característica por píxel para una región específica."
    )
    p.add_argument("--batimetria-raster", required=True)
    p.add_argument("--ref-grid", required=True)
    p.add_argument("--region-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--validity-mode",
        choices=["finite", "notna"],
        default="finite",
        help="Criterio para extraer puntos válidos del ref_grid reproyectado.",
    )
    p.add_argument(
        "--knn-mode",
        choices=["kknn_optimal", "idw_legacy"],
        default="kknn_optimal",
        help="Modo de predicción. kknn_optimal emula kknn kernel='optimal'.",
    )
    p.add_argument("--debug-grid-output", default=None)
    p.add_argument("--debug-metadata-output", default=None)
    p.add_argument("--debug-bat-points-output", default=None)
    p.add_argument("--debug-weights-output", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".csv":
        df.to_csv(path, index=False)
    elif suf == ".parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        raise ValueError(f"Extensión no soportada para salida tabular: {path}")


def reproject_raster_to_crs(src: rasterio.io.DatasetReader, dst_crs):
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
    return dst, transform


def array_bounds(arr: np.ndarray, transform) -> tuple[float, float, float, float]:
    height, width = arr.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def crop_bathymetry_to_region(bat_src: rasterio.io.DatasetReader, region_arr, region_transform):
    """
    Equivalente funcional a terra::crop(bat, region_).

    Importante: el raster GEBCO puede ser entero (por ejemplo int16), por lo
    que rasterio.mask(..., filled=True, nodata=np.nan) falla al intentar
    rellenar un arreglo entero con NaN. Por eso recortamos como masked array y
    después convertimos a float, aplicando NaN sólo fuera del dominio válido.
    """
    left, bottom, right, top = array_bounds(region_arr, region_transform)
    geom = box(left, bottom, right, top)
    cropped, cropped_transform = mask(
        bat_src,
        [mapping(geom)],
        crop=True,
        filled=False,
    )

    band = cropped[0]
    arr = band.astype("float64").filled(np.nan)

    nodata = bat_src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr[arr == float(nodata)] = np.nan

    return arr, cropped_transform


def valid_raster_points_dataframe(arr: np.ndarray, transform, validity_mode: str = "finite") -> pd.DataFrame:
    if validity_mode == "finite":
        valid_mask = np.isfinite(arr)
    elif validity_mode == "notna":
        valid_mask = ~np.isnan(arr)
    else:
        raise ValueError(f"validity_mode no soportado: {validity_mode}")

    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", "value"])
    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "row": rows.astype(int),
            "col": cols.astype(int),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "value": arr[rows, cols],
        }
    )


def kknn_optimal_rank_weights(k: int) -> np.ndarray:
    """
    Pesos por rango usados para emular kernel='optimal' validado en erosión.
    Para k=3 produce [5,3,1]/9; para k=7 produce [13,11,9,7,5,3,1]/49.
    """
    ranks = np.arange(1, k + 1, dtype=float)
    weights = 2.0 * (k - ranks) + 1.0
    return weights / weights.sum()


def scale_xy_like_kknn(train_xy: np.ndarray, pred_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # R sd(): ddof=1. kknn(scale=TRUE) escala con desviación estándar del training set.
    sd = np.nanstd(train_xy, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return train_xy / sd, pred_xy / sd, sd


def predict_kknn_optimal(train_df: pd.DataFrame, pred_df: pd.DataFrame, k: int = K_NEIGHBORS):
    train_valid = train_df[np.isfinite(train_df["bat"])].copy()
    if train_valid.empty or pred_df.empty:
        return np.full(len(pred_df), np.nan, dtype=float), np.array([]), np.array([np.nan, np.nan])

    train_xy = train_valid[["x", "y"]].to_numpy(dtype=float)
    pred_xy = pred_df[["x", "y"]].to_numpy(dtype=float)
    train_y = train_valid["bat"].to_numpy(dtype=float)

    k_eff = min(k, len(train_valid))
    train_scaled, pred_scaled, sd = scale_xy_like_kknn(train_xy, pred_xy)

    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", metric="euclidean")
    nn.fit(train_scaled)
    distances, indices = nn.kneighbors(pred_scaled, return_distance=True)

    weights = kknn_optimal_rank_weights(k_eff)
    neighbor_values = train_y[indices]

    zero_mask = distances == 0
    any_zero = zero_mask.any(axis=1)

    pred = np.empty(len(pred_df), dtype=float)
    if np.any(any_zero):
        # Si hay coincidencia exacta, kknn asigna efectivamente el valor del/los vecino(s) a distancia cero.
        zvals = neighbor_values[any_zero]
        zmask = zero_mask[any_zero]
        pred[any_zero] = np.sum(zvals * zmask, axis=1) / np.sum(zmask, axis=1)

    nonzero_rows = ~any_zero
    if np.any(nonzero_rows):
        pred[nonzero_rows] = np.sum(neighbor_values[nonzero_rows] * weights, axis=1)

    return pred, weights, sd


def predict_idw_legacy(train_df: pd.DataFrame, pred_df: pd.DataFrame, k: int = K_NEIGHBORS, p: float = DISTANCE_POWER):
    train_valid = train_df[np.isfinite(train_df["bat"])].copy()
    if train_valid.empty or pred_df.empty:
        return np.full(len(pred_df), np.nan, dtype=float), np.array([]), np.array([np.nan, np.nan])
    train_xy = train_valid[["x", "y"]].to_numpy(dtype=float)
    pred_xy = pred_df[["x", "y"]].to_numpy(dtype=float)
    train_y = train_valid["bat"].to_numpy(dtype=float)
    k_eff = min(k, len(train_valid))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", metric="euclidean")
    nn.fit(train_xy)
    distances, indices = nn.kneighbors(pred_xy, return_distance=True)
    zero_mask = distances == 0
    weights = np.zeros_like(distances, dtype=float)
    weights[~zero_mask] = 1.0 / np.power(distances[~zero_mask], p)
    any_zero = zero_mask.any(axis=1)
    if np.any(any_zero):
        weights[any_zero] = zero_mask[any_zero].astype(float)
    sums = weights.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    pred = np.sum(weights * train_y[indices], axis=1) / sums[:, 0]
    return pred, np.array([]), np.array([np.nan, np.nan])


def build_metadata(region_id: str, ref_src, bat_src, region_arr, region_transform, region_points, bat_arr, bat_points, args, weights, sd):
    left, bottom, right, top = array_bounds(region_arr, region_transform)
    bat_left, bat_bottom, bat_right, bat_top = array_bounds(bat_arr, bat_points.attrs.get("transform", region_transform)) if False else (np.nan, np.nan, np.nan, np.nan)
    meta = {
        "regionid": region_id,
        "ref_crs": str(ref_src.crs),
        "bat_crs": str(bat_src.crs),
        "ref_width": ref_src.width,
        "ref_height": ref_src.height,
        "ref_total_points": int(ref_src.width * ref_src.height),
        "ref_valid_masked": int(np.ma.masked_invalid(ref_src.read(1, masked=True)).count()),
        "region_reprojected_width": int(region_arr.shape[1]),
        "region_reprojected_height": int(region_arr.shape[0]),
        "region_total_points": int(region_arr.size),
        "region_valid_points": int(len(region_points)),
        "region_x_min": float(region_points["x"].min()) if len(region_points) else np.nan,
        "region_x_max": float(region_points["x"].max()) if len(region_points) else np.nan,
        "region_y_min": float(region_points["y"].min()) if len(region_points) else np.nan,
        "region_y_max": float(region_points["y"].max()) if len(region_points) else np.nan,
        "region_bounds_left": float(left),
        "region_bounds_bottom": float(bottom),
        "region_bounds_right": float(right),
        "region_bounds_top": float(top),
        "bat_crop_width": int(bat_arr.shape[1]),
        "bat_crop_height": int(bat_arr.shape[0]),
        "bat_crop_total_points": int(bat_arr.size),
        "bat_train_points": int(len(bat_points)),
        "knn_mode": args.knn_mode,
        "k": K_NEIGHBORS,
        "kernel": "optimal" if args.knn_mode == "kknn_optimal" else "idw_legacy",
        "scale": args.knn_mode == "kknn_optimal",
        "sd_x": float(sd[0]) if len(sd) == 2 else np.nan,
        "sd_y": float(sd[1]) if len(sd) == 2 else np.nan,
        "weights": ";".join([repr(float(w)) for w in weights]) if len(weights) else "",
    }
    return pd.DataFrame([meta])


def main() -> None:
    args = parse_args()
    bat_path = Path(args.batimetria_raster)
    ref_path = Path(args.ref_grid)
    out_path = Path(args.output)
    region_id = str(args.region_id).strip()
    validate_inputs(bat_path, ref_path)

    with rasterio.open(bat_path) as bat_src:
        if bat_src.crs is None:
            raise ValueError("El raster batimétrico no tiene CRS definido.")
        with rasterio.open(ref_path) as ref_src:
            if ref_src.crs is None:
                raise ValueError(f"El raster de referencia no tiene CRS: {ref_path}")
            region_arr, region_transform = reproject_raster_to_crs(ref_src, bat_src.crs)
            region_points = valid_raster_points_dataframe(region_arr, region_transform, args.validity_mode)
            bat_arr, bat_transform = crop_bathymetry_to_region(bat_src, region_arr, region_transform)

            bat_points = valid_raster_points_dataframe(bat_arr, bat_transform, "finite").rename(columns={"value": "bat"})

            if args.knn_mode == "kknn_optimal":
                predictions, weights, sd = predict_kknn_optimal(bat_points, region_points, K_NEIGHBORS)
            else:
                predictions, weights, sd = predict_idw_legacy(bat_points, region_points, K_NEIGHBORS, DISTANCE_POWER)

            log(f"total puntos GeoTIFF original: {ref_src.width * ref_src.height}", args.verbose)
            log(f"puntos válidos GeoTIFF original masked: {int(np.ma.masked_invalid(ref_src.read(1, masked=True)).count())}", args.verbose)
            log(f"total puntos reproyectados: {int(region_arr.size)}", args.verbose)
            log(f"puntos válidos usados en malla: {len(region_points)}", args.verbose)
            log(f"batimetría crop puntos entrenamiento: {len(bat_points)}", args.verbose)
            log(f"modo kNN: {args.knn_mode}", args.verbose)
            if len(weights):
                log(f"pesos kernel optimal k={len(weights)},d=2: {[float(w) for w in weights]}", args.verbose)
                log(f"sd entrenamiento: ({sd[0]}, {sd[1]})", args.verbose)

            if args.debug_grid_output:
                dbg_grid = region_points[["row", "col", "x", "y", "value"]].copy()
                dbg_grid.insert(0, "regionid", region_id)
                dbg_grid.insert(1, "pixid", np.arange(1, len(dbg_grid) + 1))
                save_table(dbg_grid, Path(args.debug_grid_output))
                log(f"debug grid -> {args.debug_grid_output}", args.verbose)

            if args.debug_bat_points_output:
                dbg_bat = bat_points[["row", "col", "x", "y", "bat"]].copy()
                save_table(dbg_bat, Path(args.debug_bat_points_output))
                log(f"debug bat points -> {args.debug_bat_points_output}", args.verbose)

            if args.debug_weights_output:
                save_table(pd.DataFrame({"rank": np.arange(1, len(weights) + 1), "weight": weights}), Path(args.debug_weights_output))
                log(f"debug weights -> {args.debug_weights_output}", args.verbose)

            if args.debug_metadata_output:
                meta = build_metadata(region_id, ref_src, bat_src, region_arr, region_transform, region_points, bat_arr, bat_points, args, weights, sd)
                save_table(meta, Path(args.debug_metadata_output))
                log(f"debug metadata -> {args.debug_metadata_output}", args.verbose)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(dtype=float),
            "y": region_points["y"].to_numpy(dtype=float),
            OUTPUT_FIELD: predictions,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {out_path.suffix}")
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"OK -> {out_path}")


if __name__ == "__main__":
    main()
