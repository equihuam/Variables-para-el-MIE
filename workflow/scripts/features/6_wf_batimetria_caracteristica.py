#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 6_wf_batimetria_caracteristica.py

Propósito:
    Estimar, para una región específica, la batimetría característica en cada
    píxel válido del raster de referencia a partir de un raster batimétrico base
    y exportar el resultado como tabla congruente por píxel en formato Parquet.

Equivalencia validada con R:
    - terra::project(region_, y = crs(bat), method = "near")
    - terra::crop(bat, region_)
    - kknn(bat ~ x + y, bat_points, region_points,
           distance = 2, k = 7, kernel = "optimal", scale = TRUE)

Salida principal:
    regionid, pixid, x, y, bati_char
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(
        description="Estima batimetría característica por píxel para una región específica."
    )
    parser.add_argument(
        "--batimetria-raster",
        required=True,
        help="Ruta al raster batimétrico base.",
    )
    parser.add_argument(
        "--ref-grid",
        required=True,
        help="Ruta al ref_grid.tif de la región.",
    )
    parser.add_argument(
        "--region-id",
        required=True,
        help="Identificador de la región, por ejemplo region_1.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet.",
    )
    parser.add_argument(
        "--validity-mode",
        choices=["finite", "notna"],
        default="finite",
        help="Criterio para extraer puntos válidos del ref_grid reproyectado.",
    )
    parser.add_argument(
        "--knn-mode",
        choices=["kknn_optimal", "idw_legacy"],
        default="kknn_optimal",
        help="Modo de predicción. kknn_optimal emula kknn kernel='optimal'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime diagnósticos de grilla, entrenamiento y kNN.",
    )
    return parser.parse_args()


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def reproject_raster_to_crs(src: rasterio.io.DatasetReader, dst_crs):
    transform, width, height = calculate_default_transform(
        src.crs,
        dst_crs,
        src.width,
        src.height,
        *src.bounds,
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


def crop_bathymetry_to_region(
    bat_src: rasterio.io.DatasetReader,
    region_arr: np.ndarray,
    region_transform,
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Equivalente funcional a terra::crop(bat, region_).

    El raster GEBCO puede ser entero, por ejemplo int16. Por eso no se rellena
    directamente con NaN durante rasterio.mask(); primero se recorta como masked
    array y después se convierte a float con NaN fuera del dominio válido.
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


def valid_raster_points_dataframe(
    arr: np.ndarray,
    transform,
    validity_mode: str = "finite",
) -> pd.DataFrame:
    if validity_mode == "finite":
        valid_mask = np.isfinite(arr)
    elif validity_mode == "notna":
        valid_mask = ~np.isnan(arr)
    else:
        raise ValueError(f"validity_mode no soportado: {validity_mode}")

    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", "value"])

    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "value": arr[rows, cols],
        }
    )


def kknn_optimal_rank_weights(k: int) -> np.ndarray:
    """
    Pesos por rango validados para emular kernel='optimal'.
    Para k=7 produce [13, 11, 9, 7, 5, 3, 1] / 49.
    """
    ranks = np.arange(1, k + 1, dtype=float)
    weights = 2.0 * (k - ranks) + 1.0
    return weights / weights.sum()


def scale_xy_like_kknn(
    train_xy: np.ndarray,
    pred_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # R::sd usa ddof=1; kknn(scale=TRUE) escala usando el conjunto de entrenamiento.
    sd = np.nanstd(train_xy, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return train_xy / sd, pred_xy / sd, sd


def predict_kknn_optimal(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    k: int = K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    predictions = np.empty(len(pred_df), dtype=float)

    if np.any(any_zero):
        zvals = neighbor_values[any_zero]
        zmask = zero_mask[any_zero]
        predictions[any_zero] = np.sum(zvals * zmask, axis=1) / np.sum(zmask, axis=1)

    nonzero_rows = ~any_zero
    if np.any(nonzero_rows):
        predictions[nonzero_rows] = np.sum(neighbor_values[nonzero_rows] * weights, axis=1)

    return predictions, weights, sd


def predict_idw_legacy(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    k: int = K_NEIGHBORS,
    p: float = DISTANCE_POWER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0

    predictions = np.sum(weights * train_y[indices], axis=1) / weight_sums[:, 0]
    return predictions, np.array([]), np.array([np.nan, np.nan])


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {output_path.suffix}")

    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()

    batimetria_path = Path(args.batimetria_raster)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(batimetria_path, ref_grid_path)

    with rasterio.open(batimetria_path) as bat_src:
        if bat_src.crs is None:
            raise ValueError("El raster batimétrico no tiene CRS definido.")

        with rasterio.open(ref_grid_path) as ref_src:
            if ref_src.crs is None:
                raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

            region_arr, region_transform = reproject_raster_to_crs(ref_src, bat_src.crs)
            region_points = valid_raster_points_dataframe(
                region_arr,
                region_transform,
                validity_mode=args.validity_mode,
            )

            bat_arr, bat_transform = crop_bathymetry_to_region(
                bat_src,
                region_arr,
                region_transform,
            )

    bat_points = valid_raster_points_dataframe(
        bat_arr,
        bat_transform,
        validity_mode="finite",
    ).rename(columns={"value": "bat"})

    if args.knn_mode == "kknn_optimal":
        predictions, weights, sd = predict_kknn_optimal(bat_points, region_points, K_NEIGHBORS)
    elif args.knn_mode == "idw_legacy":
        predictions, weights, sd = predict_idw_legacy(bat_points, region_points, K_NEIGHBORS, DISTANCE_POWER)
    else:
        raise ValueError(f"knn_mode no soportado: {args.knn_mode}")

    log(f"total puntos reproyectados: {int(region_arr.size)}", args.verbose)
    log(f"puntos válidos usados en malla: {len(region_points)}", args.verbose)
    log(f"batimetría crop puntos entrenamiento: {len(bat_points)}", args.verbose)
    log(f"modo kNN: {args.knn_mode}", args.verbose)
    if len(weights):
        log(f"pesos kernel optimal k={len(weights)},d=2: {[float(w) for w in weights]}", args.verbose)
        log(f"sd entrenamiento: ({sd[0]}, {sd[1]})", args.verbose)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(dtype=float),
            "y": region_points["y"].to_numpy(dtype=float),
            OUTPUT_FIELD: predictions,
        }
    )

    save_output(out, output_path)


if __name__ == "__main__":
    main()
