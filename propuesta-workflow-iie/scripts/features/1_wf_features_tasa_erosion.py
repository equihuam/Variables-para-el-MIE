#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 1_wf_features_tasa_erosion.py

Propósito:
    Calcular, para una región específica, la variable de tasa de erosión sobre
    los píxeles válidos de un raster de referencia y exportarla como tabla
    tabular congruente por píxel.

Origen:
    Refactorización para workflow de la traducción inicial a Python del script R:
    1_features_tasa_erosion.R

Resumen del flujo:
    1. Leer la tabla de tasas de erosión.
    2. Leer un raster de referencia regional.
    3. Reproyectar la plantilla regional al CRS de los puntos de erosión.
    4. Extraer los centros de píxel como tabla con coordenadas x, y.
    5. Filtrar solo las celdas válidas de la malla.
    6. Estimar la erosión en cada píxel válido mediante vecinos cercanos.
    7. Exportar la tabla resultante en Parquet.

Insumos principales:
    - tabla de tasas de erosión
    - ref_grid.tif regional

Salidas principales:
    - tabla serializada con columnas:
      regionid, pixid, x, y, erosion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.neighbors import NearestNeighbors


POINTS_CRS = "EPSG:4326"
TARGET_FIELD = "Tasa"
K_NEIGHBORS = 3
DISTANCE_POWER = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula tasa de erosión por píxel para una región específica."
    )
    parser.add_argument(
        "--erosion-table",
        required=True,
        help="Ruta a la tabla de tasas de erosión.",
    )
    parser.add_argument(
        "--ref-grid",
        required=True,
        help="Ruta al ref_grid.tif de la región.",
    )
    parser.add_argument(
        "--region-id",
        required=True,
        help="Identificador de la región, por ejemplo region_7.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet.",
    )
    return parser.parse_args()


def validate_inputs(erosion_table: Path, ref_grid: Path) -> None:
    missing = [str(p) for p in [erosion_table, ref_grid] if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_tasa_erosion(path: Path) -> pd.DataFrame:
    tasa_ero = pd.read_csv(path, sep=",", header=0, low_memory=False)

    if tasa_ero.shape[1] < 3:
        raise ValueError("La tabla de tasa de erosión no tiene al menos 3 columnas.")

    cols = list(tasa_ero.columns)
    cols[1] = "x"
    cols[2] = "y"
    tasa_ero.columns = cols

    required = {"x", "y", TARGET_FIELD}
    missing = required - set(tasa_ero.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en tasa_erosion: {missing}")

    tasa_ero["x"] = pd.to_numeric(tasa_ero["x"], errors="raise")
    tasa_ero["y"] = pd.to_numeric(tasa_ero["y"], errors="raise")
    tasa_ero[TARGET_FIELD] = pd.to_numeric(tasa_ero[TARGET_FIELD], errors="raise")

    return tasa_ero


def reproject_raster_to_epsg4326(
        src: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs,
        POINTS_CRS,
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
        dst_crs=POINTS_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )

    return dst, transform


def raster_points_dataframe(arr: np.ndarray, transform) -> pd.DataFrame:
    """
    Aproximación a as.data.frame(region_, xy = TRUE) de terra.
    Construye una fila por celda del raster reproyectado.
    """
    height, width = arr.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "x": np.asarray(xs).ravel(),
            "y": np.asarray(ys).ravel(),
            "value": arr.ravel(),
        }
    )


def filter_valid_region_points(
        region_points: pd.DataFrame,
        nodata_value: float | None = None,
) -> pd.DataFrame:
    values = pd.to_numeric(region_points["value"], errors="coerce").to_numpy()

    valid_mask = np.isfinite(values)
    if nodata_value is not None:
        valid_mask &= values != nodata_value

    out = region_points.loc[valid_mask].reset_index(drop=True).copy()
    return out


def predict_knn_weighted(
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        target_col: str = TARGET_FIELD,
        k: int = K_NEIGHBORS,
        distance_power: float = DISTANCE_POWER,
) -> np.ndarray:
    x_train = train_df[[x_col, y_col]].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    x_pred = pred_df[[x_col, y_col]].to_numpy(dtype=float)

    if len(train_df) == 0:
        return np.full(len(pred_df), np.nan, dtype=float)

    k_eff = min(k, len(train_df))

    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", metric="euclidean")
    nn.fit(x_train)

    distances, indices = nn.kneighbors(x_pred, return_distance=True)

    zero_mask = distances == 0
    weights = np.zeros_like(distances, dtype=float)
    weights[~zero_mask] = 1.0 / np.power(distances[~zero_mask], distance_power)

    any_zero = zero_mask.any(axis=1)
    if np.any(any_zero):
        weights[any_zero] = zero_mask[any_zero].astype(float)

    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0

    neighbor_values = y_train[indices]
    predictions = np.sum(weights * neighbor_values, axis=1) / weight_sums[:, 0]

    return predictions


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Este script espera salida .parquet y recibió: {output_path.suffix}"
        )

    df.to_parquet(output_path, index=False, engine="pyarrow")


def main() -> None:
    args = parse_args()

    erosion_table_path = Path(args.erosion_table)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(erosion_table_path, ref_grid_path)

    tasa_ero = load_tasa_erosion(erosion_table_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_epsg4326(src)

    region_points_all = raster_points_dataframe(region_arr, region_transform)
    region_points = filter_valid_region_points(region_points_all, nodata_value=None)

    print(f"total puntos reproyectados: {len(region_points_all)}")
    print(f"puntos válidos en malla: {len(region_points)}")

    predictions = predict_knn_weighted(
        train_df=tasa_ero,
        pred_df=region_points,
        x_col="x",
        y_col="y",
        target_col=TARGET_FIELD,
        k=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
    )

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            "erosion": predictions,
        }
    )

    save_output(out, output_path)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()