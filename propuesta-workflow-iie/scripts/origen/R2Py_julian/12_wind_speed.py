#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 12_wind_speed.py

Propósito:
    Estimar la velocidad del viento para los píxeles de cada raster regional
    mediante interpolación basada en vecinos cercanos a partir de un raster
    temático de velocidad del viento.

Origen:
    Traducción inicial a Python del script R:
    12_wind_speed.R

Resumen del flujo:
    1. Leer el raster base de velocidad del viento.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster regional al CRS del raster de viento.
    4. Recortar el raster de viento a la extensión del raster regional reproyectado.
    5. Extraer los centros de píxel como tablas con coordenadas x, y.
    6. Estimar la velocidad del viento en cada píxel mediante vecinos cercanos.
    7. Concatenar resultados regionales y serializar el resultado en PKL.

Insumos principales:
    - wind_speed.tif
    - colección regional de ref_grid.tif

Salidas principales:
    - 12_wind_speed.pkl

Supuestos y notas:
    - La interpolación se realiza en el CRS del raster de velocidad del viento.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - La aproximación inicial usa vecinos cercanos ponderados por distancia^-2.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, el kernel "optimal" de kknn no se replica exactamente; se usa
    una aproximación vectorizada basada en k vecinos cercanos con pesos por
    distancia^-2, elegida como alternativa razonable para esta fase inicial.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from sklearn.neighbors import NearestNeighbors


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
WIND_RASTER = DROPBOX_DIR / "data_crude" / "11_wind_speed" / "wind_speed.tif"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "12_wind_speed.pkl"

K_NEIGHBORS = 7
DISTANCE_POWER = 2


def load_wind(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el raster de velocidad del viento: {path}")
    return rasterio.open(path)


def list_reference_grids(ref_grid_dir: Path) -> list[Path]:
    c_list = sorted(ref_grid_dir.rglob("*.tif"))
    if not c_list:
        raise FileNotFoundError(f"No se encontraron .tif en {ref_grid_dir}")
    return c_list


def extract_region_id(path: Path) -> str:
    return path.parent.name


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


def crop_raster_to_region(src: rasterio.io.DatasetReader, region_arr: np.ndarray, region_transform):
    """
    Equivalente funcional a:
      wind_ <- crop(wind, region_)
    """
    height, width = region_arr.shape
    left, top = region_transform * (0, 0)
    right, bottom = region_transform * (width, height)

    geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
    cropped, cropped_transform = mask(src, [mapping(geom)], crop=True, filled=True)

    return cropped[0], cropped_transform


def raster_points_dataframe(arr: np.ndarray, transform) -> pd.DataFrame:
    """
    Aproximación a as.data.frame(rast, xy = TRUE) de terra.
    Se conservan todas las celdas como filas.
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


def predict_knn_weighted(
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        target_col: str = "wind_speed",
        k: int = K_NEIGHBORS,
        distance_power: float = DISTANCE_POWER,
) -> np.ndarray:
    """
    Traducción inicial de:
      kknn(wind_speed ~ x + y, wind_points, region_points, distance = 2, k = 7, kernel = "optimal")

    El kernel "optimal" de kknn no se replica exactamente aquí.
    Se usa promedio ponderado por distancia^-2 como aproximación vectorizada.
    """
    train_valid = train_df[np.isfinite(train_df[target_col])].copy()

    x_train = train_valid[[x_col, y_col]].to_numpy(dtype=float)
    y_train = train_valid[target_col].to_numpy(dtype=float)
    x_pred = pred_df[[x_col, y_col]].to_numpy(dtype=float)

    if len(train_valid) == 0:
        return np.full(len(pred_df), np.nan, dtype=float)

    k_eff = min(k, len(train_valid))

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


def process_region(region_path: Path, wind_src: rasterio.io.DatasetReader) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, wind_src.crs)

    wind_arr, wind_transform = crop_raster_to_region(wind_src, region_arr, region_transform)

    region_points = raster_points_dataframe(region_arr, region_transform)
    wind_points = raster_points_dataframe(wind_arr, wind_transform).rename(columns={"value": "wind_speed"})

    predictions = predict_knn_weighted(
        train_df=wind_points,
        pred_df=region_points,
        x_col="x",
        y_col="y",
        target_col="wind_speed",
        k=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
    )

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["wind_speed"] = predictions

    return region_points


def main() -> None:
    c_list = list_reference_grids(REF_GRID_DIR)
    df_list: list[pd.DataFrame] = []

    with load_wind(WIND_RASTER) as wind_src:
        if wind_src.crs is None:
            raise ValueError("El raster de velocidad del viento no tiene CRS definido.")

        for region in c_list:
            region_df = process_region(region, wind_src)
            df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()