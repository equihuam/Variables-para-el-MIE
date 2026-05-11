#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 6_batimetria_caracteristica.py

Propósito:
    Estimar la batimetría característica para los píxeles de cada raster
    regional mediante interpolación basada en vecinos cercanos a partir
    de un raster batimétrico base.

Origen:
    Traducción inicial a Python del script R:
    6_batimetria_caracteristica.R

Resumen del flujo:
    1. Leer el raster batimétrico base.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster regional al CRS de la batimetría.
    4. Recortar la batimetría a la extensión del raster regional reproyectado.
    5. Extraer los centros de píxel como tablas con coordenadas x, y.
    6. Estimar la batimetría en cada píxel mediante vecinos cercanos.
    7. Concatenar resultados regionales y serializar el resultado en PKL.

Insumos principales:
    - 01_GEBCO2020_SIMAR.tif
    - colección regional de ref_grid.tif

Salidas principales:
    - 6_batimetria_charact.pkl

Supuestos y notas:
    - La interpolación se realiza en el CRS del raster batimétrico.
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
BAT_RASTER = DROPBOX_DIR / "data_crude" / "13_Batimetría" / "01_GEBCO2020_SIMAR.tif"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "6_batimetria_charact.pkl"

K_NEIGHBORS = 7
DISTANCE_POWER = 2


def load_bathymetry(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el raster batimétrico: {path}")
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


def crop_bathymetry_to_region(bat_src: rasterio.io.DatasetReader, region_arr: np.ndarray, region_transform):
    """
    Equivalente funcional a:
      bat_ <- crop(bat, region_)
    usando el bbox del raster regional reproyectado.
    """
    height, width = region_arr.shape
    left, top = region_transform * (0, 0)
    right, bottom = region_transform * (width, height)

    geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
    cropped, cropped_transform = mask(bat_src, [mapping(geom)], crop=True, filled=True)

    return cropped[0], cropped_transform


def predict_knn_weighted(
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        target_col: str = "bat",
        k: int = K_NEIGHBORS,
        distance_power: float = DISTANCE_POWER,
) -> np.ndarray:
    """
    Traducción inicial de:
      kknn(bat ~ x + y, bat_points, region_points, distance = 2, k = 7, kernel = "optimal")

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


def process_region(region_path: Path, bat_src: rasterio.io.DatasetReader) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, bat_src.crs)

    bat_arr, bat_transform = crop_bathymetry_to_region(bat_src, region_arr, region_transform)

    region_points = raster_points_dataframe(region_arr, region_transform)
    bat_points = raster_points_dataframe(bat_arr, bat_transform)
    bat_points = bat_points.rename(columns={"value": "bat"})

    predictions = predict_knn_weighted(
        train_df=bat_points,
        pred_df=region_points,
        x_col="x",
        y_col="y",
        target_col="bat",
        k=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
    )

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["batimetria"] = predictions

    return region_points


def main() -> None:
    c_list = list_reference_grids(REF_GRID_DIR)
    df_list: list[pd.DataFrame] = []

    with load_bathymetry(BAT_RASTER) as bat_src:
        if bat_src.crs is None:
            raise ValueError("El raster batimétrico no tiene CRS definido.")

        for region in c_list:
            region_df = process_region(region, bat_src)
            df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()