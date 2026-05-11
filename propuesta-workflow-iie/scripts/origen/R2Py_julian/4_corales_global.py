#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 4_corales_global.py

Propósito:
    Calcular, para cada píxel de los rasters regionales de referencia, la
    distancia al coral global más cercano y serializar el resultado como
    tabla en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    4_corales_global.R

Resumen del flujo:
    1. Leer el shapefile global de corales.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster al CRS de los corales.
    4. Rasterizar los corales sobre la plantilla regional reproyectada.
    5. Calcular la distancia al coral más cercano para cada píxel.
    6. Reemplazar el valor centinela 999 en regiones sin coral por 1.5 * max_dist.
    7. Serializar el resultado final en PKL.

Insumos principales:
    - coral-global.shp
    - colección regional de ref_grid.tif

Salidas principales:
    - 4_coral_distance.pkl

Supuestos y notas:
    - La distancia se calcula en el CRS del shapefile de corales.
    - La reproyección del raster usa vecino más cercano para seguir la lógica
      de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - Se conserva el valor centinela 999 a nivel regional y se sustituye al final
      por 1.5 * max_dist, igual que en el script R.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, la distancia final usada por el script R proviene de kknn con
    k = 1 y kernel = "rectangular"; en Python se implementa directamente como
    distancia euclidiana al vecino más cercano, que es la traducción funcional
    más cercana para esta fase inicial.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
CORALES_SHP = DROPBOX_DIR / "data_crude" / "08_coral-global" / "coral-global.shp"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "4_coral_distance.pkl"

CORALS_SENTINEL = 999.0


def load_corals(path: Path) -> gpd.GeoDataFrame:
    corales = gpd.read_file(path)

    if corales.empty:
        raise ValueError(f"El shapefile de corales está vacío: {path}")
    if corales.crs is None:
        raise ValueError(f"El shapefile de corales no tiene CRS: {path}")

    return corales


def list_reference_grids(ref_grid_dir: Path) -> list[Path]:
    c_list = sorted(ref_grid_dir.rglob("*.tif"))
    if not c_list:
        raise FileNotFoundError(f"No se encontraron .tif en {ref_grid_dir}")
    return c_list


def extract_region_id(path: Path) -> str:
    # Más robusto que strsplit(...)[[1]][4], pero equivalente en intención
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
    Aproximación a as.data.frame(region_, xy = TRUE) de terra.
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


def rasterize_corals_on_region(shape, transform, corales: gpd.GeoDataFrame) -> np.ndarray:
    """
    Equivalente funcional de:
      corales_rast <- rasterize(corales, region_)
    """
    shapes = (
        (geom, 1)
        for geom in corales.geometry
        if geom is not None and not geom.is_empty
    )

    arr = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
    )

    return arr


def coral_points_from_raster(corales_rast: np.ndarray, transform) -> pd.DataFrame:
    """
    Equivalente a:
      coral_points <- as.data.frame(corales_rast, xy = TRUE)
    pero filtrando solo celdas válidas.
    """
    valid_mask = np.isfinite(corales_rast)
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", "part"])

    xs, ys = xy(transform, rows, cols, offset="center")

    coral_points = pd.DataFrame(
        {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "part": 1,
        }
    )
    return coral_points


def nearest_distance_column(points_xy: np.ndarray, coral_points: pd.DataFrame) -> np.ndarray:
    """
    Distancia euclidiana al coral más cercano.
    Traducción funcional de kknn(..., k=1, kernel='rectangular') cuando
    el script R usa modelkknn$D como resultado final.
    """
    coords = coral_points[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float)


def process_region(region_path: Path, corales: gpd.GeoDataFrame) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, corales.crs)

    region_points = raster_points_dataframe(region_arr, region_transform)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["corals"] = CORALS_SENTINEL

    corales_rast = rasterize_corals_on_region(
        shape=region_arr.shape,
        transform=region_transform,
        corales=corales,
    )

    # Equivalente a:
    # if (sum(!is.nan(values(corales_rast))) > 0) { ... }
    if np.isfinite(corales_rast).sum() > 0:
        coral_points = coral_points_from_raster(corales_rast, region_transform)

        # Replica el comportamiento del R:
        # if (nrow(coral_points) == 1) coral_points <- coral_points[c(1,1),]
        if len(coral_points) == 1:
            coral_points = pd.concat([coral_points, coral_points], ignore_index=True)

        pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)
        distances = nearest_distance_column(pred_xy, coral_points)
        region_points["corals"] = distances

    return region_points


def main() -> None:
    corales = load_corals(CORALES_SHP)
    c_list = list_reference_grids(REF_GRID_DIR)

    df_list: list[pd.DataFrame] = []

    for region in c_list:
        region_df = process_region(region, corales)
        df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    max_dist = float(full_df["corals"].max())
    full_df.loc[full_df["corals"] == CORALS_SENTINEL, "corals"] = 1.5 * max_dist

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()