#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 7_inegi_uso_suelo.py

Propósito:
    Calcular, para cada píxel de los rasters regionales de referencia, la
    distancia al uso de suelo INEGI más cercano por categoría generalizada
    y serializar el resultado como tabla en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    7_inegi_uso_suelo.R

Resumen del flujo:
    1. Leer el shapefile de uso de suelo INEGI.
    2. Reclasificar las categorías de uso de suelo en clases generales.
    3. Listar los rasters regionales ref_grid.tif.
    4. Reproyectar cada raster al CRS del shapefile de uso de suelo.
    5. Rasterizar cada clase sobre la plantilla regional reproyectada.
    6. Calcular la distancia al vecino más cercano por clase.
    7. Concatenar resultados regionales y serializar el resultado final en PKL.

Insumos principales:
    - uso_suelo_inegi.shp
    - colección regional de ref_grid.tif

Salidas principales:
    - 7_inegi_uso_suelo.pkl

Supuestos y notas:
    - La distancia se calcula en el CRS del shapefile de uso de suelo.
    - La reproyección del raster usa vecino más cercano para seguir la lógica
      de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - Las categorías de salida se construyen como area, agri y human siguiendo
      la lógica de reclasificación del script original.

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
USO_SUELO_SHP = DROPBOX_DIR / "data_crude" / "14_uso_suelo_inegi" / "uso_suelo_inegi.shp"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "7_inegi_uso_suelo.pkl"

USO_FIELD = "USO.suelo"


def load_land_use(path: Path) -> gpd.GeoDataFrame:
    uso = gpd.read_file(path)

    if uso.empty:
        raise ValueError(f"El shapefile de uso de suelo está vacío: {path}")
    if uso.crs is None:
        raise ValueError(f"El shapefile de uso de suelo no tiene CRS: {path}")
    if USO_FIELD not in uso.columns:
        raise ValueError(f"No existe el campo requerido '{USO_FIELD}' en {path.name}")

    uso = uso.copy()
    uso["type"] = np.nan

    # Traducción directa de la lógica condicional del script R.
    uso.loc[uso[USO_FIELD].isin(["Pastizal", "Vegetación de dunas costeras"]), "type"] = "area"
    uso.loc[uso[USO_FIELD].isin(["Agricultura"]), "type"] = "agri"
    uso.loc[uso[USO_FIELD].isin(["Asentamiento humano"]), "type"] = "human"

    uso = uso[uso["type"].notna()].copy()

    return uso


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


def rasterize_land_use_type(shape, transform, uso_tipo: gpd.GeoDataFrame) -> np.ndarray:
    """
    Equivalente funcional de rasterize(uso_tipo, region_).
    """
    shapes = (
        (geom, 1)
        for geom in uso_tipo.geometry
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


def land_use_points_from_raster(uso_rast: np.ndarray, transform) -> pd.DataFrame:
    """
    Equivalente a as.data.frame(uso_rast, xy = TRUE) filtrando solo celdas válidas.
    """
    valid_mask = np.isfinite(uso_rast)
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", "part"])

    xs, ys = xy(transform, rows, cols, offset="center")

    uso_points = pd.DataFrame(
        {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "part": 1,
        }
    )
    return uso_points


def nearest_distance_column(points_xy: np.ndarray, uso_points: pd.DataFrame) -> np.ndarray:
    """
    Distancia euclidiana al vecino más cercano.
    Traducción funcional de kknn(..., k=1, kernel='rectangular') cuando
    el script R usa modelkknn$D como resultado final.
    """
    if uso_points.empty:
        return np.full(points_xy.shape[0], np.nan, dtype=float)

    coords = uso_points[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float)


def process_region(region_path: Path, uso: gpd.GeoDataFrame, unique_types: list[str]) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, uso.crs)

    region_points = raster_points_dataframe(region_arr, region_transform)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    for t in unique_types:
        uso_tipo = uso[uso["type"] == t]
        uso_rast = rasterize_land_use_type(
            shape=region_arr.shape,
            transform=region_transform,
            uso_tipo=uso_tipo,
        )

        uso_points = land_use_points_from_raster(uso_rast, region_transform)
        region_points[t] = nearest_distance_column(pred_xy, uso_points)

    return region_points


def main() -> None:
    uso = load_land_use(USO_SUELO_SHP)
    unique_types = list(pd.unique(uso["type"]))
    c_list = list_reference_grids(REF_GRID_DIR)

    df_list: list[pd.DataFrame] = []

    for region in c_list:
        region_df = process_region(region, uso, unique_types)
        df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()