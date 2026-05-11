#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 7_madmex_uso_suelo_3.py

Propósito:
    Calcular, para cada píxel de los rasters regionales de referencia, la
    distancia a la clase MADMEX más cercana para grassland, agriculture y urban,
    y serializar el resultado en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    7_madmex_uso_suelo_3.R

Resumen del flujo:
    1. Leer el raster MADMEX de uso de suelo.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster regional al CRS de MADMEX.
    4. Recortar MADMEX a la extensión del raster regional reproyectado.
    5. Filtrar MADMEX a las clases 27, 28 y 29.
    6. Convertir raster regional y MADMEX recortado a tablas con coordenadas x, y.
    7. Calcular la distancia al vecino más cercano por clase.
    8. Concatenar resultados regionales y serializar el resultado final en PKL.

Insumos principales:
    - madmex_landsat_2017_31.tif
    - colección regional de ref_grid.tif

Salidas principales:
    - 7_madmex_landuse_3.pkl

Supuestos y notas:
    - La distancia se calcula en el CRS del raster MADMEX.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - Se usan las clases MADMEX: grassland (27), agriculture (28) y urban (29).

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
    También se corrige un problema evidente del script R original: en los bloques
    de agriculture y urban las distancias se escriben por error en la columna
    grassland. La versión en Python preserva la intención analítica y corrige
    esa asignación.

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
from scipy.spatial import cKDTree


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
MADMEX_RASTER = DROPBOX_DIR / "data_crude" / "16_madmex" / "madmex_landsat_2017_31.tif"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "7_madmex_landuse_3.pkl"

GRASSLAND_CLASS = 27
AGRICULTURE_CLASS = 28
URBAN_CLASS = 29
VALID_CLASSES = {GRASSLAND_CLASS, AGRICULTURE_CLASS, URBAN_CLASS}
SENTINEL = 9999.0


def load_madmex(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el raster MADMEX: {path}")
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
      madmx_ <- crop(madmx, region_)
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


def filter_madmex_classes(arr: np.ndarray) -> np.ndarray:
    """
    Equivalente funcional a:
      madmx_ <- ifel(!(madmx_ %in% c(27,28,29)), NA, madmx_)
    """
    out = arr.astype(float).copy()
    valid_mask = np.isin(out, list(VALID_CLASSES))
    out[~valid_mask] = np.nan
    return out


def nearest_distance_column(points_xy: np.ndarray, class_points: pd.DataFrame) -> np.ndarray:
    """
    Distancia euclidiana al vecino más cercano.
    Traducción funcional de kknn(..., k=1, kernel='rectangular') cuando
    el script R usa modelkknn$D como resultado final.
    """
    if class_points.empty:
        return np.full(points_xy.shape[0], SENTINEL, dtype=float)

    coords = class_points[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float)


def process_region(region_path: Path, madmx_src: rasterio.io.DatasetReader) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, madmx_src.crs)

    madmx_arr, madmx_transform = crop_raster_to_region(madmx_src, region_arr, region_transform)
    madmx_arr = filter_madmex_classes(madmx_arr)

    region_points = raster_points_dataframe(region_arr, region_transform)
    madmx_points = raster_points_dataframe(madmx_arr, madmx_transform).rename(columns={"value": "layer"})

    madmx_points["layer"] = pd.to_numeric(madmx_points["layer"], errors="coerce")

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    region_points["grassland"] = nearest_distance_column(
        pred_xy,
        madmx_points[madmx_points["layer"] == GRASSLAND_CLASS]
    )

    region_points["agriculture"] = nearest_distance_column(
        pred_xy,
        madmx_points[madmx_points["layer"] == AGRICULTURE_CLASS]
    )

    region_points["urban"] = nearest_distance_column(
        pred_xy,
        madmx_points[madmx_points["layer"] == URBAN_CLASS]
    )

    return region_points


def main() -> None:
    c_list = list_reference_grids(REF_GRID_DIR)
    df_list: list[pd.DataFrame] = []

    with load_madmex(MADMEX_RASTER) as madmx_src:
        if madmx_src.crs is None:
            raise ValueError("El raster MADMEX no tiene CRS definido.")

        for region in c_list:
            region_df = process_region(region, madmx_src)
            df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()