#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 7_madmex_uso_suelo.py

Propósito:
    Calcular, para cada píxel de los rasters regionales de referencia, el
    número de celdas MADMEX de uso de suelo de ciertas clases dentro de un
    buffer de 2500 m alrededor del píxel, y serializar el resultado en PKL.

Origen:
    Traducción inicial a Python del script R:
    7_madmex_uso_suelo.R

Resumen del flujo:
    1. Leer el raster MADMEX de uso de suelo.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster regional al CRS de MADMEX.
    4. Convertir los centros de píxel de cada raster regional en puntos.
    5. Construir un buffer de 2500 m alrededor de cada punto.
    6. Recortar MADMEX a cada buffer y contar clases de interés.
    7. Concatenar resultados regionales y serializar el resultado final en PKL.

Insumos principales:
    - madmex_landsat_2017_31.tif
    - colección regional de ref_grid.tif

Salidas principales:
    - 7_madmex_landuse.pkl

Supuestos y notas:
    - Los conteos se calculan en el CRS del raster MADMEX.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - Se contabilizan las clases: urbano (29), pastizal (27) y agricultura (28).

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, se corrigen dos problemas evidentes del script R original:
    (1) la asignación de pixid y regionid ocurre antes de crear region_points,
    y (2) el conteo de agricultura se escribe por error en agrassland en lugar
    de aagriculture. La versión en Python preserva la intención analítica y
    corrige esos errores.

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
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
MADMEX_RASTER = DROPBOX_DIR / "data_crude" / "16_madmex" / "madmex_landsat_2017_31.tif"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "7_madmex_landuse.pkl"

BUFFER_METERS = 2500.0

URBAN_CLASS = 29
GRASSLAND_CLASS = 27
AGRICULTURE_CLASS = 28


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


def raster_points_geodataframe(arr: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """
    Aproximación a as.points(region_) / as.data.frame(region_, xy = TRUE).
    Construye un GeoDataFrame con centros de píxel.
    """
    height, width = arr.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = xy(transform, rows, cols, offset="center")

    df = pd.DataFrame(
        {
            "x": np.asarray(xs).ravel(),
            "y": np.asarray(ys).ravel(),
            "value": arr.ravel(),
        }
    )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=crs,
    )
    return gdf


def count_madmex_classes_in_buffer(madmx_src: rasterio.io.DatasetReader, geom) -> tuple[int, int, int]:
    """
    Equivalente funcional a:
      madmx_masked <- crop(madmx_, r_buffer)
      counts <- as.data.frame(terra::freq(madmx_masked))
    """
    try:
        cropped, _ = mask(
            madmx_src,
            [mapping(geom)],
            crop=True,
            filled=False,
        )
    except ValueError:
        # Sin intersección
        return 0, 0, 0

    arr = cropped[0]

    if np.ma.isMaskedArray(arr):
        valid = arr.compressed()
    else:
        valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return 0, 0, 0

    values, counts = np.unique(valid, return_counts=True)
    freq = dict(zip(values.tolist(), counts.tolist()))

    urban_count = int(freq.get(URBAN_CLASS, 0))
    grassland_count = int(freq.get(GRASSLAND_CLASS, 0))
    agriculture_count = int(freq.get(AGRICULTURE_CLASS, 0))

    return urban_count, grassland_count, agriculture_count


def process_region(region_path: Path, madmx_src: rasterio.io.DatasetReader) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, madmx_src.crs)

    region_id = extract_region_id(region_path)

    region_points = raster_points_geodataframe(region_arr, region_transform, madmx_src.crs)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    region_points["aurban"] = 0
    region_points["agrassland"] = 0
    region_points["aagriculture"] = 0

    # Traducción inicial fiel al esquema del R: buffer por punto.
    for i, geom in enumerate(region_points.geometry):
        r_buffer = geom.buffer(BUFFER_METERS)

        urban_count, grassland_count, agriculture_count = count_madmex_classes_in_buffer(
            madmx_src,
            r_buffer,
        )

        region_points.iat[i, region_points.columns.get_loc("aurban")] = urban_count
        region_points.iat[i, region_points.columns.get_loc("agrassland")] = grassland_count
        region_points.iat[i, region_points.columns.get_loc("aagriculture")] = agriculture_count

    return region_points.drop(columns="geometry")


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