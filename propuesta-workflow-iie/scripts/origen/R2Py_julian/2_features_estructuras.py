#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 2_features_estructuras.py

Propósito:
    Calcular, para cada píxel de los rasters regionales de referencia, la
    distancia a la estructura costera más cercana por tipo de estructura y
    serializar el resultado como tabla en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    2_features_estructuras.R

Resumen del flujo:
    1. Leer y normalizar el shapefile de estructuras costeras.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster al CRS de las estructuras.
    4. Extraer los centros de píxel como tabla con coordenadas x, y.
    5. Calcular la distancia al vecino más cercano para cada tipo de estructura.
    6. Concatenar resultados regionales y guardar la tabla final en PKL.

Insumos principales:
    - estructuras_final_unido_.shp
    - colección regional de ref_grid.tif

Salidas principales:
    - 2_infraestructura.pkl

Supuestos y notas:
    - La distancia se calcula sobre el CRS del shapefile de estructuras.
    - La reproyección del raster usa vecino más cercano para seguir la lógica
      de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.

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
    También se corrige un detalle del script R original: dentro de los bucles
    for se reasignan el primer raster y el primer tipo de estructura, lo que
    impide recorrer todos los elementos. La versión en Python sí recorre todas
    las regiones y todos los tipos.

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
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
STRUCT_SHP = DROPBOX_DIR / "data_crude" / "05_InventarioEstructuras" / "estructuras_final_unido_.shp"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "2_infraestructura.pkl"

TIPO_FIELD = "Tipo"


def load_structures(path: Path) -> gpd.GeoDataFrame:
    struct = gpd.read_file(path)

    if struct.empty:
        raise ValueError(f"El shapefile de estructuras está vacío: {path}")
    if struct.crs is None:
        raise ValueError(f"El shapefile de estructuras no tiene CRS: {path}")
    if TIPO_FIELD not in struct.columns:
        raise ValueError(f"No existe el campo requerido '{TIPO_FIELD}' en {path.name}")

    # Normalización fiel al script R
    struct[TIPO_FIELD] = struct[TIPO_FIELD].replace(
        {
            "Escollera2": "Escollera",
            "Espigób": "Espigón",
            "espigón": "Espigón",
            "Espigón de M": "Espigón",
            "Muelle": "Puerto",
            "Rompeolas2": "Rompeolas",
        }
    )

    return struct


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


def sanitize_column_name(name: str) -> str:
    """
    Mantiene nombres de columnas razonables en pandas.
    """
    out = str(name).strip()
    out = out.replace(" ", "_")
    out = out.replace("/", "_")
    return out


def nearest_distance_column(
        points_xy: np.ndarray,
        struct_tipo: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Distancia euclidiana al vecino más cercano.
    Traducción funcional de kknn(..., k=1, kernel='rectangular') cuando
    el script R usa modelkknn$D como resultado final.
    """
    if struct_tipo.empty:
        return np.full(points_xy.shape[0], np.nan, dtype=float)

    geom = struct_tipo.geometry

    # Para máxima fidelidad inicial tomamos coordenadas de las geometrías.
    # Si hay líneas o polígonos, se usa representative_point() para asegurar
    # un punto por entidad sin perder vectorización básica.
    if geom.geom_type.isin(["Point", "MultiPoint"]).all():
        coords = np.array([(g.x, g.y) for g in geom], dtype=float)
    else:
        reps = geom.representative_point()
        coords = np.array([(g.x, g.y) for g in reps], dtype=float)

    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)

    return distances.astype(float)


def process_region(region_path: Path, struct: gpd.GeoDataFrame, unique_strus: list[str]) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, struct.crs)

    region_points = raster_points_dataframe(region_arr, region_transform)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    for estructura in unique_strus:
        struct_tipo = struct[struct[TIPO_FIELD] == estructura]
        region_points[sanitize_column_name(estructura)] = nearest_distance_column(pred_xy, struct_tipo)

    return region_points


def main() -> None:
    struct = load_structures(STRUCT_SHP)
    unique_strus = list(pd.unique(struct[TIPO_FIELD]))
    c_list = list_reference_grids(REF_GRID_DIR)

    df_list: list[pd.DataFrame] = []

    for region in c_list:
        region_df = process_region(region, struct, unique_strus)
        df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()