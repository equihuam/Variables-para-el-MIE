#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 1_features_tasa_erosion.py

Propósito:
    Estimar el atributo de erosión costera para los píxeles de cada raster
    regional mediante interpolación basada en vecinos cercanos a partir de
    una tabla de puntos observados.

Origen:
    Traducción inicial a Python del script R:
    1_features_tasa_erosion.R

Resumen del flujo:
    1. Leer la tabla de tasas de erosión.
    2. Reproyectar cada raster regional al CRS geográfico de los puntos.
    3. Extraer los centros de píxel del raster reproyectado.
    4. Estimar el valor de erosión en cada píxel mediante k vecinos cercanos.
    5. Concatenar resultados regionales y serializar el resultado en formato PKL.

Insumos principales:
    - Tasas_erosionMEX_Actualizado2018.txt
    - colección regional de ref_grid.tif

Salidas principales:
    - 1_tasa_erosion.pkl

Supuestos y notas:
    - Los puntos de erosión se interpretan en EPSG:4326.
    - La traducción usa una aproximación vectorizada basada en vecinos cercanos.
    - El kernel "optimal" de kknn en R no se replica exactamente en esta fase;
      se usa una aproximación ponderada por distancia para conservar la lógica general.
    - La salida se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    En este caso, la operación de interpolación no replica exactamente el kernel
    "optimal" de kknn, sino una aproximación vectorizada basada en vecinos
    cercanos ponderados por distancia, elegida por ser la alternativa más cercana
    disponible en Python para esta fase inicial.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.neighbors import NearestNeighbors


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
EROSION_TXT = DROPBOX_DIR / "data_crude" / "04_Erosion_acresion" / "Tasas_erosionMEX_Actualizado2018.txt"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_RDS_EQUIV = DROPBOX_DIR / "data_features" / "1_tasa_erosion.pkl"

POINTS_CRS = "EPSG:4326"
K_NEIGHBORS = 3
DISTANCE_POWER = 2
TARGET_FIELD = "Tasa"
NODATA_VALUE = -9999.0


def load_tasa_erosion(path: Path) -> pd.DataFrame:
    tasa_ero = pd.read_csv(path, sep=",", header=0, low_memory=False)

    if tasa_ero.shape[1] < 3:
        raise ValueError("El archivo de tasa de erosión no tiene al menos 3 columnas.")

    # Replica:
    # names(tasa_ero)[2] <- "x"
    # names(tasa_ero)[3] <- "y"
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


def list_reference_grids(ref_grid_dir: Path) -> list[Path]:
    c_list = sorted(ref_grid_dir.rglob("*.tif"))
    if not c_list:
        raise FileNotFoundError(f"No se encontraron .tif en {ref_grid_dir}")
    return c_list


def extract_region_id(path: Path) -> str:
    # En R:
    # region_id <- strsplit(region, split = "/")[[1]][4]
    # Aquí tomamos el nombre del directorio padre, que es más robusto y
    # consistente con region_<id>/ref_grid.tif
    return path.parent.name


def reproject_raster_to_epsg4326(src: rasterio.io.DatasetReader):
    dst_crs = POINTS_CRS

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
    Equivalente aproximado a:
      as.data.frame(region_, xy = TRUE)
    en terra.
    Incluye todas las celdas, incluso si luego contienen NaN.
    """
    height, width = arr.shape

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

    df = pd.DataFrame(
        {
            "x": np.asarray(xs).ravel(),
            "y": np.asarray(ys).ravel(),
            "value": arr.ravel(),
        }
    )
    return df


def predict_knn_weighted(
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        target_col: str = TARGET_FIELD,
        k: int = K_NEIGHBORS,
        distance_power: float = DISTANCE_POWER,
) -> np.ndarray:
    """
    Traducción inicial del:
      kknn(Tasa ~ x + y, tasa_ero, region_points, distance = 2, k = 3, kernel = "optimal")

    Nota: el kernel 'optimal' de kknn no se replica exactamente aquí.
    Se usa promedio ponderado por distancia^-2, que es la aproximación inicial
    más razonable y vectorizable.
    """
    x_train = train_df[[x_col, y_col]].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    x_pred = pred_df[[x_col, y_col]].to_numpy(dtype=float)

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(x_train)

    distances, indices = nn.kneighbors(x_pred, return_distance=True)

    # Manejo robusto para distancia cero
    zero_mask = distances == 0
    weights = np.zeros_like(distances, dtype=float)
    weights[~zero_mask] = 1.0 / np.power(distances[~zero_mask], distance_power)

    # Si hay coincidencia exacta, dar peso 1 a las coincidencias exactas
    # y 0 al resto en esa fila
    any_zero = zero_mask.any(axis=1)
    if np.any(any_zero):
        weights[any_zero] = zero_mask[any_zero].astype(float)

    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0

    neighbor_values = y_train[indices]
    predictions = np.sum(weights * neighbor_values, axis=1) / weight_sums[:, 0]

    return predictions


def process_region(region_path: Path, tasa_ero: pd.DataFrame) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_epsg4326(src)

    region_points = raster_points_dataframe(region_arr, region_transform)

    # En R se predice sobre todos los puntos del raster reproyectado.
    predictions = predict_knn_weighted(
        train_df=tasa_ero,
        pred_df=region_points,
        x_col="x",
        y_col="y",
        target_col=TARGET_FIELD,
        k=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
    )

    region_id = extract_region_id(region_path)

    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["erosion"] = predictions

    return region_points


def main() -> None:
    tasa_ero = load_tasa_erosion(EROSION_TXT)
    c_list = list_reference_grids(REF_GRID_DIR)

    df_list: list[pd.DataFrame] = []

    for region in c_list:
        region_df = process_region(region, tasa_ero)
        df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_RDS_EQUIV.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_RDS_EQUIV)

    print(f"OK -> {OUTPUT_RDS_EQUIV}")


if __name__ == "__main__":
    main()