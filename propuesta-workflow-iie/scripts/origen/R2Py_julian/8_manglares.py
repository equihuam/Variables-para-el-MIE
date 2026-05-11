#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 8_manglares.py

Propósito:
    Estimar, para cada píxel de los rasters regionales de referencia, una
    probabilidad de pertenencia o cercanía funcional a manglares a partir de
    un clasificador k-NN construido sobre un shapefile de manglares rasterizado
    en la región, y serializar el resultado en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    8_manglares.R

Resumen del flujo:
    1. Leer el shapefile de manglares.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster regional al CRS de manglares.
    4. Convertir el raster regional a tabla de píxeles x, y.
    5. Rasterizar manglares sobre la plantilla regional.
    6. Construir una tabla de entrenamiento binaria 0/1 sobre la región.
    7. Ajustar un clasificador k-NN y obtener probabilidad para la clase positiva.
    8. Concatenar resultados regionales y serializar el resultado final en PKL.

Insumos principales:
    - cm-conabio.shp
    - colección regional de ref_grid.tif

Salidas principales:
    - 8_manglares.pkl

Supuestos y notas:
    - La clasificación se realiza en el CRS del shapefile de manglares.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - La columna de salida se interpreta como probabilidad asociada a manglares.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, el kernel "optimal" de kknn y la extracción de modelkknn$prob[,2]
    no se replican exactamente; se usa un clasificador k-NN de scikit-learn con
    ponderación por distancia como aproximación inicial.
    La probabilidad de manglares se toma como la probabilidad de la clase positiva
    (valor 1) en la tabla binaria construida a partir del rasterizado de manglares.

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
from sklearn.neighbors import KNeighborsClassifier


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
MANGLARES_SHP = DROPBOX_DIR / "data_crude" / "02_cm-conabio" / "cm-conabio.shp"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "8_manglares.pkl"

K_NEIGHBORS = 30


def load_mangroves(path: Path) -> gpd.GeoDataFrame:
    manglares = gpd.read_file(path)

    if manglares.empty:
        raise ValueError(f"El shapefile de manglares está vacío: {path}")
    if manglares.crs is None:
        raise ValueError(f"El shapefile de manglares no tiene CRS: {path}")

    return manglares


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


def raster_points_dataframe(arr: np.ndarray, transform, include_na: bool = True) -> pd.DataFrame:
    """
    Aproximación a as.data.frame(region_, xy = TRUE, na.rm = FALSE/TRUE) de terra.
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

    if not include_na:
        df = df[np.isfinite(df["value"])].copy()

    return df


def rasterize_mangroves(shape, transform, manglares: gpd.GeoDataFrame) -> np.ndarray:
    """
    Equivalente funcional de:
      manglares_rast <- rasterize(manglares, region_)
    """
    shapes = (
        (geom, 1)
        for geom in manglares.geometry
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


def build_training_dataframe(
        manglares_rast: np.ndarray,
        region_points_na: pd.DataFrame,
        transform,
) -> pd.DataFrame:
    """
    Traducción cercana a:
      manglares_points <- as.data.frame(manglares_rast, xy = TRUE, na.rm = FALSE)
      manglares_points$layer[is.na(manglares_points$layer) & !is.na(region_points_na$OID_1)] <- 0
      manglares_points$layer <- as.factor(manglares_points$layer)

    Aquí se construye una tabla binaria 0/1 sobre todas las celdas de la región.
    """
    manglares_points = raster_points_dataframe(manglares_rast, transform, include_na=True)
    manglares_points = manglares_points.rename(columns={"value": "layer"})

    # Para celdas dentro de la región sin manglar: layer = 0
    inside_region = np.isfinite(region_points_na["value"].to_numpy())
    layer = manglares_points["layer"].to_numpy(dtype=float)

    missing_in_region = np.isnan(layer) & inside_region
    layer[missing_in_region] = 0.0

    manglares_points["layer"] = layer

    # Nos quedamos solo con celdas etiquetadas 0/1
    manglares_points = manglares_points[np.isfinite(manglares_points["layer"])].copy()
    manglares_points["layer"] = manglares_points["layer"].astype(int)

    return manglares_points


def fit_knn_probabilities(
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    Traducción inicial de:
      kknn(layer ~ x + y, manglares_points, region_points, distance = 2, k=30, kernel = "optimal")
      modelkknn$prob[,2]

    Nota: la equivalencia no es exacta.
    """
    if train_df.empty:
        return np.zeros(len(pred_df), dtype=float)

    x_train = train_df[["x", "y"]].to_numpy(dtype=float)
    y_train = train_df["layer"].to_numpy(dtype=int)
    x_pred = pred_df[["x", "y"]].to_numpy(dtype=float)

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        # Si solo hay una clase, asignamos probabilidad 1 o 0 según esa clase.
        only_class = unique_classes[0]
        return np.full(len(pred_df), float(only_class), dtype=float)

    k_eff = min(k, len(train_df))
    clf = KNeighborsClassifier(
        n_neighbors=k_eff,
        weights="distance",
        algorithm="auto",
        metric="euclidean",
    )
    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_pred)

    # Queremos la probabilidad de clase 1 (manglar)
    classes = clf.classes_
    if 1 not in classes:
        return np.zeros(len(pred_df), dtype=float)

    class_index = int(np.where(classes == 1)[0][0])
    return probs[:, class_index].astype(float)


def process_region(region_path: Path, manglares: gpd.GeoDataFrame) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, manglares.crs)

    region_points = raster_points_dataframe(region_arr, region_transform, include_na=False)
    region_points_na = raster_points_dataframe(region_arr, region_transform, include_na=True)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["manglares"] = 0.0

    manglares_rast = rasterize_mangroves(
        shape=region_arr.shape,
        transform=region_transform,
        manglares=manglares,
    )

    if np.isfinite(manglares_rast).sum() > 0:
        manglares_points = build_training_dataframe(
            manglares_rast=manglares_rast,
            region_points_na=region_points_na,
            transform=region_transform,
        )

        # Replica del tratamiento especial del R cuando nrow == 1
        if len(manglares_points) == 1:
            manglares_points = pd.concat([manglares_points, manglares_points], ignore_index=True)

        probabilities = fit_knn_probabilities(
            train_df=manglares_points,
            pred_df=region_points,
            k=K_NEIGHBORS,
        )

        region_points["manglares"] = probabilities

    return region_points


def main() -> None:
    manglares = load_mangroves(MANGLARES_SHP)
    c_list = list_reference_grids(REF_GRID_DIR)

    df_list: list[pd.DataFrame] = []

    for region in c_list:
        region_df = process_region(region, manglares)
        df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()