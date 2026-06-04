#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 11_tipo_costa.py

Propósito:
    Clasificar, para cada píxel de los rasters regionales de referencia, el
    tipo de costa más probable a partir del shapefile de tipología costera y
    serializar el resultado en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    11_tipo_costa.R

Resumen del flujo:
    1. Leer el shapefile de tipo de costa.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster regional al CRS del shapefile de costas.
    4. Convertir el raster regional a tabla con coordenadas x, y.
    5. Rasterizar el atributo TipoCosta sobre la plantilla regional.
    6. Ajustar un clasificador 1-NN sobre los píxeles etiquetados.
    7. Predecir una clase de tipo de costa para cada píxel.
    8. Concatenar resultados regionales y serializar el resultado final en PKL.

Insumos principales:
    - TipoCosta.SHP
    - colección regional de ref_grid.tif

Salidas principales:
    - 11_tipo_costa.pkl

Supuestos y notas:
    - La clasificación se realiza en el CRS del shapefile de tipo de costa.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - La salida `tipo_costa` se guarda como etiqueta categórica predicha.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, el script R usa kknn con k = 1 y kernel = "rectangular" y luego
    toma fitted.values; en Python se implementa directamente como clasificación
    1-NN, que es la traducción funcional más cercana para esta fase inicial.

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
COSTAS_SHP = DROPBOX_DIR / "data_crude" / "12_Tipo_de_costa" / "TipoCosta.SHP"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "11_tipo_costa.pkl"

FIELD_NAME = "TipoCosta"
K_NEIGHBORS = 1


def load_coast_types(path: Path) -> gpd.GeoDataFrame:
    costas = gpd.read_file(path)

    if costas.empty:
        raise ValueError(f"El shapefile de tipo de costa está vacío: {path}")
    if costas.crs is None:
        raise ValueError(f"El shapefile de tipo de costa no tiene CRS: {path}")
    if FIELD_NAME not in costas.columns:
        raise ValueError(f"No existe el campo requerido '{FIELD_NAME}' en {path.name}")

    return costas


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


def rasterize_coast_types(shape, transform, costas: gpd.GeoDataFrame) -> np.ndarray:
    """
    Equivalente funcional de:
      costas_rast <- rasterize(costas, region_, field="TipoCosta")
    """
    categories = pd.Categorical(costas[FIELD_NAME].astype(str))
    codes = categories.codes + 1  # 0 reservado para 'sin dato'

    shapes = (
        (geom, code)
        for geom, code in zip(costas.geometry, codes)
        if geom is not None and not geom.is_empty
    )

    arr = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )

    code_to_label = {int(code + 1): str(cat) for code, cat in enumerate(categories.categories)}
    return arr, code_to_label


def coast_points_from_raster(costas_rast: np.ndarray, transform, code_to_label: dict[int, str]) -> pd.DataFrame:
    """
    Equivalente a:
      costas_table <- as.data.frame(costas_rast, xy = TRUE)
      costas_table$TipoCosta <- as.factor(costas_table$TipoCosta)

    Solo conserva celdas con etiqueta válida.
    """
    valid_mask = costas_rast > 0
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", FIELD_NAME])

    xs, ys = xy(transform, rows, cols, offset="center")
    codes = costas_rast[rows, cols]

    labels = [code_to_label[int(c)] for c in codes]

    return pd.DataFrame(
        {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            FIELD_NAME": labels,
        }
    )


def fit_knn_labels(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    """
    Traducción inicial de:
      kknn(TipoCosta ~ x + y, costas_table, region_points, distance = 2, k=1,
           kernel = "rectangular")
      costa_prediction <- modelkknn$fitted.values

    Con k = 1, la traducción funcional más cercana es clasificación 1-NN.
    """
    if train_df.empty:
        return np.full(len(pred_df), None, dtype=object)

    x_train = train_df[["x", "y"]].to_numpy(dtype=float)
    y_train = train_df[FIELD_NAME].astype(str).to_numpy()
    x_pred = pred_df[["x", "y"]].to_numpy(dtype=float)

    clf = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights="uniform",
        algorithm="auto",
        metric="euclidean",
    )
    clf.fit(x_train, y_train)

    pred = clf.predict(x_pred)
    return pred.astype(object)


def process_region(region_path: Path, costas: gpd.GeoDataFrame) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, costas.crs)

    region_points = raster_points_dataframe(region_arr, region_transform)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["tipo_costa"] = ""

    costas_rast, code_to_label = rasterize_coast_types(
        shape=region_arr.shape,
        transform=region_transform,
        costas=costas,
    )

    costas_table = coast_points_from_raster(costas_rast, region_transform, code_to_label)

    if not costas_table.empty:
        costa_prediction = fit_knn_labels(costas_table, region_points)
        region_points["tipo_costa"] = costa_prediction

    return region_points


def main() -> None:
    costas = load_coast_types(COSTAS_SHP)
    c_list = list_reference_grids(REF_GRID_DIR)

    df_list: list[pd.DataFrame] = []

    for region in c_list:
        region_df = process_region(region, costas)
        df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()