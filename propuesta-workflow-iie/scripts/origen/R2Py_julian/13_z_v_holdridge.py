#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 13_z_v_holdridge.py

Propósito:
    Clasificar, para cada píxel de los rasters regionales de referencia, la
    zona de vida Holdridge más probable a partir de un raster temático de
    zonas de vida y serializar el resultado en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    13_z_v_holdridge.R

Resumen del flujo:
    1. Leer el shapefile de manglares para obtener el CRS de trabajo.
    2. Leer el raster de zonas de vida Holdridge.
    3. Reproyectar el raster Holdridge al CRS de manglares.
    4. Listar los rasters regionales ref_grid.tif.
    5. Reproyectar cada raster regional al CRS de manglares.
    6. Recortar el raster Holdridge a la extensión de la región.
    7. Ajustar un clasificador 1-NN sobre los píxeles etiquetados de Holdridge.
    8. Predecir una clase de zona de vida para cada píxel regional.
    9. Concatenar resultados regionales y serializar el resultado final en PKL.

Insumos principales:
    - cm-conabio.shp
    - zvh_mx3gw.tif
    - colección regional de ref_grid.tif

Salidas principales:
    - 13_zvh.pkl

Supuestos y notas:
    - La clasificación se realiza en el CRS del shapefile de manglares.
    - Tanto el raster Holdridge como los rasters regionales se reproyectan con
      vecino más cercano para seguir la lógica de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - La salida `zvh` se guarda como etiqueta categórica predicha.

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
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from sklearn.neighbors import KNeighborsClassifier


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
MANGLARES_SHP = DROPBOX_DIR / "data_crude" / "02_cm-conabio" / "cm-conabio.shp"
ZVH_RASTER = DROPBOX_DIR / "data_crude" / "07_zvh_mx3gw" / "zvh_mx3gw.tif"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "13_zvh.pkl"

K_NEIGHBORS = 1


def load_mangroves_crs(path: Path):
    manglares = gpd.read_file(path)
    if manglares.empty:
        raise ValueError(f"El shapefile de manglares está vacío: {path}")
    if manglares.crs is None:
        raise ValueError(f"El shapefile de manglares no tiene CRS: {path}")
    return manglares.crs


def load_zvh(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el raster Holdridge: {path}")
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
      zvh_ <- crop(zvh, region_)
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


def fit_knn_labels(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    """
    Traducción inicial de:
      kknn(layer ~ x + y, zvh_points, region_points, distance = 2, k=1,
           kernel = "rectangular")
      modelkknn$fitted.values

    Con k = 1, la traducción funcional más cercana es clasificación 1-NN.
    """
    train_valid = train_df[np.isfinite(train_df["layer"])].copy()
    if train_valid.empty:
        return np.full(len(pred_df), np.nan, dtype=float)

    x_train = train_valid[["x", "y"]].to_numpy(dtype=float)
    y_train = train_valid["layer"].astype(int).to_numpy()
    x_pred = pred_df[["x", "y"]].to_numpy(dtype=float)

    clf = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights="uniform",
        algorithm="auto",
        metric="euclidean",
    )
    clf.fit(x_train, y_train)

    pred = clf.predict(x_pred)
    return pred.astype(float)


def process_region(region_path: Path, zvh_src: rasterio.io.DatasetReader, target_crs) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, target_crs)

    zvh_arr, zvh_transform = crop_raster_to_region(zvh_src, region_arr, region_transform)

    region_points = raster_points_dataframe(region_arr, region_transform)
    zvh_points = raster_points_dataframe(zvh_arr, zvh_transform).rename(columns={"value": "layer"})

    predictions = fit_knn_labels(zvh_points, region_points)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["zvh"] = predictions

    return region_points


def main() -> None:
    target_crs = load_mangroves_crs(MANGLARES_SHP)
    c_list = list_reference_grids(REF_GRID_DIR)
    df_list: list[pd.DataFrame] = []

    with load_zvh(ZVH_RASTER) as zvh_raw:
        if zvh_raw.crs is None:
            raise ValueError("El raster de zonas de vida Holdridge no tiene CRS definido.")

        # Replica:
        # zvh <- project(zvh, y = crs(manglares), method = "near")
        zvh_arr, zvh_transform = reproject_raster_to_crs(zvh_raw, target_crs)

        profile = zvh_raw.profile.copy()
        profile.update(
            driver="GTiff",
            height=zvh_arr.shape[0],
            width=zvh_arr.shape[1],
            count=1,
            dtype="float32",
            crs=target_crs,
            transform=zvh_transform,
            nodata=np.nan,
        )

        from rasterio.io import MemoryFile

        with MemoryFile() as memfile:
            with memfile.open(**profile) as zvh_src:
                zvh_src.write(zvh_arr, 1)

                for region in c_list:
                    region_df = process_region(region, zvh_src, target_crs)
                    df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()