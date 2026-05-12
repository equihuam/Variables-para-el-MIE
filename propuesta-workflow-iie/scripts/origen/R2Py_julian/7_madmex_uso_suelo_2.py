 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 7_madmex_uso_suelo_2.py

Propósito:
    Estimar, para cada píxel de los rasters regionales de referencia, una
    probabilidad derivada de clasificación k-NN usando clases MADMEX de uso
    de suelo recortadas a la región y serializar el resultado en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    7_madmex_uso_suelo_2.R

Resumen del flujo:
    1. Leer el raster MADMEX de uso de suelo.
    2. Listar los rasters regionales ref_grid.tif.
    3. Reproyectar cada raster regional al CRS de MADMEX.
    4. Recortar MADMEX a la extensión del raster regional reproyectado.
    5. Convertir MADMEX recortado y raster regional a tablas con coordenadas x, y.
    6. Ajustar un clasificador k-NN sobre las clases MADMEX.
    7. Extraer una probabilidad por píxel y serializar el resultado en PKL.

Insumos principales:
    - madmex_landsat_2017_31.tif
    - colección regional de ref_grid.tif

Salidas principales:
    - 7_madmex_landuse_2.pkl

Supuestos y notas:
    - La clasificación se realiza en el CRS del raster MADMEX.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - La salida se guarda en una columna llamada manglares por correspondencia
      con el script original, aunque el nombre depende de la interpretación
      de la segunda clase del clasificador.

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
    También se corrige un problema evidente del script R original: dentro del
    bucle for se reasigna region <- c_list[1], lo que impide procesar todas las
    regiones. La versión en Python sí recorre todos los rasters.

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
from sklearn.neighbors import KNeighborsClassifier


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
MADMEX_RASTER = DROPBOX_DIR / "data_crude" / "16_madmex" / "madmex_landsat_2017_31.tif"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "7_madmex_landuse_2.pkl"

K_NEIGHBORS = 1000


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


def fit_knn_probabilities(
        madmx_points: pd.DataFrame,
        region_points: pd.DataFrame,
        k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    Traducción inicial de:
      kknn(layer ~ x + y, madmx_points, region_points, distance = 2, k=1000, kernel="optimal")
      modelkknn$prob[,2]

    Nota: la equivalencia no es exacta.
    """
    train = madmx_points[np.isfinite(madmx_points["layer"])].copy()
    if train.empty:
        return np.full(len(region_points), np.nan, dtype=float)

    x_train = train[["x", "y"]].to_numpy(dtype=float)
    y_train = train["layer"].astype(int).to_numpy()
    x_pred = region_points[["x", "y"]].to_numpy(dtype=float)

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        # En R prob[,2] implicaría una segunda clase; si no existe, devolvemos 0
        return np.zeros(len(region_points), dtype=float)

    k_eff = min(k, len(train))
    clf = KNeighborsClassifier(
        n_neighbors=k_eff,
        weights="distance",
        algorithm="auto",
        metric="euclidean",
    )
    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_pred)

    # Aproximación a prob[,2] del R:
    # tomamos la segunda columna del orden interno de clases.
    if probs.shape[1] < 2:
        return np.zeros(len(region_points), dtype=float)

    return probs[:, 1].astype(float)


def process_region(region_path: Path, madmx_src: rasterio.io.DatasetReader) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_crs(src, madmx_src.crs)

    madmx_arr, madmx_transform = crop_raster_to_region(madmx_src, region_arr, region_transform)

    region_points = raster_points_dataframe(region_arr, region_transform)
    madmx_points = raster_points_dataframe(madmx_arr, madmx_transform).rename(columns={"value": "layer"})

    # En R:
    # madmx_points$layer <- as.factor(madmx_points$layer)
    madmx_points["layer"] = pd.to_numeric(madmx_points["layer"], errors="coerce")

    probabilities = fit_knn_probabilities(madmx_points, region_points, k=K_NEIGHBORS)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points["manglares"] = probabilities

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