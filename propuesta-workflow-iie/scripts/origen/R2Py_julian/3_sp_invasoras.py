#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 3_sp_invasoras.py

Propósito:
    Calcular, para cada píxel de los rasters regionales de referencia, la
    distancia a la ocurrencia más cercana de cada especie invasora, normalizar
    esas distancias por especie y derivar un índice agregado de potencial de
    especies invasoras.

Origen:
    Traducción inicial a Python del script R:
    3_sp_invasoras.R

Resumen del flujo:
    1. Leer el listado de especies invasoras desde Excel.
    2. Leer la base filtrada de plantas invasoras y preparar coordenadas x, y.
    3. Listar los rasters regionales ref_grid.tif.
    4. Reproyectar cada raster al CRS de los puntos de especies invasoras.
    5. Extraer los centros de píxel como tabla con coordenadas x, y.
    6. Calcular la distancia al vecino más cercano por especie invasora.
    7. Normalizar por columna las distancias y calcular sp_inv_potential.
    8. Serializar el resultado final en PKL.

Insumos principales:
    - especies_invasoras.xlsx
    - plantas_invasoras.csv
    - colección regional de ref_grid.tif

Salidas principales:
    - 3_sp_inv_potential.pkl

Supuestos y notas:
    - Los puntos de especies invasoras se interpretan en EPSG:4326.
    - La reproyección del raster usa vecino más cercano para seguir la lógica
      de project(..., method = "near") en R.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - La normalización por especie replica la función normalize(x) del script R
      y luego aplica 1 - normalize(distancia), como en la versión original.

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

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
SP_INV_XLSX = DROPBOX_DIR / "data_crude" / "15_plantas_snib" / "especies_invasoras.xlsx"
SP_INV_CSV = DROPBOX_DIR / "data_crude" / "15_plantas_snib" / "plantas_invasoras.csv"
REF_GRID_DIR = DROPBOX_DIR / "data" / "06_DunasCost250116_malla_ref_50m"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "3_sp_inv_potential.pkl"

SPECIES_FIELD = "especievalida"
POINTS_CRS = "EPSG:4326"


def normalize_series(values: pd.Series) -> pd.Series:
    """
    Replica:
      normalize <- function(x, na.rm = TRUE) {
        return((x - min(x)) /(max(x)-min(x)))
      }
    """
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)

    if pd.isna(vmin) or pd.isna(vmax):
        return pd.Series(np.nan, index=values.index)

    if vmax == vmin:
        return pd.Series(0.0, index=values.index)

    return (values - vmin) / (vmax - vmin)


def load_species_catalog(path: Path) -> pd.DataFrame:
    """
    Replica:
      sp_inv_s <- read_excel(..., col_names = FALSE)
      sp_inv_s$sp[i] <- word(sp_inv_s$...1[i], 1,2, sep=" ")
    """
    sp_inv_s = pd.read_excel(path, header=None)

    if sp_inv_s.empty:
        raise ValueError(f"El archivo Excel de especies invasoras está vacío: {path}")

    first_col = sp_inv_s.iloc[:, 0].astype(str)
    sp_inv_s["sp"] = first_col.str.split().str[:2].str.join(" ")

    return sp_inv_s


def load_invasive_points(path: Path) -> pd.DataFrame:
    sp_inv = pd.read_csv(path, sep=",", header=0, low_memory=False)

    if sp_inv.shape[1] < 13:
        raise ValueError("El archivo plantas_invasoras.csv no tiene al menos 13 columnas.")

    cols = list(sp_inv.columns)
    cols[11] = "x"
    cols[12] = "y"
    sp_inv.columns = cols

    required = {"x", "y", SPECIES_FIELD}
    missing = required - set(sp_inv.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en plantas_invasoras.csv: {missing}")

    sp_inv["x"] = pd.to_numeric(sp_inv["x"], errors="raise")
    sp_inv["y"] = pd.to_numeric(sp_inv["y"], errors="raise")

    return sp_inv


def list_reference_grids(ref_grid_dir: Path) -> list[Path]:
    c_list = sorted(ref_grid_dir.rglob("*.tif"))
    if not c_list:
        raise FileNotFoundError(f"No se encontraron .tif en {ref_grid_dir}")
    return c_list


def extract_region_id(path: Path) -> str:
    return path.parent.name


def reproject_raster_to_epsg4326(src: rasterio.io.DatasetReader):
    transform, width, height = calculate_default_transform(
        src.crs,
        POINTS_CRS,
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
        dst_crs=POINTS_CRS,
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
    out = str(name).strip()
    out = out.replace(" ", "_")
    out = out.replace("/", "_")
    return out


def nearest_distance_column(points_xy: np.ndarray, species_df: pd.DataFrame) -> np.ndarray:
    if species_df.empty:
        return np.full(points_xy.shape[0], np.nan, dtype=float)

    coords = species_df[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float)


def process_region(region_path: Path, sp_inv: pd.DataFrame, unique_inv: list[str]) -> pd.DataFrame:
    print(region_path)

    with rasterio.open(region_path) as src:
        region_arr, region_transform = reproject_raster_to_epsg4326(src)

    region_points = raster_points_dataframe(region_arr, region_transform)

    region_id = extract_region_id(region_path)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    for sp in unique_inv:
        sp_inv_f = sp_inv[sp_inv[SPECIES_FIELD] == sp]
        region_points[sanitize_column_name(sp)] = nearest_distance_column(pred_xy, sp_inv_f)

    return region_points


def main() -> None:
    # Se lee el catálogo por fidelidad al R, aunque en esta fase no se usa para filtrar,
    # porque el R tampoco lo aplica finalmente (las líneas de filtro están comentadas).
    _ = load_species_catalog(SP_INV_XLSX)

    sp_inv = load_invasive_points(SP_INV_CSV)
    unique_inv = list(pd.unique(sp_inv[SPECIES_FIELD]))
    c_list = list_reference_grids(REF_GRID_DIR)

    df_list: list[pd.DataFrame] = []

    for region in c_list:
        region_df = process_region(region, sp_inv, unique_inv)
        df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)

    # En el R esto ocurre en columnas 6:17.
    # Aquí lo hacemos sobre las columnas de especies detectadas realmente.
    species_cols = [sanitize_column_name(sp) for sp in unique_inv]

    for col in species_cols:
        full_df[col] = 1.0 - normalize_series(full_df[col])

    full_df["sp_inv_potential"] = full_df[species_cols].sum(axis=1, skipna=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()