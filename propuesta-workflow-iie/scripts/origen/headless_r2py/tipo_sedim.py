# -*- coding: utf-8 -*-
"""
Lee un shapefile de puntos de tipo de sedimento, lo reproyecta al CRS de
reg_unidas y rasteriza el atributo seleccionado sobre la malla del raster base.

Traducción headless del script R original.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(".")

POINTS_PATH = BASE_DIR / "data" / "tipo_sed" / "tipo_sed_reproy.shp"
BASE_RASTER = BASE_DIR / "data" / "regiones_unidas" / "reg_unidas.tif"
OUTPUT_RASTER = BASE_DIR / "data" / "tipo_sed" / "tipo_sed.tif"

SOURCE_FIELD = "raster"
ENCODE_FACTOR_FIELD = "id_estruct"

# Si True, rasteriza el campo categórico codificado numéricamente.
# Si False, intenta rasterizar directamente SOURCE_FIELD.
USE_ENCODED_FIELD = True

NODATA_VALUE = 0
OUTPUT_DTYPE = "uint16"


# ---------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def validate_input_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"No existe el archivo de entrada ({label}): {path}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def factorize_column(series: pd.Series) -> pd.Series:
    """
    Codifica una columna categórica a enteros positivos, preservando NA.
    """
    codes, uniques = pd.factorize(series, sort=True)
    out = pd.Series(codes + 1, index=series.index, dtype="float64")
    out[codes == -1] = np.nan
    return out


# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def main() -> int:
    validate_input_file(POINTS_PATH, "shapefile de puntos")
    validate_input_file(BASE_RASTER, "raster base")
    ensure_parent(OUTPUT_RASTER)

    log("Leyendo puntos...")
    pts = gpd.read_file(POINTS_PATH)

    if pts.empty:
        fail("El shapefile de puntos está vacío.")

    if SOURCE_FIELD not in pts.columns:
        fail(f"No existe el campo requerido en el shapefile: {SOURCE_FIELD}")

    log("Leyendo raster base...")
    with rasterio.open(BASE_RASTER) as base:
        base_crs = base.crs
        base_transform = base.transform
        base_shape = (base.height, base.width)
        base_profile = base.profile.copy()

        if base_crs is None:
            fail("El raster base no tiene CRS definido.")

        log("Reproyectando puntos al CRS del raster base...")
        pts = pts.to_crs(base_crs)

        log(f"Codificando {SOURCE_FIELD} a entero en {ENCODE_FACTOR_FIELD}...")
        pts[ENCODE_FACTOR_FIELD] = factorize_column(pts[SOURCE_FIELD])

        # TODO: Confirmar si la rasterización final debe usar el campo categórico
        # original SOURCE_FIELD o el campo codificado ENCODE_FACTOR_FIELD.
        # El script R crea id_estruct pero luego rasteriza con 'raster', lo que
        # sugiere una inconsistencia metodológica que conviene resolver explícitamente.
        field_to_use = ENCODE_FACTOR_FIELD if USE_ENCODED_FIELD else SOURCE_FIELD

        if field_to_use not in pts.columns:
            fail(f"No existe el campo a rasterizar: {field_to_use}")

        valid = pts.geometry.notnull() & pts[field_to_use].notnull()
        pts_valid = pts.loc[valid].copy()

        if pts_valid.empty:
            fail("No hay puntos válidos para rasterizar.")

        shapes = [
            (geom, value)
            for geom, value in zip(pts_valid.geometry, pts_valid[field_to_use])
        ]

        log(f"Rasterizando usando el campo: {field_to_use}")
        out_arr = rasterize(
            shapes=shapes,
            out_shape=base_shape,
            transform=base_transform,
            fill=NODATA_VALUE,
            dtype=OUTPUT_DTYPE,
        )

        base_profile.update(
            dtype=OUTPUT_DTYPE,
            count=1,
            nodata=NODATA_VALUE,
            compress="lzw",
        )

        log("Guardando raster de salida...")
        with rasterio.open(OUTPUT_RASTER, "w", **base_profile) as dst:
            dst.write(out_arr, 1)

    log("\n----------------------------------------")
    log("Proceso completado.")
    log(f"Raster guardado en: {OUTPUT_RASTER}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
        