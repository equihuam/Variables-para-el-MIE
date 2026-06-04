# -*- coding: utf-8 -*-
"""
Construye un CSV tabular a partir de un conjunto de rasters alineados y una serie
de mallas raster. Para cada malla:
- recorta cada raster de variables a la extensión de la malla,
- conserva solo píxeles donde la malla tiene valor no nulo / no NoData,
- extrae x, y y valores por píxel,
- agrega un identificador de malla.

Versión headless para workflow-iie.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(".")
TIFS_DIR = BASE_DIR / "data" / "alineadas_regunidas"
MALLAS_DIR = BASE_DIR / "data" / "malla_reg_unidas"
OUTPUT_CSV = BASE_DIR / "results" / "tables" / "final_data_v3.csv"

# Nombre esperado del raster de malla una vez leído
MALLA_ID_NAME = "OID_1"

# Si True, ignora píxeles donde la malla es 0 además de NoData
IGNORE_ZERO_IN_MALLA = False


# ---------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def list_tifs(folder: Path) -> list[Path]:
    if not folder.exists():
        fail(f"No existe el directorio: {folder}")
    return sorted([p for p in folder.rglob("*.tif") if p.is_file()])


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_pixel_centers(transform, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys)


def read_masked_window(src, bounds):
    """
    Lee una ventana del raster usando los límites dados.
    """
    window = from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()

    data = src.read(1, window=window, masked=True)
    transform = src.window_transform(window)
    return data, transform


def build_valid_mask(malla_arr: np.ma.MaskedArray, nodata) -> np.ndarray:
    """
    Construye la máscara de píxeles válidos a partir de la malla.
    """
    valid = ~np.ma.getmaskarray(malla_arr)

    if nodata is not None:
        valid &= (malla_arr.data != nodata)

    if IGNORE_ZERO_IN_MALLA:
        valid &= (malla_arr.data != 0)

    return valid


def get_malla_id(malla_values: np.ndarray) -> float | int | None:
    # TODO: Confirmar que cada archivo de malla contiene un único ID válido.
    vals = malla_values[~np.isnan(malla_values)] if np.issubdtype(malla_values.dtype, np.floating) else malla_values
    if vals.size == 0:
        return None

    unique_vals = np.unique(vals)
    if unique_vals.size > 1:
        raise RuntimeError(
            "La malla contiene más de un valor válido de ID; revisar la lógica de asignación de mallaid."
        )

    return unique_vals[0]

    

# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def process_one_malla(malla_path: Path, tif_paths: list[Path]) -> pd.DataFrame:
    log(f"\nProcesando malla: {malla_path.name}")

    with rasterio.open(malla_path) as malla_src:
        malla_bounds = malla_src.bounds
        malla_arr = malla_src.read(1, masked=True)
        malla_transform = malla_src.transform
        malla_nodata = malla_src.nodata

        valid_mask = build_valid_mask(malla_arr, malla_nodata)

        if valid_mask.sum() == 0:
            log("  La malla no contiene píxeles válidos; se omite.")
            return pd.DataFrame()

        xs, ys = get_pixel_centers(malla_transform, malla_src.width, malla_src.height)

        xy_df = pd.DataFrame({
            "x": xs[valid_mask],
            "y": ys[valid_mask],
        })

        malla_vals = malla_arr.data[valid_mask]
        malla_id = get_malla_id(malla_vals)

    variable_data = {}

    for tif_path in tif_paths:
        var_name = tif_path.stem
        log(f"  Leyendo variable: {var_name}")

        with rasterio.open(tif_path) as tif_src:
            tif_arr, tif_transform = read_masked_window(tif_src, malla_bounds)

            # Validación mínima de consistencia espacial
            if tif_arr.shape != malla_arr.shape:
                fail(
                    f"El raster {tif_path.name} recortado a {malla_path.name} "
                    f"no coincide en forma con la malla: {tif_arr.shape} vs {malla_arr.shape}"
                )

            variable_data[var_name] = tif_arr.data[valid_mask]

    tif_df = pd.concat([xy_df, pd.DataFrame(variable_data)], axis=1)
    tif_df["mallaid"] = malla_id

    return tif_df


def main() -> int:
    tif_paths = list_tifs(TIFS_DIR)
    malla_paths = list_tifs(MALLAS_DIR)

    if not tif_paths:
        fail(f"No se encontraron TIFF en {TIFS_DIR}")
    if not malla_paths:
        fail(f"No se encontraron mallas TIFF en {MALLAS_DIR}")

    ensure_parent(OUTPUT_CSV)

    final_data_list = []

    for malla_path in malla_paths:
        df = process_one_malla(malla_path, tif_paths)
        if not df.empty:
            final_data_list.append(df)

    if not final_data_list:
        fail("No se generaron datos de salida; todas las mallas quedaron vacías.")

    final_data = pd.concat(final_data_list, ignore_index=True)
    final_data.to_csv(OUTPUT_CSV, index=False)

    log("\n----------------------------------------")
    log("Proceso completado.")
    log(f"CSV guardado en: {OUTPUT_CSV}")
    log(f"Filas generadas: {len(final_data)}")
    log(f"Columnas: {list(final_data.columns)}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)