# -*- coding: utf-8 -*-
"""
Agrega a una tabla de puntos (x, y) valores extraídos desde múltiples rasters
de distancia a estructuras. Crea una columna por raster.

Traducción headless del script R original.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(".")

INPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v8_batimetria.csv"
REF_RASTER = BASE_DIR / "data" / "regiones_unidas" / "reg_unidas.tif"
ESTRUCT_DIR = BASE_DIR / "data" / "estructuras_separadas" / "dist_estruct_compress"

OUTPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v9_3.csv"

OLD_COL = "estruct_dist_dunas_aligned"


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


def validate_input_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        fail(f"No existe el directorio de entrada ({label}): {path}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_tifs(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*.tif") if p.is_file()])


def sample_array_at_points(arr: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    values = np.full(xs.shape[0], np.nan, dtype=float)
    for i, (x, y) in enumerate(zip(xs, ys)):
        row, col = rasterio.transform.rowcol(transform, x, y)
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            values[i] = arr[row, col]
    return values


def align_raster_to_reference_if_needed(ref_path: Path, src_path: Path) -> tuple[np.ndarray, dict]:
    """
    Si el raster de entrada no coincide con la malla de referencia, lo reproyecta
    en memoria a la malla exacta de reg_unidas usando vecino más cercano.
    """
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

        with rasterio.open(src_path) as src:
            same_grid = (
                src.crs == ref_crs
                and src.transform == ref_transform
                and src.width == ref_width
                and src.height == ref_height
            )

            # TODO: Confirmar si todas las capas de distancia a estructuras deben estar
            # previamente alineadas a reg_unidas. Si esa precondición se garantiza upstream,
            # esta reproyección en memoria podría simplificarse.
            if same_grid:
                arr = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                return arr, {
                    "transform": src.transform,
                    "crs": src.crs,
                }

            dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )

            return dst, {
                "transform": ref_transform,
                "crs": ref_crs,
            }


# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def add_structure_distance_columns(
    df: pd.DataFrame,
    ref_raster: Path,
    estruct_files: list[Path],
) -> pd.DataFrame:
    if not {"x", "y"}.issubset(df.columns):
        fail("La tabla debe contener columnas x e y.")

    out = df.copy()
    xs = out["x"].to_numpy()
    ys = out["y"].to_numpy()

    for f in estruct_files:
        colname = f.stem
        log(f"\nProcesando raster: {f.name}")

        arr, meta = align_raster_to_reference_if_needed(ref_raster, f)
        vals = sample_array_at_points(arr, meta["transform"], xs, ys)
        out[colname] = vals

        na_count = int(np.isnan(vals).sum())
        non_na_count = int((~np.isnan(vals)).sum())

        # TODO: Definir si columnas completamente vacías tras la extracción deben
        # disparar advertencia o error de validación formal.
        if non_na_count == 0:
            log(f"  Advertencia: la columna {colname} quedó completamente vacía.")

        log(f"  Valores no NA: {non_na_count}")
        log(f"  Valores NA   : {na_count}")
        log(f"  Valores no NA: {non_na_count}")
        log(f"  Valores NA   : {na_count}")

    return out


def main() -> int:
    validate_input_file(INPUT_CSV, "tabla final")
    validate_input_file(REF_RASTER, "raster de referencia")
    validate_input_dir(ESTRUCT_DIR, "directorio de rasters de estructuras")
    ensure_parent(OUTPUT_CSV)

    estruct_files = list_tifs(ESTRUCT_DIR)

    log("Rasters encontrados:")
    for f in estruct_files:
        log(f"  - {f}")
    log(f"Total de rasters: {len(estruct_files)}")

    if len(estruct_files) == 0:
        fail(f"No se encontraron archivos .tif en {ESTRUCT_DIR}")

    log("\nLeyendo tabla...")
    df = pd.read_csv(INPUT_CSV)

    if OLD_COL in df.columns:
        log(f"Eliminando columna previa: {OLD_COL}")
        df = df.drop(columns=[OLD_COL])

    log("Extrayendo valores de rasters de estructuras...")
    df = add_structure_distance_columns(
        df=df,
        ref_raster=REF_RASTER,
        estruct_files=estruct_files,
    )

    log("\nGuardando CSV final...")
    df.to_csv(OUTPUT_CSV, index=False)

    log("\n----------------------------------------")
    log("Proceso completado.")
    log(f"CSV guardado en: {OUTPUT_CSV}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)