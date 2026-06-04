# -*- coding: utf-8 -*-
"""
Extrae batimetría desde un raster GEBCO para una tabla de puntos (x, y),
rellena faltantes con vecino espacial más cercano (k=1) y guarda la tabla final.

Traducción headless del script R original.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.neighbors import KNeighborsRegressor


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(".")

INPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v7_batimetria.csv"
REF_RASTER = BASE_DIR / "data" / "regiones_unidas" / "reg_unidas.tif"
BATHY_RASTER = BASE_DIR / "data" / "batimetria_gebco" / "GEBCO_compressed.tif"

OUTPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v8_batimetria.csv"

OLD_BATHY_COL = "batimetria_aligned"
NEW_BATHY_COL = "batimetria_gebco"


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


def sample_raster_at_points(src, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    coords = list(zip(xs.tolist(), ys.tolist()))
    sampled = list(src.sample(coords))
    out = np.array([v[0] if len(v) > 0 else np.nan for v in sampled], dtype=float)

    nodata = src.nodata
    if nodata is not None:
        out[out == nodata] = np.nan

    return out


def build_aligned_bathy_if_needed(ref_path: Path, bathy_path: Path) -> tuple[np.ndarray, dict]:
    """
    Si bathy no coincide en CRS/resolución/malla con ref, lo reproyecta en memoria
    a la malla de referencia.
    """

    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

        with rasterio.open(bathy_path) as bathy:
            same_grid = (
                bathy.crs == ref_crs 
                and bathy.transform == ref_transform
                and bathy.width == ref_width
                and bathy.height == ref_height 
            )

            # TODO: Confirmar si la extracción de batimetría debe hacerse sobre raster
            # completamente alineado a reg_unidas, como aquí, o si basta con reproyección
            # por CRS y recorte por extensión como en
            if same_grid:
                arr = bathy.read(1).astype(np.float32)
                nodata = bathy.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                return arr, {
                    "transform": bathy.transform,
                    "crs": bathy.crs,
                    "nodata": np.nan,
                }

            dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(bathy, 1),
                destination=dst,
                src_transform=bathy.transform,
                src_crs=bathy.crs,
                src_nodata=bathy.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

            return dst, {
                "transform": ref_transform,
                "crs": ref_crs,
                "nodata": np.nan,
            }


def sample_array_at_points(arr: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    values = np.full(xs.shape[0], np.nan, dtype=float)
    for i, (x, y) in enumerate(zip(xs, ys)):
        row, col = rasterio.transform.rowcol(transform, x, y)
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            values[i] = arr[row, col]
    return values


def impute_knn_spatial(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    na_mask = df[target_col].isna()

    if not na_mask.any():
        log("No hay valores faltantes que imputar.")
        return df

    train = df.loc[~na_mask, ["x", "y", target_col]].copy()
    pred = df.loc[na_mask, ["x", "y"]].copy()

    if train.empty:
        fail(f"No hay datos válidos para entrenar la imputación de {target_col}")

    model = KNeighborsRegressor(
        n_neighbors=1,
        weights="uniform",
        metric="minkowski",
        p=2,
    )
    model.fit(train[["x", "y"]], train[target_col])

    predicted = model.predict(pred[["x", "y"]])

    out = df.copy()
    out.loc[na_mask, target_col] = predicted
    return out


# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def main() -> int:
    validate_input_file(INPUT_CSV, "tabla final")
    validate_input_file(REF_RASTER, "raster de referencia")
    validate_input_file(BATHY_RASTER, "raster de batimetría")
    ensure_parent(OUTPUT_CSV)

    log("Leyendo tabla...")
    variables_ie = pd.read_csv(INPUT_CSV)

    required_cols = {"x", "y"}
    if not required_cols.issubset(variables_ie.columns):
        fail("La tabla debe contener columnas x e y.")

    if OLD_BATHY_COL in variables_ie.columns:
        log(f"Eliminando columna previa: {OLD_BATHY_COL}")
        variables_ie = variables_ie.drop(columns=[OLD_BATHY_COL])

    log("Alineando batimetría a la malla de referencia si hace falta...")
    bathy_arr, bathy_meta = build_aligned_bathy_if_needed(REF_RASTER, BATHY_RASTER)

    log("Extrayendo batimetría en puntos x,y...")
    variables_ie[NEW_BATHY_COL] = sample_array_at_points(
        arr=bathy_arr,
        transform=bathy_meta["transform"],
        xs=variables_ie["x"].to_numpy(),
        ys=variables_ie["y"].to_numpy(),
    )

    log("Imputando NA con vecino espacial más cercano (k=1)...")
    variables_ie = impute_knn_spatial(variables_ie, NEW_BATHY_COL)

    log("Guardando tabla final...")
    variables_ie.to_csv(OUTPUT_CSV, index=False, na_rep="")

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