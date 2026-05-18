# -*- coding: utf-8 -*-
"""
Imputa valores faltantes de tasas de erosión mediante vecino espacial más cercano
(k = 1 sobre coordenadas x, y), luego rasteriza el resultado sobre la malla
de referencia definida por reg_unidas.

Traducción headless del script R original.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from sklearn.neighbors import KNeighborsRegressor


# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(".")
INPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v3_completa.csv"
REF_RASTER = BASE_DIR / "data" / "regiones_unidas" / "reg_unidas.tif"

OUTPUT_RASTER = BASE_DIR / "data" / "prueba_erosion_3.tif"
OUTPUT_CSV = BASE_DIR / "data" / "tablas_finales" / "final_data_v2.csv"

TARGET_COL = "tasaserosion2_aligned"
DROP_COLS_BY_INDEX = [5, 6, 7, 8, 9]   # equivalente a -(6:10) en R, base 1
NODATA_VALUE = -9999.0


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


def drop_columns_by_zero_based_index(df: pd.DataFrame, idxs: list[int]) -> pd.DataFrame:
    cols = df.columns.tolist()
    to_drop = [cols[i] for i in idxs if i < len(cols)]
    return df.drop(columns=to_drop)


def build_template_profile(ref_path: Path) -> dict:
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
        return {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "profile": profile,
        }


def rasterize_points_to_template_mean(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    ref_path: Path,
    out_path: Path,
    nodata_value: float = NODATA_VALUE,
) -> None:
    """
    Rasteriza una tabla de puntos asignando a cada celda el promedio de los valores
    que caen en ella.
    """
    with rasterio.open(ref_path) as ref:
        height = ref.height
        width = ref.width
        transform = ref.transform
        crs = ref.crs
        profile = ref.profile.copy()

    sums = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int32)

    for row in df.itertuples(index=False):
        x = getattr(row, x_col)
        y = getattr(row, y_col)
        value = getattr(row, value_col)

        if pd.isna(x) or pd.isna(y) or pd.isna(value):
            continue

        r, c = rasterio.transform.rowcol(transform, x, y)
        if 0 <= r < height and 0 <= c < width:
            sums[r, c] += float(value)
            counts[r, c] += 1

    out_arr = np.full((height, width), nodata_value, dtype=np.float32)
    mask = counts > 0
    out_arr[mask] = (sums[mask] / counts[mask]).astype(np.float32)

    profile.update(
        dtype="float32",
        count=1,
        nodata=nodata_value,
        compress="lzw"
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)


# ---------------------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------------------

def impute_erosion_knn(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # TODO: Verificar si la imputación de tasas de erosión debe seguir siendo
    # estrictamente espacial (x, y) o si en una versión futura conviene incorporar
    # covariables ambientales como predictores.
    
    if target_col not in df.columns:
        fail(f"La columna objetivo no existe: {target_col}")

    if not {"x", "y"}.issubset(df.columns):
        fail("La tabla debe contener columnas x e y.")

    na_mask = df[target_col].isna()

    train = df.loc[~na_mask, ["x", "y", target_col]].copy()
    pred = df.loc[na_mask, ["x", "y"]].copy()

    if train.empty:
        fail(f"No hay datos de entrenamiento con valores válidos en {target_col}")

    if pred.empty:
        log("No hay valores faltantes que imputar.")
        return df

    model = KNeighborsRegressor(
        n_neighbors=1,
        weights="uniform",
        metric="minkowski",
        p=2
    )
    model.fit(train[["x", "y"]], train[target_col])

    predicted = model.predict(pred[["x", "y"]])
    out = df.copy()
    out.loc[na_mask, target_col] = predicted

    return out


def main() -> int:
    validate_input_file(INPUT_CSV, "tabla final")
    validate_input_file(REF_RASTER, "raster de referencia")

    ensure_parent(OUTPUT_RASTER)
    ensure_parent(OUTPUT_CSV)

    log("Leyendo tabla...")
    variables_ie = pd.read_csv(INPUT_CSV)

    if TARGET_COL not in variables_ie.columns:
        fail(f"No existe la columna {TARGET_COL} en {INPUT_CSV}")

    # TODO: Reemplazar la eliminación por índices de columna por una lista explícita
    # de nombres; la lógica actual depende del orden del CSV de entrada.
    log("Eliminando columnas 6:10 del script original...")
    variables_ie = drop_columns_by_zero_based_index(variables_ie, DROP_COLS_BY_INDEX)

    log("Imputando valores faltantes con k-NN espacial (k=1)...")
    variables_ie = impute_erosion_knn(variables_ie, TARGET_COL)

    log("Rasterizando resultado sobre la malla de referencia...")
    rasterize_points_to_template_mean(
        df=variables_ie,
        x_col="x",
        y_col="y",
        value_col=TARGET_COL,
        ref_path=REF_RASTER,
        out_path=OUTPUT_RASTER,
        nodata_value=NODATA_VALUE,
    )

    log("Guardando tabla final...")
    variables_ie.to_csv(OUTPUT_CSV, index=False)

    log("\n----------------------------------------")
    log("Proceso completado.")
    log(f"Raster guardado en: {OUTPUT_RASTER}")
    log(f"CSV guardado en: {OUTPUT_CSV}")
    log("----------------------------------------")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)