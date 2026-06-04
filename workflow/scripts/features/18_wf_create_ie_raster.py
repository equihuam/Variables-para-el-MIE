#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 18_wf_create_ie_raster.py

Propósito:
    Generar GeoTIFF regionales del índice de integridad ecosistémica (IE)
    a partir de una tabla base espacialmente congruente, un archivo externo
    de predicciones exportado por Netica y una colección de mallas de
    referencia regionales.

Origen:
    Refactorización para workflow basada en:
    - create_ie_raster.py
    - 18_create_ei_rasters.py

Resumen del flujo:
    1. Leer la tabla base con regionid/regionId, x, y.
    2. Leer las predicciones externas de IE exportadas por Netica.
    3. Normalizar las predicciones al rango 0–1.
    4. Construir una grilla fuente regular en EPSG:4326 a partir de los puntos.
    5. Reproyectar la grilla fuente a cada plantilla regional ref_grid.tif.
    6. Escribir un GeoTIFF por región y generar histogramas PNG.
    7. Opcionalmente, exportar una tabla enriquecida con la columna ie.

Insumos principales:
    - bn_input.csv u otra tabla base equivalente
    - predictions.csv externo producido por Netica
    - colección regional de ref_grid.tif

Salidas principales:
    - eicoastal_<region>.tif
    - hist_ie_global.png
    - hist_eicoastal_<region>.png
    - opcionalmente: tabla enriquecida con ie

Supuestos y notas:
    - Las coordenadas x, y de entrada están en EPSG:4326.
    - La salida hereda el CRS y la grilla de cada raster de referencia.
    - Se usa nodata = -9999.0.
    - La correspondencia entre tabla base y predicciones se asume idéntica en
      longitud y orden, igual que en el flujo heredado.
    - Se acepta 'regionid' o 'regionId' como nombre de la columna de región.

Fidelidad de la traducción:
    Esta versión conserva la lógica espacial ya probada en create_ie_raster.py:
    construir una grilla fuente en coordenadas geográficas y hacer warp a la
    plantilla regional, lo que evita desalineaciones respecto a la malla de
    referencia. La principal diferencia respecto a las versiones previas es que
    aquí se parametrizan entradas y salidas para integrarse limpiamente en un
    workflow orquestado.

Observaciones:
    Este script está pensado para ejecución headless y para integrarse en un
    workflow orquestado, por ejemplo con Snakemake. La inferencia bayesiana
    se asume externa al DAG en esta etapa.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.warp import Resampling, reproject


SOURCE_CRS = "EPSG:4326"
NODATA_VALUE = -9999.0
REGION_CANDIDATES = ["regionid", "regionId"]
X_FIELD = "x"
Y_FIELD = "y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruye GeoTIFFs regionales del IE a partir de una tabla base y predicciones externas."
    )
    parser.add_argument(
        "--training-table",
        required=True,
        help="Ruta a la tabla base (.csv o .parquet) con regionid/regionId, x, y.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Ruta al archivo CSV de predicciones externas de IE.",
    )
    parser.add_argument(
        "--ref-grid-dir",
        required=True,
        help="Directorio raíz que contiene subdirectorios regionales con ref_grid.tif.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se escribirán los GeoTIFF finales y los histogramas.",
    )
    parser.add_argument(
        "--region-id",
        required=False,
        default=None,
        help="Si se especifica, procesa solo una región, por ejemplo region_7.",
    )
    parser.add_argument(
        "--output-table",
        required=False,
        default=None,
        help="Ruta opcional (.parquet o .csv) para exportar la tabla enriquecida con la columna ie.",
    )
    parser.add_argument(
        "--prediction-column",
        required=False,
        default=None,
        help="Nombre explícito de la columna de predicción en el CSV externo. "
             "Si no se da, se intentará inferir.",
    )
    parser.add_argument(
        "--normalize-from",
        required=False,
        default="1.5,5.5",
        help="Rango origen para normalizar las predicciones a 0–1, por ejemplo '1.5,5.5'.",
    )
    return parser.parse_args()


def parse_normalization_range(spec: str) -> tuple[float, float]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Formato inválido para --normalize-from: {spec}")
    lo, hi = map(float, parts)
    if hi == lo:
        raise ValueError("El rango de normalización no puede tener extremos iguales.")
    return lo, hi


def normalize_ie_predictions(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normaliza valores del rango [lo, hi] a 0–1."""
    return (values - lo) / (hi - lo)


def infer_region_field(columns: list[str]) -> str:
    for col in REGION_CANDIDATES:
        if col in columns:
            return col
    raise ValueError(
        f"No se encontró columna de región. Se esperaba una de: {REGION_CANDIDATES}"
    )


def load_training_table(path: Path) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, header=0, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Formato no soportado para training table: {path.suffix}. Use .csv o .parquet"
        )

    region_field = infer_region_field(df.columns.tolist())

    required = [region_field, X_FIELD, Y_FIELD]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en training table: {missing}")

    df = df[[region_field, X_FIELD, Y_FIELD]].copy()
    df["idx"] = np.arange(1, len(df) + 1)

    df[X_FIELD] = pd.to_numeric(df[X_FIELD], errors="raise")
    df[Y_FIELD] = pd.to_numeric(df[Y_FIELD], errors="raise")
    df[region_field] = df[region_field].astype(str)

    return df, region_field


def infer_prediction_column(df: pd.DataFrame) -> str:
    preferred = [
        "ie_2026",
        "E[ei_qnint_map]",
        "E[ei_map]",
        "E[ie]",
        "ie",
        "ei",
        "eii",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    if len(df.columns) == 1:
        return df.columns[0]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"No se pudo inferir la columna de predicción. Columnas disponibles: {df.columns.tolist()}"
    )


def load_ie_predictions(path: Path, prediction_column: str | None, lo: float, hi: float) -> np.ndarray:
    ie_pred_df = pd.read_csv(path, header=0, low_memory=False)

    if ie_pred_df.empty:
        raise ValueError(f"El archivo de predicciones de IE está vacío: {path}")

    pred_col = prediction_column or infer_prediction_column(ie_pred_df)
    values = pd.to_numeric(ie_pred_df[pred_col], errors="raise").to_numpy(dtype=float)
    return normalize_ie_predictions(values, lo, hi)


def build_reference_grid_map(ref_grid_dir: Path) -> dict[str, Path]:
    """
    Crea un mapa region -> ref_grid.tif usando el nombre del subdirectorio.
    """
    mapping: dict[str, Path] = {}

    for region_dir in sorted(ref_grid_dir.iterdir()):
        if not region_dir.is_dir():
            continue

        tif_path = region_dir / "ref_grid.tif"
        if not tif_path.exists():
            continue

        mapping[region_dir.name.strip().lower()] = tif_path

    if not mapping:
        raise ValueError(f"No se encontraron rasters de referencia en {ref_grid_dir}")

    return mapping


def find_reference_raster(region: str, mapping: dict[str, Path]) -> Path:
    region_key = str(region).strip().lower()

    if region_key not in mapping:
        raise FileNotFoundError(
            f"No encontré raster de referencia para la región '{region}'. "
            f"Disponibles: {list(mapping.keys())}"
        )

    return mapping[region_key]


def sort_regions_numerically(regions: list[str]) -> list[str]:
    return sorted(regions, key=lambda s: int(str(s).split("_")[-1]))


def save_histogram(
        values: np.ndarray,
        output_png: Path,
        title: str,
        xlabel: str = "IE normalizado",
        bins: int = 50,
) -> None:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def save_array_histogram(
        arr: np.ndarray,
        output_png: Path,
        title: str,
        nodata_value: float = NODATA_VALUE,
        bins: int = 50,
) -> None:
    valid = arr[np.isfinite(arr) & (arr != nodata_value)]
    if valid.size == 0:
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(valid, bins=bins)
    plt.title(title)
    plt.xlabel("IE normalizado")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def build_source_grid_from_points(
        region_df: pd.DataFrame,
        ie_pred_values: np.ndarray,
        nodata_value: float = NODATA_VALUE,
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Construye una grilla raster en EPSG:4326 a partir de puntos x, y, z.
    La resolución se infiere del espaciamiento mínimo positivo de x e y.
    Si varias observaciones caen en la misma celda, promedia sus valores.
    """
    idx0 = region_df["idx"].to_numpy() - 1
    z = ie_pred_values[idx0]

    xs = region_df[X_FIELD].to_numpy(dtype=float)
    ys = region_df[Y_FIELD].to_numpy(dtype=float)

    if len(xs) == 0:
        raise ValueError("No hay puntos para construir la grilla fuente.")

    unique_x = np.sort(np.unique(xs))
    unique_y = np.sort(np.unique(ys))

    if len(unique_x) < 2 or len(unique_y) < 2:
        raise ValueError("No hay suficientes coordenadas únicas para inferir resolución de grilla.")

    dx_candidates = np.diff(unique_x)
    dy_candidates = np.diff(unique_y)

    dx_candidates = dx_candidates[dx_candidates > 0]
    dy_candidates = dy_candidates[dy_candidates > 0]

    if len(dx_candidates) == 0 or len(dy_candidates) == 0:
        raise ValueError("No fue posible inferir resolución positiva de la grilla fuente.")

    dx = float(dx_candidates.min())
    dy = float(dy_candidates.min())

    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())

    width = int(round((xmax - xmin) / dx)) + 1
    height = int(round((ymax - ymin) / dy)) + 1

    transform = from_origin(
        west=xmin - dx / 2.0,
        north=ymax + dy / 2.0,
        xsize=dx,
        ysize=dy,
    )

    sums = np.zeros((height, width), dtype="float64")
    counts = np.zeros((height, width), dtype="uint32")

    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    z_valid = z[valid_mask]

    np.add.at(sums, (rows, cols), z_valid)
    np.add.at(counts, (rows, cols), 1)

    source_array = np.full((height, width), nodata_value, dtype="float32")
    mask = counts > 0
    source_array[mask] = (sums[mask] / counts[mask]).astype("float32")

    return source_array, transform


def warp_source_to_template(
        source_array: np.ndarray,
        source_transform,
        template_path: Path,
        output_path: Path,
        histogram_png_path: Path | None = None,
        histogram_title: str | None = None,
        source_crs: str = SOURCE_CRS,
        nodata_value: float = NODATA_VALUE,
) -> tuple[int, float | None, float | None]:
    """
    Reproyecta una grilla fuente a la plantilla usando nearest neighbour.
    Devuelve cantidad de celdas válidas y rango de valores válidos.
    """
    with rasterio.open(template_path) as src:
        meta = src.meta.copy()
        template_transform = src.transform
        template_crs = src.crs
        height = src.height
        width = src.width

        if template_crs is None:
            raise ValueError(f"El raster plantilla no tiene CRS: {template_path}")

        dest_array = np.full((height, width), nodata_value, dtype="float32")

        reproject(
            source=source_array,
            destination=dest_array,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=nodata_value,
            dst_transform=template_transform,
            dst_crs=template_crs,
            dst_nodata=nodata_value,
            resampling=Resampling.nearest,
        )

        valid = dest_array[dest_array != nodata_value]
        valid_count = int(valid.size)
        min_valid = float(valid.min()) if valid_count > 0 else None
        max_valid = float(valid.max()) if valid_count > 0 else None

        meta.update(
            dtype="float32",
            count=1,
            nodata=nodata_value,
            compress="lzw",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            output_path.unlink()

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(dest_array, 1)

    if histogram_png_path is not None:
        save_array_histogram(
            dest_array,
            output_png=histogram_png_path,
            title=histogram_title or output_path.stem,
            nodata_value=nodata_value,
        )

    return valid_count, min_valid, max_valid


def rasterize_points_to_template(
        region_df: pd.DataFrame,
        ie_pred_values: np.ndarray,
        template_path: Path,
        output_path: Path,
        histogram_png_path: Path | None = None,
        histogram_title: str | None = None,
        source_crs: str = SOURCE_CRS,
        nodata_value: float = NODATA_VALUE,
) -> tuple[int, float | None, float | None]:
    """
    Flujo completo:
    1. arma raster fuente en CRS original de puntos
    2. lo reproyecta a la plantilla
    """
    source_array, source_transform = build_source_grid_from_points(
        region_df=region_df,
        ie_pred_values=ie_pred_values,
        nodata_value=nodata_value,
    )

    return warp_source_to_template(
        source_array=source_array,
        source_transform=source_transform,
        template_path=template_path,
        output_path=output_path,
        histogram_png_path=histogram_png_path,
        histogram_title=histogram_title,
        source_crs=source_crs,
        nodata_value=nodata_value,
    )


def save_enriched_table(df: pd.DataFrame, ie_pred_values: np.ndarray, output_table: Path) -> None:
    enriched = df.copy()
    enriched["ie"] = ie_pred_values

    output_table.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_table.suffix.lower()
    if suffix == ".parquet":
        enriched.to_parquet(output_table, index=False, engine="pyarrow")
    elif suffix == ".csv":
        enriched.to_csv(output_table, index=False)
    else:
        raise ValueError(
            f"Formato no soportado para output_table: {output_table.suffix}. Use .parquet o .csv"
        )


def main() -> None:
    t_inicio = time.time()
    args = parse_args()

    training_table = Path(args.training_table)
    predictions_csv = Path(args.predictions)
    ref_grid_dir = Path(args.ref_grid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lo, hi = parse_normalization_range(args.normalize_from)

    df, region_field = load_training_table(training_table)
    ie_pred_values = load_ie_predictions(predictions_csv, args.prediction_column, lo, hi)
    ref_map = build_reference_grid_map(ref_grid_dir)

    if df["idx"].max() > len(ie_pred_values):
        raise ValueError(
            f"El máximo idx ({df['idx'].max()}) excede el número de predicciones IE ({len(ie_pred_values)})"
        )

    save_histogram(
        ie_pred_values,
        output_dir / "hist_ie_global.png",
        title="Histograma global de IE normalizado",
        )

    if args.output_table:
        save_enriched_table(df, ie_pred_values, Path(args.output_table))

    region_groups = {
        region: group.copy()
        for region, group in df.groupby(region_field, sort=False)
    }

    regiones = sort_regions_numerically(list(region_groups.keys()))

    if args.region_id is not None:
        target = str(args.region_id).strip()
        if target not in region_groups:
            raise KeyError(
                f"La región solicitada '{target}' no está en la tabla base. "
                f"Disponibles: {regiones}"
            )
        regiones = [target]

    for region in regiones:
        region_df = region_groups[region]
        template_path = find_reference_raster(region, ref_map)
        output_path = output_dir / f"eicoastal_{region}.tif"
        hist_path = output_dir / f"hist_eicoastal_{region}.png"

        valid_count, min_valid, max_valid = rasterize_points_to_template(
            region_df=region_df,
            ie_pred_values=ie_pred_values,
            template_path=template_path,
            output_path=output_path,
            histogram_png_path=hist_path,
            histogram_title=f"Histograma IE {region}",
            source_crs=SOURCE_CRS,
            nodata_value=NODATA_VALUE,
        )

        print(
            f"{region}: {valid_count} celdas válidas, "
            f"rango [{min_valid}, {max_valid}] -> {output_path}"
        )

    print(f"✅ Script 18_wf_create_ie_raster terminado en {time.time() - t_inicio:.2f}s")


if __name__ == "__main__":
    main()