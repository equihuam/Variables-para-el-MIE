#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 1_wf_features_tasa_erosion.py

Propósito:
    Calcular, para una región específica, la variable de tasa de erosión sobre
    los píxeles válidos de un raster de referencia y exportarla como tabla
    tabular congruente por píxel.

Origen:
    Refactorización para workflow de la traducción inicial a Python del script R:
    1_features_tasa_erosion.R

Resumen del flujo:
    1. Leer la tabla de tasas de erosión.
    2. Leer un raster de referencia regional.
    3. Reproyectar la plantilla regional al CRS de los puntos de erosión.
    4. Extraer los centros de píxel solo para celdas válidas.
    5. Estimar la erosión en cada píxel válido mediante una emulación de kknn de R.
    6. Exportar la tabla resultante en Parquet.

Modo diagnóstico opcional:
    El script es compatible con Snakemake sin argumentos de depuración.
    Opcionalmente exporta una tabla de grilla con ref_value y metadatos de
    reproyección para comparar contra la salida de referencia generada en R.

Modo kNN canónico:
    Por defecto usa una emulación de kknn::kknn(Tasa ~ x + y, ...,
    k = 3, distance = 2, kernel = "optimal", scale = TRUE).

Insumos principales:
    - tabla de tasas de erosión
    - ref_grid.tif regional

Salidas principales:
    - tabla serializada con columnas:
      regionid, pixid, x, y, erosion
    - opcional: tabla diagnóstica de grilla con columnas:
      regionid, pixid, x, y, ref_value
    - opcional: metadatos diagnósticos de raster/grilla en CSV, incluyendo
      conteos antes y después de reproyectar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.neighbors import NearestNeighbors


POINTS_CRS = "EPSG:4326"
TARGET_FIELD = "Tasa"
K_NEIGHBORS = 3
DISTANCE_POWER = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula tasa de erosión por píxel para una región específica."
    )
    parser.add_argument(
        "--erosion-table",
        required=True,
        help="Ruta a la tabla de tasas de erosión.",
    )
    parser.add_argument(
        "--ref-grid",
        required=True,
        help="Ruta al ref_grid.tif de la región.",
    )
    parser.add_argument(
        "--region-id",
        required=True,
        help="Identificador de la región, por ejemplo region_7.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet.",
    )
    parser.add_argument(
        "--debug-grid-output",
        default=None,
        help=(
            "Ruta opcional para exportar la grilla diagnóstica con ref_value. "
            "Acepta .parquet o .csv."
        ),
    )
    parser.add_argument(
        "--debug-metadata-output",
        default=None,
        help=(
            "Ruta opcional para exportar metadatos diagnósticos de la grilla. "
            "Acepta .csv."
        ),
    )
    parser.add_argument(
        "--validity-mode",
        choices=["finite", "mask"],
        default="finite",
        help=(
            "Criterio para seleccionar pixeles de la grilla reproyectada. "
            "finite: conserva sólo valores finitos del raster reproyectado. "
            "mask: conserva pixeles válidos según la máscara GDAL reproyectada. "
            "Use mask para aproximar mejor el comportamiento de terra/as.data.frame "
            "cuando hay discrepancias de NoData."
        ),
    )
    parser.add_argument(
        "--knn-mode",
        choices=["kknn_optimal", "idw_legacy"],
        default="kknn_optimal",
        help=(
            "Método de predicción. kknn_optimal emula kknn de R con "
            "k=3, distance=2, kernel='optimal', scale=TRUE. "
            "idw_legacy conserva el promedio ponderado 1/distancia^2 de versiones previas."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Imprime diagnósticos de grilla y kNN en consola. "
            "Por defecto el script sólo imprime la confirmación final, "
            "para mantener limpia la salida de Snakemake."
        ),
    )
    return parser.parse_args()


def validate_inputs(erosion_table: Path, ref_grid: Path) -> None:
    missing = [str(p) for p in [erosion_table, ref_grid] if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def source_raster_counts(src: rasterio.io.DatasetReader) -> dict[str, int | float | str | None]:
    """Calcula una firma rápida del GeoTIFF original, antes de reproyectar."""
    arr = src.read(1, masked=True)
    valid_mask = ~arr.mask
    data = arr.filled(np.nan).astype(float)
    read_mask = src.read_masks(1) > 0

    return {
        "src_total_cells": int(src.width * src.height),
        "src_valid_masked_points": int(valid_mask.sum()),
        "src_finite_points": int(np.isfinite(data).sum()),
        "src_non_nan_points": int((~np.isnan(data)).sum()),
        "src_gdal_mask_valid_points": int(read_mask.sum()),
        "src_dtype": str(src.dtypes[0]),
        "src_count": int(src.count),
    }


def load_tasa_erosion(path: Path) -> pd.DataFrame:
    tasa_ero = pd.read_csv(path, sep=",", header=0, low_memory=False)

    if tasa_ero.shape[1] < 3:
        raise ValueError("La tabla de tasa de erosión no tiene al menos 3 columnas.")

    cols = list(tasa_ero.columns)
    cols[1] = "x"
    cols[2] = "y"
    tasa_ero.columns = cols

    required = {"x", "y", TARGET_FIELD}
    missing = required - set(tasa_ero.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en tasa_erosion: {missing}")

    tasa_ero["x"] = pd.to_numeric(tasa_ero["x"], errors="raise")
    tasa_ero["y"] = pd.to_numeric(tasa_ero["y"], errors="raise")
    tasa_ero[TARGET_FIELD] = pd.to_numeric(tasa_ero[TARGET_FIELD], errors="raise")

    return tasa_ero


def reproject_raster_to_epsg4326(
    src: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs,
        POINTS_CRS,
        src.width,
        src.height,
        *src.bounds,
    )

    dst = np.full((height, width), np.nan, dtype=np.float32)

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

    src_mask = src.read_masks(1).astype(np.uint8)
    dst_mask = np.zeros((height, width), dtype=np.uint8)

    reproject(
        source=src_mask,
        destination=dst_mask,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=0,
        dst_transform=transform,
        dst_crs=POINTS_CRS,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    return dst, dst_mask, transform


def valid_raster_points_dataframe(
    arr: np.ndarray,
    transform,
    validity_mode: str = "finite",
    mask_arr: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Construye tabla xy solo para celdas válidas del raster reproyectado.

    La columna ref_value conserva el valor de la plantilla/máscara de referencia
    después de reproyectar. Es útil para comparar contra la tabla generada por R.
    """
    if validity_mode == "finite":
        valid_mask = np.isfinite(arr)
    elif validity_mode == "mask":
        if mask_arr is None:
            raise ValueError("validity_mode='mask' requiere mask_arr.")
        valid_mask = mask_arr > 0
    else:
        raise ValueError(f"Modo de validez no reconocido: {validity_mode}")
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", "ref_value"])

    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "row": rows.astype(np.int64),
            "col": cols.astype(np.int64),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "ref_value": arr[rows, cols].astype(float),
        }
    )


def build_grid_metadata(
    *,
    region_id: str,
    ref_grid_path: Path,
    src: rasterio.io.DatasetReader,
    region_arr: np.ndarray,
    region_transform: rasterio.Affine,
    valid_points: int,
    validity_mode: str,
    mask_valid_points: int,
    finite_points: int,
    non_nan_points: int,
    source_counts: dict[str, int | float | str | None] | None = None,
) -> pd.DataFrame:
    """Devuelve metadatos compactos para diagnosticar diferencias de grilla."""
    height, width = region_arr.shape
    left, bottom, right, top = rasterio.transform.array_bounds(
        height, width, region_transform
    )

    src_bounds = src.bounds
    row = {
                "regionid": region_id,
                "ref_grid": str(ref_grid_path),
                "src_crs": str(src.crs),
                "dst_crs": POINTS_CRS,
                "src_width": src.width,
                "src_height": src.height,
                "dst_width": width,
                "dst_height": height,
                "total_points": int(region_arr.size),
                "validity_mode": validity_mode,
                "valid_points": int(valid_points),
                "finite_points": int(finite_points),
                "non_nan_points": int(non_nan_points),
                "mask_valid_points": int(mask_valid_points),
                "src_nodata": src.nodata,
                "src_left": src_bounds.left,
                "src_bottom": src_bounds.bottom,
                "src_right": src_bounds.right,
                "src_top": src_bounds.top,
                "dst_left": left,
                "dst_bottom": bottom,
                "dst_right": right,
                "dst_top": top,
                "dst_res_x": region_transform.a,
                "dst_res_y": region_transform.e,
                "dst_transform": repr(region_transform),
                "valid_min_x": np.nan,
                "valid_max_x": np.nan,
                "valid_min_y": np.nan,
                "valid_max_y": np.nan,
            }
    if source_counts:
        row.update(source_counts)
    return pd.DataFrame([row])


def update_metadata_with_valid_ranges(
    metadata: pd.DataFrame,
    region_points: pd.DataFrame,
) -> pd.DataFrame:
    metadata = metadata.copy()
    if len(region_points) > 0:
        metadata.loc[0, "valid_min_x"] = region_points["x"].min()
        metadata.loc[0, "valid_max_x"] = region_points["x"].max()
        metadata.loc[0, "valid_min_y"] = region_points["y"].min()
        metadata.loc[0, "valid_max_y"] = region_points["y"].max()
    return metadata


def kknn_optimal_kernel_weights(k: int, d: int = 2) -> np.ndarray:
    """
    Replica optKernel(k, d) del paquete R kknn.

    Fórmula en R:
        1/k * (1 + d/2 - d/(2*k^(2/d)) *
               ((1:k)^(1+2/d) - (0:(k-1))^(1+2/d)))

    Para Tasa ~ x + y, d = 2.
    """
    ranks = np.arange(1, k + 1, dtype=float)
    prev = np.arange(0, k, dtype=float)
    weights = (1.0 / k) * (
        1.0
        + d / 2.0
        - d / (2.0 * (k ** (2.0 / d)))
        * (ranks ** (1.0 + 2.0 / d) - prev ** (1.0 + 2.0 / d))
    )
    return weights.astype(float)


def scale_like_kknn_train_valid(
    x_train: np.ndarray,
    x_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replica el escalamiento básico de kknn(scale=TRUE) para predictores numéricos.

    kknn calcula varianza muestral en el conjunto de entrenamiento y divide tanto
    train como test entre sqrt(var). No centra las variables; sólo escala por SD.
    Para variables con varianza cero usa divisor 1.
    """
    if x_train.ndim != 2 or x_pred.ndim != 2:
        raise ValueError("x_train y x_pred deben ser matrices 2D.")

    # R stats::var usa varianza muestral: ddof=1.
    if x_train.shape[0] > 1:
        sd = np.nanstd(x_train, axis=0, ddof=1)
    else:
        sd = np.ones(x_train.shape[1], dtype=float)

    sd = np.asarray(sd, dtype=float)
    sd[(~np.isfinite(sd)) | (sd == 0)] = 1.0

    return x_train / sd, x_pred / sd, sd


def predict_knn_idw_legacy(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    target_col: str = TARGET_FIELD,
    k: int = K_NEIGHBORS,
    distance_power: float = DISTANCE_POWER,
) -> np.ndarray:
    """Predicción heredada: pesos 1/distancia^distance_power, sin escala kknn."""
    x_train = train_df[[x_col, y_col]].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    x_pred = pred_df[[x_col, y_col]].to_numpy(dtype=float)

    if len(train_df) == 0:
        return np.full(len(pred_df), np.nan, dtype=float)

    k_eff = min(k, len(train_df))

    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", metric="euclidean")
    nn.fit(x_train)

    distances, indices = nn.kneighbors(x_pred, return_distance=True)

    zero_mask = distances == 0
    weights = np.zeros_like(distances, dtype=float)
    weights[~zero_mask] = 1.0 / np.power(distances[~zero_mask], distance_power)

    any_zero = zero_mask.any(axis=1)
    if np.any(any_zero):
        weights[any_zero] = zero_mask[any_zero].astype(float)

    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0

    neighbor_values = y_train[indices]
    predictions = np.sum(weights * neighbor_values, axis=1) / weight_sums[:, 0]

    return predictions


def predict_knn_kknn_optimal(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    target_col: str = TARGET_FIELD,
    k: int = K_NEIGHBORS,
    distance: int = 2,
    kernel_dimension: int = 2,
) -> np.ndarray:
    """
    Emula el caso usado en R:

        kknn(Tasa ~ x + y, tasa_ero, region_points,
             distance = 2, k = 3, kernel = "optimal", scale = TRUE)

    Aspectos replicados:
      - usa x,y como predictores numéricos;
      - escala train/test dividiendo entre SD muestral del train;
      - busca vecinos en distancia euclidiana cuando distance=2;
      - usa pesos de rango optKernel(k, d=2), independientes de la distancia;
      - predicción continua = sum(W * valores_vecinos) / sum(W).

    Nota: puede haber diferencias residuales por empates en vecinos y por la
    implementación interna de búsqueda de vecinos respecto a kknn en R.
    """
    if distance != 2:
        raise NotImplementedError("Esta emulación sólo implementa distance=2.")

    x_train = train_df[[x_col, y_col]].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    x_pred = pred_df[[x_col, y_col]].to_numpy(dtype=float)

    if len(train_df) == 0:
        return np.full(len(pred_df), np.nan, dtype=float)
    if len(pred_df) == 0:
        return np.empty(0, dtype=float)

    valid_train = (
        np.isfinite(x_train).all(axis=1)
        & np.isfinite(y_train)
    )
    x_train = x_train[valid_train]
    y_train = y_train[valid_train]

    if len(y_train) == 0:
        return np.full(len(pred_df), np.nan, dtype=float)

    k_eff = min(k, len(y_train))

    x_train_scaled, x_pred_scaled, _sd = scale_like_kknn_train_valid(x_train, x_pred)

    # kknn calcula internamente k+1 distancias para obtener maxdist y luego usa
    # los primeros k vecinos. Para kernel="optimal", maxdist no afecta los pesos,
    # pero pedir k+1 puede hacer más comparable el orden interno en casos de borde.
    n_neighbors_query = min(k_eff + 1, len(y_train))
    nn = NearestNeighbors(
        n_neighbors=n_neighbors_query,
        algorithm="auto",
        metric="euclidean",
    )
    nn.fit(x_train_scaled)

    _distances, indices_all = nn.kneighbors(x_pred_scaled, return_distance=True)
    indices = indices_all[:, :k_eff]

    weights = kknn_optimal_kernel_weights(k_eff, d=kernel_dimension)
    neighbor_values = y_train[indices]

    predictions = np.sum(neighbor_values * weights.reshape(1, -1), axis=1) / np.sum(weights)
    return predictions


def predict_erosion(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    if mode == "kknn_optimal":
        return predict_knn_kknn_optimal(train_df=train_df, pred_df=pred_df)
    if mode == "idw_legacy":
        return predict_knn_idw_legacy(train_df=train_df, pred_df=pred_df)
    raise ValueError(f"Modo kNN no reconocido: {mode}")


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow")
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Extensión no soportada para {output_path}. Use .parquet o .csv."
        )


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Este script espera salida .parquet y recibió: {output_path.suffix}"
        )
    save_table(df, output_path)


def main() -> None:
    args = parse_args()

    erosion_table_path = Path(args.erosion_table)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    debug_grid_output = Path(args.debug_grid_output) if args.debug_grid_output else None
    debug_metadata_output = (
        Path(args.debug_metadata_output) if args.debug_metadata_output else None
    )
    region_id = str(args.region_id).strip()

    validate_inputs(erosion_table_path, ref_grid_path)

    tasa_ero = load_tasa_erosion(erosion_table_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        source_counts = source_raster_counts(src)
        region_arr, region_mask, region_transform = reproject_raster_to_epsg4326(src)
        finite_points = int(np.isfinite(region_arr).sum())
        non_nan_points = int((~np.isnan(region_arr)).sum())
        mask_valid_points = int((region_mask > 0).sum())
        region_points = valid_raster_points_dataframe(
            region_arr,
            region_transform,
            validity_mode=args.validity_mode,
            mask_arr=region_mask,
        )
        grid_metadata = build_grid_metadata(
            region_id=region_id,
            ref_grid_path=ref_grid_path,
            src=src,
            region_arr=region_arr,
            region_transform=region_transform,
            valid_points=len(region_points),
            validity_mode=args.validity_mode,
            mask_valid_points=mask_valid_points,
            finite_points=finite_points,
            non_nan_points=non_nan_points,
            source_counts=source_counts,
        )

    grid_metadata = update_metadata_with_valid_ranges(grid_metadata, region_points)
    grid_metadata.loc[0, "knn_mode"] = args.knn_mode
    grid_metadata.loc[0, "knn_k"] = K_NEIGHBORS
    grid_metadata.loc[0, "knn_distance"] = DISTANCE_POWER
    grid_metadata.loc[0, "knn_kernel"] = "optimal" if args.knn_mode == "kknn_optimal" else "idw"
    grid_metadata.loc[0, "knn_scale_like_r"] = args.knn_mode == "kknn_optimal"
    if args.knn_mode == "kknn_optimal":
        grid_metadata.loc[0, "knn_optimal_weights"] = ";".join(
            str(v) for v in kknn_optimal_kernel_weights(K_NEIGHBORS, d=2)
        )

    total_points = int(region_arr.size)
    if args.verbose:
        print(f"total puntos GeoTIFF original: {source_counts['src_total_cells']}")
        print(
            "puntos válidos GeoTIFF original masked: "
            f"{source_counts['src_valid_masked_points']}"
        )
        print(f"puntos finitos GeoTIFF original: {source_counts['src_finite_points']}")
        print(
            "puntos válidos máscara GDAL original: "
            f"{source_counts['src_gdal_mask_valid_points']}"
        )
        print(f"total puntos reproyectados: {total_points}")
        print(f"modo de validez: {args.validity_mode}")
        print(f"puntos finitos: {finite_points}")
        print(f"puntos no-NaN: {non_nan_points}")
        print(f"puntos válidos por máscara GDAL: {mask_valid_points}")
        print(f"puntos válidos usados en malla: {len(region_points)}")
        if len(region_points) > 0:
            print(
                "rango x/y válido: "
                f"x=[{region_points['x'].min()}, {region_points['x'].max()}], "
                f"y=[{region_points['y'].min()}, {region_points['y'].max()}]"
            )

        print(f"modo kNN: {args.knn_mode}")
        if args.knn_mode == "kknn_optimal":
            print(
                "pesos kernel optimal k=3,d=2: "
                f"{kknn_optimal_kernel_weights(K_NEIGHBORS, d=2).tolist()}"
            )

    predictions = predict_erosion(
        train_df=tasa_ero,
        pred_df=region_points,
        mode=args.knn_mode,
    )

    pixid = np.arange(1, len(region_points) + 1)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": pixid,
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            "erosion": predictions,
        }
    )

    if debug_grid_output is not None:
        debug_grid = pd.DataFrame(
            {
                "regionid": region_id,
                "pixid": pixid,
                "row": region_points["row"].to_numpy(dtype=np.int64),
                "col": region_points["col"].to_numpy(dtype=np.int64),
                "x": region_points["x"].to_numpy(),
                "y": region_points["y"].to_numpy(),
                "ref_value": region_points["ref_value"].to_numpy(),
            }
        )
        save_table(debug_grid, debug_grid_output)
        print(f"debug grid -> {debug_grid_output}")

    if debug_metadata_output is not None:
        if debug_metadata_output.suffix.lower() != ".csv":
            raise ValueError("--debug-metadata-output debe tener extensión .csv")
        save_table(grid_metadata, debug_metadata_output)
        print(f"debug metadata -> {debug_metadata_output}")

    save_output(out, output_path)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
