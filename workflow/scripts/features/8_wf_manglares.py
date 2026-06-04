#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 8_wf_manglares.py

Propósito:
    Calcular, para una región específica, la probabilidad de manglar
    (`p_manglares`) en cada pixel válido de la malla regional y exportar
    una tabla Parquet congruente por pixel.

Lógica validada contra R:
    1. Leer shapefile de manglares.
    2. Leer `ref_grid.tif` regional.
    3. Reproyectar la malla regional al CRS de manglares con vecino cercano.
    4. Rasterizar manglares sobre esa malla.
    5. Etiquetar celdas válidas de la región como manglar/no manglar (1/0).
    6. Estimar probabilidad con una emulación de
       `kknn(layer ~ x + y, k = 30, distance = 2, kernel = "optimal", scale = TRUE)`.
    7. Exportar `regionid, pixid, x, y, p_manglares`.

Notas de compatibilidad:
    - La interfaz canónica usa `--ref-grid` y `--region-id`.
    - Para compatibilidad transitoria con reglas antiguas de Snakemake, también
      acepta `--base-table` y deriva de ahí la ruta esperada del `ref_grid.tif`
      cuando `--ref-grid` no se proporciona.
    - La ejecución productiva es silenciosa por defecto y sólo imprime `OK -> ...`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.neighbors import NearestNeighbors


OUTPUT_FIELD = "p_manglares"
K_NEIGHBORS = 30
POINTS_SOURCE_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula probabilidad de manglares por píxel para una región específica."
    )
    parser.add_argument("--mangroves-shp", required=True, help="Ruta al shapefile de manglares.")
    parser.add_argument("--ref-grid", default=None, help="Ruta al ref_grid.tif regional.")
    parser.add_argument("--region-id", default=None, help="Identificador regional, por ejemplo region_1.")
    parser.add_argument(
        "--base-table",
        default=None,
        help="Ruta opcional a tabla base regional; usada sólo para derivar ref_grid si --ref-grid no se proporciona.",
    )
    parser.add_argument("--output", required=True, help="Ruta de salida .parquet.")
    parser.add_argument(
        "--debug-grid-output",
        default=None,
        help="CSV opcional con grilla válida y etiqueta manglar 0/1.",
    )
    parser.add_argument(
        "--debug-metadata-output",
        default=None,
        help="CSV opcional con metadatos de validación.",
    )
    parser.add_argument(
        "--debug-mangrove-raster-points-output",
        default=None,
        help="CSV opcional con puntos rasterizados de manglar.",
    )
    parser.add_argument(
        "--knn-mode",
        choices=["kknn_optimal", "uniform_legacy"],
        default="kknn_optimal",
        help="Modo de probabilidad kNN. kknn_optimal replica kknn(kernel='optimal', scale=TRUE).",
    )
    parser.add_argument("--verbose", action="store_true", help="Imprime diagnósticos.")
    return parser.parse_args()


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def derive_ref_grid_from_base_table(base_table: Path) -> Path:
    """Deriva results/reference/<region>/ref_grid.tif desde results/features/<feature>/<region>.parquet."""
    region = base_table.stem
    try:
        data_repo = base_table.parents[3]
    except IndexError as exc:
        raise ValueError(
            "No se pudo derivar data_repo desde --base-table. "
            "Proporciona --ref-grid explícitamente."
        ) from exc
    return data_repo / "results" / "reference" / region / "ref_grid.tif"


def log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def load_mangroves(path: Path) -> gpd.GeoDataFrame:
    manglares = gpd.read_file(path)
    if manglares.empty:
        raise ValueError(f"El shapefile de manglares está vacío: {path}")
    if manglares.crs is None:
        raise ValueError(f"El shapefile de manglares no tiene CRS: {path}")
    return manglares


def reproject_raster_to_crs(src: rasterio.io.DatasetReader, dst_crs) -> tuple[np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs,
        dst_crs,
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
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )

    return dst, transform


def valid_raster_points_dataframe(arr: np.ndarray, transform) -> pd.DataFrame:
    valid_mask = np.isfinite(arr)
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", "ref_value"])

    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "row": rows.astype(int),
            "col": cols.astype(int),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "ref_value": arr[rows, cols],
        }
    )


def rasterize_mangroves(shape: tuple[int, int], transform, manglares: gpd.GeoDataFrame) -> np.ndarray:
    shapes: Iterable[tuple[object, int]] = (
        (geom, 1) for geom in manglares.geometry if geom is not None and not geom.is_empty
    )
    return rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )


def optimal_rank_weights(k: int) -> np.ndarray:
    # kknn::kernel="optimal" para distance=2 genera pesos proporcionales a
    # 2*(k-rank)+1; suman k^2. Para k=3: 5,3,1 / 9.
    ranks = np.arange(1, k + 1, dtype=float)
    weights = 2.0 * (k - ranks) + 1.0
    return weights / weights.sum()


def scale_train_test(x_train: np.ndarray, x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # R scale=TRUE en kknn escala por sd muestral del train, sin centrar.
    sd = np.std(x_train, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return x_train / sd, x_pred / sd, sd


def predict_mangrove_probability(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    k: int = K_NEIGHBORS,
    knn_mode: str = "kknn_optimal",
) -> tuple[np.ndarray, np.ndarray]:
    if train_df.empty or pred_df.empty:
        return np.full(len(pred_df), np.nan, dtype=float), np.array([np.nan, np.nan])

    labels = train_df["label"].to_numpy(dtype=float)
    unique = np.unique(labels[np.isfinite(labels)])
    if len(unique) == 0:
        return np.full(len(pred_df), np.nan, dtype=float), np.array([np.nan, np.nan])
    if len(unique) == 1:
        return np.full(len(pred_df), float(unique[0]), dtype=float), np.array([np.nan, np.nan])

    x_train = train_df[["x", "y"]].to_numpy(dtype=float)
    x_pred = pred_df[["x", "y"]].to_numpy(dtype=float)
    x_train_s, x_pred_s, sd = scale_train_test(x_train, x_pred)

    k_eff = min(k, len(train_df))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", metric="euclidean")
    nn.fit(x_train_s)
    _, indices = nn.kneighbors(x_pred_s, return_distance=True)

    neighbor_labels = labels[indices]

    if knn_mode == "uniform_legacy":
        probs = np.mean(neighbor_labels, axis=1)
    else:
        weights = optimal_rank_weights(k_eff)
        probs = np.sum(neighbor_labels * weights.reshape(1, -1), axis=1)

    return probs.astype(float), sd


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        raise ValueError(f"Extensión no soportada: {path.suffix}")


def main() -> None:
    args = parse_args()

    mangroves_path = Path(args.mangroves_shp)
    output_path = Path(args.output)

    if args.ref_grid:
        ref_grid_path = Path(args.ref_grid)
    elif args.base_table:
        ref_grid_path = derive_ref_grid_from_base_table(Path(args.base_table))
    else:
        raise ValueError("Debes proporcionar --ref-grid o --base-table.")

    if args.region_id:
        region_id = str(args.region_id).strip()
    else:
        region_id = ref_grid_path.parent.name

    validate_inputs(mangroves_path, ref_grid_path)
    manglares = load_mangroves(mangroves_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
        src_total = int(src.width * src.height)
        src_masked = src.read(1, masked=True)
        src_valid = int(np.sum(~src_masked.mask))
        region_arr, region_transform = reproject_raster_to_crs(src, manglares.crs)

    region_points = valid_raster_points_dataframe(region_arr, region_transform)
    mangrove_rast = rasterize_mangroves(region_arr.shape, region_transform, manglares)

    valid_rows = region_points["row"].to_numpy(dtype=int)
    valid_cols = region_points["col"].to_numpy(dtype=int)
    labels = mangrove_rast[valid_rows, valid_cols].astype(int)

    train_points = region_points[["x", "y"]].copy()
    train_points["label"] = labels

    probs, sd = predict_mangrove_probability(
        train_df=train_points,
        pred_df=region_points,
        k=K_NEIGHBORS,
        knn_mode=args.knn_mode,
    )

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            OUTPUT_FIELD: probs,
        }
    )

    save_table(out, output_path)

    n_mangrove_cells = int(labels.sum())
    n_valid = int(len(region_points))

    log(args.verbose, f"total puntos GeoTIFF original: {src_total}")
    log(args.verbose, f"puntos válidos GeoTIFF original masked: {src_valid}")
    log(args.verbose, f"total puntos reproyectados: {int(region_arr.size)}")
    log(args.verbose, f"puntos válidos usados en malla: {n_valid}")
    log(args.verbose, f"celdas manglar rasterizadas en malla válida: {n_mangrove_cells}")
    log(args.verbose, f"modo kNN: {args.knn_mode}")
    log(args.verbose, f"sd entrenamiento: ({sd[0]}, {sd[1]})")

    if args.debug_grid_output:
        dbg = out.copy()
        dbg["row"] = region_points["row"].to_numpy(dtype=int)
        dbg["col"] = region_points["col"].to_numpy(dtype=int)
        dbg["ref_value"] = region_points["ref_value"].to_numpy()
        dbg["label"] = labels
        save_table(dbg, Path(args.debug_grid_output))
        log(args.verbose, f"debug grid -> {args.debug_grid_output}")

    if args.debug_mangrove_raster_points_output:
        rows, cols = np.where(mangrove_rast == 1)
        if len(rows) == 0:
            pts = pd.DataFrame(columns=["row", "col", "x", "y", "layer"])
        else:
            xs, ys = xy(region_transform, rows, cols, offset="center")
            pts = pd.DataFrame(
                {
                    "row": rows.astype(int),
                    "col": cols.astype(int),
                    "x": np.asarray(xs, dtype=float),
                    "y": np.asarray(ys, dtype=float),
                    "layer": 1,
                }
            )
        save_table(pts, Path(args.debug_mangrove_raster_points_output))
        log(args.verbose, f"debug mangrove raster points -> {args.debug_mangrove_raster_points_output}")

    if args.debug_metadata_output:
        meta = pd.DataFrame(
            [
                {
                    "regionid": region_id,
                    "src_total_points": src_total,
                    "src_valid_points": src_valid,
                    "dst_total_points": int(region_arr.size),
                    "dst_valid_points": n_valid,
                    "mangrove_valid_cells": n_mangrove_cells,
                    "non_mangrove_valid_cells": int(n_valid - n_mangrove_cells),
                    "k": K_NEIGHBORS,
                    "knn_mode": args.knn_mode,
                    "sd_x": float(sd[0]) if len(sd) > 0 else np.nan,
                    "sd_y": float(sd[1]) if len(sd) > 1 else np.nan,
                    "p_min": float(np.nanmin(probs)) if len(probs) else np.nan,
                    "p_max": float(np.nanmax(probs)) if len(probs) else np.nan,
                    "p_mean": float(np.nanmean(probs)) if len(probs) else np.nan,
                }
            ]
        )
        save_table(meta, Path(args.debug_metadata_output))
        log(args.verbose, f"debug metadata -> {args.debug_metadata_output}")

    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
