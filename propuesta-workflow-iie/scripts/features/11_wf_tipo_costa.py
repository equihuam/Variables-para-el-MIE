#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree

FIELD_NAME = "TipoCosta"
OUTPUT_FIELD = "tipo_costa"
K_NEIGHBORS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica tipo de costa por píxel para una región específica."
    )
    parser.add_argument("--coast-types-shp", required=True, help="Ruta al shapefile de tipología costera.")
    parser.add_argument("--ref-grid", required=True, help="Ruta al ref_grid.tif de la región.")
    parser.add_argument("--region-id", required=True, help="Identificador de la región, por ejemplo region_1.")
    parser.add_argument("--output", required=True, help="Ruta de salida .parquet.")
    parser.add_argument(
        "--distance-mode",
        choices=["kknn_scaled", "raw"],
        default="kknn_scaled",
        help="Modo de distancia para el vecino más cercano. kknn_scaled replica scale=TRUE de kknn.",
    )
    parser.add_argument("--debug-grid-output", default=None, help="CSV opcional con la grilla regional.")
    parser.add_argument("--debug-metadata-output", default=None, help="CSV opcional con metadatos de diagnóstico.")
    parser.add_argument("--debug-coast-points-output", default=None, help="CSV opcional con puntos rasterizados de tipo de costa.")
    parser.add_argument("--verbose", action="store_true", help="Imprime diagnósticos detallados.")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_coast_types(path: Path) -> gpd.GeoDataFrame:
    costas = gpd.read_file(path)
    if costas.empty:
        raise ValueError(f"El shapefile de tipo de costa está vacío: {path}")
    if costas.crs is None:
        raise ValueError(f"El shapefile de tipo de costa no tiene CRS: {path}")
    if FIELD_NAME not in costas.columns:
        raise ValueError(f"No existe el campo requerido '{FIELD_NAME}' en {path.name}")

    costas = costas.copy()
    costas[FIELD_NAME] = costas[FIELD_NAME].astype(str)
    costas = costas[costas.geometry.notna() & ~costas.geometry.is_empty].copy()
    if costas.empty:
        raise ValueError("No quedaron geometrías válidas de tipo de costa.")
    return costas


def reproject_raster_to_crs(src: rasterio.io.DatasetReader, dst_crs: Any) -> tuple[np.ndarray, rasterio.Affine, dict[str, Any]]:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
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
    meta = {
        "src_crs": str(src.crs),
        "dst_crs": str(dst_crs),
        "src_width": src.width,
        "src_height": src.height,
        "dst_width": width,
        "dst_height": height,
        "src_nodata": src.nodata,
    }
    return dst, transform, meta


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


def rasterize_coast_types(shape, transform, costas: gpd.GeoDataFrame) -> tuple[np.ndarray, dict[int, str]]:
    labels = costas[FIELD_NAME].astype(str)
    categories = pd.Categorical(labels)
    codes = categories.codes + 1
    shapes = (
        (geom, int(code))
        for geom, code in zip(costas.geometry, codes)
        if geom is not None and not geom.is_empty
    )
    arr = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    code_to_label = {int(code + 1): str(cat) for code, cat in enumerate(categories.categories)}
    return arr, code_to_label


def coast_points_from_raster(costas_rast: np.ndarray, transform, code_to_label: dict[int, str]) -> pd.DataFrame:
    valid_mask = costas_rast > 0
    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", FIELD_NAME])
    xs, ys = xy(transform, rows, cols, offset="center")
    codes = costas_rast[rows, cols]
    labels = [code_to_label[int(c)] for c in codes]
    return pd.DataFrame(
        {
            "row": rows.astype(int),
            "col": cols.astype(int),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            FIELD_NAME: labels,
            "code": codes.astype(int),
        }
    )


def scaled_xy(train_xy: np.ndarray, pred_xy: np.ndarray, distance_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if distance_mode == "raw":
        return train_xy, pred_xy, np.ones(train_xy.shape[1], dtype=float)
    sd = np.nanstd(train_xy, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return train_xy / sd, pred_xy / sd, sd


def fit_knn_labels(train_df: pd.DataFrame, pred_df: pd.DataFrame, distance_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if train_df.empty:
        return np.full(len(pred_df), None, dtype=object), np.array([np.nan, np.nan], dtype=float)
    train_xy = train_df[["x", "y"]].to_numpy(dtype=float)
    pred_xy = pred_df[["x", "y"]].to_numpy(dtype=float)
    train_scaled, pred_scaled, sd = scaled_xy(train_xy, pred_xy, distance_mode)

    tree = cKDTree(train_scaled)
    _, idx = tree.query(pred_scaled, k=1)
    labels = train_df[FIELD_NAME].astype(str).to_numpy()
    pred = labels[np.asarray(idx, dtype=int)]
    return pred.astype(object), sd


def write_optional_csv(df: pd.DataFrame, path: str | None) -> None:
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {output_path.suffix}")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()
    coast_types_path = Path(args.coast_types_shp)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(coast_types_path, ref_grid_path)
    costas = load_coast_types(coast_types_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
        original_masked = src.read(1, masked=True)
        original_valid = int(np.sum(~original_masked.mask))
        region_arr, region_transform, reproj_meta = reproject_raster_to_crs(src, costas.crs)

    region_points = valid_raster_points_dataframe(region_arr, region_transform)
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    costas_rast, code_to_label = rasterize_coast_types(region_arr.shape, region_transform, costas)
    costas_table = coast_points_from_raster(costas_rast, region_transform, code_to_label)

    coast_prediction, sd = fit_knn_labels(costas_table, region_points, args.distance_mode)
    region_points[OUTPUT_FIELD] = coast_prediction

    out = region_points[["regionid", "pixid", "x", "y", OUTPUT_FIELD]].copy()

    metadata = {
        "regionid": region_id,
        "distance_mode": args.distance_mode,
        "original_valid_points": original_valid,
        "reprojected_total_points": int(region_arr.size),
        "reprojected_valid_points": int(len(region_points)),
        "coast_raster_points": int(len(costas_table)),
        "n_coast_labels": int(costas_table[FIELD_NAME].nunique()) if not costas_table.empty else 0,
        "sd_x": float(sd[0]) if len(sd) > 0 else np.nan,
        "sd_y": float(sd[1]) if len(sd) > 1 else np.nan,
        **reproj_meta,
    }
    meta_df = pd.DataFrame([metadata])

    log(f"puntos válidos originales: {original_valid}", args.verbose)
    log(f"puntos válidos reproyectados: {len(region_points)}", args.verbose)
    log(f"puntos rasterizados tipo costa: {len(costas_table)}", args.verbose)
    log(f"etiquetas: {sorted(costas_table[FIELD_NAME].dropna().unique().tolist()) if not costas_table.empty else []}", args.verbose)
    log(f"modo distancia: {args.distance_mode}; sd=({sd[0] if len(sd)>0 else np.nan}, {sd[1] if len(sd)>1 else np.nan})", args.verbose)

    write_optional_csv(region_points[["regionid", "pixid", "row", "col", "x", "y", "ref_value"]], args.debug_grid_output)
    write_optional_csv(costas_table, args.debug_coast_points_output)
    write_optional_csv(meta_df, args.debug_metadata_output)

    save_output(out, output_path)


if __name__ == "__main__":
    main()
