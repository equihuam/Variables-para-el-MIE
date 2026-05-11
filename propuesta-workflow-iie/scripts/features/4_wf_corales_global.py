#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 4_wf_corales_global.py

Propósito:
    Calcular, para una región específica, la distancia al coral global más cercano
    para cada píxel válido de un raster de referencia y exportarla como tabla
    tabular congruente por píxel.

Origen:
    Refactorización para workflow de la traducción inicial a Python del script R:
    4_corales_global.R
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree


CORALS_SENTINEL = 999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia a corales por píxel para una región específica."
    )
    parser.add_argument(
        "--corals-shp",
        required=True,
        help="Ruta al shapefile global de corales.",
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
    return parser.parse_args()


def validate_inputs(corals_shp: Path, ref_grid: Path) -> None:
    missing = [str(p) for p in [corals_shp, ref_grid] if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_corals(path: Path) -> gpd.GeoDataFrame:
    corales = gpd.read_file(path)

    if corales.empty:
        raise ValueError(f"El shapefile de corales está vacío: {path}")
    if corales.crs is None:
        raise ValueError(f"El shapefile de corales no tiene CRS: {path}")

    return corales


def reproject_raster_to_crs(
        src: rasterio.io.DatasetReader, dst_crs
) -> tuple[np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs,
        dst_crs,
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
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )

    return dst, transform


def valid_raster_points_dataframe(arr: np.ndarray, transform) -> pd.DataFrame:
    """
    Construye tabla xy solo para celdas válidas del raster.
    Evita materializar toda la grilla en memoria.
    """
    valid_mask = np.isfinite(arr)
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", "value"])

    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "value": arr[rows, cols],
        }
    )


def rasterize_corals_on_region(shape, transform, corales: gpd.GeoDataFrame) -> np.ndarray:
    shapes = (
        (geom, 1)
        for geom in corales.geometry
        if geom is not None and not geom.is_empty
    )

    arr = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
    )

    return arr


def coral_points_from_raster(corales_rast: np.ndarray, transform) -> pd.DataFrame:
    valid_mask = np.isfinite(corales_rast)
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", "part"])

    xs, ys = xy(transform, rows, cols, offset="center")

    coral_points = pd.DataFrame(
        {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "part": 1,
        }
    )
    return coral_points


def nearest_distance_column(points_xy: np.ndarray, coral_points: pd.DataFrame) -> np.ndarray:
    coords = coral_points[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float)


def finalize_corals(distances: np.ndarray) -> np.ndarray:
    out = distances.copy()
    if np.any(out == CORALS_SENTINEL):
        max_dist = float(np.max(out))
        out[out == CORALS_SENTINEL] = 1.5 * max_dist
    return out


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Este script requiere salida .parquet. Recibido: {output_path.suffix}"
        )

    df.to_parquet(output_path, index=False, engine="pyarrow")


def main() -> None:
    args = parse_args()

    corals_path = Path(args.corals_shp)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(corals_path, ref_grid_path)
    corales = load_corals(corals_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_crs(src, corales.crs)

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)

    print(f"total puntos reproyectados: {total_points}")
    print(f"puntos válidos en malla: {len(region_points)}")

    corales_rast = rasterize_corals_on_region(
        shape=region_arr.shape,
        transform=region_transform,
        corales=corales,
    )

    distances = np.full(len(region_points), CORALS_SENTINEL, dtype=float)

    if np.isfinite(corales_rast).sum() > 0:
        coral_points = coral_points_from_raster(corales_rast, region_transform)

        if len(coral_points) == 1:
            coral_points = pd.concat([coral_points, coral_points], ignore_index=True)

        pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)
        distances = nearest_distance_column(pred_xy, coral_points)

    distances = finalize_corals(distances)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            "corals": distances,
        }
    )

    save_output(out, output_path)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()