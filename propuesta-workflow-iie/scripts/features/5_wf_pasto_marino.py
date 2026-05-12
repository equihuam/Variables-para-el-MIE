#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 5_wf_pasto_marino.py

Propósito:
    Calcular, para una región específica, la distancia al pasto marino más
    cercano para cada píxel válido del raster de referencia y exportar el
    resultado como tabla congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script Python inicial:
    5_pasto_marino.py

Resumen del flujo:
    1. Leer el shapefile de pasto marino.
    2. Leer el ref_grid.tif de una región.
    3. Reproyectar el raster regional al CRS del shapefile.
    4. Extraer solo los centros de píxel válidos de la malla regional.
    5. Rasterizar el pasto marino sobre la plantilla regional.
    6. Calcular la distancia al pasto marino más cercano para cada píxel válido.
    7. Reemplazar el valor centinela 999, cuando no hay pasto en la región,
       por 1.5 * max_dist de la región.
    8. Exportar la tabla regional en Parquet.

Insumos principales:
    - shapefile de pasto marino
    - ref_grid.tif regional

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y, pasto

Supuestos y notas:
    - La distancia se calcula en el CRS del shapefile de pasto marino.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica del flujo original.
    - Solo se conservan celdas válidas de la malla regional.
    - La distancia final se aproxima como distancia euclidiana al vecino más cercano.
    - Se conserva el valor centinela 999 a nivel regional y se sustituye al final
      por 1.5 * max_dist, igual que en el flujo original.

Observaciones:
    - Este script está diseñado para integrarse en un workflow Snakemake.
    - La ejecución es por región y con rutas parametrizadas.
    - La salida es compatible con el contrato mínimo del proyecto para tablas
      de features congruentes por píxel.
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


PASTO_SENTINEL = 999.0
OUTPUT_FIELD = "d_pastosmarinos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia a pasto marino por píxel para una región específica."
    )
    parser.add_argument(
        "--pasto-marino-shp",
        required=True,
        help="Ruta al shapefile de pasto marino.",
    )
    parser.add_argument(
        "--ref-grid",
        required=True,
        help="Ruta al ref_grid.tif de la región.",
    )
    parser.add_argument(
        "--region-id",
        required=True,
        help="Identificador de la región, por ejemplo region_1.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet.",
    )
    return parser.parse_args()


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_pasto_marino(path: Path) -> gpd.GeoDataFrame:
    pasto = gpd.read_file(path)

    if pasto.empty:
        raise ValueError(f"El shapefile de pasto marino está vacío: {path}")
    if pasto.crs is None:
        raise ValueError(f"El shapefile de pasto marino no tiene CRS: {path}")

    return pasto


def reproject_raster_to_crs(
        src: rasterio.io.DatasetReader,
        dst_crs,
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


def rasterize_pasto_on_region(shape, transform, pasto: gpd.GeoDataFrame) -> np.ndarray:
    shapes = (
        (geom, 1)
        for geom in pasto.geometry
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


def pasto_points_from_raster(pasto_rast: np.ndarray, transform) -> pd.DataFrame:
    valid_mask = np.isfinite(pasto_rast)
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", "part"])

    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "part": 1,
        }
    )


def nearest_distance_column(points_xy: np.ndarray, pasto_points: pd.DataFrame) -> np.ndarray:
    coords = pasto_points[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float)


def finalize_pasto(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    sentinel_mask = out == PASTO_SENTINEL

    if np.any(~sentinel_mask):
        max_dist = float(np.nanmax(out[~sentinel_mask]))
        out[sentinel_mask] = 1.5 * max_dist

    return out


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Este script requiere salida .parquet. Recibido: {output_path.suffix}"
        )

    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()

    pasto_path = Path(args.pasto_marino_shp)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(pasto_path, ref_grid_path)
    pasto = load_pasto_marino(pasto_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_crs(src, pasto.crs)

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)

    print(f"total puntos reproyectados: {total_points}")
    print(f"puntos válidos en malla: {len(region_points)}")

    distances = np.full(len(region_points), PASTO_SENTINEL, dtype=float)

    pasto_rast = rasterize_pasto_on_region(
        shape=region_arr.shape,
        transform=region_transform,
        pasto=pasto,
    )

    if np.isfinite(pasto_rast).sum() > 0:
        pasto_points = pasto_points_from_raster(pasto_rast, region_transform)

        if len(pasto_points) == 1:
            pasto_points = pd.concat([pasto_points, pasto_points], ignore_index=True)

        pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)
        distances = nearest_distance_column(pred_xy, pasto_points)

    distances = finalize_pasto(distances)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            OUTPUT_FIELD: distances,
        }
    )

    save_output(out, output_path)


if __name__ == "__main__":
    main()