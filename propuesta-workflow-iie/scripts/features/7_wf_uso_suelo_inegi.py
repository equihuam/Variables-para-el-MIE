#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 7_wf_uso_suelo_inegi.py

Propósito:
    Calcular, para una región específica, la distancia al uso de suelo INEGI
    más cercano para tres categorías generalizadas y exportar el resultado
    como tabla congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script Python inicial:
    7_inegi_uso_suelo.py

Resumen del flujo:
    1. Leer el shapefile de uso de suelo INEGI.
    2. Reclasificar las categorías de uso de suelo en clases generales.
    3. Leer el ref_grid.tif de una región.
    4. Reproyectar el raster regional al CRS del shapefile de uso de suelo.
    5. Extraer solo los centros de píxel válidos de la malla regional.
    6. Rasterizar cada clase de uso de suelo sobre la plantilla regional.
    7. Calcular la distancia al vecino más cercano por clase.
    8. Exportar la tabla regional en Parquet.

Insumos principales:
    - shapefile de uso de suelo INEGI
    - ref_grid.tif regional

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y,
      d_grassland, d_agriculture, d_urban

Supuestos y notas:
    - La distancia se calcula en el CRS del shapefile de uso de suelo.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica del flujo original.
    - Solo se conservan celdas válidas de la malla regional.
    - La distancia final se aproxima como distancia euclidiana al vecino más cercano.
    - Las clases originales se generalizan siguiendo la lógica del script base:
      Pastizal y Vegetación de dunas costeras -> grassland,
      Agricultura -> agriculture,
      Asentamiento humano -> urban.

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


USO_FIELD = "USO.suelo"

TYPE_TO_OUTPUT = {
    "grassland": "d_grassland",
    "agriculture": "d_agriculture",
    "urban": "d_urban",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia a clases de uso de suelo INEGI por píxel para una región específica."
    )
    parser.add_argument(
        "--uso-suelo-shp",
        required=True,
        help="Ruta al shapefile de uso de suelo INEGI.",
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


def load_land_use(path: Path) -> gpd.GeoDataFrame:
    uso = gpd.read_file(path)

    if uso.empty:
        raise ValueError(f"El shapefile de uso de suelo está vacío: {path}")
    if uso.crs is None:
        raise ValueError(f"El shapefile de uso de suelo no tiene CRS: {path}")
    if USO_FIELD not in uso.columns:
        raise ValueError(f"No existe el campo requerido '{USO_FIELD}' en {path.name}")

    uso = uso.copy()
    uso["type"] = np.nan

    # Lógica de reclasificación del script base
    uso.loc[
        uso[USO_FIELD].isin(["Pastizal", "Vegetación de dunas costeras"]),
        "type",
    ] = "grassland"
    uso.loc[uso[USO_FIELD].isin(["Agricultura"]), "type"] = "agriculture"
    uso.loc[uso[USO_FIELD].isin(["Asentamiento humano"]), "type"] = "urban"

    uso = uso[uso["type"].notna()].copy()

    if uso.empty:
        raise ValueError("No quedaron categorías válidas después de reclasificar uso de suelo.")

    return uso


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


def rasterize_land_use_type(shape, transform, uso_tipo: gpd.GeoDataFrame) -> np.ndarray:
    shapes = (
        (geom, 1)
        for geom in uso_tipo.geometry
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


def land_use_points_from_raster(uso_rast: np.ndarray, transform) -> pd.DataFrame:
    valid_mask = np.isfinite(uso_rast)
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


def nearest_distance_column(points_xy: np.ndarray, uso_points: pd.DataFrame) -> np.ndarray:
    if uso_points.empty:
        return np.full(points_xy.shape[0], np.nan, dtype=float)

    coords = uso_points[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float)


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

    uso_suelo_path = Path(args.uso_suelo_shp)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(uso_suelo_path, ref_grid_path)
    uso = load_land_use(uso_suelo_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_crs(src, uso.crs)

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)

    print(f"total puntos reproyectados: {total_points}")
    print(f"puntos válidos en malla: {len(region_points)}")

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
        }
    )

    for tipo, output_col in TYPE_TO_OUTPUT.items():
        uso_tipo = uso[uso["type"] == tipo]
        uso_rast = rasterize_land_use_type(
            shape=region_arr.shape,
            transform=region_transform,
            uso_tipo=uso_tipo,
        )
        uso_points = land_use_points_from_raster(uso_rast, region_transform)
        out[output_col] = nearest_distance_column(pred_xy, uso_points)

    save_output(out, output_path)


if __name__ == "__main__":
    main()