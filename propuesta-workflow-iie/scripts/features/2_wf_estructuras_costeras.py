#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 2_wf_estructuras_costeras.py

Propósito:
    Calcular, para una región específica, la distancia a la estructura costera
    más cercana por tipo de estructura y exportar el resultado como tabla
    congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script Python inicial:
    2_features_estructuras.py

Resumen del flujo:
    1. Leer y normalizar el shapefile de estructuras costeras.
    2. Leer el ref_grid.tif de una región.
    3. Reproyectar el raster regional al CRS de las estructuras.
    4. Extraer solo los centros de píxel válidos de la malla regional.
    5. Calcular la distancia euclidiana al vecino más cercano para cada tipo de estructura.
    6. Exportar la tabla regional en Parquet.

Insumos principales:
    - shapefile de estructuras costeras
    - ref_grid.tif regional

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y,
      escollera, espigon, muro, rompeolas, puerto

Supuestos y notas:
    - La distancia se calcula sobre el CRS del shapefile de estructuras.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica del flujo original.
    - Solo se conservan celdas válidas de la malla regional.
    - La distancia final se aproxima como distancia euclidiana al vecino más cercano.
    - Los nombres de columnas temáticas se normalizan a una convención canónica
      sin acentos para facilitar su integración al flujo BN.

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
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree


TIPO_FIELD = "Tipo"

CANONICAL_TYPES = {
    "Escollera": "escollera",
    "Espigón": "espigon",
    "Muro": "muro",
    "Rompeolas": "rompeolas",
    "Puerto": "puerto",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia a estructuras costeras por píxel para una región específica."
    )
    parser.add_argument(
        "--structures-shp",
        required=True,
        help="Ruta al shapefile de estructuras costeras.",
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


def load_structures(path: Path) -> gpd.GeoDataFrame:
    struct = gpd.read_file(path)

    if struct.empty:
        raise ValueError(f"El shapefile de estructuras está vacío: {path}")
    if struct.crs is None:
        raise ValueError(f"El shapefile de estructuras no tiene CRS: {path}")
    if TIPO_FIELD not in struct.columns:
        raise ValueError(f"No existe el campo requerido '{TIPO_FIELD}' en {path.name}")

    struct = struct.copy()

    # Normalización fiel al script R
    struct[TIPO_FIELD] = struct[TIPO_FIELD].replace(
        {
            "Escollera2": "Escollera",
            "Espigób": "Espigón",
            "espigón": "Espigón",
            "Espigón de M": "Espigón",
            "Muelle": "Puerto",
            "Rompeolas2": "Rompeolas",
        }
    )

    struct[TIPO_FIELD] = struct[TIPO_FIELD].astype(str)

    # Conservar solo tipos reconocidos por el flujo canónico
    struct = struct[struct[TIPO_FIELD].isin(CANONICAL_TYPES.keys())].copy()

    if struct.empty:
        raise ValueError("No quedaron estructuras válidas después de normalizar tipos.")

    return struct


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


def nearest_distance_column(points_xy: np.ndarray, struct_tipo: gpd.GeoDataFrame) -> np.ndarray:
    if struct_tipo.empty:
        return np.full(points_xy.shape[0], np.nan, dtype=float)

    geom = struct_tipo.geometry

    if geom.geom_type.isin(["Point", "MultiPoint"]).all():
        coords = np.array([(g.x, g.y) for g in geom], dtype=float)
    else:
        reps = geom.representative_point()
        coords = np.array([(g.x, g.y) for g in reps], dtype=float)

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

    structures_path = Path(args.structures_shp)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(structures_path, ref_grid_path)
    struct = load_structures(structures_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_crs(src, struct.crs)

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

    for tipo_original, tipo_canonico in CANONICAL_TYPES.items():
        struct_tipo = struct[struct[TIPO_FIELD] == tipo_original]
        out[tipo_canonico] = nearest_distance_column(pred_xy, struct_tipo)

    save_output(out, output_path)


if __name__ == "__main__":
    main()