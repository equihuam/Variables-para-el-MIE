#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 3_wf_sp_invasoras.py

Propósito:
    Calcular, para una región específica, un índice agregado de potencial de
    especies invasoras a partir de la distancia a la ocurrencia más cercana de
    cada especie invasora y exportarlo como tabla congruente por píxel en
    formato Parquet.

Origen:
    Adaptación a workflow del script Python inicial:
    3_sp_invasoras.py

Resumen del flujo:
    1. Leer el catálogo de especies invasoras por fidelidad al flujo original.
    2. Leer la base de plantas invasoras y preparar coordenadas x, y.
    3. Leer el ref_grid.tif de una región.
    4. Reproyectar el raster regional a EPSG:4326.
    5. Extraer solo los centros de píxel válidos de la malla regional.
    6. Calcular la distancia al vecino más cercano por especie invasora.
    7. Normalizar por especie las distancias y aplicar 1 - normalize(distancia).
    8. Sumar las contribuciones por especie para derivar sp_inv_potential.
    9. Exportar la tabla regional en Parquet.

Insumos principales:
    - catálogo Excel de especies invasoras
    - base CSV de plantas invasoras
    - ref_grid.tif regional

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y, sp_inv_potential

Supuestos y notas:
    - Los puntos de especies invasoras se interpretan en EPSG:4326.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica del flujo original.
    - Solo se conservan celdas válidas de la malla regional.
    - La distancia se aproxima como distancia euclidiana al vecino más cercano.
    - La normalización replica la función normalize(x) del script original y
      luego aplica 1 - normalize(distancia).

Observaciones:
    - Este script está diseñado para integrarse en un workflow Snakemake.
    - La ejecución es por región y con rutas parametrizadas.
    - La salida es compatible con el contrato mínimo del proyecto para tablas
      de features congruentes por píxel.
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
from scipy.spatial import cKDTree


SPECIES_FIELD = "especievalida"
POINTS_CRS = "EPSG:4326"
OUTPUT_FIELD = "sp_inv_pot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula potencial de especies invasoras por píxel para una región específica."
    )
    parser.add_argument(
        "--species-points-csv",
        required=True,
        help="Ruta a la base CSV de plantas invasoras.",
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


def normalize_series(values: pd.Series) -> pd.Series:
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)

    if pd.isna(vmin) or pd.isna(vmax):
        return pd.Series(np.nan, index=values.index)

    if vmax == vmin:
        return pd.Series(0.0, index=values.index)

    return (values - vmin) / (vmax - vmin)



def load_invasive_points(path: Path) -> pd.DataFrame:
    sp_inv = pd.read_csv(path, sep=",", header=0, low_memory=False)

    if sp_inv.shape[1] < 13:
        raise ValueError("El archivo plantas_invasoras.csv no tiene al menos 13 columnas.")

    cols = list(sp_inv.columns)
    cols[11] = "x"
    cols[12] = "y"
    sp_inv.columns = cols

    required = {"x", "y", SPECIES_FIELD}
    missing = required - set(sp_inv.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en plantas_invasoras.csv: {missing}")

    sp_inv["x"] = pd.to_numeric(sp_inv["x"], errors="raise")
    sp_inv["y"] = pd.to_numeric(sp_inv["y"], errors="raise")
    sp_inv[SPECIES_FIELD] = sp_inv[SPECIES_FIELD].astype(str)

    return sp_inv


def reproject_raster_to_epsg4326(
        src: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs,
        POINTS_CRS,
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
        dst_crs=POINTS_CRS,
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


def nearest_distance_column(points_xy: np.ndarray, species_df: pd.DataFrame) -> np.ndarray:
    if species_df.empty:
        return np.full(points_xy.shape[0], np.nan, dtype=float)

    coords = species_df[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(points_xy, k=1)

    return distances.astype(float)


def compute_sp_inv_potential(region_points: pd.DataFrame, sp_inv: pd.DataFrame) -> np.ndarray:
    unique_inv = list(pd.unique(sp_inv[SPECIES_FIELD]))
    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    species_scores: list[pd.Series] = []

    for sp in unique_inv:
        sp_inv_f = sp_inv[sp_inv[SPECIES_FIELD] == sp]
        distances = pd.Series(nearest_distance_column(pred_xy, sp_inv_f))
        score = 1.0 - normalize_series(distances)
        species_scores.append(score)

    if not species_scores:
        return np.full(len(region_points), np.nan, dtype=float)

    score_df = pd.concat(species_scores, axis=1)
    return score_df.sum(axis=1, skipna=True).to_numpy(dtype=float)


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

    species_points_path = Path(args.species_points_csv)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(species_points_path, ref_grid_path)

    sp_inv = load_invasive_points(species_points_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_epsg4326(src)

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)

    print(f"total puntos reproyectados: {total_points}")
    print(f"puntos válidos en malla: {len(region_points)}")

    sp_inv_pot = compute_sp_inv_potential(region_points, sp_inv)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            OUTPUT_FIELD: sp_inv_pot,
        }
    )

    save_output(out, output_path)


if __name__ == "__main__":
    main()