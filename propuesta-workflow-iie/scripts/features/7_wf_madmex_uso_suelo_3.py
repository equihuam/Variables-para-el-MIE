#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 7_wf_madmex_uso_suelo_3.py

Propósito:
    Calcular, para una región específica, la distancia a la clase MADMEX más
    cercana para pastizal, agricultura y urbano, usando como referencia una
    tabla base regional ya alineada, y exportar el resultado como tabla
    congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script Python inicial:
    7_madmex_uso_suelo_3.py

Resumen del flujo:
    1. Leer el raster MADMEX de uso de suelo.
    2. Leer una tabla base regional ya alineada.
    3. Verificar el contrato mínimo de la tabla base.
    4. Transformar las coordenadas x, y de la tabla base al CRS de MADMEX.
    5. Recortar MADMEX a la extensión de los puntos transformados.
    6. Filtrar MADMEX a las clases 27, 28 y 29.
    7. Calcular la distancia al vecino más cercano por clase.
    8. Exportar la tabla regional en Parquet conservando exactamente la misma
       malla tabular de la base.

Insumos principales:
    - raster MADMEX de uso de suelo
    - tabla base regional alineada

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y,
      d_grassland, d_agriculture, d_urban

Supuestos y notas:
    - La distancia se calcula en el CRS del raster MADMEX.
    - La malla canónica se toma de la tabla base regional.
    - Se usan las clases MADMEX:
      27 = pastizales,
      28 = tierras agrícolas,
      29 = urbano y construido.
    - La distancia final se aproxima como distancia euclidiana al vecino más cercano.

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
import pyarrow.parquet as pq
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import transform as rio_transform
from shapely.geometry import box, mapping
from scipy.spatial import cKDTree


GRASSLAND_CLASS = 27
AGRICULTURE_CLASS = 28
URBAN_CLASS = 29
VALID_CLASSES = {GRASSLAND_CLASS, AGRICULTURE_CLASS, URBAN_CLASS}
SENTINEL = 9999.0

KEY_COLUMNS = ["regionid", "pixid", "x", "y"]

OUTPUT_COLS = {
    GRASSLAND_CLASS: "d_grassland",
    AGRICULTURE_CLASS: "d_agriculture",
    URBAN_CLASS: "d_urban",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia a clases MADMEX usando una tabla base regional."
    )
    parser.add_argument(
        "--madmex-raster",
        required=True,
        help="Ruta al raster MADMEX de uso de suelo.",
    )
    parser.add_argument(
        "--base-table",
        required=True,
        help="Ruta a la tabla base regional .parquet con regionid, pixid, x, y.",
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


def read_base_table(path: Path) -> pd.DataFrame:
    table = pq.read_table(path, use_threads=False)
    df = table.to_pandas()

    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"La tabla base no cumple el contrato mínimo. Faltan columnas: {missing}"
        )

    out = df[KEY_COLUMNS].copy()
    out["x"] = pd.to_numeric(out["x"], errors="raise")
    out["y"] = pd.to_numeric(out["y"], errors="raise")

    if out.empty:
        raise ValueError(f"La tabla base está vacía: {path}")

    return out


def transform_points_to_crs(
        xs: np.ndarray,
        ys: np.ndarray,
        src_crs,
        dst_crs,
) -> tuple[np.ndarray, np.ndarray]:
    tx, ty = rio_transform(src_crs, dst_crs, xs.tolist(), ys.tolist())
    return np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)


def crop_raster_to_points_extent(
        src: rasterio.io.DatasetReader,
        xs: np.ndarray,
        ys: np.ndarray,
) -> tuple[np.ndarray, rasterio.Affine]:
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No hay puntos válidos para definir la extensión de recorte.")

    geom = box(float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))
    cropped, cropped_transform = mask(src, [mapping(geom)], crop=True, filled=True)

    return cropped[0], cropped_transform


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


def filter_madmex_classes(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    valid_mask = np.isin(out, list(VALID_CLASSES))
    out[~valid_mask] = np.nan
    return out


def nearest_distance_column(points_xy: np.ndarray, class_points: pd.DataFrame) -> np.ndarray:
    if class_points.empty:
        return np.full(points_xy.shape[0], SENTINEL, dtype=float)

    coords = class_points[["x", "y"]].to_numpy(dtype=float)
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

    madmex_path = Path(args.madmex_raster)
    base_table_path = Path(args.base_table)
    output_path = Path(args.output)

    validate_inputs(madmex_path, base_table_path)

    base = read_base_table(base_table_path)
    print(f"filas en base regional: {len(base)}")

    with rasterio.open(madmex_path) as madmex_src:
        if madmex_src.crs is None:
            raise ValueError("El raster MADMEX no tiene CRS definido.")

        # La tabla base viene de las otras features ya alineadas en EPSG:4326
        # dentro del flujo actual. Si en el futuro cambia la convención,
        # esta parte deberá parametrizarse.
        tx, ty = transform_points_to_crs(
            base["x"].to_numpy(),
            base["y"].to_numpy(),
            "EPSG:4326",
            madmex_src.crs,
        )

        pred_xy = np.column_stack([tx, ty])

        madmex_arr, madmex_transform = crop_raster_to_points_extent(
            madmex_src,
            tx,
            ty,
        )
        madmex_arr = filter_madmex_classes(madmex_arr)

    madmex_points = valid_raster_points_dataframe(
        madmex_arr,
        madmex_transform,
    ).rename(columns={"value": "layer"})

    madmex_points["layer"] = pd.to_numeric(madmex_points["layer"], errors="coerce")

    out = base.copy()

    for class_code, out_col in OUTPUT_COLS.items():
        out[out_col] = nearest_distance_column(
            pred_xy,
            madmex_points[madmex_points["layer"] == class_code],
        )

    save_output(out, output_path)


if __name__ == "__main__":
    main()