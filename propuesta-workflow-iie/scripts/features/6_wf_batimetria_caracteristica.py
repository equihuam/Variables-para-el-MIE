#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 6_wf_batimetria_caracteristica.py

Propósito:
    Estimar, para una región específica, la batimetría característica en cada
    píxel válido del raster de referencia a partir de un raster batimétrico base
    y exportar el resultado como tabla congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script Python inicial:
    6_batimetria_caracteristica.py

Resumen del flujo:
    1. Leer el raster batimétrico base.
    2. Leer el ref_grid.tif de una región.
    3. Reproyectar el raster regional al CRS de la batimetría.
    4. Extraer solo los centros de píxel válidos de la malla regional.
    5. Recortar la batimetría a la extensión del raster regional reproyectado.
    6. Convertir la batimetría recortada a tabla de puntos de entrenamiento.
    7. Estimar la batimetría en cada píxel válido mediante vecinos cercanos
       ponderados por distancia.
    8. Exportar la tabla regional en Parquet.

Insumos principales:
    - raster batimétrico base
    - ref_grid.tif regional

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y, batimetria

Supuestos y notas:
    - La interpolación se realiza en el CRS del raster batimétrico.
    - La reproyección del raster regional usa vecino más cercano para seguir
      la lógica del flujo original.
    - Solo se conservan celdas válidas de la malla regional.
    - La aproximación usa k vecinos cercanos ponderados por distancia^-2.
    - La salida `batimetria` se conserva como variable continua.

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
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from sklearn.neighbors import NearestNeighbors


OUTPUT_FIELD = "bati_char"
K_NEIGHBORS = 7
DISTANCE_POWER = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima batimetría característica por píxel para una región específica."
    )
    parser.add_argument(
        "--batimetria-raster",
        required=True,
        help="Ruta al raster batimétrico base.",
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


def crop_bathymetry_to_region(
        bat_src: rasterio.io.DatasetReader,
        region_arr: np.ndarray,
        region_transform,
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Equivalente funcional a:
      bat_ <- crop(bat, region_)
    usando el bbox del raster regional reproyectado.
    """
    height, width = region_arr.shape
    left, top = region_transform * (0, 0)
    right, bottom = region_transform * (width, height)

    geom = box(
        min(left, right),
        min(bottom, top),
        max(left, right),
        max(bottom, top),
    )
    cropped, cropped_transform = mask(
        bat_src,
        [mapping(geom)],
        crop=True,
        filled=True,
    )

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


def predict_knn_weighted(
        train_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        target_col: str = "bat",
        k: int = K_NEIGHBORS,
        distance_power: float = DISTANCE_POWER,
) -> np.ndarray:
    train_valid = train_df[np.isfinite(train_df[target_col])].copy()

    if len(train_valid) == 0:
        return np.full(len(pred_df), np.nan, dtype=float)

    x_train = train_valid[[x_col, y_col]].to_numpy(dtype=float)
    y_train = train_valid[target_col].to_numpy(dtype=float)
    x_pred = pred_df[[x_col, y_col]].to_numpy(dtype=float)

    k_eff = min(k, len(train_valid))

    nn = NearestNeighbors(
        n_neighbors=k_eff,
        algorithm="auto",
        metric="euclidean",
    )
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

    batimetria_path = Path(args.batimetria_raster)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(batimetria_path, ref_grid_path)

    with rasterio.open(batimetria_path) as bat_src:
        if bat_src.crs is None:
            raise ValueError("El raster batimétrico no tiene CRS definido.")

        with rasterio.open(ref_grid_path) as ref_src:
            if ref_src.crs is None:
                raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

            region_arr, region_transform = reproject_raster_to_crs(ref_src, bat_src.crs)

        total_points = int(region_arr.size)
        region_points = valid_raster_points_dataframe(region_arr, region_transform)

        print(f"total puntos reproyectados: {total_points}")
        print(f"puntos válidos en malla: {len(region_points)}")

        bat_arr, bat_transform = crop_bathymetry_to_region(
            bat_src,
            region_arr,
            region_transform,
        )

    bat_points = valid_raster_points_dataframe(
        bat_arr,
        bat_transform,
    ).rename(columns={"value": "bat"})

    predictions = predict_knn_weighted(
        train_df=bat_points,
        pred_df=region_points,
        x_col="x",
        y_col="y",
        target_col="bat",
        k=K_NEIGHBORS,
        distance_power=DISTANCE_POWER,
    )

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            OUTPUT_FIELD: predictions,
        }
    )

    save_output(out, output_path)


if __name__ == "__main__":
    main()