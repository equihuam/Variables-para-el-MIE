#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 13_wf_z_v_holdridge.py

Propósito:
    Clasificar, para una región específica, la zona de vida Holdridge más
    probable para cada píxel válido del raster de referencia y exportar el
    resultado como tabla congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script Python inicial:
    13_z_v_holdridge.py

Resumen del flujo:
    1. Leer el shapefile de manglares para obtener el CRS de trabajo.
    2. Leer el raster de zonas de vida Holdridge.
    3. Leer el ref_grid.tif de una región.
    4. Reproyectar el ref_grid regional al CRS de trabajo.
    5. Extraer solo los centros de píxel válidos de la malla regional.
    6. Reproyectar el raster Holdridge al mismo CRS de trabajo.
    7. Recortar el raster Holdridge a la extensión de la región.
    8. Convertir el raster Holdridge recortado a tabla de puntos etiquetados.
    9. Ajustar un clasificador 1-NN sobre los píxeles etiquetados de Holdridge.
    10. Predecir una clase de zona de vida para cada píxel válido de la región.
    11. Exportar la tabla regional en Parquet.

Insumos principales:
    - shapefile de manglares, usado para definir el CRS de trabajo
    - raster de zonas de vida Holdridge
    - ref_grid.tif regional

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y, zvh

Supuestos y notas:
    - La clasificación se realiza en el CRS del shapefile de manglares.
    - Tanto el raster Holdridge como el raster regional se reproyectan con
      vecino más cercano para seguir la lógica del flujo original.
    - Solo se conservan celdas válidas de la malla regional.
    - La predicción sigue la lógica funcional de clasificación 1-NN.
    - La salida `zvh` se conserva como clase numérica predicha.

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
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from sklearn.neighbors import KNeighborsClassifier


OUTPUT_FIELD = "zvh"
K_NEIGHBORS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica zona de vida Holdridge por píxel para una región específica."
    )
    parser.add_argument(
        "--mangroves-shp",
        required=True,
        help="Ruta al shapefile usado para definir el CRS de trabajo.",
    )
    parser.add_argument(
        "--zvh-raster",
        required=True,
        help="Ruta al raster de zonas de vida Holdridge.",
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


def load_target_crs(path: Path):
    manglares = gpd.read_file(path)
    if manglares.empty:
        raise ValueError(f"El shapefile de manglares está vacío: {path}")
    if manglares.crs is None:
        raise ValueError(f"El shapefile de manglares no tiene CRS: {path}")
    return manglares.crs


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


def crop_raster_to_region(
        src: rasterio.io.DatasetReader,
        region_arr: np.ndarray,
        region_transform,
) -> tuple[np.ndarray, rasterio.Affine]:
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
        src,
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


def fit_knn_labels(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    train_valid = train_df[np.isfinite(train_df["layer"])].copy()

    if train_valid.empty:
        return np.full(len(pred_df), np.nan, dtype=float)

    x_train = train_valid[["x", "y"]].to_numpy(dtype=float)
    y_train = train_valid["layer"].astype(int).to_numpy()
    x_pred = pred_df[["x", "y"]].to_numpy(dtype=float)

    clf = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights="uniform",
        algorithm="auto",
        metric="euclidean",
    )
    clf.fit(x_train, y_train)

    pred = clf.predict(x_pred)
    return pred.astype(float)


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

    mangroves_path = Path(args.mangroves_shp)
    zvh_raster_path = Path(args.zvh_raster)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(mangroves_path, zvh_raster_path, ref_grid_path)

    target_crs = load_target_crs(mangroves_path)

    with rasterio.open(ref_grid_path) as ref_src:
        if ref_src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_crs(ref_src, target_crs)

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)

    print(f"total puntos reproyectados: {total_points}")
    print(f"puntos válidos en malla: {len(region_points)}")

    with rasterio.open(zvh_raster_path) as zvh_raw:
        if zvh_raw.crs is None:
            raise ValueError("El raster de zonas de vida Holdridge no tiene CRS definido.")

        zvh_arr, zvh_transform = reproject_raster_to_crs(zvh_raw, target_crs)

        profile = zvh_raw.profile.copy()
        profile.update(
            driver="GTiff",
            height=zvh_arr.shape[0],
            width=zvh_arr.shape[1],
            count=1,
            dtype="float32",
            crs=target_crs,
            transform=zvh_transform,
            nodata=np.nan,
        )

        from rasterio.io import MemoryFile

        with MemoryFile() as memfile:
            with memfile.open(**profile) as zvh_src:
                zvh_src.write(zvh_arr, 1)

                zvh_region_arr, zvh_region_transform = crop_raster_to_region(
                    zvh_src,
                    region_arr,
                    region_transform,
                )

    zvh_points = valid_raster_points_dataframe(
        zvh_region_arr,
        zvh_region_transform,
    ).rename(columns={"value": "layer"})

    predictions = fit_knn_labels(zvh_points, region_points)

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