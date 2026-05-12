#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from sklearn.neighbors import KNeighborsClassifier


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
    return parser.parse_args()


def validate_inputs(coast_types_shp: Path, ref_grid: Path) -> None:
    missing = [str(p) for p in [coast_types_shp, ref_grid] if not p.exists()]
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
    return costas


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


def rasterize_coast_types(
        shape,
        transform,
        costas: gpd.GeoDataFrame,
) -> tuple[np.ndarray, dict[int, str]]:
    categories = pd.Categorical(costas[FIELD_NAME].astype(str))
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
    )

    code_to_label = {
        int(code + 1): str(cat)
        for code, cat in enumerate(categories.categories)
    }
    return arr, code_to_label


def coast_points_from_raster(
        costas_rast: np.ndarray,
        transform,
        code_to_label: dict[int, str],
) -> pd.DataFrame:
    valid_mask = costas_rast > 0
    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", FIELD_NAME])

    xs, ys = xy(transform, rows, cols, offset="center")
    codes = costas_rast[rows, cols]
    labels = [code_to_label[int(c)] for c in codes]

    return pd.DataFrame(
        {
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            FIELD_NAME: labels,
        }
    )


def fit_knn_labels(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> np.ndarray:
    if train_df.empty:
        return np.full(len(pred_df), None, dtype=object)

    x_train = train_df[["x", "y"]].to_numpy(dtype=float)
    y_train = train_df[FIELD_NAME].astype(str).to_numpy()
    x_pred = pred_df[["x", "y"]].to_numpy(dtype=float)

    clf = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights="uniform",
        algorithm="auto",
        metric="euclidean",
    )
    clf.fit(x_train, y_train)

    pred = clf.predict(x_pred)
    return pred.astype(object)


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

    coast_types_path = Path(args.coast_types_shp)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(coast_types_path, ref_grid_path)
    costas = load_coast_types(coast_types_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_crs(src, costas.crs)

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)

    print(f"total puntos reproyectados: {total_points}")
    print(f"puntos válidos en malla: {len(region_points)}")

    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id
    region_points[OUTPUT_FIELD] = None

    costas_rast, code_to_label = rasterize_coast_types(
        shape=region_arr.shape,
        transform=region_transform,
        costas=costas,
    )

    costas_table = coast_points_from_raster(costas_rast, region_transform, code_to_label)

    if not costas_table.empty:
        coast_prediction = fit_knn_labels(costas_table, region_points)
        region_points[OUTPUT_FIELD] = coast_prediction

    out = region_points[["regionid", "pixid", "x", "y", OUTPUT_FIELD]].copy()
    save_output(out, output_path)


if __name__ == "__main__":
    main()