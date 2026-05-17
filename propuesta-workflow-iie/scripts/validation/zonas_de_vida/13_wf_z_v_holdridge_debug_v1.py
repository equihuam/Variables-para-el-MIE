#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug v1 para 13_z_v_holdridge.

Replica la lógica funcional del R:
  zvh <- project(zvh, y = crs(manglares), method = "near")
  region_ <- project(region_, y = crs(manglares), method = "near")
  zvh_ <- crop(zvh, region_)
  zvh_points$layer <- as.factor(zvh_points$layer)
  kknn(layer ~ x + y, zvh_points, region_points,
       distance = 2, k = 1, kernel = "rectangular")

Notas:
- Clasificación categórica 1-NN.
- `distance-mode=kknn_scaled` escala x/y con la desviación estándar del
  entrenamiento, emulando `scale=TRUE` de kknn.
- Intenta preservar etiquetas categóricas del raster si están disponibles vía
  GDAL CategoryNames. Si no, usa los códigos raster como etiquetas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from sklearn.neighbors import NearestNeighbors

OUTPUT_FIELD = "zvh"
K_NEIGHBORS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica zona de vida Holdridge por pixel para una región específica."
    )
    parser.add_argument("--mangroves-shp", required=True, help="Shapefile usado para definir CRS de trabajo.")
    parser.add_argument("--zvh-raster", required=True, help="Raster de zonas de vida Holdridge.")
    parser.add_argument("--ref-grid", required=True, help="ref_grid.tif regional.")
    parser.add_argument("--region-id", required=True, help="Identificador regional, por ejemplo region_1.")
    parser.add_argument("--output", required=True, help="Salida .parquet.")
    parser.add_argument(
        "--distance-mode",
        choices=["kknn_scaled", "raw"],
        default="kknn_scaled",
        help="Modo de distancia para 1-NN. kknn_scaled emula scale=TRUE de kknn.",
    )
    parser.add_argument(
        "--debug-grid-output",
        default=None,
        help="CSV opcional con la grilla regional reproyectada.",
    )
    parser.add_argument(
        "--debug-zvh-points-output",
        default=None,
        help="CSV opcional con puntos de entrenamiento ZVH recortados.",
    )
    parser.add_argument(
        "--debug-metadata-output",
        default=None,
        help="CSV opcional con metadatos de la corrida.",
    )
    parser.add_argument("--verbose", action="store_true", help="Imprime diagnósticos.")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_target_crs(path: Path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"El shapefile está vacío: {path}")
    if gdf.crs is None:
        raise ValueError(f"El shapefile no tiene CRS: {path}")
    return gdf.crs


def read_category_names(path: Path) -> dict[int, str]:
    """Intenta leer nombres categóricos GDAL; si no existen, regresa {}."""
    mapping_out: dict[int, str] = {}
    try:
        from osgeo import gdal  # type: ignore

        ds = gdal.Open(str(path))
        if ds is None:
            return {}
        band = ds.GetRasterBand(1)
        cats = band.GetCategoryNames()
        if cats:
            for i, cat in enumerate(cats):
                if cat is not None and str(cat).strip() != "":
                    mapping_out[int(i)] = str(cat)
        ds = None
    except Exception:
        return {}
    return mapping_out


def reproject_raster_to_crs(src: rasterio.io.DatasetReader, dst_crs) -> tuple[np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    dst = np.full((height, width), np.nan, dtype=np.float32)
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


def crop_array_to_region_bbox(
    arr: np.ndarray,
    transform,
    crs,
    region_arr: np.ndarray,
    region_transform,
) -> tuple[np.ndarray, rasterio.Affine]:
    height, width = region_arr.shape
    left, top = region_transform * (0, 0)
    right, bottom = region_transform * (width, height)
    geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as src:
            src.write(arr.astype(np.float32), 1)
            cropped, cropped_transform = mask(
                src,
                [mapping(geom)],
                crop=True,
                filled=False,
            )
            out = cropped[0].astype("float64")
            out = np.where(np.ma.getmaskarray(out), np.nan, np.asarray(out, dtype=float))
    return out, cropped_transform


def valid_raster_points_dataframe(arr: np.ndarray, transform, value_col: str = "value") -> pd.DataFrame:
    valid_mask = np.isfinite(arr)
    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", value_col])
    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "row": rows.astype(int),
            "col": cols.astype(int),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            value_col: arr[rows, cols],
        }
    )


def scale_xy(train_xy: np.ndarray, pred_xy: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "raw":
        return train_xy, pred_xy, np.array([1.0, 1.0], dtype=float)
    sd = np.nanstd(train_xy, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return train_xy / sd, pred_xy / sd, sd


def predict_1nn_labels(train_df: pd.DataFrame, pred_df: pd.DataFrame, distance_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if train_df.empty:
        return np.full(len(pred_df), None, dtype=object), np.array([np.nan, np.nan])
    train_xy = train_df[["x", "y"]].to_numpy(dtype=float)
    pred_xy = pred_df[["x", "y"]].to_numpy(dtype=float)
    train_scaled, pred_scaled, sd = scale_xy(train_xy, pred_xy, distance_mode)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(train_scaled)
    _, idx = nn.kneighbors(pred_scaled, return_distance=True)
    labels = train_df["label"].to_numpy(dtype=object)[idx[:, 0]]
    return labels.astype(object), sd


def label_from_code(value: Any, cat_map: dict[int, str]) -> str:
    if pd.isna(value):
        return ""
    code_float = float(value)
    code_int = int(round(code_float))
    if abs(code_float - code_int) < 1e-6 and code_int in cat_map:
        return cat_map[code_int]
    if abs(code_float - code_int) < 1e-6:
        return str(code_int)
    return str(code_float)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Extensión no soportada: {path.suffix}")


def main() -> None:
    args = parse_args()
    mangroves_path = Path(args.mangroves_shp)
    zvh_path = Path(args.zvh_raster)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(mangroves_path, zvh_path, ref_grid_path)
    target_crs = load_target_crs(mangroves_path)
    cat_map = read_category_names(zvh_path)

    with rasterio.open(ref_grid_path) as ref_src:
        if ref_src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
        original_valid = int(np.isfinite(ref_src.read(1, masked=True).filled(np.nan)).sum())
        region_arr, region_transform = reproject_raster_to_crs(ref_src, target_crs)

    region_points = valid_raster_points_dataframe(region_arr, region_transform, value_col="ref_value")
    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    with rasterio.open(zvh_path) as zvh_src:
        if zvh_src.crs is None:
            raise ValueError("El raster ZVH no tiene CRS definido.")
        zvh_arr, zvh_transform = reproject_raster_to_crs(zvh_src, target_crs)
        zvh_region_arr, zvh_region_transform = crop_array_to_region_bbox(
            zvh_arr, zvh_transform, target_crs, region_arr, region_transform
        )

    zvh_points = valid_raster_points_dataframe(zvh_region_arr, zvh_region_transform, value_col="layer")
    zvh_points["label"] = [label_from_code(v, cat_map) for v in zvh_points["layer"]]

    predictions, sd = predict_1nn_labels(zvh_points, region_points, args.distance_mode)

    out = pd.DataFrame(
        {
            "regionid": region_points["regionid"].to_numpy(),
            "pixid": region_points["pixid"].to_numpy(),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            OUTPUT_FIELD: predictions,
        }
    )

    if args.debug_grid_output:
        save_table(region_points[["regionid", "pixid", "row", "col", "x", "y", "ref_value"]], Path(args.debug_grid_output))
    if args.debug_zvh_points_output:
        save_table(zvh_points[["row", "col", "x", "y", "layer", "label"]], Path(args.debug_zvh_points_output))
    if args.debug_metadata_output:
        vc = pd.Series(predictions).value_counts(dropna=False).to_dict()
        meta = {
            "regionid": region_id,
            "target_crs": str(target_crs),
            "original_valid_points": original_valid,
            "reprojected_valid_points": int(len(region_points)),
            "zvh_training_points": int(len(zvh_points)),
            "n_labels_training": int(zvh_points["label"].nunique(dropna=True)) if len(zvh_points) else 0,
            "distance_mode": args.distance_mode,
            "sd_x": float(sd[0]) if len(sd) else np.nan,
            "sd_y": float(sd[1]) if len(sd) else np.nan,
            "category_map_found": bool(cat_map),
            "prediction_counts_json": json.dumps({str(k): int(v) for k, v in vc.items()}, ensure_ascii=False),
        }
        save_table(pd.DataFrame([meta]), Path(args.debug_metadata_output))

    log(f"puntos válidos originales: {original_valid}", args.verbose)
    log(f"puntos válidos reproyectados: {len(region_points)}", args.verbose)
    log(f"puntos entrenamiento ZVH: {len(zvh_points)}", args.verbose)
    log(f"etiquetas entrenamiento: {sorted(zvh_points['label'].dropna().unique().tolist())[:20]}", args.verbose)
    log(f"modo distancia: {args.distance_mode}; sd=({sd[0]}, {sd[1]})", args.verbose)

    save_table(out, output_path)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
