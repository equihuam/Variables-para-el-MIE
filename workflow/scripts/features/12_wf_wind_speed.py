#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug: 12_wf_wind_speed_debug_v2.py

Valida la extracción de velocidad media del viento por región contra el flujo R:
  wspeed <- rast(nc)
  wspeed_mean <- app(wspeed, mean)
  wspeed_reproj <- project(wspeed_mean, y = crs(struct), method = "near")
  wspeed_reproj <- crop(wspeed_reproj, struct)  # recorte por extensión, no máscara
  region_ <- project(region_, y = crs(struct), method = "near")
  extracted <- extract(wspeed_reproj, as.points(region_))

Salida principal:
  regionid, pixid, x, y, windspeed
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling

OUTPUT_FIELD = "windspeed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae velocidad media del viento por píxel para una región específica."
    )
    parser.add_argument("--structures-shp", required=True)
    parser.add_argument("--wind-nc", required=True)
    parser.add_argument("--ref-grid", required=True)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-grid-output", default=None)
    parser.add_argument("--debug-metadata-output", default=None)
    parser.add_argument("--debug-wind-raster-output", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


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
    return struct


def valid_raster_points_dataframe(arr: np.ndarray, transform) -> pd.DataFrame:
    valid_mask = np.isfinite(arr)
    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", "value"])
    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "row": rows.astype(int),
            "col": cols.astype(int),
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "value": arr[rows, cols],
        }
    )


def reproject_raster_to_crs(
    src: rasterio.io.DatasetReader,
    dst_crs,
    resampling=Resampling.nearest,
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
        resampling=resampling,
    )
    return dst, transform


def open_wind_dataset(nc_path: Path):
    """Abre NetCDF con rasterio; si hay subdatasets, abre el primero.

    terra::rast(nc) suele exponer la variable como raster multilayer. En algunos
    NetCDF, rasterio.open(path) devuelve subdatasets. En ese caso usamos el primer
    subdataset disponible, que debe corresponder a la variable principal del archivo.
    """
    src = rasterio.open(nc_path)
    if src.subdatasets:
        subdataset = src.subdatasets[0]
        src.close()
        src = rasterio.open(subdataset)
    return src


def build_time_mean_from_nc(nc_path: Path, fallback_crs: str = "EPSG:4326") -> tuple[np.ndarray, dict, dict[str, Any]]:
    with open_wind_dataset(nc_path) as src:
        src_crs = src.crs if src.crs is not None else fallback_crs
        if src.count < 1:
            raise ValueError("El NetCDF de viento no contiene bandas.")

        arr = src.read().astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Mean of empty slice",
                category=RuntimeWarning,
            )
            mean_arr = np.nanmean(arr, axis=0).astype(np.float32)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.nan,
            crs=src_crs,
        )
        meta = {
            "wind_src_crs": str(src_crs),
            "wind_src_count": int(src.count),
            "wind_src_width": int(src.width),
            "wind_src_height": int(src.height),
            "wind_src_nodata": nodata,
            "wind_mean_min": float(np.nanmin(mean_arr)),
            "wind_mean_max": float(np.nanmax(mean_arr)),
            "wind_mean_mean": float(np.nanmean(mean_arr)),
        }
    return mean_arr, profile, meta


def reproject_and_crop_wind_mean(
    wind_mean_arr: np.ndarray,
    wind_profile: dict,
    struct_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, dict, dict[str, Any]]:
    target_crs = struct_gdf.crs
    with MemoryFile() as memfile_in:
        with memfile_in.open(**wind_profile) as src:
            src.write(wind_mean_arr, 1)
            reproj_arr, reproj_transform = reproject_raster_to_crs(
                src,
                target_crs,
                resampling=Resampling.nearest,
            )
            reproj_profile = src.profile.copy()
            reproj_profile.update(
                driver="GTiff",
                height=reproj_arr.shape[0],
                width=reproj_arr.shape[1],
                count=1,
                dtype="float32",
                crs=target_crs,
                transform=reproj_transform,
                nodata=np.nan,
            )

    # terra::crop(raster, vector) recorta por la extensión del vector; no aplica
    # máscara geométrica. Por eso aquí usamos únicamente el bounding box de
    # struct_gdf y leemos esa ventana del raster reproyectado.
    with MemoryFile() as memfile_reproj:
        with memfile_reproj.open(**reproj_profile) as reproj_src:
            reproj_src.write(reproj_arr, 1)

            minx, miny, maxx, maxy = struct_gdf.total_bounds
            window = from_bounds(
                left=float(minx),
                bottom=float(miny),
                right=float(maxx),
                top=float(maxy),
                transform=reproj_src.transform,
            ).round_offsets().round_lengths()

            cropped_arr = reproj_src.read(
                1,
                window=window,
                boundless=True,
                fill_value=np.nan,
            ).astype(np.float32)
            cropped_transform = reproj_src.window_transform(window)

            cropped_profile = reproj_profile.copy()
            cropped_profile.update(
                height=cropped_arr.shape[0],
                width=cropped_arr.shape[1],
                transform=cropped_transform,
                nodata=np.nan,
            )
    meta = {
        "wind_reproj_width": int(reproj_arr.shape[1]),
        "wind_reproj_height": int(reproj_arr.shape[0]),
        "wind_crop_width": int(cropped_arr.shape[1]),
        "wind_crop_height": int(cropped_arr.shape[0]),
        "wind_crop_valid": int(np.isfinite(cropped_arr).sum()),
        "wind_crop_min": float(np.nanmin(cropped_arr)) if np.isfinite(cropped_arr).any() else np.nan,
        "wind_crop_max": float(np.nanmax(cropped_arr)) if np.isfinite(cropped_arr).any() else np.nan,
        "wind_crop_mean": float(np.nanmean(cropped_arr)) if np.isfinite(cropped_arr).any() else np.nan,
    }
    return cropped_arr, cropped_profile, meta


def sample_raster_at_points(raster_arr: np.ndarray, raster_profile: dict, points_xy: np.ndarray) -> np.ndarray:
    with MemoryFile() as memfile:
        with memfile.open(**raster_profile) as src:
            src.write(raster_arr, 1)
            samples = list(src.sample(points_xy))
            values = np.array([s[0] if len(s) else np.nan for s in samples], dtype=float)
    return values


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow")
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Extensión no soportada: {output_path.suffix}")


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {output_path.suffix}")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()
    structures_path = Path(args.structures_shp)
    wind_nc_path = Path(args.wind_nc)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(structures_path, wind_nc_path, ref_grid_path)

    struct_gdf = load_structures(structures_path)
    wind_mean_arr, wind_profile, wind_mean_meta = build_time_mean_from_nc(wind_nc_path)
    wind_reproj_arr, wind_reproj_profile, wind_crop_meta = reproject_and_crop_wind_mean(
        wind_mean_arr, wind_profile, struct_gdf
    )

    target_crs = struct_gdf.crs
    with rasterio.open(ref_grid_path) as ref_src:
        if ref_src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
        src_total = int(ref_src.width * ref_src.height)
        src_masked = ref_src.read(1, masked=True)
        src_valid = int((~np.ma.getmaskarray(src_masked)).sum())
        region_arr, region_transform = reproject_raster_to_crs(
            ref_src,
            target_crs,
            resampling=Resampling.nearest,
        )

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)
    region_points.insert(0, "regionid", region_id)
    region_points.insert(1, "pixid", np.arange(1, len(region_points) + 1))

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)
    extracted = sample_raster_at_points(wind_reproj_arr, wind_reproj_profile, pred_xy)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(),
            "y": region_points["y"].to_numpy(),
            OUTPUT_FIELD: extracted,
        }
    )

    if args.debug_grid_output:
        debug_grid = out.copy()
        debug_grid["ref_value"] = region_points["value"].to_numpy()
        save_table(debug_grid, Path(args.debug_grid_output))
        log(f"debug grid -> {args.debug_grid_output}", args.verbose)

    if args.debug_wind_raster_output:
        wind_points = valid_raster_points_dataframe(wind_reproj_arr, wind_reproj_profile["transform"])
        save_table(wind_points, Path(args.debug_wind_raster_output))
        log(f"debug wind raster -> {args.debug_wind_raster_output}", args.verbose)

    if args.debug_metadata_output:
        meta = {
            "regionid": region_id,
            "structures_crs": str(struct_gdf.crs),
            "ref_src_total": src_total,
            "ref_src_valid_masked": src_valid,
            "region_reprojected_total": total_points,
            "region_reprojected_valid": int(len(region_points)),
            "output_min": float(np.nanmin(extracted)) if np.isfinite(extracted).any() else np.nan,
            "output_max": float(np.nanmax(extracted)) if np.isfinite(extracted).any() else np.nan,
            "output_mean": float(np.nanmean(extracted)) if np.isfinite(extracted).any() else np.nan,
            **wind_mean_meta,
            **wind_crop_meta,
        }
        save_table(pd.DataFrame([meta]), Path(args.debug_metadata_output))
        log(f"debug metadata -> {args.debug_metadata_output}", args.verbose)

    if args.verbose:
        log(f"total puntos GeoTIFF original: {src_total}", True)
        log(f"puntos válidos GeoTIFF original masked: {src_valid}", True)
        log(f"total puntos reproyectados: {total_points}", True)
        log(f"puntos válidos usados en malla: {len(region_points)}", True)
        log(f"windspeed rango: [{np.nanmin(extracted)}, {np.nanmax(extracted)}]", True)

    save_output(out, output_path)


if __name__ == "__main__":
    main()
