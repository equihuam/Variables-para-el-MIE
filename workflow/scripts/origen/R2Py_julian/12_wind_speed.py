#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 12_wf_wind_speed.py

Propósito:
    Estimar, para una región específica, la velocidad media del viento en cada
    píxel válido del raster de referencia a partir de un NetCDF mensual de
    velocidad del viento, y exportar el resultado como tabla congruente por
    píxel en formato Parquet.

Origen:
    Adaptación a workflow del script R original:
    12_wind_speed.R

Resumen del flujo:
    1. Leer el shapefile de estructuras para definir el CRS de trabajo.
    2. Leer el NetCDF mensual de velocidad del viento.
    3. Calcular la media temporal del cubo de viento.
    4. Reproyectar el raster medio al CRS de trabajo.
    5. Recortar el raster medio al shapefile de estructuras.
    6. Leer el ref_grid.tif de una región.
    7. Reproyectar el ref_grid regional al mismo CRS de trabajo.
    8. Extraer solo los centros de píxel válidos de la malla regional.
    9. Muestrear el raster medio de viento en los puntos regionales.
    10. Exportar la tabla regional en Parquet.

Insumos principales:
    - shapefile de estructuras
    - NetCDF mensual de velocidad del viento
    - ref_grid.tif regional

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y, wind_speed

Supuestos y notas:
    - La lógica sigue el script R original: promedio temporal + extracción
      raster en puntos.
    - No se usa k-NN en esta variable.
    - Solo se conservan celdas válidas de la malla regional.
    - La salida `wind_speed` se conserva como variable continua.

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
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling


OUTPUT_FIELD = "wind_speed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae velocidad media del viento por píxel para una región específica."
    )
    parser.add_argument(
        "--structures-shp",
        required=True,
        help="Ruta al shapefile de estructuras usado para definir CRS y recorte.",
    )
    parser.add_argument(
        "--wind-nc",
        required=True,
        help="Ruta al NetCDF mensual de velocidad del viento.",
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
    return struct


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


def build_time_mean_from_nc(nc_path: Path) -> tuple[np.ndarray, dict]:
    """
    Abre el NetCDF con rasterio, lee todas las bandas y calcula la media temporal.
    Replica conceptualmente:
      wspeed <- rast(...)
      wspeed_mean <- app(wspeed, mean)
    """
    with rasterio.open(nc_path) as src:
        if src.crs is None:
            raise ValueError("El NetCDF de viento no tiene CRS definido.")
        if src.count < 1:
            raise ValueError("El NetCDF de viento no contiene bandas.")

        arr = src.read().astype(np.float32)  # shape: (bands, rows, cols)

        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        mean_arr = np.nanmean(arr, axis=0).astype(np.float32)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.nan,
        )

    return mean_arr, profile


def reproject_and_crop_wind_mean(
        wind_mean_arr: np.ndarray,
        wind_profile: dict,
        struct_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, dict]:
    """
    Replica conceptualmente:
      wspeed_reproj <- project(wspeed_mean, y = crs(struct), method = "near")
      wspeed_reproj <- crop(wspeed_reproj, struct)
    """
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

    with MemoryFile() as memfile_reproj:
        with memfile_reproj.open(**reproj_profile) as reproj_src:
            reproj_src.write(reproj_arr, 1)

            cropped, cropped_transform = mask(
                reproj_src,
                struct_gdf.geometry,
                crop=True,
                filled=True,
                nodata=np.nan,
            )

            cropped_arr = cropped[0].astype(np.float32)

            cropped_profile = reproj_profile.copy()
            cropped_profile.update(
                height=cropped_arr.shape[0],
                width=cropped_arr.shape[1],
                transform=cropped_transform,
                nodata=np.nan,
            )

    return cropped_arr, cropped_profile


def sample_raster_at_points(raster_arr: np.ndarray, raster_profile: dict, points_xy: np.ndarray) -> np.ndarray:
    """
    Replica conceptualmente:
      points_ <- as.points(region_)
      extracted <- extract(wspeed_reproj, points_)
      region_points$windspeed <- extracted$mean
    """
    with MemoryFile() as memfile:
        with memfile.open(**raster_profile) as src:
            src.write(raster_arr, 1)
            samples = list(src.sample(points_xy))
            values = np.array([s[0] if len(s) else np.nan for s in samples], dtype=float)
    return values


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
    wind_nc_path = Path(args.wind_nc)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(structures_path, wind_nc_path, ref_grid_path)

    struct_gdf = load_structures(structures_path)

    wind_mean_arr, wind_profile = build_time_mean_from_nc(wind_nc_path)
    wind_reproj_arr, wind_reproj_profile = reproject_and_crop_wind_mean(
        wind_mean_arr,
        wind_profile,
        struct_gdf,
    )

    target_crs = struct_gdf.crs

    with rasterio.open(ref_grid_path) as ref_src:
        if ref_src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        region_arr, region_transform = reproject_raster_to_crs(
            ref_src,
            target_crs,
            resampling=Resampling.nearest,
        )

    total_points = int(region_arr.size)
    region_points = valid_raster_points_dataframe(region_arr, region_transform)

    print(f"total puntos reproyectados: {total_points}")
    print(f"puntos válidos en malla: {len(region_points)}")

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

    save_output(out, output_path)


if __name__ == "__main__":
    main()