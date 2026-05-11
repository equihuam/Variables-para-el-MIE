#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 9_condicion_dunas.py

Propósito:
    Construir, para cada región costera, una tabla de píxeles con coordenadas
    x, y y el valor rasterizado de condición de dunas (CONSERV_ED), proyectada
    al CRS de manglares, y serializar el resultado en formato PKL.

Origen:
    Traducción inicial a Python del script R:
    9_condicion_dunas.R

Resumen del flujo:
    1. Leer el raster base de dunas INEGI y los insumos vectoriales.
    2. Reproyectar dunas y regiones al CRS del raster base.
    3. Para cada región, construir una plantilla raster regional con buffer 10 km.
    4. Rasterizar el atributo CONSERV_ED sobre la plantilla y aplicar máscara regional.
    5. Reproyectar el raster resultante al CRS de manglares.
    6. Convertir el raster reproyectado a tabla con coordenadas x, y.
    7. Concatenar resultados regionales, ordenar por regionid y serializar en PKL.

Insumos principales:
    - cdv_usuev250sVII_dunas.tif
    - DunasCosteras250116.shp
    - RegionesCosteras40km.shp
    - cm-conabio.shp

Salidas principales:
    - 9_condicion_dunas.pkl

Supuestos y notas:
    - Se conserva la lógica de crop + extend + disagg del script original.
    - La rasterización del atributo CONSERV_ED se realiza en el CRS del raster
      base de dunas INEGI y luego se reproyecta al CRS de manglares.
    - Se serializa en .pkl en lugar de .rds por congruencia con el flujo Python.
    - Se conserva el orden final por regionid.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, la secuencia crop + extend + disagg de terra se aproxima en
    Python mediante construcción explícita de una plantilla alineada a la grilla
    del raster base y posterior desagregación con nearest neighbour.

Observaciones:
    Este script está pensado para ejecución headless y forma parte del flujo
    de adaptación R -> Python dentro del proyecto.
"""

from __future__ import annotations

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, xy
from rasterio.warp import calculate_default_transform, reproject


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
DELTAVP_RASTER = DROPBOX_DIR / "data" / "5_deltavp250m" / "1delt_vp_250m.tif"  # se valida por fidelidad, no se usa
DUNES_INEGI_RASTER = DROPBOX_DIR / "data" / "6_inegiSVII_dunes" / "cdv_usuev250sVII_dunas.tif"
DUNES_OTHER_SHP = DROPBOX_DIR / "data" / "7_dunes250116" / "DunasCosteras250116.shp"
COASTAL_REGIONS_SHP = DROPBOX_DIR / "data" / "3_misc_cesia" / "RegionesCosteras40km.shp"
MANGLARES_SHP = DROPBOX_DIR / "data_crude" / "02_cm-conabio" / "cm-conabio.shp"
OUTPUT_PKL = DROPBOX_DIR / "data_features" / "9_condicion_dunas.pkl"

BUFFER_METERS = 10000
DISAGG_FACTOR = 5
FIELD_NAME = "CONSERV_ED"
REGION_ID_FIELD = "myid"
NODATA_VALUE = -9999.0


def validate_inputs() -> None:
    required = [
        DELTAVP_RASTER,
        DUNES_INEGI_RASTER,
        DUNES_OTHER_SHP,
        COASTAL_REGIONS_SHP,
        MANGLARES_SHP,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_vectors(target_crs):
    dunesother = gpd.read_file(DUNES_OTHER_SHP)
    coastalr = gpd.read_file(COASTAL_REGIONS_SHP)
    manglares = gpd.read_file(MANGLARES_SHP)

    if dunesother.crs is None:
        raise ValueError(f"{DUNES_OTHER_SHP.name} no tiene CRS definido.")
    if manglares.crs is None:
        raise ValueError(f"{MANGLARES_SHP.name} no tiene CRS definido.")

    # Replica el R:
    # coastalr <- st_set_crs(coastalr, 4326)
    coastalr = coastalr.set_crs(4326, allow_override=True)

    dunesother_reproj = dunesother.to_crs(target_crs)
    coastalr_reproj = coastalr.to_crs(target_crs)

    if FIELD_NAME not in dunesother_reproj.columns:
        raise ValueError(f"No existe el campo '{FIELD_NAME}' en {DUNES_OTHER_SHP.name}")
    if REGION_ID_FIELD not in coastalr_reproj.columns:
        raise ValueError(f"No existe el campo '{REGION_ID_FIELD}' en {COASTAL_REGIONS_SHP.name}")

    return dunesother_reproj, coastalr_reproj, manglares


def align_bounds_to_grid(bounds, transform: Affine):
    left, bottom, right, top = bounds

    xres = transform.a
    yres = abs(transform.e)
    x0 = transform.c
    y0 = transform.f

    aligned_left = x0 + math.floor((left - x0) / xres) * xres
    aligned_right = x0 + math.ceil((right - x0) / xres) * xres
    aligned_top = y0 - math.floor((y0 - top) / yres) * yres
    aligned_bottom = y0 - math.ceil((y0 - bottom) / yres) * yres

    return aligned_left, aligned_bottom, aligned_right, aligned_top


def build_extended_template(src: rasterio.io.DatasetReader, geom, factor: int):
    """
    Aproximación fiel a:
      dinegi_cropped <- extend(crop(dunesinegi, coastr_poly), coastr_poly)
      dinegi_disagg <- disagg(dinegi_cropped, fact=5)
    """
    left, bottom, right, top = align_bounds_to_grid(geom.bounds, src.transform)

    xres = src.transform.a
    yres = abs(src.transform.e)

    width = int(round((right - left) / xres))
    height = int(round((top - bottom) / yres))

    coarse_transform = Affine(xres, 0.0, left, 0.0, -yres, top)

    coarse_array = np.full(
        (height, width),
        src.nodata if src.nodata is not None else 0,
        dtype=src.dtypes[0],
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=coarse_array,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=coarse_transform,
        dst_crs=src.crs,
        dst_nodata=src.nodata,
        resampling=Resampling.nearest,
    )

    fine_width = width * factor
    fine_height = height * factor
    fine_transform = Affine(
        coarse_transform.a / factor,
        0.0,
        coarse_transform.c,
        0.0,
        coarse_transform.e / factor,
        coarse_transform.f,
        )

    fine_array = np.full(
        (fine_height, fine_width),
        src.nodata if src.nodata is not None else 0,
        dtype=src.dtypes[0],
    )

    reproject(
        source=coarse_array,
        destination=fine_array,
        src_transform=coarse_transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=fine_transform,
        dst_crs=src.crs,
        dst_nodata=src.nodata,
        resampling=Resampling.nearest,
    )

    return fine_array, fine_transform


def rasterize_polygon_mask(shape, transform, geom):
    """
    Equivalente a:
      coastr_rast <- rasterize(coastr_poly, dinegi_disagg)
    """
    return rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )


def rasterize_dunes_condition(shape, transform, dunesother_reproj: gpd.GeoDataFrame):
    """
    Equivalente a:
      dother_rast <- rasterize(dother, coastr_rast, field="CONSERV_ED")
    """
    shapes = (
        (geom, value)
        for geom, value in zip(dunesother_reproj.geometry, dunesother_reproj[FIELD_NAME])
        if geom is not None and not geom.is_empty
    )

    return rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=NODATA_VALUE,
        dtype="float32",
    )


def reproject_array_to_crs(arr: np.ndarray, src_transform, src_crs, dst_crs, nodata_value: float = NODATA_VALUE):
    height, width = arr.shape

    left, top = src_transform * (0, 0)
    right, bottom = src_transform * (width, height)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        width,
        height,
        left,
        bottom,
        right,
        top,
    )

    dst_arr = np.full((dst_height, dst_width), nodata_value, dtype="float32")

    reproject(
        source=arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=nodata_value,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=nodata_value,
        resampling=Resampling.nearest,
    )

    return dst_arr, dst_transform


def raster_points_dataframe(arr: np.ndarray, transform) -> pd.DataFrame:
    """
    Aproximación a as.data.frame(rast, xy = TRUE) de terra.
    Se conservan todas las celdas como filas.
    """
    height, width = arr.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "x": np.asarray(xs).ravel(),
            "y": np.asarray(ys).ravel(),
            "value": arr.ravel(),
        }
    )


def process_region(row, dunesinegi, dunesother_reproj, manglares_crs) -> pd.DataFrame:
    region_geom = row.geometry
    region_id = row[REGION_ID_FIELD]

    if region_geom is None or region_geom.is_empty:
        return pd.DataFrame(columns=["x", "y", "value", "pixid", "regionid"])

    # coastr_poly <- buffer(vect(coastalr_reproj[i,]), width=10000)
    coastr_poly = region_geom.buffer(BUFFER_METERS)

    # dinegi_cropped + extend + disagg
    dinegi_disagg, template_transform = build_extended_template(
        dunesinegi,
        coastr_poly,
        DISAGG_FACTOR,
    )

    # coastr_rast <- rasterize(coastr_poly, dinegi_disagg)
    coastr_rast = rasterize_polygon_mask(
        shape=dinegi_disagg.shape,
        transform=template_transform,
        geom=coastr_poly,
    )

    # dother_rast <- rasterize(dother, coastr_rast, field="CONSERV_ED")
    dother_rast = rasterize_dunes_condition(
        shape=coastr_rast.shape,
        transform=template_transform,
        dunesother_reproj=dunesother_reproj,
    )

    # dother_rast[is.na(coastr_rast)] <- NA
    dother_rast = dother_rast.astype("float32")
    dother_rast[coastr_rast == 0] = NODATA_VALUE

    # dother_rast_ <- project(dother_rast, y = crs(manglares), method = "near")
    dother_rast_proj, proj_transform = reproject_array_to_crs(
        arr=dother_rast,
        src_transform=template_transform,
        src_crs=dunesinegi.crs,
        dst_crs=manglares_crs,
        nodata_value=NODATA_VALUE,
    )

    # region_points <- as.data.frame(dother_rast_, xy = TRUE)
    region_points = raster_points_dataframe(dother_rast_proj, proj_transform)

    region_points["pixid"] = np.arange(1, len(region_points) + 1)
    region_points["regionid"] = region_id

    return region_points


def main() -> None:
    validate_inputs()

    # Se conserva por fidelidad al R, aunque no se use en la lógica.
    _ = DELTAVP_RASTER

    with rasterio.open(DUNES_INEGI_RASTER) as dunesinegi:
        if dunesinegi.crs is None:
            raise ValueError("El raster de dunas INEGI no tiene CRS.")

        dunesother_reproj, coastalr_reproj, manglares = load_vectors(dunesinegi.crs)

        df_list: list[pd.DataFrame] = []

        for i, (_, row) in enumerate(coastalr_reproj.iterrows(), start=1):
            print(i)
            region_df = process_region(
                row=row,
                dunesinegi=dunesinegi,
                dunesother_reproj=dunesother_reproj,
                manglares_crs=manglares.crs,
            )
            df_list.append(region_df)

    full_df = pd.concat(df_list, ignore_index=True)
    full_df = full_df.sort_values("regionid", ascending=True).reset_index(drop=True)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_pickle(OUTPUT_PKL)

    print(f"OK -> {OUTPUT_PKL}")


if __name__ == "__main__":
    main()