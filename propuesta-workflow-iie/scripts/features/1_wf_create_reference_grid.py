#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 1_wf_create_reference_grid.py

Propósito:
    Generar un raster de referencia congruente para una región específica a partir
    del raster base de dunas INEGI, polígonos de dunas costeras y regiones costeras.

Origen:
    Traducción inicial a Python del script R:
    1_create_reference_grid2.R

Resumen del flujo:
    1. Leer el raster base de dunas INEGI y los insumos vectoriales.
    2. Reproyectar dunas y regiones al CRS del raster base.
    3. Seleccionar una región específica y construir un buffer de 10 km.
    4. Construir una plantilla raster equivalente a crop + extend + disagg.
    5. Rasterizar el atributo CONSERV_ED sobre la plantilla.
    6. Aplicar la máscara regional y escribir ref_grid.tif.

Insumos principales:
    - raster base de dunas INEGI
    - shapefile de dunas costeras
    - shapefile de regiones costeras

Salidas principales:
    - ref_grid.tif para una región específica

Supuestos y notas:
    - Se asume que el shapefile de regiones costeras debe interpretarse en EPSG:4326
      antes de reproyectarse al CRS del raster base, siguiendo la lógica del script R.
    - Se usa nodata = -9999.0.
    - La región se identifica por el valor del campo myid transformado a region_<id>.

Fidelidad de la traducción:
    Traducción inicial con alta fidelidad lógica respecto al script R original.
    La secuencia analítica, los insumos principales y los productos esperados
    buscan corresponder lo más posible con la versión en R. Cuando alguna
    operación no tiene equivalencia directa entre bibliotecas de R y Python,
    se adopta una implementación funcionalmente equivalente o la aproximación
    más cercana disponible, procurando conservar el resultado analítico esperado.
    En este caso, la secuencia crop + extend + disagg de terra se aproxima en
    Python mediante la construcción explícita de una plantilla alineada a la
    grilla del raster base y una desagregación posterior con nearest neighbour.

Observaciones:
    Este script está pensado para ejecución headless y para integrarse en un
    workflow orquestado, por ejemplo con Snakemake.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject


BUFFER_METERS = 10000
DISAGG_FACTOR = 5
FIELD_NAME = "CONSERV_ED"
REGION_ID_FIELD = "myid"
NODATA_VALUE = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera ref_grid.tif para una región específica."
    )
    parser.add_argument(
        "--dunes-inegi",
        required=True,
        help="Ruta al raster base de dunas INEGI.",
    )
    parser.add_argument(
        "--dunes-other",
        required=True,
        help="Ruta al shapefile de dunas costeras.",
    )
    parser.add_argument(
        "--coastal-regions",
        required=True,
        help="Ruta al shapefile de regiones costeras.",
    )
    parser.add_argument(
        "--region-id",
        required=True,
        help="Identificador de región en formato region_<id>, por ejemplo region_7.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida del ref_grid.tif.",
    )
    return parser.parse_args()


def normalize_region_id(region_id: str) -> str:
    return str(region_id).strip().lower()


def make_region_key_from_value(value: object) -> str:
    return f"region_{value}".strip().lower()


def validate_inputs(
        dunes_inegi: Path,
        dunes_other: Path,
        coastal_regions: Path,
) -> None:
    missing = [str(p) for p in [dunes_inegi, dunes_other, coastal_regions] if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_vectors(
        dunes_other_path: Path,
        coastal_regions_path: Path,
        target_crs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    dunes_other = gpd.read_file(dunes_other_path)
    coastal_regions = gpd.read_file(coastal_regions_path)

    if dunes_other.empty:
        raise ValueError(f"El shapefile está vacío: {dunes_other_path}")
    if coastal_regions.empty:
        raise ValueError(f"El shapefile está vacío: {coastal_regions_path}")

    if dunes_other.crs is None:
        raise ValueError(f"{dunes_other_path.name} no tiene CRS definido.")

    # Replica la lógica del script R:
    # coastalr <- st_set_crs(coastalr, 4326)
    coastal_regions = coastal_regions.set_crs(4326, allow_override=True)

    dunes_other_reproj = dunes_other.to_crs(target_crs)
    coastal_regions_reproj = coastal_regions.to_crs(target_crs)

    if FIELD_NAME not in dunes_other_reproj.columns:
        raise ValueError(
            f"No existe el campo requerido '{FIELD_NAME}' en {dunes_other_path.name}"
        )

    if REGION_ID_FIELD not in coastal_regions_reproj.columns:
        raise ValueError(
            f"No existe el campo requerido '{REGION_ID_FIELD}' en {coastal_regions_path.name}"
        )

    return dunes_other_reproj, coastal_regions_reproj


def select_region(coastal_regions_reproj: gpd.GeoDataFrame, region_id: str):
    target = normalize_region_id(region_id)

    region_keys = coastal_regions_reproj[REGION_ID_FIELD].map(make_region_key_from_value)
    matches = coastal_regions_reproj.loc[region_keys == target]

    if matches.empty:
        available = sorted(region_keys.unique().tolist())
        raise KeyError(
            f"No se encontró la región '{region_id}'. Disponibles: {available}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"La región '{region_id}' aparece más de una vez en {REGION_ID_FIELD}."
        )

    return matches.iloc[0]


def align_bounds_to_grid(bounds, transform: Affine) -> tuple[float, float, float, float]:
    """
    Alinea el bbox de la geometría a la grilla del raster fuente, aproximando
    el comportamiento crop + extend del script R.
    """
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


def build_extended_template(
        src: rasterio.io.DatasetReader,
        geom,
        factor: int,
) -> tuple[np.ndarray, Affine]:
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


def rasterize_polygon_mask(shape, transform, geom) -> np.ndarray:
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


def rasterize_dunes_condition(
        shape,
        transform,
        dunes_other_reproj: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Equivalente a:
      dother_rast <- rasterize(dother, coastr_rast, field="CONSERV_ED")
    """
    shapes = (
        (geom, value)
        for geom, value in zip(dunes_other_reproj.geometry, dunes_other_reproj[FIELD_NAME])
        if geom is not None and not geom.is_empty
    )

    return rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=NODATA_VALUE,
        dtype="float32",
    )


def write_output(output_path: Path, arr: np.ndarray, transform, crs) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    meta = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA_VALUE,
        "compress": "LZW",
    }

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(arr.astype("float32"), 1)


def main() -> None:
    args = parse_args()

    dunes_inegi_path = Path(args.dunes_inegi)
    dunes_other_path = Path(args.dunes_other)
    coastal_regions_path = Path(args.coastal_regions)
    output_path = Path(args.output)
    region_id = args.region_id

    validate_inputs(dunes_inegi_path, dunes_other_path, coastal_regions_path)

    with rasterio.open(dunes_inegi_path) as dunes_inegi:
        if dunes_inegi.crs is None:
            raise ValueError("El raster de dunas INEGI no tiene CRS.")

        dunes_other_reproj, coastal_regions_reproj = load_vectors(
            dunes_other_path=dunes_other_path,
            coastal_regions_path=coastal_regions_path,
            target_crs=dunes_inegi.crs,
        )

        region_row = select_region(coastal_regions_reproj, region_id)
        region_geom = region_row.geometry

        if region_geom is None or region_geom.is_empty:
            raise ValueError(f"La región '{region_id}' no tiene geometría válida.")

        # coastr_poly <- buffer(vect(coastalr_reproj[i,]), width=10000)
        coastal_buffer = region_geom.buffer(BUFFER_METERS)

        # dinegi_cropped + extend + disagg
        dunes_template, template_transform = build_extended_template(
            dunes_inegi,
            coastal_buffer,
            DISAGG_FACTOR,
        )

        # coastr_rast <- rasterize(coastr_poly, dinegi_disagg)
        coastal_mask = rasterize_polygon_mask(
            shape=dunes_template.shape,
            transform=template_transform,
            geom=coastal_buffer,
        )

        # dother_rast <- rasterize(dother, coastr_rast, field="CONSERV_ED")
        dunes_raster = rasterize_dunes_condition(
            shape=coastal_mask.shape,
            transform=template_transform,
            dunes_other_reproj=dunes_other_reproj,
        )

        # dother_rast[is.na(coastr_rast)] <- NA
        dunes_raster = dunes_raster.astype("float32")
        dunes_raster[coastal_mask == 0] = NODATA_VALUE

        write_output(
            output_path=output_path,
            arr=dunes_raster,
            transform=template_transform,
            crs=dunes_inegi.crs,
        )

    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()