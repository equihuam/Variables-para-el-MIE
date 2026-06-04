#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject
from shapely.geometry import box


DROPBOX_DIR = Path(r"C:/Users/equih/1 Nubes/Dropbox/ei-coastal")
WRITEPATH = DROPBOX_DIR / "data" / "8_ref_grid_50m"

# Se conserva por congruencia con el script R, aunque no participa en la lógica.
DELTAVP_RASTER = DROPBOX_DIR / "data" / "5_deltavp250m" / "1delt_vp_250m.tif"

DUNES_INEGI_RASTER = DROPBOX_DIR / "data" / "6_inegiSVII_dunes" / "cdv_usuev250sVII_dunas.tif"
DUNES_OTHER_SHP = DROPBOX_DIR / "data" / "7_dunes250116" / "DunasCosteras250116.shp"
COASTAL_REGIONS_SHP = DROPBOX_DIR / "data" / "3_misc_cesia" / "RegionesCosteras40km.shp"

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
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def load_vectors(target_crs):
    dunesother = gpd.read_file(DUNES_OTHER_SHP)
    coastalr = gpd.read_file(COASTAL_REGIONS_SHP)

    if dunesother.crs is None:
        raise ValueError(f"{DUNES_OTHER_SHP.name} no tiene CRS definido.")

    # Replica la lógica del R:
    # coastalr <- st_set_crs(coastalr, 4326)
    coastalr = coastalr.set_crs(4326, allow_override=True)

    dunesother_reproj = dunesother.to_crs(target_crs)
    coastalr_reproj = coastalr.to_crs(target_crs)

    if FIELD_NAME not in dunesother_reproj.columns:
        raise ValueError(f"No existe el campo '{FIELD_NAME}' en {DUNES_OTHER_SHP.name}")

    if REGION_ID_FIELD not in coastalr_reproj.columns:
        raise ValueError(f"No existe el campo '{REGION_ID_FIELD}' en {COASTAL_REGIONS_SHP.name}")

    return dunesother_reproj, coastalr_reproj


def align_bounds_to_grid(bounds, transform: Affine):
    """
    Replica de forma cercana el comportamiento crop + extend de terra:
    toma el bbox del polígono y lo alinea a la grilla del raster fuente.
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


def build_extended_template(src: rasterio.io.DatasetReader, geom, factor: int):
    """
    Equivalente aproximado y muy cercano a:
      dinegi_cropped <- crop(dunesinegi, coastr_poly)
      dinegi_cropped <- extend(dinegi_cropped, coastr_poly)
      dinegi_disagg <- disagg(dinegi_cropped, fact=5)

    Se construye primero la plantilla a resolución original,
    alineada a la grilla del raster fuente y extendida al bbox del buffer.
    Luego se desagrega por nearest neighbour.
    """
    aligned_bounds = align_bounds_to_grid(geom.bounds, src.transform)
    left, bottom, right, top = aligned_bounds

    xres = src.transform.a
    yres = abs(src.transform.e)

    width = int(round((right - left) / xres))
    height = int(round((top - bottom) / yres))

    coarse_transform = Affine(xres, 0.0, left, 0.0, -yres, top)

    coarse_array = np.full((height, width), src.nodata if src.nodata is not None else 0, dtype=src.dtypes[0])

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

    fine_array = np.full((fine_height, fine_width), src.nodata if src.nodata is not None else 0, dtype=src.dtypes[0])

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
    mask_arr = rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask_arr


def rasterize_dunes_other(shape, transform, dunesother_reproj: gpd.GeoDataFrame):
    """
    Equivalente a:
      dother_rast <- rasterize(dother, coastr_rast, field="CONSERV_ED")
    """
    shapes = (
        (geom, value)
        for geom, value in zip(dunesother_reproj.geometry, dunesother_reproj[FIELD_NAME])
        if geom is not None and not geom.is_empty
    )

    out = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=NODATA_VALUE,
        dtype="float32",
    )
    return out


def write_output(output_path: Path, arr: np.ndarray, transform, crs):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    # Si tu GDAL soporta COG, esto se parece más al of=COG del R.
    meta = {
        "driver": "COG",
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
    validate_inputs()
    WRITEPATH.mkdir(parents=True, exist_ok=True)

    # Se valida existencia por fidelidad al R, aunque no se use después.
    _ = DELTAVP_RASTER

    with rasterio.open(DUNES_INEGI_RASTER) as dunesinegi:
        if dunesinegi.crs is None:
            raise ValueError("El raster de dunas INEGI no tiene CRS.")

        dunesother_reproj, coastalr_reproj = load_vectors(dunesinegi.crs)

        for i, (_, row) in enumerate(coastalr_reproj.iterrows(), start=1):
            region_geom = row.geometry
            region_id = row[REGION_ID_FIELD]

            if region_geom is None or region_geom.is_empty:
                print(f"Aviso: región {region_id} sin geometría; se omite.")
                continue

            # coastr_poly <- buffer(vect(coastalr_reproj[i,]), width=10000)
            coastr_poly = region_geom.buffer(BUFFER_METERS)

            # dinegi_cropped <- extend(crop(dunesinegi, coastr_poly), coastr_poly)
            # dinegi_disagg <- disagg(dinegi_cropped, fact=5)
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

            # dother <- vect(dunesother_reproj)
            # dother_rast <- rasterize(dother, coastr_rast, field="CONSERV_ED")
            dother_rast = rasterize_dunes_other(
                shape=coastr_rast.shape,
                transform=template_transform,
                dunesother_reproj=dunesother_reproj,
            )

            # dother_rast[is.na(coastr_rast)] <- NA
            # En Python coastr_rast es 0 fuera / 1 dentro, así que el equivalente es:
            dother_rast = dother_rast.astype("float32")
            dother_rast[coastr_rast == 0] = NODATA_VALUE

            out_dir = WRITEPATH / f"region_{region_id}"
            out_path = out_dir / "ref_grid.tif"

            write_output(
                output_path=out_path,
                arr=dother_rast,
                transform=template_transform,
                crs=dunesinegi.crs,
            )

            print(f"OK -> {out_path}")


if __name__ == "__main__":
    main()