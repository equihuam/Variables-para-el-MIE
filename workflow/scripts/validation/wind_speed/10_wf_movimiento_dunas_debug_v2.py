#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 10_wf_movimiento_dunas_debug_v1.py

Propósito:
    Versión diagnóstica para reproducir de forma más cercana la lógica de
    10_movimiento_dunas.R.

Notas:
    A diferencia de otros scripts del workflow, el R original no usa ref_grid.tif
    ni una tabla base regional. Construye una malla por región costera a partir de:
      - raster INEGI de dunas
      - polígonos de regiones costeras
      - polígonos de dunas costeras con NESTB_EDO
      - CRS de manglares como CRS destino final

    Flujo equivalente:
      1. Leer raster INEGI de dunas.
      2. Leer regiones costeras y reproyectarlas al CRS del raster INEGI.
      3. Seleccionar una región costera por índice o id.
      4. Aplicar buffer de 10 km.
      5. Recortar/expandir la malla INEGI a ese buffer.
      6. Disagregar la malla por factor 5.
      7. Rasterizar el polígono costero sobre la malla disgregada.
      8. Rasterizar dunas con campo NESTB_EDO sobre esa malla.
      9. Enmascarar fuera del polígono costero.
     10. Reproyectar al CRS destino.
     11. Exportar x, y, NESTB_EDO, pixid, regionid.

Limitaciones:
    rasterio/rasterize usa códigos enteros para atributos categóricos; el script
    conserva un diccionario de códigos a etiquetas y devuelve etiquetas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy, Affine
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import calculate_default_transform, reproject, Resampling


OUTPUT_FIELD = "NESTB_EDO"
DEFAULT_BUFFER = 10000.0
DEFAULT_DISAGG = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug: reproduce movimiento de dunas según la lógica del R original."
    )
    p.add_argument("--dunes-inegi-raster", required=True, help="Raster INEGI de dunas.")
    p.add_argument("--dunes-other-shp", required=True, help="Shapefile de dunas costeras con NESTB_EDO.")
    p.add_argument("--coastal-regions-shp", required=True, help="Shapefile de regiones costeras con myid.")
    p.add_argument("--target-crs-shp", required=True, help="Capa usada sólo para definir CRS final, como manglares.")
    p.add_argument("--region-index", type=int, default=None, help="Índice 1-based de región costera, como en el bucle R.")
    p.add_argument("--region-id", default=None, help="Valor de campo myid para seleccionar región.")
    p.add_argument("--region-id-field", default="myid", help="Campo de id en regiones costeras.")
    p.add_argument("--buffer-width", type=float, default=DEFAULT_BUFFER, help="Buffer en unidades del CRS INEGI.")
    p.add_argument("--disagg-factor", type=int, default=DEFAULT_DISAGG, help="Factor de disgregación de la malla.")
    p.add_argument("--output", required=True, help="Salida .parquet.")
    p.add_argument("--debug-metadata-output", default=None, help="CSV opcional de metadatos.")
    p.add_argument("--debug-codebook-output", default=None, help="CSV opcional de código-etiqueta.")
    p.add_argument("--na-policy", choices=["r_default", "valid_only"], default="r_default",
                   help="r_default conserva todas las celdas del raster reproyectado; valid_only conserva sólo NESTB_EDO no NA.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def select_region(coastal: gpd.GeoDataFrame, region_index: int | None, region_id: str | None, id_field: str) -> gpd.GeoDataFrame:
    if region_id is not None:
        if id_field not in coastal.columns:
            raise ValueError(f"No existe el campo de región '{id_field}'.")
        sel = coastal[coastal[id_field].astype(str) == str(region_id)].copy()
        if sel.empty:
            raise ValueError(f"No se encontró region_id={region_id} en {id_field}.")
        return sel.iloc[[0]].copy()

    if region_index is None:
        region_index = 1

    if region_index < 1 or region_index > len(coastal):
        raise ValueError(f"region-index fuera de rango: {region_index}; n={len(coastal)}")

    return coastal.iloc[[region_index - 1]].copy()


def read_window_as_float(src: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, Affine]:
    win = from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets().round_lengths()

    # Leer como masked array para compatibilidad con versiones de rasterio en las
    # que DatasetReader.read() no acepta filled=True. Luego convertir a float y
    # representar NoData/fuera de ventana como NaN.
    arr = src.read(1, window=win, boundless=True, masked=True)
    tr = window_transform(win, src.transform)

    if np.ma.isMaskedArray(arr):
        out = arr.astype("float64").filled(np.nan)
    else:
        out = arr.astype("float64")

    nodata = src.nodata
    if nodata is not None:
        out[out == nodata] = np.nan

    return out, tr


def disaggregate_nearest(arr: np.ndarray, transform: Affine, factor: int) -> tuple[np.ndarray, Affine]:
    if factor <= 1:
        return arr, transform
    out = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    new_transform = transform * Affine.scale(1 / factor, 1 / factor)
    return out, new_transform


def make_codebook(values: pd.Series) -> tuple[dict[Any, int], dict[int, Any]]:
    cats = pd.Categorical(values.astype("object"))
    labels = list(cats.categories)
    value_to_code = {label: i + 1 for i, label in enumerate(labels)}
    code_to_value = {i + 1: label for i, label in enumerate(labels)}
    return value_to_code, code_to_value


def rasterize_categorical(gdf: gpd.GeoDataFrame, field: str, out_shape, transform: Affine) -> tuple[np.ndarray, dict[int, Any]]:
    value_to_code, code_to_value = make_codebook(gdf[field])
    shapes = (
        (geom, int(value_to_code[val]))
        for geom, val in zip(gdf.geometry, gdf[field])
        if geom is not None and not geom.is_empty and pd.notna(val)
    )
    arr = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return arr, code_to_value


def reproject_codes_to_crs(
    code_arr: np.ndarray,
    src_transform: Affine,
    src_crs,
    dst_crs,
) -> tuple[np.ndarray, Affine]:
    transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        code_arr.shape[1],
        code_arr.shape[0],
        *rasterio.transform.array_bounds(code_arr.shape[0], code_arr.shape[1], src_transform),
    )
    dst = np.zeros((height, width), dtype="int32")
    reproject(
        source=code_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=0,
        dst_transform=transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst, transform


def codes_to_dataframe(code_arr: np.ndarray, transform: Affine, code_to_value: dict[int, Any], na_policy: str) -> pd.DataFrame:
    if na_policy == "valid_only":
        rows, cols = np.where(code_arr > 0)
    else:
        rows, cols = np.indices(code_arr.shape)
        rows = rows.ravel()
        cols = cols.ravel()

    if len(rows) == 0:
        return pd.DataFrame(columns=["x", "y", OUTPUT_FIELD])

    xs, ys = xy(transform, rows, cols, offset="center")
    codes = code_arr[rows, cols]
    labels = [code_to_value.get(int(c), np.nan) if int(c) != 0 else np.nan for c in codes]

    return pd.DataFrame({"x": np.asarray(xs), "y": np.asarray(ys), OUTPUT_FIELD: labels})


def main() -> None:
    args = parse_args()

    dunes_inegi_path = Path(args.dunes_inegi_raster)
    dunes_other_path = Path(args.dunes_other_shp)
    coastal_regions_path = Path(args.coastal_regions_shp)
    target_crs_path = Path(args.target_crs_shp)
    output_path = Path(args.output)

    validate_inputs(dunes_inegi_path, dunes_other_path, coastal_regions_path, target_crs_path)

    dunes_other = gpd.read_file(dunes_other_path)
    coastal = gpd.read_file(coastal_regions_path)
    target_layer = gpd.read_file(target_crs_path)

    if OUTPUT_FIELD not in dunes_other.columns:
        raise ValueError(f"No existe el campo {OUTPUT_FIELD} en {dunes_other_path.name}")
    if target_layer.crs is None:
        raise ValueError("La capa target-crs-shp no tiene CRS.")
    if coastal.crs is None:
        # R hace st_set_crs(coastalr, 4326)
        coastal = coastal.set_crs("EPSG:4326")

    with rasterio.open(dunes_inegi_path) as src:
        if src.crs is None:
            raise ValueError("El raster INEGI de dunas no tiene CRS.")

        dunes_other_reproj = dunes_other.to_crs(src.crs)
        coastal_reproj = coastal.to_crs(src.crs)

        region_gdf = select_region(coastal_reproj, args.region_index, args.region_id, args.region_id_field)
        original_region_id = (
            str(region_gdf.iloc[0][args.region_id_field])
            if args.region_id_field in region_gdf.columns
            else str(args.region_index if args.region_index is not None else 1)
        )

        # R: coastr_poly <- buffer(vect(coastalr_reproj[i,]), width=10000)
        coastr_poly = region_gdf.copy()
        coastr_poly["geometry"] = coastr_poly.geometry.buffer(args.buffer_width)

        bounds = tuple(coastr_poly.total_bounds)
        dinegi_crop, dinegi_crop_transform = read_window_as_float(src, bounds)

        # R: dinegi_disagg <- disagg(dinegi_cropped, fact=5)
        dinegi_disagg, disagg_transform = disaggregate_nearest(dinegi_crop, dinegi_crop_transform, args.disagg_factor)
        shape = dinegi_disagg.shape

        # R: coastr_rast <- rasterize(coastr_poly, dinegi_disagg)
        coastr_mask = rasterize(
            ((geom, 1) for geom in coastr_poly.geometry if geom is not None and not geom.is_empty),
            out_shape=shape,
            transform=disagg_transform,
            fill=0,
            dtype="uint8",
        )

        # R: dother_rast <- rasterize(dother, coastr_rast, field="NESTB_EDO")
        dother_codes, code_to_value = rasterize_categorical(
            dunes_other_reproj,
            OUTPUT_FIELD,
            out_shape=shape,
            transform=disagg_transform,
        )

        # R: dother_rast[is.na(coastr_rast)] <- NA
        # Here coastr_rast outside region is 0, so set dune codes outside region to 0.
        dother_codes[coastr_mask == 0] = 0

        # R: project(dother_rast, y = crs(manglares), method="near")
        reproj_codes, reproj_transform = reproject_codes_to_crs(
            dother_codes,
            disagg_transform,
            src.crs,
            target_layer.crs,
        )

    out = codes_to_dataframe(reproj_codes, reproj_transform, code_to_value, args.na_policy)
    out["pixid"] = np.arange(1, len(out) + 1)
    out["regionid"] = original_region_id
    out = out[["regionid", "pixid", "x", "y", OUTPUT_FIELD]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")

    if args.debug_codebook_output:
        cb = pd.DataFrame(
            [{"code": code, OUTPUT_FIELD: value} for code, value in code_to_value.items()]
        )
        Path(args.debug_codebook_output).parent.mkdir(parents=True, exist_ok=True)
        cb.to_csv(args.debug_codebook_output, index=False)

    if args.debug_metadata_output:
        meta = pd.DataFrame([{
            "regionid": original_region_id,
            "region_index": args.region_index,
            "buffer_width": args.buffer_width,
            "disagg_factor": args.disagg_factor,
            "inegi_crs": str(src.crs) if 'src' in locals() else "",
            "target_crs": str(target_layer.crs),
            "crop_rows": int(dinegi_crop.shape[0]),
            "crop_cols": int(dinegi_crop.shape[1]),
            "disagg_rows": int(dinegi_disagg.shape[0]),
            "disagg_cols": int(dinegi_disagg.shape[1]),
            "reproj_rows": int(reproj_codes.shape[0]),
            "reproj_cols": int(reproj_codes.shape[1]),
            "out_rows": int(len(out)),
            "non_na_NESTB_EDO": int(out[OUTPUT_FIELD].notna().sum()),
            "unique_labels": "|".join(map(str, sorted(out[OUTPUT_FIELD].dropna().unique()))),
        }])
        Path(args.debug_metadata_output).parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(args.debug_metadata_output, index=False)

    log(f"regionid: {original_region_id}", args.verbose)
    log(f"salida filas: {len(out)}", args.verbose)
    log(f"NESTB_EDO no NA: {out[OUTPUT_FIELD].notna().sum()}", args.verbose)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
