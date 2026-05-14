#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 2_wf_estructuras_costeras.py

Propósito:
    Calcular, para una región específica, la distancia al elemento más cercano
    de estructuras costeras por tipo de estructura y exportar una tabla Parquet
    congruente por píxel.

Notas de equivalencia con R:
    - El script R usa terra::geom(struct_tipo) y kknn(..., k = 1,
      kernel = "rectangular") para recuperar modelkknn$D.
    - Para aproximar ese comportamiento, el modo canónico de este script usa
      todos los vértices de las geometrías como conjunto de referencia y calcula
      la distancia al vértice más cercano en el espacio escalado como kknn(scale=TRUE).
    - El modo legacy basado en representative_point se conserva sólo para
      diagnóstico/comparación.

Interfaz Snakemake:
    Requiere únicamente --structures-shp, --ref-grid, --region-id y --output.
    Los argumentos debug son opcionales y no afectan la regla productiva.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry


TIPO_FIELD = "Tipo"

# Nombres esperados por el workflow. Las llaves son los nombres normalizados
# que quedan en el campo Tipo después de limpiar variantes del shapefile.
CANONICAL_TYPES = {
    "Escollera": "escollera",
    "Espigón": "espigon",
    "Muro": "muro",
    "Rompeolas": "rompeolas",
    "Puerto": "puerto",
}

TYPE_REPLACEMENTS = {
    "Escollera2": "Escollera",
    "Espigób": "Espigón",
    "espigón": "Espigón",
    "Espigón de M": "Espigón",
    "Muelle": "Puerto",
    "Rompeolas2": "Rompeolas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia a estructuras costeras por píxel para una región específica."
    )
    parser.add_argument(
        "--structures-shp",
        required=True,
        help="Ruta al shapefile de estructuras costeras.",
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
    parser.add_argument(
        "--debug-grid-output",
        default=None,
        help="Ruta opcional para guardar CSV diagnóstico de la grilla usada.",
    )
    parser.add_argument(
        "--debug-metadata-output",
        default=None,
        help="Ruta opcional para guardar CSV diagnóstico de metadatos.",
    )
    parser.add_argument(
        "--debug-structures-output",
        default=None,
        help="Ruta opcional para guardar CSV con conteos de estructuras/vértices por tipo.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=["vertices", "representative"],
        default="vertices",
        help=(
            "Modo para construir puntos de referencia. "
            "vertices replica mejor terra::geom(); representative reproduce el enfoque legacy."
        ),
    )
    parser.add_argument(
        "--distance-mode",
        choices=["kknn_scaled", "raw"],
        default="kknn_scaled",
        help=(
            "Modo de distancia. kknn_scaled replica kknn(scale=TRUE) y devuelve "
            "modelkknn$D; raw usa distancia euclidiana directa en coordenadas."
        ),
    )
    parser.add_argument(
        "--validity-mode",
        choices=["finite", "notnan"],
        default="finite",
        help="Criterio para seleccionar celdas válidas después de reproyectar.",
    )
    parser.add_argument(
        "--column-naming",
        choices=["canonical", "r"],
        default="canonical",
        help="Nombres de columnas de salida: canonical sin acentos o r con nombres del campo Tipo.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime diagnósticos en consola.",
    )
    return parser.parse_args()


def log(message: str, verbose: bool = False) -> None:
    if verbose:
        print(message)


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
    if TIPO_FIELD not in struct.columns:
        raise ValueError(f"No existe el campo requerido '{TIPO_FIELD}' en {path.name}")

    struct = struct.copy()
    struct[TIPO_FIELD] = struct[TIPO_FIELD].replace(TYPE_REPLACEMENTS).astype(str)
    struct = struct[struct[TIPO_FIELD].isin(CANONICAL_TYPES.keys())].copy()
    struct = struct[~struct.geometry.is_empty & struct.geometry.notna()].copy()

    if struct.empty:
        raise ValueError("No quedaron estructuras válidas después de normalizar tipos.")

    return struct


def original_raster_valid_counts(src: rasterio.io.DatasetReader) -> dict[str, int]:
    arr_masked = src.read(1, masked=True)
    arr = np.asarray(arr_masked.filled(np.nan), dtype=float)
    mask = np.asarray(src.dataset_mask(), dtype=np.uint8) > 0
    return {
        "src_total_points": int(src.width * src.height),
        "src_valid_masked": int(np.ma.count(arr_masked)),
        "src_finite_points": int(np.isfinite(arr).sum()),
        "src_gdal_mask_valid": int(mask.sum()),
    }


def reproject_raster_to_crs(
    src: rasterio.io.DatasetReader,
    dst_crs,
) -> tuple[np.ndarray, rasterio.Affine, dict[str, object]]:
    transform, width, height = calculate_default_transform(
        src.crs,
        dst_crs,
        src.width,
        src.height,
        *src.bounds,
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

    metadata = {
        "dst_width": int(width),
        "dst_height": int(height),
        "dst_total_points": int(width * height),
        "dst_transform_a": float(transform.a),
        "dst_transform_b": float(transform.b),
        "dst_transform_c": float(transform.c),
        "dst_transform_d": float(transform.d),
        "dst_transform_e": float(transform.e),
        "dst_transform_f": float(transform.f),
    }
    return dst, transform, metadata


def valid_raster_points_dataframe(
    arr: np.ndarray,
    transform,
    validity_mode: str = "finite",
) -> pd.DataFrame:
    if validity_mode == "finite":
        valid_mask = np.isfinite(arr)
    elif validity_mode == "notnan":
        valid_mask = ~np.isnan(arr)
    else:
        raise ValueError(f"Modo de validez no reconocido: {validity_mode}")

    rows, cols = np.where(valid_mask)

    if len(rows) == 0:
        return pd.DataFrame(columns=["row", "col", "x", "y", "ref_value"])

    xs, ys = xy(transform, rows, cols, offset="center")

    return pd.DataFrame(
        {
            "row": rows.astype(np.int64),
            "col": cols.astype(np.int64),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "ref_value": arr[rows, cols],
        }
    )


def _coords_from_geometry(geom: BaseGeometry) -> Iterable[tuple[float, float]]:
    """Extrae coordenadas de vértices en forma similar a terra::geom()."""
    if geom is None or geom.is_empty:
        return []

    gtype = geom.geom_type

    if gtype == "Point":
        return [(float(geom.x), float(geom.y))]

    if gtype in {"LineString", "LinearRing"}:
        return [(float(x), float(y)) for x, y, *_ in geom.coords]

    if gtype == "Polygon":
        coords: list[tuple[float, float]] = []
        coords.extend((float(x), float(y)) for x, y, *_ in geom.exterior.coords)
        for interior in geom.interiors:
            coords.extend((float(x), float(y)) for x, y, *_ in interior.coords)
        return coords

    if gtype in {"MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection"}:
        coords = []
        for part in geom.geoms:
            coords.extend(_coords_from_geometry(part))
        return coords

    raise ValueError(f"Tipo de geometría no soportado: {gtype}")


def coordinates_from_structures(
    struct_tipo: gpd.GeoDataFrame,
    geometry_mode: str = "vertices",
) -> np.ndarray:
    if struct_tipo.empty:
        return np.empty((0, 2), dtype=float)

    if geometry_mode == "representative":
        reps = struct_tipo.geometry.representative_point()
        coords = np.array([(g.x, g.y) for g in reps], dtype=float)
    elif geometry_mode == "vertices":
        coords_list: list[tuple[float, float]] = []
        for geom in struct_tipo.geometry:
            coords_list.extend(_coords_from_geometry(geom))
        coords = np.asarray(coords_list, dtype=float)
    else:
        raise ValueError(f"geometry_mode no reconocido: {geometry_mode}")

    if coords.size == 0:
        return np.empty((0, 2), dtype=float)

    coords = coords.reshape((-1, 2))
    coords = coords[np.isfinite(coords).all(axis=1)]
    return coords


def scale_like_kknn_train_valid(
    x_train: np.ndarray,
    x_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replica el escalamiento básico de kknn(scale=TRUE).

    kknn calcula la varianza muestral del conjunto de entrenamiento y divide
    train y test entre sqrt(var). No centra las coordenadas.
    """
    if x_train.ndim != 2 or x_pred.ndim != 2:
        raise ValueError("x_train y x_pred deben ser matrices 2D.")

    if x_train.shape[0] > 1:
        sd = np.nanstd(x_train, axis=0, ddof=1)
    else:
        sd = np.ones(x_train.shape[1], dtype=float)

    sd = np.asarray(sd, dtype=float)
    sd[(~np.isfinite(sd)) | (sd == 0)] = 1.0

    return x_train / sd, x_pred / sd, sd


def nearest_distance_column(
    points_xy: np.ndarray,
    reference_xy: np.ndarray,
    distance_mode: str = "kknn_scaled",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula distancia al vecino más cercano.

    En distance_mode='kknn_scaled' devuelve la distancia en el espacio escalado,
    equivalente a modelkknn$D con k=1, distance=2, kernel='rectangular', scale=TRUE.
    También devuelve el vector de desviaciones estándar usado para x/y.
    """
    if reference_xy.size == 0:
        return np.full(points_xy.shape[0], np.nan, dtype=float), np.array([np.nan, np.nan])

    if distance_mode == "raw":
        ref = reference_xy
        pts = points_xy
        sd = np.array([1.0, 1.0], dtype=float)
    elif distance_mode == "kknn_scaled":
        ref, pts, sd = scale_like_kknn_train_valid(reference_xy, points_xy)
    else:
        raise ValueError(f"distance_mode no reconocido: {distance_mode}")

    tree = cKDTree(ref)
    distances, _ = tree.query(pts, k=1)
    return distances.astype(float), sd.astype(float)


def make_structures_summary(
    struct: gpd.GeoDataFrame,
    geometry_mode: str,
) -> pd.DataFrame:
    rows = []
    for tipo_original, tipo_canonico in CANONICAL_TYPES.items():
        struct_tipo = struct[struct[TIPO_FIELD] == tipo_original]
        coords = coordinates_from_structures(struct_tipo, geometry_mode=geometry_mode)
        rows.append(
            {
                "tipo_original": tipo_original,
                "tipo_canonico": tipo_canonico,
                "n_features": int(len(struct_tipo)),
                "n_reference_points": int(coords.shape[0]),
                "geometry_mode": geometry_mode,
            }
        )
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow")
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Extensión de salida no soportada: {output_path.suffix}")


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
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    region_id = str(args.region_id).strip()

    validate_inputs(structures_path, ref_grid_path)
    struct = load_structures(structures_path)

    metadata: dict[str, object] = {
        "regionid": region_id,
        "structures_path": str(structures_path),
        "ref_grid_path": str(ref_grid_path),
        "output_path": str(output_path),
        "tipo_field": TIPO_FIELD,
        "geometry_mode": args.geometry_mode,
        "distance_mode": args.distance_mode,
        "validity_mode": args.validity_mode,
        "column_naming": args.column_naming,
        "structures_crs": str(struct.crs),
        "n_structures_total_normalized": int(len(struct)),
    }

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")

        metadata.update(original_raster_valid_counts(src))
        metadata.update(
            {
                "src_crs": str(src.crs),
                "src_width": int(src.width),
                "src_height": int(src.height),
                "src_nodata": src.nodata,
                "src_bounds_left": float(src.bounds.left),
                "src_bounds_bottom": float(src.bounds.bottom),
                "src_bounds_right": float(src.bounds.right),
                "src_bounds_top": float(src.bounds.top),
            }
        )

        region_arr, region_transform, reproj_metadata = reproject_raster_to_crs(
            src, struct.crs
        )
        metadata.update(reproj_metadata)

    region_points = valid_raster_points_dataframe(
        region_arr,
        region_transform,
        validity_mode=args.validity_mode,
    )

    metadata.update(
        {
            "dst_finite_points": int(np.isfinite(region_arr).sum()),
            "dst_notnan_points": int((~np.isnan(region_arr)).sum()),
            "valid_points_used": int(len(region_points)),
        }
    )

    if len(region_points) > 0:
        metadata.update(
            {
                "valid_x_min": float(region_points["x"].min()),
                "valid_x_max": float(region_points["x"].max()),
                "valid_y_min": float(region_points["y"].min()),
                "valid_y_max": float(region_points["y"].max()),
            }
        )

    log(f"total puntos GeoTIFF original: {metadata['src_total_points']}", args.verbose)
    log(f"puntos válidos GeoTIFF original masked: {metadata['src_valid_masked']}", args.verbose)
    log(f"total puntos reproyectados: {metadata['dst_total_points']}", args.verbose)
    log(f"modo de validez: {args.validity_mode}", args.verbose)
    log(f"puntos válidos usados en malla: {len(region_points)}", args.verbose)
    log(f"modo geometría: {args.geometry_mode}", args.verbose)
    log(f"modo distancia: {args.distance_mode}", args.verbose)

    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "regionid": region_id,
            "pixid": np.arange(1, len(region_points) + 1),
            "x": region_points["x"].to_numpy(dtype=float),
            "y": region_points["y"].to_numpy(dtype=float),
        }
    )

    structures_summary = []
    for tipo_original, tipo_canonico in CANONICAL_TYPES.items():
        struct_tipo = struct[struct[TIPO_FIELD] == tipo_original]
        coords = coordinates_from_structures(struct_tipo, geometry_mode=args.geometry_mode)
        colname = tipo_canonico if args.column_naming == "canonical" else tipo_original
        distances, sd_xy = nearest_distance_column(
            pred_xy,
            coords,
            distance_mode=args.distance_mode,
        )
        out[colname] = distances
        structures_summary.append(
            {
                "tipo_original": tipo_original,
                "tipo_canonico": tipo_canonico,
                "output_column": colname,
                "n_features": int(len(struct_tipo)),
                "n_reference_points": int(coords.shape[0]),
                "geometry_mode": args.geometry_mode,
                "distance_mode": args.distance_mode,
                "sd_x_kknn": float(sd_xy[0]) if len(sd_xy) > 0 and np.isfinite(sd_xy[0]) else np.nan,
                "sd_y_kknn": float(sd_xy[1]) if len(sd_xy) > 1 and np.isfinite(sd_xy[1]) else np.nan,
            }
        )
        log(
            f"{tipo_original} -> {colname}: features={len(struct_tipo)}, "
            f"puntos_ref={coords.shape[0]}, sd=({sd_xy[0]:.8g}, {sd_xy[1]:.8g})",
            args.verbose,
        )

    if args.debug_grid_output:
        debug_grid = region_points.copy()
        debug_grid.insert(0, "pixid", np.arange(1, len(debug_grid) + 1))
        debug_grid.insert(0, "regionid", region_id)
        save_table(debug_grid, Path(args.debug_grid_output))
        log(f"debug grid -> {args.debug_grid_output}", args.verbose)

    if args.debug_metadata_output:
        save_table(pd.DataFrame([metadata]), Path(args.debug_metadata_output))
        log(f"debug metadata -> {args.debug_metadata_output}", args.verbose)

    if args.debug_structures_output:
        save_table(pd.DataFrame(structures_summary), Path(args.debug_structures_output))
        log(f"debug structures -> {args.debug_structures_output}", args.verbose)

    save_output(out, output_path)


if __name__ == "__main__":
    main()
