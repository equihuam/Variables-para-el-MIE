#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 7_wf_madmex_uso_suelo_3.py

Propósito:
    Calcular, para una región específica, la distancia a la clase MADMEX más
    cercana para pastizal, agricultura y urbano, usando como referencia una
    tabla base regional ya alineada, y exportar el resultado como tabla
    congruente por píxel en formato Parquet.

Salida:
    regionid, pixid, x, y, d_grassland, d_agriculture, d_urban

Notas de equivalencia:
    - La tabla base regional define la malla canónica de salida.
    - Las coordenadas de la tabla base se interpretan por defecto en EPSG:4326.
    - Las distancias se calculan en el CRS del raster MADMEX.
    - El modo canónico usa distancia equivalente a kknn(..., k = 1,
      distance = 2, kernel = "rectangular", scale = TRUE).
    - Las clases MADMEX usadas son:
        27 = d_grassland
        28 = d_agriculture
        29 = d_urban
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import pyarrow.parquet as pq
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy
from rasterio.warp import transform as rio_transform
from shapely.geometry import box, mapping
from scipy.spatial import cKDTree


GRASSLAND_CLASS = 27
AGRICULTURE_CLASS = 28
URBAN_CLASS = 29
VALID_CLASSES = {GRASSLAND_CLASS, AGRICULTURE_CLASS, URBAN_CLASS}
SENTINEL = 9999.0
KEY_COLUMNS = ["regionid", "pixid", "x", "y"]
BASE_CRS = "EPSG:4326"

OUTPUT_COLS = {
    GRASSLAND_CLASS: "d_grassland",
    AGRICULTURE_CLASS: "d_agriculture",
    URBAN_CLASS: "d_urban",
}

CLASS_NAMES = {
    GRASSLAND_CLASS: "grassland",
    AGRICULTURE_CLASS: "agriculture",
    URBAN_CLASS: "urban",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia a clases MADMEX usando una tabla base regional."
    )
    parser.add_argument(
        "--madmex-raster",
        required=True,
        help="Ruta al raster MADMEX de uso de suelo.",
    )
    parser.add_argument(
        "--base-table",
        required=True,
        help="Ruta a la tabla base regional .parquet con regionid, pixid, x, y.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet.",
    )
    parser.add_argument(
        "--distance-mode",
        choices=["kknn_scaled", "raw"],
        default="kknn_scaled",
        help="kknn_scaled emula kknn(scale=TRUE); raw usa distancia euclidiana cruda.",
    )
    parser.add_argument(
        "--source-crs",
        default=BASE_CRS,
        help="CRS de las coordenadas x/y de la tabla base. Default: EPSG:4326.",
    )
    parser.add_argument("--debug-metadata-output", default=None)
    parser.add_argument("--debug-madmex-points-output", default=None)
    parser.add_argument("--debug-base-output", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def validate_inputs(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def read_base_table(path: Path) -> pd.DataFrame:
    table = pq.read_table(path, use_threads=False)
    df = table.to_pandas()

    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"La tabla base no cumple el contrato mínimo. Faltan columnas: {missing}"
        )

    out = df[KEY_COLUMNS].copy()
    out["x"] = pd.to_numeric(out["x"], errors="raise")
    out["y"] = pd.to_numeric(out["y"], errors="raise")

    if out.empty:
        raise ValueError(f"La tabla base está vacía: {path}")

    return out


def transform_points_to_crs(
    xs: np.ndarray,
    ys: np.ndarray,
    src_crs,
    dst_crs,
) -> tuple[np.ndarray, np.ndarray]:
    tx, ty = rio_transform(src_crs, dst_crs, xs.tolist(), ys.tolist())
    return np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)


def crop_raster_to_points_extent(
    src: rasterio.io.DatasetReader,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, rasterio.Affine]:
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No hay puntos válidos para definir la extensión de recorte.")

    geom = box(float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))
    cropped, cropped_transform = mask(src, [mapping(geom)], crop=True, filled=True)

    return cropped[0], cropped_transform


def raster_points_dataframe(arr: np.ndarray, transform) -> pd.DataFrame:
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


def filter_madmex_classes(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float, copy=True)
    valid_mask = np.isin(out, list(VALID_CLASSES))
    out[~valid_mask] = np.nan
    return out


def scaled_xy(
    train_xy: np.ndarray,
    pred_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sd = np.std(train_xy, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd != 0), sd, 1.0)
    return train_xy / sd, pred_xy / sd, sd


def nearest_distance_column(
    points_xy: np.ndarray,
    class_points: pd.DataFrame,
    distance_mode: str,
) -> tuple[np.ndarray, tuple[float, float]]:
    if class_points.empty:
        return np.full(points_xy.shape[0], SENTINEL, dtype=float), (np.nan, np.nan)

    coords = class_points[["x", "y"]].to_numpy(dtype=float)

    if distance_mode == "kknn_scaled":
        coords_query, points_query, sd = scaled_xy(coords, points_xy)
    elif distance_mode == "raw":
        coords_query = coords
        points_query = points_xy
        sd = np.array([1.0, 1.0], dtype=float)
    else:
        raise ValueError(f"distance_mode no reconocido: {distance_mode}")

    tree = cKDTree(coords_query)
    distances, _ = tree.query(points_query, k=1)
    return distances.astype(float), (float(sd[0]), float(sd[1]))


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow")
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Extensión no soportada: {output_path.suffix}")


def main() -> None:
    args = parse_args()

    madmex_path = Path(args.madmex_raster)
    base_table_path = Path(args.base_table)
    output_path = Path(args.output)

    validate_inputs(madmex_path, base_table_path)
    base = read_base_table(base_table_path)

    with rasterio.open(madmex_path) as madmex_src:
        if madmex_src.crs is None:
            raise ValueError("El raster MADMEX no tiene CRS definido.")

        tx, ty = transform_points_to_crs(
            base["x"].to_numpy(),
            base["y"].to_numpy(),
            args.source_crs,
            madmex_src.crs,
        )
        pred_xy = np.column_stack([tx, ty])

        madmex_arr_raw, madmex_transform = crop_raster_to_points_extent(
            madmex_src,
            tx,
            ty,
        )
        madmex_arr = filter_madmex_classes(madmex_arr_raw)

        metadata = {
            "madmex_crs": str(madmex_src.crs),
            "madmex_nodata": madmex_src.nodata,
            "base_rows": len(base),
            "crop_width": int(madmex_arr.shape[1]),
            "crop_height": int(madmex_arr.shape[0]),
            "distance_mode": args.distance_mode,
            "source_crs": args.source_crs,
        }

    madmex_points = raster_points_dataframe(madmex_arr, madmex_transform).rename(
        columns={"value": "layer"}
    )
    madmex_points["layer"] = pd.to_numeric(madmex_points["layer"], errors="coerce")

    out = base.copy()
    class_rows: list[dict[str, object]] = []

    for class_code, out_col in OUTPUT_COLS.items():
        class_points = madmex_points[madmex_points["layer"] == class_code]
        distances, sd = nearest_distance_column(pred_xy, class_points, args.distance_mode)
        out[out_col] = distances

        class_rows.append(
            {
                "class_code": class_code,
                "class_name": CLASS_NAMES[class_code],
                "output_col": out_col,
                "n_points": int(len(class_points)),
                "sd_x": sd[0],
                "sd_y": sd[1],
                "min_distance": float(np.nanmin(distances)) if len(distances) else np.nan,
                "max_distance": float(np.nanmax(distances)) if len(distances) else np.nan,
                "n_sentinel": int(np.sum(distances == SENTINEL)),
            }
        )

    class_summary = pd.DataFrame(class_rows)

    for _, row in class_summary.iterrows():
        log(
            f"clase {int(row['class_code'])} -> {row['output_col']}: "
            f"puntos={int(row['n_points'])}, sd=({row['sd_x']}, {row['sd_y']}), "
            f"sentinel={int(row['n_sentinel'])}",
            args.verbose,
        )

    save_table(out, output_path)

    if args.debug_madmex_points_output:
        save_table(madmex_points.copy(), Path(args.debug_madmex_points_output))
        log(f"debug madmex points -> {args.debug_madmex_points_output}", args.verbose)

    if args.debug_base_output:
        debug_base = base.copy()
        debug_base["x_madmex"] = tx
        debug_base["y_madmex"] = ty
        save_table(debug_base, Path(args.debug_base_output))
        log(f"debug base -> {args.debug_base_output}", args.verbose)

    if args.debug_metadata_output:
        meta_df = pd.DataFrame([metadata])
        for _, row in class_summary.iterrows():
            prefix = str(row["class_name"])
            meta_df[f"{prefix}_n_points"] = int(row["n_points"])
            meta_df[f"{prefix}_sd_x"] = float(row["sd_x"])
            meta_df[f"{prefix}_sd_y"] = float(row["sd_y"])
            meta_df[f"{prefix}_n_sentinel"] = int(row["n_sentinel"])
        save_table(meta_df, Path(args.debug_metadata_output))
        log(f"debug metadata -> {args.debug_metadata_output}", args.verbose)

    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
