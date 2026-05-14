#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 3_wf_spp_invasoras_global_stats.py

Propósito:
    Calcular estadísticos globales min/max de distancia por especie invasora
    sobre todas las regiones de referencia. Estos estadísticos permiten replicar
    la normalización global del script R 3_sp_invasoras.R, donde las distancias
    por especie se normalizan después de unir todas las regiones.

Salida:
    CSV con columnas:
      species, species_col, min, max, n_regions, n_pixels, n_points, sd_x, sd_y
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree


SPECIES_FIELD = "especievalida"
POINTS_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula min/max global de distancias por especie invasora."
    )
    parser.add_argument("--species-points-csv", required=True,
                        help="Ruta a plantas_invasoras.csv.")
    parser.add_argument("--ref-grids", required=True, nargs="+",
                        help="Lista de ref_grid.tif regionales.")
    parser.add_argument("--output", required=True,
                        help="Ruta de salida CSV con min/max global por especie.")
    parser.add_argument("--validity-mode", choices=["finite", "notnan"], default="finite",
                        help="Criterio de celdas válidas tras reproyección. Default: finite.")
    parser.add_argument("--distance-mode", choices=["kknn_scaled", "raw"], default="kknn_scaled",
                        help="Modo de distancia. Default: kknn_scaled para emular kknn(scale=TRUE).")
    parser.add_argument("--verbose", action="store_true",
                        help="Imprime diagnósticos detallados.")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def validate_inputs(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))


def safe_colname(value: str) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "species"


def make_unique(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = name
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def load_invasive_points(path: Path) -> pd.DataFrame:
    sp_inv = pd.read_csv(path, sep=",", header=0, low_memory=False)

    if sp_inv.shape[1] < 13:
        raise ValueError("El archivo plantas_invasoras.csv no tiene al menos 13 columnas.")

    cols = list(sp_inv.columns)
    cols[11] = "x"
    cols[12] = "y"
    sp_inv.columns = cols

    required = {"x", "y", SPECIES_FIELD}
    missing = required - set(sp_inv.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en plantas_invasoras.csv: {missing}")

    sp_inv = sp_inv.copy()
    sp_inv["x"] = pd.to_numeric(sp_inv["x"], errors="coerce")
    sp_inv["y"] = pd.to_numeric(sp_inv["y"], errors="coerce")
    sp_inv[SPECIES_FIELD] = sp_inv[SPECIES_FIELD].astype(str)
    sp_inv = sp_inv.dropna(subset=["x", "y", SPECIES_FIELD]).copy()

    if sp_inv.empty:
        raise ValueError("No quedaron puntos válidos de especies invasoras después de limpiar x/y.")

    return sp_inv


def reproject_raster_to_epsg4326(src: rasterio.io.DatasetReader) -> tuple[np.ndarray, rasterio.Affine]:
    transform, width, height = calculate_default_transform(
        src.crs, POINTS_CRS, src.width, src.height, *src.bounds
    )
    dst = np.empty((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=transform,
        dst_crs=POINTS_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return dst, transform


def valid_raster_points_dataframe(
    arr: np.ndarray,
    transform: rasterio.Affine,
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
        return pd.DataFrame(columns=["x", "y"])

    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame({"x": np.asarray(xs, dtype=float), "y": np.asarray(ys, dtype=float)})


def kknn_scaled_distances(points_xy: np.ndarray, train_xy: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    if train_xy.shape[0] == 0:
        return np.full(points_xy.shape[0], np.nan, dtype=float), (np.nan, np.nan)

    sd = np.std(train_xy, axis=0, ddof=1)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    tree = cKDTree(train_xy / sd)
    distances, _ = tree.query(points_xy / sd, k=1)
    return distances.astype(float), (float(sd[0]), float(sd[1]))


def raw_distances(points_xy: np.ndarray, train_xy: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    if train_xy.shape[0] == 0:
        return np.full(points_xy.shape[0], np.nan, dtype=float), (np.nan, np.nan)
    tree = cKDTree(train_xy)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float), (1.0, 1.0)


def main() -> None:
    args = parse_args()

    species_points_path = Path(args.species_points_csv)
    ref_grid_paths = [Path(p) for p in args.ref_grids]
    output_path = Path(args.output)

    validate_inputs([species_points_path, *ref_grid_paths])

    sp_inv = load_invasive_points(species_points_path)
    species = list(pd.unique(sp_inv[SPECIES_FIELD]))
    species_cols = make_unique([safe_colname(s) for s in species])

    accum: dict[str, dict[str, object]] = {}
    for sp, sp_col in zip(species, species_cols):
        sp_df = sp_inv[sp_inv[SPECIES_FIELD] == sp]
        train_xy = sp_df[["x", "y"]].to_numpy(dtype=float)
        sd = np.std(train_xy, axis=0, ddof=1)
        sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
        accum[sp] = {
            "species": sp,
            "species_col": sp_col,
            "min": np.inf,
            "max": -np.inf,
            "n_regions": 0,
            "n_pixels": 0,
            "n_points": int(len(sp_df)),
            "sd_x": float(sd[0]),
            "sd_y": float(sd[1]),
        }

    for ref_grid_path in ref_grid_paths:
        with rasterio.open(ref_grid_path) as src:
            if src.crs is None:
                raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
            arr, transform = reproject_raster_to_epsg4326(src)

        region_points = valid_raster_points_dataframe(arr, transform, args.validity_mode)
        points_xy = region_points[["x", "y"]].to_numpy(dtype=float)
        log(f"{ref_grid_path}: puntos válidos reproyectados={len(region_points)}", args.verbose)

        if len(region_points) == 0:
            continue

        for sp in species:
            sp_df = sp_inv[sp_inv[SPECIES_FIELD] == sp]
            train_xy = sp_df[["x", "y"]].to_numpy(dtype=float)
            if args.distance_mode == "kknn_scaled":
                distances, _ = kknn_scaled_distances(points_xy, train_xy)
            elif args.distance_mode == "raw":
                distances, _ = raw_distances(points_xy, train_xy)
            else:
                raise ValueError(f"distance_mode no reconocido: {args.distance_mode}")

            valid = distances[np.isfinite(distances)]
            if valid.size == 0:
                continue
            accum[sp]["min"] = min(float(accum[sp]["min"]), float(np.min(valid)))
            accum[sp]["max"] = max(float(accum[sp]["max"]), float(np.max(valid)))
            accum[sp]["n_pixels"] = int(accum[sp]["n_pixels"]) + int(valid.size)
            accum[sp]["n_regions"] = int(accum[sp]["n_regions"]) + 1

    rows = []
    for sp in species:
        row = dict(accum[sp])
        if row["min"] == np.inf:
            row["min"] = np.nan
        if row["max"] == -np.inf:
            row["max"] = np.nan
        row["validity_mode"] = args.validity_mode
        row["distance_mode"] = args.distance_mode
        rows.append(row)

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("La salida de estadísticos globales debe ser .csv")
    out.to_csv(output_path, index=False)
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
