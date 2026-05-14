#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 3_wf_spp_invasoras_debug_v1.py

Propósito:
    Calcular, para una región específica, el potencial agregado de especies
    invasoras, emulando la lógica principal del script R 3_sp_invasoras.R:
    distancia al vecino más cercano por especie con kknn(k=1, distance=2,
    kernel='rectangular', scale=TRUE), normalización 1 - normalize(distancia)
    y suma por píxel.

Notas de validación:
    - Por defecto usa distancia kknn escalada, no distancia cruda.
    - La normalización por defecto es regional para facilitar validación por región.
      El script R original normaliza después de unir todas las regiones, por lo que
      para equivalencia global estricta se deben proporcionar estadísticos globales
      por especie mediante --normalization-stats.
    - La salida productiva conserva sólo: regionid, pixid, x, y, sp_inv_pot.
    - Los archivos debug opcionales pueden guardar grilla, metadatos, resumen de
      especies y distancias/scores por especie.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import rasterio
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree


SPECIES_FIELD = "especievalida"
POINTS_CRS = "EPSG:4326"
OUTPUT_FIELD = "sp_inv_pot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula potencial de especies invasoras por píxel para una región específica."
    )
    parser.add_argument("--species-points-csv", required=True,
                        help="Ruta a plantas_invasoras.csv.")
    parser.add_argument("--ref-grid", required=True,
                        help="Ruta al ref_grid.tif de la región.")
    parser.add_argument("--region-id", required=True,
                        help="Identificador de la región, por ejemplo region_1.")
    parser.add_argument("--output", required=True,
                        help="Ruta de salida .parquet.")
    parser.add_argument("--validity-mode", choices=["finite", "notnan"], default="finite",
                        help="Criterio de celdas válidas tras reproyección. Default: finite.")
    parser.add_argument("--distance-mode", choices=["kknn_scaled", "raw"], default="kknn_scaled",
                        help="Modo de distancia. Default: kknn_scaled para emular kknn(scale=TRUE).")
    parser.add_argument("--normalization-mode", choices=["region", "global_stats", "none"], default="region",
                        help=("Normalización de distancias. 'region' normaliza con min/max de la región; "
                              "'global_stats' usa --normalization-stats; 'none' no normaliza y exporta suma de distancias."))
    parser.add_argument("--normalization-stats", default=None,
                        help="CSV con columnas species,min,max para normalización global.")
    parser.add_argument("--debug-grid-output", default=None,
                        help="CSV opcional con grilla reproyectada válida.")
    parser.add_argument("--debug-metadata-output", default=None,
                        help="CSV opcional con metadatos de grilla y ejecución.")
    parser.add_argument("--debug-species-output", default=None,
                        help="CSV opcional con resumen por especie.")
    parser.add_argument("--debug-wide-output", default=None,
                        help="Parquet/CSV opcional con distancias y scores por especie.")
    parser.add_argument("--verbose", action="store_true",
                        help="Imprime diagnósticos detallados.")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def validate_inputs(*paths: Path | None) -> None:
    missing = [str(p) for p in paths if p is not None and not p.exists()]
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
        return pd.DataFrame(columns=["row", "col", "x", "y", "ref_value"])

    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "row": rows.astype(int),
            "col": cols.astype(int),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "ref_value": arr[rows, cols],
        }
    )


def kknn_scaled_distances(points_xy: np.ndarray, train_xy: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Distancia a vecino más cercano emulando kknn(..., scale=TRUE) para k=1.

    kknn escala predictores con la desviación estándar del conjunto de entrenamiento.
    Para k=1 no hay ponderación que calcular: sólo importa la distancia escalada al
    vecino más cercano, que es lo que R guarda en modelkknn$D.
    """
    if train_xy.shape[0] == 0:
        return np.full(points_xy.shape[0], np.nan, dtype=float), (np.nan, np.nan)

    sd = np.std(train_xy, axis=0, ddof=1)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)

    train_scaled = train_xy / sd
    points_scaled = points_xy / sd

    tree = cKDTree(train_scaled)
    distances, _ = tree.query(points_scaled, k=1)
    return distances.astype(float), (float(sd[0]), float(sd[1]))


def raw_distances(points_xy: np.ndarray, train_xy: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    if train_xy.shape[0] == 0:
        return np.full(points_xy.shape[0], np.nan, dtype=float), (np.nan, np.nan)
    tree = cKDTree(train_xy)
    distances, _ = tree.query(points_xy, k=1)
    return distances.astype(float), (1.0, 1.0)


def normalize_values(values: pd.Series, vmin: float | None = None, vmax: float | None = None) -> pd.Series:
    if vmin is None:
        vmin = values.min(skipna=True)
    if vmax is None:
        vmax = values.max(skipna=True)

    if pd.isna(vmin) or pd.isna(vmax):
        return pd.Series(np.nan, index=values.index, dtype=float)
    if vmax == vmin:
        return pd.Series(0.0, index=values.index, dtype=float)
    return (values - vmin) / (vmax - vmin)


def load_normalization_stats(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None:
        return {}
    stats = pd.read_csv(path)
    required = {"species", "min", "max"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"El CSV de normalización requiere columnas {required}; faltan {missing}")
    return {
        str(row["species"]): (float(row["min"]), float(row["max"]))
        for _, row in stats.iterrows()
    }


def compute_species_tables(
    region_points: pd.DataFrame,
    sp_inv: pd.DataFrame,
    distance_mode: str,
    normalization_mode: str,
    normalization_stats: dict[str, tuple[float, float]],
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    species = list(pd.unique(sp_inv[SPECIES_FIELD]))
    species_cols = make_unique([safe_colname(s) for s in species])
    pred_xy = region_points[["x", "y"]].to_numpy(dtype=float)

    wide = pd.DataFrame(index=region_points.index)
    summary_rows: list[dict[str, object]] = []
    score_cols: list[str] = []

    for sp, sp_col in zip(species, species_cols):
        sp_inv_f = sp_inv[sp_inv[SPECIES_FIELD] == sp]
        train_xy = sp_inv_f[["x", "y"]].to_numpy(dtype=float)

        if distance_mode == "kknn_scaled":
            distances, sd_xy = kknn_scaled_distances(pred_xy, train_xy)
        elif distance_mode == "raw":
            distances, sd_xy = raw_distances(pred_xy, train_xy)
        else:
            raise ValueError(f"distance_mode no reconocido: {distance_mode}")

        dist_col = f"dist__{sp_col}"
        score_col = f"score__{sp_col}"
        wide[dist_col] = distances

        dser = pd.Series(distances, index=wide.index)
        if normalization_mode == "region":
            norm = normalize_values(dser)
        elif normalization_mode == "global_stats":
            if sp not in normalization_stats:
                raise ValueError(f"Faltan estadísticos globales para especie: {sp}")
            vmin, vmax = normalization_stats[sp]
            norm = normalize_values(dser, vmin=vmin, vmax=vmax)
        elif normalization_mode == "none":
            norm = dser
        else:
            raise ValueError(f"normalization_mode no reconocido: {normalization_mode}")

        if normalization_mode == "none":
            score = norm
        else:
            score = 1.0 - norm

        wide[score_col] = score.to_numpy(dtype=float)
        score_cols.append(score_col)

        summary_rows.append(
            {
                "species": sp,
                "species_col": sp_col,
                "n_points": int(len(sp_inv_f)),
                "sd_x": sd_xy[0],
                "sd_y": sd_xy[1],
                "distance_min_region": float(np.nanmin(distances)) if len(distances) else np.nan,
                "distance_max_region": float(np.nanmax(distances)) if len(distances) else np.nan,
                "normalization_mode": normalization_mode,
            }
        )
        log(f"{sp}: n={len(sp_inv_f)}, sd=({sd_xy[0]:.8g}, {sd_xy[1]:.8g})", verbose)

    if not score_cols:
        potential = np.full(len(region_points), np.nan, dtype=float)
    else:
        potential = wide[score_cols].sum(axis=1, skipna=True).to_numpy(dtype=float)

    return wide, pd.DataFrame(summary_rows), potential


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Formato no soportado para {path}; use .csv o .parquet")


def main() -> None:
    args = parse_args()

    species_points_path = Path(args.species_points_csv)
    ref_grid_path = Path(args.ref_grid)
    output_path = Path(args.output)
    stats_path = Path(args.normalization_stats) if args.normalization_stats else None
    region_id = str(args.region_id).strip()

    validate_inputs(species_points_path, ref_grid_path, stats_path)
    sp_inv = load_invasive_points(species_points_path)
    normalization_stats = load_normalization_stats(stats_path)

    with rasterio.open(ref_grid_path) as src:
        if src.crs is None:
            raise ValueError(f"El raster de referencia no tiene CRS: {ref_grid_path}")
        src_total = int(src.width * src.height)
        src_arr = src.read(1, masked=True)
        src_valid_masked = int(np.ma.count(src_arr))
        src_bounds = src.bounds
        src_crs = str(src.crs)
        src_nodata = src.nodata
        region_arr, region_transform = reproject_raster_to_epsg4326(src)

    region_points = valid_raster_points_dataframe(region_arr, region_transform, args.validity_mode)
    region_points.insert(0, "pixid", np.arange(1, len(region_points) + 1))
    region_points.insert(0, "regionid", region_id)

    log(f"total puntos GeoTIFF original: {src_total}", args.verbose)
    log(f"puntos válidos GeoTIFF original masked: {src_valid_masked}", args.verbose)
    log(f"total puntos reproyectados: {int(region_arr.size)}", args.verbose)
    log(f"modo de validez: {args.validity_mode}", args.verbose)
    log(f"puntos válidos usados en malla: {len(region_points)}", args.verbose)
    log(f"modo distancia: {args.distance_mode}", args.verbose)
    log(f"modo normalización: {args.normalization_mode}", args.verbose)

    wide, species_summary, potential = compute_species_tables(
        region_points=region_points,
        sp_inv=sp_inv,
        distance_mode=args.distance_mode,
        normalization_mode=args.normalization_mode,
        normalization_stats=normalization_stats,
        verbose=args.verbose,
    )

    out = pd.DataFrame(
        {
            "regionid": region_points["regionid"].to_numpy(),
            "pixid": region_points["pixid"].to_numpy(),
            "x": region_points["x"].to_numpy(dtype=float),
            "y": region_points["y"].to_numpy(dtype=float),
            OUTPUT_FIELD: potential,
        }
    )
    save_table(out, output_path)

    if args.debug_grid_output:
        save_table(region_points[["regionid", "pixid", "row", "col", "x", "y", "ref_value"]], Path(args.debug_grid_output))
        log(f"debug grid -> {args.debug_grid_output}", args.verbose)

    if args.debug_species_output:
        save_table(species_summary, Path(args.debug_species_output))
        log(f"debug species -> {args.debug_species_output}", args.verbose)

    if args.debug_wide_output:
        wide_out = pd.concat(
            [region_points[["regionid", "pixid", "x", "y"]].reset_index(drop=True), wide.reset_index(drop=True)],
            axis=1,
        )
        save_table(wide_out, Path(args.debug_wide_output))
        log(f"debug wide -> {args.debug_wide_output}", args.verbose)

    if args.debug_metadata_output:
        meta = pd.DataFrame([
            {
                "regionid": region_id,
                "src_crs": src_crs,
                "dst_crs": POINTS_CRS,
                "src_total_points": src_total,
                "src_valid_masked": src_valid_masked,
                "src_nodata": src_nodata,
                "src_bounds_left": src_bounds.left,
                "src_bounds_bottom": src_bounds.bottom,
                "src_bounds_right": src_bounds.right,
                "src_bounds_top": src_bounds.top,
                "dst_total_points": int(region_arr.size),
                "dst_valid_points": int(len(region_points)),
                "validity_mode": args.validity_mode,
                "distance_mode": args.distance_mode,
                "normalization_mode": args.normalization_mode,
                "n_species": int(species_summary.shape[0]),
                "x_min": float(region_points["x"].min()) if len(region_points) else np.nan,
                "x_max": float(region_points["x"].max()) if len(region_points) else np.nan,
                "y_min": float(region_points["y"].min()) if len(region_points) else np.nan,
                "y_max": float(region_points["y"].max()) if len(region_points) else np.nan,
            }
        ])
        save_table(meta, Path(args.debug_metadata_output))
        log(f"debug metadata -> {args.debug_metadata_output}", args.verbose)

    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()
