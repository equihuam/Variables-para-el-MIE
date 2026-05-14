#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 10_wf_movimiento_dunas_basegrid_v2.py

Propósito:
    Asignar, para una región específica, la condición de dunas costeras
    `NESTB_EDO` sobre una malla regional común ya alineada, y exportar el
    resultado como tabla Parquet compatible con el workflow.

Decisión metodológica:
    El script R original de movimiento de dunas construye una malla propia a
    partir de regiones costeras, buffer, raster INEGI y polígonos de dunas. Esa
    malla no coincide con los `ref_grid.tif` regionales usados por el resto del
    workflow. Esta versión adapta la variable al marco regional común del
    proyecto: usa una tabla base con `regionid, pixid, x, y` y asigna
    `NESTB_EDO` por intersección espacial con los polígonos de dunas.

Entrada:
    --dunes-shp    Shapefile de dunas costeras con campo NESTB_EDO.
    --base-table   Tabla regional .parquet con regionid, pixid, x, y.
    --output       Tabla .parquet de salida.

Salida:
    regionid, pixid, x, y, NESTB_EDO

Notas:
    - La tabla base se asume en EPSG:4326, igual que las features validadas
      previamente en el workflow.
    - Si un pixel no intersecta ninguna duna, NESTB_EDO queda como NA.
    - Si un pixel intersecta varias geometrías, se conserva el primer valor no
      nulo en el orden devuelto por el join espacial.
    - `--verbose` habilita conteos diagnósticos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import pyarrow.parquet as pq


KEY_COLUMNS = ["regionid", "pixid", "x", "y"]
OUTPUT_FIELD = "NESTB_EDO"
SOURCE_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asigna NESTB_EDO a una malla base regional del workflow."
    )
    parser.add_argument(
        "--dunes-shp",
        required=True,
        help="Ruta al shapefile de dunas costeras con campo NESTB_EDO.",
    )
    parser.add_argument(
        "--base-table",
        required=True,
        help="Ruta a tabla regional .parquet con regionid, pixid, x, y.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet.",
    )
    parser.add_argument(
        "--source-crs",
        default=SOURCE_CRS,
        help="CRS de las coordenadas x/y de la tabla base. Default: EPSG:4326.",
    )
    parser.add_argument(
        "--debug-summary-output",
        default=None,
        help="CSV opcional con conteos por clase NESTB_EDO.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime diagnósticos de conteo.",
    )
    return parser.parse_args()


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

    # Mantener orden original de la malla.
    out = out.reset_index(drop=True)
    return out


def load_dunes(path: Path) -> gpd.GeoDataFrame:
    dunes = gpd.read_file(path)

    if dunes.empty:
        raise ValueError(f"El shapefile de dunas está vacío: {path}")
    if dunes.crs is None:
        raise ValueError(f"El shapefile de dunas no tiene CRS: {path}")
    if OUTPUT_FIELD not in dunes.columns:
        raise ValueError(f"No existe el campo requerido '{OUTPUT_FIELD}' en {path.name}")

    dunes = dunes[[OUTPUT_FIELD, "geometry"]].copy()
    dunes = dunes[dunes.geometry.notna() & ~dunes.geometry.is_empty].copy()

    if dunes.empty:
        raise ValueError("No quedaron geometrías válidas de dunas después de filtrar.")

    # Asegurar dtype estable para salida Parquet.
    dunes[OUTPUT_FIELD] = dunes[OUTPUT_FIELD].astype("object")
    return dunes


def label_points_with_dune_condition(
    base: pd.DataFrame,
    dunes: gpd.GeoDataFrame,
    source_crs: str = SOURCE_CRS,
) -> pd.Series:
    pts = gpd.GeoDataFrame(
        base.copy(),
        geometry=gpd.points_from_xy(base["x"], base["y"]),
        crs=source_crs,
    )
    pts = pts.to_crs(dunes.crs)

    joined = gpd.sjoin(
        pts,
        dunes,
        how="left",
        predicate="intersects",
    )

    # En caso de múltiples intersecciones, conservar el primer no nulo.
    labeled = (
        joined.groupby(joined.index, sort=False)[OUTPUT_FIELD]
        .agg(lambda s: next((v for v in s if pd.notna(v)), np.nan))
        .reindex(base.index)
    )

    return labeled.astype("object")


def write_summary(out: pd.DataFrame, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts = (
        out[OUTPUT_FIELD]
        .fillna("<NA>")
        .value_counts(dropna=False)
        .rename_axis(OUTPUT_FIELD)
        .reset_index(name="n")
    )
    counts.to_csv(summary_path, index=False, encoding="utf-8")


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

    dunes_path = Path(args.dunes_shp)
    base_table_path = Path(args.base_table)
    output_path = Path(args.output)

    validate_inputs(dunes_path, base_table_path)

    base = read_base_table(base_table_path)
    dunes = load_dunes(dunes_path)

    labeled = label_points_with_dune_condition(
        base=base,
        dunes=dunes,
        source_crs=args.source_crs,
    )

    out = base.copy()
    out[OUTPUT_FIELD] = labeled

    if args.debug_summary_output:
        write_summary(out, Path(args.debug_summary_output))

    if args.verbose:
        n_total = len(out)
        n_labeled = int(out[OUTPUT_FIELD].notna().sum())
        n_na = int(out[OUTPUT_FIELD].isna().sum())
        print(f"filas base: {n_total}")
        print(f"NESTB_EDO no NA: {n_labeled}")
        print(f"NESTB_EDO NA: {n_na}")
        print(out[OUTPUT_FIELD].fillna("<NA>").value_counts(dropna=False).to_string())

    save_output(out, output_path)


if __name__ == "__main__":
    main()
