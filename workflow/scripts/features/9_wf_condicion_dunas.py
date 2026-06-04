#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 9_wf_condicion_dunas.py

Propósito:
    Asignar, para una región específica, el estimador inicial observado de
    condición de dunas a partir del campo CONSERV_ED de la cartografía de
    dunas costeras, exportándolo con el nombre canónico ei_qnint.

Decisión metodológica:
    El script R original de condición de dunas construye una malla propia a
    partir de regiones costeras, buffer, raster INEGI y polígonos de dunas. Para
    integrarlo al workflow headless actual, esta versión adopta la malla regional
    común ya validada (`regionid`, `pixid`, `x`, `y`) y asigna el atributo
    `CONSERV_ED` mediante intersección espacial con los polígonos de dunas y
    lo publica como `ei_qnint` para integrarlo a la tabla BN.

Insumos:
    - shapefile de dunas costeras con campo CONSERV_ED
    - tabla base regional alineada en Parquet con columnas regionid, pixid, x, y

Salida:
    - Parquet con columnas regionid, pixid, x, y, ei_qnint
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
SOURCE_FIELD = "CONSERV_ED"
OUTPUT_FIELD = "ei_qnint"
SOURCE_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asigna CONSERV_ED como ei_qnint usando una tabla base regional común."
    )
    parser.add_argument(
        "--dunes-shp",
        required=True,
        help="Ruta al shapefile de dunas costeras con campo CONSERV_ED.",
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
        "--field",
        default=SOURCE_FIELD,
        help="Campo del shapefile a asignar. Por defecto: CONSERV_ED.",
    )
    parser.add_argument(
        "--output-field",
        default=OUTPUT_FIELD,
        help="Nombre de la columna de salida. Por defecto: ei_qnint.",
    )
    parser.add_argument(
        "--source-crs",
        default=SOURCE_CRS,
        help="CRS de las coordenadas x/y de la tabla base. Por defecto: EPSG:4326.",
    )
    parser.add_argument(
        "--debug-summary-output",
        default=None,
        help="CSV opcional con conteos por categoría.",
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

    return out


def load_dunes(path: Path, field: str) -> gpd.GeoDataFrame:
    dunes = gpd.read_file(path)

    if dunes.empty:
        raise ValueError(f"El shapefile de dunas está vacío: {path}")
    if dunes.crs is None:
        raise ValueError(f"El shapefile de dunas no tiene CRS: {path}")
    if field not in dunes.columns:
        raise ValueError(f"No existe el campo requerido '{field}' en {path.name}")

    out = dunes[[field, "geometry"]].copy()
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    return out


def first_non_null(values: pd.Series):
    for value in values:
        if pd.notna(value):
            return value
    return pd.NA


def stringify_nullable(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def label_points_with_dune_condition(
    base: pd.DataFrame,
    dunes: gpd.GeoDataFrame,
    field: str,
    source_crs: str,
) -> pd.Series:
    pts = gpd.GeoDataFrame(
        base.reset_index(drop=True).copy(),
        geometry=gpd.points_from_xy(base["x"], base["y"]),
        crs=source_crs,
    )
    pts["__rowid"] = np.arange(len(pts), dtype=np.int64)

    pts = pts.to_crs(dunes.crs)

    joined = gpd.sjoin(
        pts,
        dunes[[field, "geometry"]],
        how="left",
        predicate="intersects",
    )

    labeled = (
        joined.groupby("__rowid", sort=False)[field]
        .agg(first_non_null)
        .reindex(np.arange(len(base), dtype=np.int64))
    )

    return labeled.map(stringify_nullable).astype("string")


def build_summary(df: pd.DataFrame, field: str) -> pd.DataFrame:
    counts = df[field].value_counts(dropna=False).rename_axis(field).reset_index(name="n")
    counts[field] = counts[field].astype("string")
    return counts


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {output_path.suffix}")

    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()

    dunes_path = Path(args.dunes_shp)
    base_table_path = Path(args.base_table)
    output_path = Path(args.output)
    field = str(args.field).strip()
    output_field = str(args.output_field).strip()

    validate_inputs(dunes_path, base_table_path)

    base = read_base_table(base_table_path)
    dunes = load_dunes(dunes_path, field)

    labeled = label_points_with_dune_condition(
        base=base,
        dunes=dunes,
        field=field,
        source_crs=args.source_crs,
    )

    out = base.copy()
    out[output_field] = labeled

    if args.verbose:
        non_na = int(out[output_field].notna().sum())
        na_count = int(out[output_field].isna().sum())
        print(f"filas base: {len(out)}")
        print(f"{output_field} no NA: {non_na}")
        print(f"{output_field} NA: {na_count}")
        print(out[output_field].value_counts(dropna=False).to_string())

    if args.debug_summary_output:
        summary_path = Path(args.debug_summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        build_summary(out, output_field).to_csv(summary_path, index=False)

    save_output(out[[*KEY_COLUMNS, output_field]], output_path)


if __name__ == "__main__":
    main()
