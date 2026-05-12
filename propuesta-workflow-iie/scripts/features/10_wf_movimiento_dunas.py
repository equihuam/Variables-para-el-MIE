#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 10_wf_movimiento_dunas.py

Propósito:
    Asignar, para una región específica, la condición de las dunas a partir del
    atributo NESTB_EDO de la cartografía de dunas costeras y exportar el
    resultado como tabla congruente por píxel en formato Parquet.

Origen:
    Adaptación a workflow del script R original:
    10_movimiento_dunas.R

Resumen del flujo:
    1. Leer el shapefile de dunas costeras.
    2. Leer una tabla base regional ya alineada.
    3. Verificar el contrato mínimo de la tabla base.
    4. Transformar las coordenadas x, y de la tabla base al CRS de las dunas.
    5. Etiquetar cada punto base con el atributo NESTB_EDO según su intersección
       espacial con los polígonos de dunas.
    6. Resolver múltiples intersecciones conservando un único valor por punto.
    7. Exportar la tabla regional en Parquet conservando exactamente la misma
       malla tabular de la base.

Insumos principales:
    - shapefile de dunas costeras con campo NESTB_EDO
    - tabla base regional alineada

Salida principal:
    - tabla .parquet con columnas:
      regionid, pixid, x, y, NESTB_EDO

Supuestos y notas:
    - La malla canónica se toma de la tabla base regional.
    - La asignación es espacial por intersección punto-polígono.
    - Si un punto no intersecta ninguna duna, NESTB_EDO queda como NA.
    - Si un punto intersecta varias dunas, se conserva el primer valor no nulo.

Observaciones:
    - Este script está diseñado para integrarse en un workflow Snakemake.
    - La ejecución es por región y con rutas parametrizadas.
    - La salida es compatible con el contrato mínimo del proyecto para tablas
      de features congruentes por píxel.
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
        description="Asigna NESTB_EDO usando una tabla base regional."
    )
    parser.add_argument(
        "--dunes-shp",
        required=True,
        help="Ruta al shapefile de dunas costeras.",
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


def load_dunes(path: Path) -> gpd.GeoDataFrame:
    dunes = gpd.read_file(path)

    if dunes.empty:
        raise ValueError(f"El shapefile de dunas está vacío: {path}")
    if dunes.crs is None:
        raise ValueError(f"El shapefile de dunas no tiene CRS: {path}")
    if OUTPUT_FIELD not in dunes.columns:
        raise ValueError(f"No existe el campo requerido '{OUTPUT_FIELD}' en {path.name}")

    return dunes[[OUTPUT_FIELD, "geometry"]].copy()


def label_points_with_dune_condition(base: pd.DataFrame, dunes: gpd.GeoDataFrame) -> pd.Series:
    pts = gpd.GeoDataFrame(
        base.copy(),
        geometry=gpd.points_from_xy(base["x"], base["y"]),
        crs=SOURCE_CRS,
    )

    pts = pts.to_crs(dunes.crs)

    joined = gpd.sjoin(
        pts,
        dunes,
        how="left",
        predicate="intersects",
    )

    labeled = (
        joined.groupby(joined.index, sort=False)[OUTPUT_FIELD]
        .agg(lambda s: next((v for v in s if pd.notna(v)), np.nan))
        .reindex(base.index)
    )

    return labeled.astype("object")


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
    print(f"filas en base regional: {len(base)}")

    dunes = load_dunes(dunes_path)
    labeled = label_points_with_dune_condition(base, dunes)

    out = base.copy()
    out[OUTPUT_FIELD] = labeled

    save_output(out, output_path)


if __name__ == "__main__":
    main()