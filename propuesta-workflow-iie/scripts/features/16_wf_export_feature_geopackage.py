#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 16_wf_export_feature_geopackage.py

Propósito:
    Exportar un GeoPackage por variable a partir de la colección regional de
    archivos Parquet ya generados en results/features/<variable>/.

Origen:
    Nuevo componente del workflow para inspección SIG y validación espacial.

Resumen del flujo:
    1. Leer todos los Parquet regionales de una variable.
    2. Validar el contrato mínimo de columnas espaciales.
    3. Concatenar las regiones.
    4. Construir geometría puntual a partir de x, y.
    5. Exportar un GeoPackage único por variable.

Insumos principales:
    - directorio de una variable en results/features/<variable>/

Salida principal:
    - un archivo .gpkg por variable

Supuestos y notas:
    - Cada parquet regional contiene regionid, pixid, x, y y una o más columnas temáticas.
    - La geometría puntual se construye en EPSG:4326, que es la convención
      operativa actual de la tabla regional.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


BASE_COLUMNS = ["regionid", "pixid", "x", "y"]
OUTPUT_LAYER = "feature_data"
OUTPUT_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta un GeoPackage por variable a partir de parquets regionales."
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        help="Directorio de la variable, por ejemplo results/features/corales",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta del archivo .gpkg de salida.",
    )
    return parser.parse_args()


def validate_feature_dir(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el directorio de feature: {path}")
    if not path.is_dir():
        raise ValueError(f"La ruta no es un directorio: {path}")

    files = sorted(path.glob("region_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos region_*.parquet en {path}")

    return files


def read_feature_tables(files: list[Path]) -> pd.DataFrame:
    parts = []
    for fp in files:
        df = pd.read_parquet(fp)

        missing = [c for c in BASE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"El archivo {fp} no cumple el contrato mínimo. Faltan columnas: {missing}"
            )

        parts.append(df)

    out = pd.concat(parts, ignore_index=True)

    if out.empty:
        raise ValueError("La concatenación de tablas regionales quedó vacía.")

    return out


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=OUTPUT_CRS,
    )
    return gdf


def main() -> None:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    output_path = Path(args.output)

    files = validate_feature_dir(feature_dir)
    df = read_feature_tables(files)
    gdf = build_geodataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, layer=OUTPUT_LAYER, driver="GPKG")

    print(f"feature_dir: {feature_dir}")
    print(f"archivos regionales: {len(files)}")
    print(f"filas exportadas: {len(gdf)}")
    print(f"OK -> {output_path}")


if __name__ == "__main__":
    main()