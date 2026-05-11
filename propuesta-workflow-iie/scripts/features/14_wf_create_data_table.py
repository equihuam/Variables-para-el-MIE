#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 14_wf_create_data_table.py

Propósito:
    Integrar tablas de features congruentes por píxel en una sola tabla maestra
    consolidada, preservando las llaves espaciales y exportando el resultado
    en formato Parquet.

Estrategia:
    1. Detectar regiones disponibles en los directorios de features.
    2. Leer por pares regionales (erosión + corales).
    3. Validar esquema y alineación ligera.
    4. Combinar por posición dentro de cada región.
    5. Reindexar pixid por región.
    6. Concatenar todas las regiones.
    7. Escribir master_features.parquet.

Notas:
    - Esta versión asume que las features regionales ya fueron corregidas para
      conservar solo celdas válidas de la malla útil.
    - La combinación por posición se apoya en que ambas features provienen de la
      misma malla regional y mismo orden de filas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401
import pyarrow.parquet as pq


KEY_COLUMNS = ["regionid", "pixid", "x", "y"]
CORALES_VALUE_COL = "corals"
EROSION_VALUE_COL = "erosion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra features congruentes por píxel en una tabla master consolidada."
    )
    parser.add_argument(
        "--erosion-dir",
        required=True,
        help="Directorio con archivos regionales .parquet de erosión.",
    )
    parser.add_argument(
        "--corales-dir",
        required=True,
        help="Directorio con archivos regionales .parquet de corales.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet para la tabla master consolidada.",
    )
    return parser.parse_args()


def validate_input_dirs(*dirs: Path) -> None:
    missing = [str(d) for d in dirs if not d.exists()]
    if missing:
        raise FileNotFoundError("Faltan directorios de entrada:\n" + "\n".join(missing))


def list_parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .parquet en {directory}")
    return files


def build_region_map(directory: Path) -> dict[str, Path]:
    files = list_parquet_files(directory)
    region_map: dict[str, Path] = {}
    for path in files:
        region_map[path.stem] = path
    return region_map


def validate_columns(df: pd.DataFrame, feature_name: str, value_col: str) -> None:
    required = KEY_COLUMNS + [value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"La feature '{feature_name}' no cumple el contrato mínimo. "
            f"Faltan columnas: {missing}"
        )


def read_parquet_safe(path: Path, feature_name: str) -> pd.DataFrame:
    print(f"Leyendo {feature_name}: {path}")
    table = pq.read_table(path, use_threads=False)
    df = table.to_pandas()
    print(f"  -> shape {df.shape}")
    return df


def lightweight_alignment_check(erosion: pd.DataFrame, corales: pd.DataFrame) -> None:
    if len(erosion) != len(corales):
        raise ValueError(
            f"Las features no tienen el mismo número de filas: "
            f"erosion={len(erosion)}, corales={len(corales)}"
        )

    if len(erosion) == 0:
        return

    if erosion["regionid"].iloc[0] != corales["regionid"].iloc[0]:
        raise ValueError(
            f"Las features no pertenecen a la misma región: "
            f"{erosion['regionid'].iloc[0]} vs {corales['regionid'].iloc[0]}"
        )

    sample_idx = [0, len(erosion) // 2, len(erosion) - 1]
    for idx in sample_idx:
        if idx < 0 or idx >= len(erosion):
            continue

        e = erosion.iloc[idx]
        c = corales.iloc[idx]

        for col in KEY_COLUMNS:
            if e[col] != c[col]:
                raise ValueError(
                    f"Desalineación detectada en fila {idx}, columna {col}: "
                    f"{e[col]} != {c[col]}"
                )


def combine_by_position(erosion: pd.DataFrame, corales: pd.DataFrame) -> pd.DataFrame:
    out = erosion[KEY_COLUMNS + [EROSION_VALUE_COL]].copy()
    out[CORALES_VALUE_COL] = corales[CORALES_VALUE_COL].to_numpy()
    return out


def reindex_pixid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["regionid", "pixid"]).reset_index(drop=True).copy()
    out["pixid"] = out.groupby("regionid").cumcount() + 1
    return out


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {output_path.suffix}")

    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()

    erosion_dir = Path(args.erosion_dir)
    corales_dir = Path(args.corales_dir)
    output_path = Path(args.output)

    print(f"erosion_dir: {erosion_dir}")
    print(f"corales_dir: {corales_dir}")
    print(f"output: {output_path}")

    validate_input_dirs(erosion_dir, corales_dir)
    print("1) directorios validados")

    erosion_map = build_region_map(erosion_dir)
    corales_map = build_region_map(corales_dir)

    erosion_regions = set(erosion_map.keys())
    corales_regions = set(corales_map.keys())

    if erosion_regions != corales_regions:
        missing_in_corales = sorted(erosion_regions - corales_regions)
        missing_in_erosion = sorted(corales_regions - erosion_regions)
        raise ValueError(
            "Las regiones disponibles no coinciden entre features.\n"
            f"Faltan en corales: {missing_in_corales}\n"
            f"Faltan en erosion: {missing_in_erosion}"
        )

    region_names = sorted(erosion_regions, key=lambda s: int(s.split("_")[-1]))
    print(f"2) regiones detectadas: {region_names}")

    merged_regions: list[pd.DataFrame] = []

    for region in region_names:
        print(f"\nProcesando región: {region}")

        erosion = read_parquet_safe(erosion_map[region], "erosion")
        validate_columns(erosion, "erosion", EROSION_VALUE_COL)
        print("  -> esquema erosión OK")

        corales = read_parquet_safe(corales_map[region], "corales")
        validate_columns(corales, "corales", CORALES_VALUE_COL)
        print("  -> esquema corales OK")

        lightweight_alignment_check(erosion, corales)
        print("  -> alineación ligera OK")

        dat_region = combine_by_position(erosion, corales)
        print(f"  -> combinación regional: {dat_region.shape}")

        dat_region = reindex_pixid(dat_region)
        print(f"  -> reindexado regional: {dat_region.shape}")

        merged_regions.append(dat_region)

        del erosion
        del corales
        del dat_region

    print("\n3) concatenando regiones...")
    dat = pd.concat(merged_regions, ignore_index=True)
    print(f"  -> shape master: {dat.shape}")

    save_output(dat, output_path)


if __name__ == "__main__":
    main()