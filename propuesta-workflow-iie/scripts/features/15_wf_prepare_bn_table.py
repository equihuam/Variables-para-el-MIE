#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 15_wf_prepare_bn_table.py

Propósito:
    Preparar una tabla de entrada para entrenamiento o inferencia de una red
    bayesiana a partir de la base maestra de features congruentes, exportando
    una versión tabular en CSV y una versión trazable en Parquet.

Origen:
    Refactorización para workflow inspirada en:
    - 15_one_class_ei.R
    - 16_final_training_data_v1.R
    - misc_functions.R

Resumen del flujo:
    1. Leer la base maestra de features en Parquet.
    2. Seleccionar y ordenar las columnas requeridas para la red bayesiana.
    3. Convertir variables categóricas a tipo category.
    4. Discretizar variables continuas con 5 intervalos iguales.
    5. Normalizar el nombre de la columna objetivo del índice si existe.
    6. Exportar una tabla preparada en Parquet y un CSV para el motor bayesiano.

Insumos principales:
    - master_features.parquet

Salidas principales:
    - bn_input.parquet
    - bn_input.csv

Supuestos y notas:
    - Se usa Parquet como formato normal de entrada y salida interna.
    - La discretización sigue la lógica de misc_functions.R:
      5 intervalos iguales por variable continua.
    - La columna del índice puede venir como ie, ei o eii; internamente se
      normaliza a ie cuando existe.
    - Este script no entrena la red; solo prepara la tabla de entrada.

Fidelidad de la traducción:
    Traducción orientada a preservar la intención analítica del flujo R.
    La lógica de factorCols() se replica como coerción simple a category y
    la de discretizeCols() como discretización por intervalos iguales.
    Esta versión es más explícita y robusta para workflow que las variantes
    originales basadas en posiciones de columnas.

Observaciones:
    Este script está pensado para ejecución headless y para integrarse en un
    workflow orquestado, por ejemplo con Snakemake.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401


CANONICAL_TARGET = "ie"
TARGET_SYNONYMS = ["ie", "ei", "eii", "ei_qnint"]

KEY_COLUMNS = ["regionid", "pixid", "x", "y"]

CAT_VARS = ["NESTB_EDO", "tipo_costa", "zvh", "CONSERV_ED", "ie"]

CON_VARS = [
    "erosion",
    "movdunas",
    "windspeed",
    "escollera",
    "espigon",
    "muro",
    "rompeolas",
    "puerto",
    "sp_inv_pot",
    "d_corales",
    "d_pastosmarinos",
    "bati_char",
    "d_grassland",
    "d_agriculture",
    "d_urban",
    "p_manglares",
]

DEFAULT_ORDER = [
    "x",
    "y",
    "regionid",
    "pixid",
    "NESTB_EDO",
    "CONSERV_ED",
    "erosion",
    "movdunas",
    "tipo_costa",
    "windspeed",
    "zvh",
    "escollera",
    "espigon",
    "muro",
    "rompeolas",
    "puerto",
    "sp_inv_pot",
    "d_corales",
    "d_pastosmarinos",
    "bati_char",
    "d_grassland",
    "d_agriculture",
    "d_urban",
    "p_manglares",
    "ie",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara la tabla de entrada para la red bayesiana."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al master_features.parquet.",
    )
    parser.add_argument(
        "--output-parquet",
        required=True,
        help="Ruta de salida .parquet para la tabla preparada.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Ruta de salida .csv para la tabla preparada.",
    )
    return parser.parse_args()


def validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe la tabla de entrada: {input_path}")


def canonicalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    present = [c for c in TARGET_SYNONYMS if c in out.columns]
    if not present:
        return out

    if CANONICAL_TARGET in out.columns:
        return out

    source = present[0]
    out = out.rename(columns={source: CANONICAL_TARGET})
    return out


def validate_contract(df: pd.DataFrame) -> None:
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"La tabla maestra no cumple el contrato mínimo. Faltan columnas: {missing}"
        )

    if df.duplicated(KEY_COLUMNS).any():
        dup_count = int(df.duplicated(KEY_COLUMNS).sum())
        raise ValueError(
            f"La tabla maestra contiene {dup_count} duplicados en la llave espacial."
        )


def factor_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def discretize_cols(
        df: pd.DataFrame,
        cols: list[str],
        breaks: int = 5,
        method: str = "interval",
) -> pd.DataFrame:
    """
    Aproximación a misc_functions.R:
      discretizeCols(..., breaks_vec=rep(5,...), method="interval")
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue

        s = pd.to_numeric(out[col], errors="coerce")

        if s.nunique(dropna=True) <= 1:
            out[col] = pd.Series(["1"] * len(s), index=s.index, dtype="category")
            continue

        if method != "interval":
            raise ValueError("Esta traducción usa method='interval' como en misc_functions.R.")

        disc = pd.cut(s, bins=breaks, labels=False, include_lowest=True)
        out[col] = (disc + 1).astype("Int64").astype(str).astype("category")

    return out


def select_and_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DEFAULT_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    return df[cols + remaining].copy()


def save_outputs(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError(f"Salida parquet inválida: {parquet_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Salida csv inválida: {csv_path}")

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    df.to_csv(csv_path, index=False)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_parquet = Path(args.output_parquet)
    output_csv = Path(args.output_csv)

    validate_input(input_path)

    dat = pd.read_parquet(input_path)
    validate_contract(dat)

    dat = canonicalize_target_column(dat)
    dat = select_and_order_columns(dat)

    dat = factor_cols(dat, CAT_VARS)
    dat = discretize_cols(dat, CON_VARS, breaks=5, method="interval")

    save_outputs(dat, output_parquet, output_csv)
    print(f"OK -> {output_parquet}")
    print(f"OK -> {output_csv}")


if __name__ == "__main__":
    main()