#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nombre: 15_wf_prepare_bn_table.py

Propósito:
    Preparar una tabla de entrada para Netica / red bayesiana a partir de
    master_features.parquet, preservando variables continuas y categóricas sin
    discretización automática.

Decisiones canónicas:
    - No discretiza variables continuas.
    - Conserva categorías reales como texto/categoría nullable.
    - Conserva `ei_qnint` como variable observada de condición inicial, no la
      renombra a `ie`.
    - Sólo normaliza a `ie` columnas objetivo explícitas `ie`, `ei` o `eii`
      cuando existan.
    - Escribe CSV para Netica con `*` como marcador de dato faltante.
    - Entrecomilla campos alfanuméricos en el CSV, dejando numéricos sin comillas.
    - Escribe Parquet interno con NA nativos, sin sustituirlos por `*`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401


CANONICAL_TARGET = "ie"
# Importante: `ei_qnint` NO debe estar aquí. Es variable observada/categórica.
TARGET_SYNONYMS = ["ie", "ei", "eii"]

KEY_COLUMNS = ["regionid", "pixid", "x", "y"]

# Variables categóricas reales que deben conservarse como etiquetas/estados.
CAT_VARS = [
    "regionid",
    "NESTB_EDO",
    "tipo_costa",
    "zvh",
    "ei_qnint",
    "CONSERV_ED",  # respaldo transitorio por si aparece en alguna tabla vieja
]

# Variables continuas esperadas. Se fuerzan a numéricas si existen, pero no se
# discretizan ni redondean.
CONTINUOUS_VARS = [
    "x",
    "y",
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
    CANONICAL_TARGET,
]

DEFAULT_ORDER = [
    "x",
    "y",
    "regionid",
    "pixid",
    "NESTB_EDO",
    "ei_qnint",
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

NETICA_MISSING = "*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara la tabla de entrada para Netica preservando continuas."
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
        help="Ruta de salida .csv para Netica.",
    )
    parser.add_argument(
        "--missing-marker",
        default=NETICA_MISSING,
        help="Marcador de datos faltantes para CSV de Netica. Por defecto: '*'.",
    )
    parser.add_argument(
        "--csv-quoting",
        choices=["nonnumeric", "minimal", "all"],
        default="nonnumeric",
        help=(
            "Modo de entrecomillado del CSV. "
            "'nonnumeric' entrecomilla alfanuméricos y deja números sin comillas; "
            "es el valor recomendado para Netica."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime resumen de tipos y faltantes.",
    )
    return parser.parse_args()


def validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe la tabla de entrada: {input_path}")


def canonicalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra una columna objetivo explícita a `ie` cuando aplique.

    `ei_qnint` se excluye deliberadamente porque representa la condición
    inicial observada de dunas, no la predicción/objetivo final de IE.
    """
    out = df.copy()

    if CANONICAL_TARGET in out.columns:
        return out

    present = [c for c in TARGET_SYNONYMS if c in out.columns]
    if not present:
        return out

    source = present[0]
    if source != CANONICAL_TARGET:
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


def clean_integer_like_string(value):
    """Convierte categorías a strings estables preservando faltantes.

    Ejemplos:
      1.0 -> "1"
      15.0 -> "15"
      "Arena o grava" -> "Arena o grava"
      NaN -> pd.NA
    """
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>", ""}:
        return pd.NA
    # Limpieza conservadora de strings que llegan como "14.0".
    try:
        f = float(text)
        if f.is_integer() and text.replace(".", "", 1).replace("-", "", 1).isdigit():
            return str(int(f))
    except ValueError:
        pass
    return text


def coerce_categorical_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(clean_integer_like_string).astype("string")
    return out


def coerce_continuous_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def coerce_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pixid" in out.columns:
        out["pixid"] = pd.to_numeric(out["pixid"], errors="raise").astype("int64")
    if "regionid" in out.columns:
        out["regionid"] = out["regionid"].map(clean_integer_like_string).astype("string")
    return out


def normalize_unexpected_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte object no declarado a string nullable para evitar fallas Arrow.

    No se aplica a columnas numéricas ya coercionadas.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].map(clean_integer_like_string).astype("string")
    return out


def select_and_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DEFAULT_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    return df[cols + remaining].copy()


def prepare_bn_table(df: pd.DataFrame) -> pd.DataFrame:
    out = canonicalize_target_column(df)
    out = select_and_order_columns(out)
    out = coerce_key_columns(out)
    out = coerce_categorical_columns(out, CAT_VARS)
    out = coerce_continuous_columns(out, CONTINUOUS_VARS)
    out = normalize_unexpected_object_columns(out)
    return out


def dataframe_for_netica_csv(df: pd.DataFrame, missing_marker: str) -> pd.DataFrame:
    """Crea una copia para CSV con faltantes marcados como `*`.

    El Parquet interno conserva NA nativo; sólo el CSV reemplaza NA por el
    marcador requerido por Netica.
    """
    out = df.copy()

    # Asegurar categorías como texto limpio antes de fillna.
    # Nota: pandas deprecó pd.api.types.is_categorical_dtype; usar
    # isinstance(dtype, pd.CategoricalDtype) evita DeprecationWarning.
    for col in out.columns:
        dtype = out[col].dtype
        if pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            out[col] = out[col].astype("string")

    return out.fillna(missing_marker)


def csv_quoting_constant(mode: str) -> int:
    if mode == "nonnumeric":
        return csv.QUOTE_NONNUMERIC
    if mode == "minimal":
        return csv.QUOTE_MINIMAL
    if mode == "all":
        return csv.QUOTE_ALL
    raise ValueError(f"Modo de quoting CSV no soportado: {mode}")


def save_outputs(
        df: pd.DataFrame,
        parquet_path: Path,
        csv_path: Path,
        missing_marker: str,
        csv_quoting: str,
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError(f"Salida parquet inválida: {parquet_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Salida csv inválida: {csv_path}")

    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    csv_df = dataframe_for_netica_csv(df, missing_marker=missing_marker)
    csv_df.to_csv(
        csv_path,
        index=False,
        na_rep=missing_marker,
        quoting=csv_quoting_constant(csv_quoting),
    )


def print_summary(df: pd.DataFrame) -> None:
    print("Tipos finales:")
    print(df.dtypes.to_string())
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nFaltantes por columna:")
        print(missing.to_string())
    else:
        print("\nSin faltantes detectados.")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_parquet = Path(args.output_parquet)
    output_csv = Path(args.output_csv)

    validate_input(input_path)

    dat = pd.read_parquet(input_path)
    validate_contract(dat)

    dat = prepare_bn_table(dat)

    if args.verbose:
        print_summary(dat)

    save_outputs(
        dat,
        output_parquet,
        output_csv,
        missing_marker=args.missing_marker,
        csv_quoting=args.csv_quoting,
    )
    print(f"OK -> {output_parquet}")
    print(f"OK -> {output_csv}")


if __name__ == "__main__":
    main()
