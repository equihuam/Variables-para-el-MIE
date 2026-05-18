#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401
import pyarrow.parquet as pq


KEY_COLUMNS = ["regionid", "pixid", "x", "y"]

# Estas columnas deben conservarse como variables continuas / numéricas.
# El integrador NO discretiza, NO redondea y NO recodifica estas columnas.
CONTINUOUS_COLUMNS = {
    "erosion",
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
}

# Estas columnas son discretas/categóricas por definición y se escriben como
# texto nullable estable para evitar errores de PyArrow por mezcla string/NaN.
CATEGORICAL_COLUMNS = {
    "NESTB_EDO",
    "tipo_costa",
    "zvh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra múltiples features congruentes por píxel en una tabla master consolidada."
    )
    parser.add_argument(
        "--features-dir",
        required=True,
        help="Directorio raíz que contiene subdirectorios por variable.",
    )
    parser.add_argument(
        "--variables",
        required=True,
        help="Lista separada por comas de variables a integrar, por ejemplo tasa_erosion,corales,tipo_costa",
    )
    parser.add_argument(
        "--regions",
        required=False,
        default=None,
        help="Lista separada por comas de regiones a integrar. Si se omite, se infiere de la primera variable.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida .parquet para la tabla master consolidada.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime diagnóstico de tipos de columnas antes de guardar.",
    )
    return parser.parse_args()


def validate_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el directorio requerido: {path}")


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_region_map(variable_dir: Path) -> dict[str, Path]:
    files = sorted(variable_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron .parquet en {variable_dir}")
    return {path.stem: path for path in files}


def read_parquet_safe(path: Path, label: str, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        print(f"Leyendo {label}: {path}")
    table = pq.read_table(path, use_threads=False)
    df = table.to_pandas()
    if verbose:
        print(f"  -> shape {df.shape}")
    return df


def validate_contract(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"La feature '{label}' no cumple el contrato mínimo. "
            f"Faltan columnas: {missing}"
        )


def get_value_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEY_COLUMNS]


def lightweight_alignment_check(base: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    if len(base) != len(other):
        raise ValueError(
            f"La feature '{label}' no tiene el mismo número de filas que la base: "
            f"base={len(base)}, other={len(other)}"
        )

    if len(base) == 0:
        return

    if base["regionid"].iloc[0] != other["regionid"].iloc[0]:
        raise ValueError(
            f"La feature '{label}' no pertenece a la misma región que la base: "
            f"{base['regionid'].iloc[0]} vs {other['regionid'].iloc[0]}"
        )

    sample_idx = [0, len(base) // 2, len(base) - 1]
    for idx in sample_idx:
        if idx < 0 or idx >= len(base):
            continue

        b = base.iloc[idx]
        o = other.iloc[idx]

        for col in KEY_COLUMNS:
            if b[col] != o[col]:
                raise ValueError(
                    f"Desalineación con '{label}' en fila {idx}, columna {col}: "
                    f"{b[col]} != {o[col]}"
                )


def append_value_columns(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    out = base.copy()
    value_cols = get_value_columns(other)

    if not value_cols:
        raise ValueError(f"La feature '{label}' no tiene columnas temáticas.")

    for col in value_cols:
        if col in out.columns:
            raise ValueError(
                f"La columna '{col}' de la feature '{label}' ya existe en la base."
            )
        out[col] = other[col].to_numpy()

    return out


def reindex_pixid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["regionid", "pixid"]).reset_index(drop=True).copy()
    out["pixid"] = out.groupby("regionid").cumcount() + 1
    return out


def stringify_nullable_value(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_output_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Prepara tipos para Parquet sin discretizar variables continuas.

    - Las columnas continuas conocidas se escriben como numéricas.
    - Las columnas categóricas conocidas se escriben como string nullable.
    - Otras columnas object se intentan convertir a numéricas; si no se puede,
      se escriben como string nullable.
    """
    out = df.copy()

    # Claves: mantener contrato estable.
    if "regionid" in out.columns:
        out["regionid"] = out["regionid"].map(stringify_nullable_value).astype("string")
    if "pixid" in out.columns:
        out["pixid"] = pd.to_numeric(out["pixid"], errors="raise").astype("Int64")
    for col in ["x", "y"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")

    for col in sorted(CONTINUOUS_COLUMNS & set(out.columns)):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    for col in sorted(CATEGORICAL_COLUMNS & set(out.columns)):
        out[col] = out[col].map(stringify_nullable_value).astype("string")

    # Resolver cualquier object restante sin tocar columnas ya tipadas.
    for col in out.select_dtypes(include=["object"]).columns:
        if col in CATEGORICAL_COLUMNS:
            continue
        numeric_candidate = pd.to_numeric(out[col], errors="coerce")
        non_na_original = out[col].notna()
        non_na_numeric = numeric_candidate.notna()
        # Si todos los no-NA son convertibles, mantener como numérico.
        if bool((non_na_numeric[non_na_original]).all()):
            out[col] = numeric_candidate.astype("float64")
        else:
            out[col] = out[col].map(stringify_nullable_value).astype("string")

    if verbose:
        print("Tipos de salida:")
        print(out.dtypes.to_string())

    return out


def save_output(df: pd.DataFrame, output_path: Path, verbose: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Este script requiere salida .parquet. Recibido: {output_path.suffix}")

    df_out = normalize_output_dtypes(df, verbose=verbose)
    df_out.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"OK -> {output_path}")


def main() -> None:
    args = parse_args()

    features_dir = Path(args.features_dir)
    output_path = Path(args.output)
    variables = parse_csv_list(args.variables)
    requested_regions = parse_csv_list(args.regions)

    validate_dir(features_dir)

    if not variables:
        raise ValueError("Debes indicar al menos una variable en --variables")

    if args.verbose:
        print(f"features_dir: {features_dir}")
        print(f"variables: {variables}")
        print(f"output: {output_path}")

    variable_maps: dict[str, dict[str, Path]] = {}

    for var in variables:
        var_dir = features_dir / var
        validate_dir(var_dir)
        variable_maps[var] = build_region_map(var_dir)

    base_var = variables[0]
    base_regions = set(variable_maps[base_var].keys())

    if requested_regions:
        region_names = sorted(requested_regions, key=lambda s: int(s.split("_")[-1]))
    else:
        region_names = sorted(base_regions, key=lambda s: int(s.split("_")[-1]))

    for var in variables:
        available = set(variable_maps[var].keys())
        missing = [r for r in region_names if r not in available]
        if missing:
            raise ValueError(
                f"La variable '{var}' no tiene todas las regiones solicitadas. "
                f"Faltan: {missing}"
            )

    if args.verbose:
        print(f"1) regiones a integrar: {region_names}")

    merged_regions: list[pd.DataFrame] = []

    for region in region_names:
        if args.verbose:
            print(f"\nProcesando región: {region}")

        base = read_parquet_safe(variable_maps[base_var][region], base_var, verbose=args.verbose)
        validate_contract(base, base_var)

        for var in variables[1:]:
            other = read_parquet_safe(variable_maps[var][region], var, verbose=args.verbose)
            validate_contract(other, var)
            lightweight_alignment_check(base, other, var)
            base = append_value_columns(base, other, var)

        base = reindex_pixid(base)
        merged_regions.append(base)

    if args.verbose:
        print("\n2) concatenando regiones...")
    dat = pd.concat(merged_regions, ignore_index=True)
    if args.verbose:
        print(f"  -> shape master: {dat.shape}")

    save_output(dat, output_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
